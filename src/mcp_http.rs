// Optional authenticated Streamable HTTP transport for the same read-only MCP
// dispatcher as `synty mcp` stdio. The default bind is loopback; wider binds
// require an explicit flag because the corpus can contain source and tool data.

use anyhow::{Result, bail};

#[cfg(feature = "mcp-http")]
use crate::mcp;
#[cfg(feature = "mcp-http")]
use serde_json::{Value, json};
#[cfg(feature = "mcp-http")]
use std::io::Read;
#[cfg(feature = "mcp-http")]
use tiny_http::{Header, Method, Request, Response, Server as HttpServer, StatusCode};

#[cfg(feature = "mcp-http")]
pub fn run(
    model_id: String,
    bind: &str,
    token: &str,
    listen_public: bool,
    role: crate::policy::McpRole,
    scope: crate::policy::ReadScope,
    redaction: crate::redact::Profile,
) -> Result<()> {
    if token.is_empty() {
        bail!("HTTP MCP requires --token or SYNTY_MCP_TOKEN");
    }
    if !listen_public && !is_loopback_bind(bind) {
        bail!("refusing non-loopback bind {bind}; pass --listen-public behind TLS");
    }
    let server = HttpServer::http(bind).map_err(|e| anyhow::anyhow!("bind {bind}: {e}"))?;
    let mut dispatcher = mcp::Server::new(model_id, role, scope, redaction);
    eprintln!("synty mcp: serving authenticated HTTP at http://{bind}/mcp");
    for request in server.incoming_requests() {
        handle(request, &mut dispatcher, token)?;
    }
    Ok(())
}

#[cfg(not(feature = "mcp-http"))]
pub fn run(
    _model_id: String,
    _bind: &str,
    _token: &str,
    _listen_public: bool,
    _role: crate::policy::McpRole,
    _scope: crate::policy::ReadScope,
    _redaction: crate::redact::Profile,
) -> Result<()> {
    bail!("HTTP MCP is not in this binary; rebuild with --features mcp-http")
}

#[cfg(feature = "mcp-http")]
fn handle(mut request: Request, dispatcher: &mut mcp::Server, token: &str) -> Result<()> {
    let path = request.url().split('?').next().unwrap_or(request.url());
    if request.method() == &Method::Get && path == "/health" {
        let body = json!({
            "status": "ok",
            "name": "synty",
            "version": env!("CARGO_PKG_VERSION"),
        })
        .to_string();
        return respond(request, StatusCode(200), body, true);
    }
    if path != "/mcp" {
        return respond(request, StatusCode(404), "not found".into(), false);
    }
    if request.method() != &Method::Post {
        return respond(request, StatusCode(405), "method not allowed".into(), false);
    }
    if !authorized(&request, token) {
        return respond(request, StatusCode(401), "unauthorized".into(), false);
    }
    if !origin_allowed(&request) {
        return respond(request, StatusCode(403), "origin forbidden".into(), false);
    }
    let mut body = String::new();
    request.as_reader().take(4 * 1024 * 1024).read_to_string(&mut body)?;
    let value: Value = match serde_json::from_str(&body) {
        Ok(value) => value,
        Err(error) => {
            return respond(
                request,
                StatusCode(400),
                json!({"error": format!("invalid JSON: {error}")}).to_string(),
                true,
            );
        }
    };
    match dispatcher.handle(&value) {
        Some(response) => respond(request, StatusCode(200), response.to_string(), true),
        None => respond(request, StatusCode(202), String::new(), false),
    }
}

#[cfg(feature = "mcp-http")]
fn respond(request: Request, status: StatusCode, body: String, json_body: bool) -> Result<()> {
    let mut response = Response::from_string(body).with_status_code(status);
    if json_body {
        response.add_header(
            Header::from_bytes("Content-Type", "application/json")
                .map_err(|_| anyhow::anyhow!("invalid response header"))?,
        );
    }
    request.respond(response)?;
    Ok(())
}

#[cfg(feature = "mcp-http")]
fn header<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
    let want = name.to_ascii_lowercase();
    request.headers().iter().find_map(|header| {
        let field = header.field.as_str().to_ascii_lowercase();
        (field == want).then_some(header.value.as_str())
    })
}

#[cfg(feature = "mcp-http")]
fn authorized(request: &Request, expected: &str) -> bool {
    let Some(actual) = header(request, "Authorization").and_then(|value| value.strip_prefix("Bearer "))
    else {
        return false;
    };
    constant_time_eq(actual.as_bytes(), expected.as_bytes())
}

#[cfg(feature = "mcp-http")]
fn origin_allowed(request: &Request) -> bool {
    header(request, "Origin").is_none_or(|origin| {
        origin == "null"
            || origin.starts_with("http://127.0.0.1:")
            || origin.starts_with("http://localhost:")
            || origin.starts_with("https://")
    })
}

#[cfg(any(feature = "mcp-http", test))]
fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |difference, (a, b)| difference | (a ^ b))
        == 0
}

#[cfg(any(feature = "mcp-http", test))]
fn is_loopback_bind(bind: &str) -> bool {
    let host = bind.rsplit_once(':').map(|(host, _)| host).unwrap_or(bind);
    matches!(host.trim_matches(['[', ']']), "127.0.0.1" | "::1" | "localhost")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_comparison_and_loopback_guard_fail_closed() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"prefix"));
        assert!(!constant_time_eq(b"short", b"longer"));
        assert!(is_loopback_bind("127.0.0.1:8765"));
        assert!(is_loopback_bind("[::1]:8765"));
        assert!(!is_loopback_bind("0.0.0.0:8765"));
    }
}
