// Optional authenticated Streamable HTTP transport for the same read-only MCP
// dispatcher as `synty mcp` stdio. The default bind is loopback; wider binds
// require an explicit flag because the corpus can contain source and tool data.

use anyhow::{Result, bail};

#[cfg(feature = "mcp-http")]
use crate::mcp;
#[cfg(feature = "mcp-http")]
use serde_json::{Value, json};
#[cfg(any(feature = "mcp-http", test))]
use std::collections::HashMap;
#[cfg(feature = "mcp-http")]
use std::io::Read;
#[cfg(feature = "mcp-http")]
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc::{self, RecvTimeoutError, SyncSender, TrySendError},
};
#[cfg(any(feature = "mcp-http", test))]
use std::time::{Duration, Instant};
#[cfg(feature = "mcp-http")]
use tiny_http::{Header, Method, Request, Response, Server as HttpServer, StatusCode};

#[cfg(feature = "mcp-http")]
const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;
#[cfg(feature = "mcp-http")]
const MAX_IN_FLIGHT: usize = 32;
#[cfg(feature = "mcp-http")]
const DISPATCH_QUEUE: usize = 8;
#[cfg(feature = "mcp-http")]
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
#[cfg(any(feature = "mcp-http", test))]
const RATE_WINDOW: Duration = Duration::from_secs(60);
#[cfg(any(feature = "mcp-http", test))]
const RATE_REQUESTS: u32 = 120;

#[cfg(feature = "mcp-http")]
struct DispatchJob {
    request: Value,
    response: SyncSender<Option<Value>>,
    cancelled: Arc<AtomicBool>,
}

#[cfg(feature = "mcp-http")]
#[derive(Clone)]
struct Dispatcher {
    sender: SyncSender<DispatchJob>,
}

#[cfg(feature = "mcp-http")]
#[derive(Debug)]
enum DispatchError {
    Busy,
    Timeout,
    Stopped,
}

#[cfg(feature = "mcp-http")]
impl Dispatcher {
    fn start(name: &str, mut server: mcp::Server) -> Result<Self> {
        let (sender, receiver) = mpsc::sync_channel::<DispatchJob>(DISPATCH_QUEUE);
        std::thread::Builder::new()
            .name(format!("synty-mcp-{name}"))
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    if job.cancelled.load(Ordering::Acquire) {
                        continue;
                    }
                    let response = server.handle(&job.request);
                    let _ = job.response.send(response);
                }
            })
            .map_err(|error| anyhow::anyhow!("start {name} MCP dispatcher: {error}"))?;
        Ok(Self { sender })
    }

    fn call(&self, request: Value) -> std::result::Result<Option<Value>, DispatchError> {
        let (response, result) = mpsc::sync_channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let job = DispatchJob { request, response, cancelled: Arc::clone(&cancelled) };
        match self.sender.try_send(job) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => return Err(DispatchError::Busy),
            Err(TrySendError::Disconnected(_)) => return Err(DispatchError::Stopped),
        }
        match result.recv_timeout(REQUEST_TIMEOUT) {
            Ok(response) => Ok(response),
            Err(RecvTimeoutError::Timeout) => {
                cancelled.store(true, Ordering::Release);
                Err(DispatchError::Timeout)
            }
            Err(RecvTimeoutError::Disconnected) => Err(DispatchError::Stopped),
        }
    }
}

#[cfg(any(feature = "mcp-http", test))]
#[derive(Default)]
struct RateLimiter {
    clients: HashMap<String, RateWindow>,
}

#[cfg(any(feature = "mcp-http", test))]
struct RateWindow {
    started: Instant,
    requests: u32,
}

#[cfg(any(feature = "mcp-http", test))]
impl RateLimiter {
    fn allow(&mut self, client: &str, now: Instant) -> bool {
        self.clients.retain(|_, window| now.duration_since(window.started) < RATE_WINDOW);
        let window = self.clients.entry(client.to_string()).or_insert(RateWindow {
            started: now,
            requests: 0,
        });
        if now.duration_since(window.started) >= RATE_WINDOW {
            window.started = now;
            window.requests = 0;
        }
        if window.requests >= RATE_REQUESTS {
            return false;
        }
        window.requests += 1;
        true
    }
}

#[cfg_attr(not(feature = "mcp-http"), allow(dead_code))]
pub struct Opts {
    pub model_id: String,
    pub bind: String,
    pub token: String,
    pub listen_public: bool,
    pub role: crate::policy::McpRole,
    pub scope: crate::policy::ReadScope,
    pub redaction: crate::redact::Profile,
    pub allowed_origins: Vec<String>,
    pub bucket: Option<String>,
}

#[cfg(feature = "mcp-http")]
pub fn run(opts: Opts) -> Result<()> {
    anyhow::ensure!(opts.token.len() >= 32, "HTTP MCP bearer token must contain at least 32 bytes");
    if !opts.listen_public && !is_loopback_bind(&opts.bind) {
        bail!("refusing non-loopback bind {}; pass --listen-public behind TLS", opts.bind);
    }
    let server = HttpServer::http(&opts.bind).map_err(|e| anyhow::anyhow!("bind {}: {e}", opts.bind))?;
    let require_read_model = opts.bucket.is_some();
    mcp::start_bucket_refresh(opts.bucket.clone());
    let dispatcher = Dispatcher::start("semantic", mcp::Server::new(
        opts.model_id.clone(),
        opts.role,
        opts.scope.clone(),
        opts.redaction,
        false,
        opts.bucket.clone(),
    ))?;
    // Raw-derived tools may scan a large local history. Keep them serialized,
    // but on a separate dispatcher so they cannot block semantic search.
    let raw_dispatcher = Dispatcher::start("raw", mcp::Server::new(
        opts.model_id, opts.role, opts.scope, opts.redaction, false, opts.bucket,
    ))?;
    let token = Arc::new(opts.token);
    let allowed_origins = Arc::new(opts.allowed_origins);
    let in_flight = Arc::new(AtomicUsize::new(0));
    let rate_limiter = Arc::new(Mutex::new(RateLimiter::default()));
    eprintln!("synty mcp: serving authenticated HTTP at http://{}/mcp", opts.bind);
    for request in server.incoming_requests() {
        if in_flight.fetch_add(1, Ordering::AcqRel) >= MAX_IN_FLIGHT {
            in_flight.fetch_sub(1, Ordering::AcqRel);
            let _ = respond(request, StatusCode(503), "server busy".into(), false);
            continue;
        }
        let dispatcher = dispatcher.clone();
        let raw_dispatcher = raw_dispatcher.clone();
        let token = Arc::clone(&token);
        let allowed_origins = Arc::clone(&allowed_origins);
        let in_flight = Arc::clone(&in_flight);
        let rate_limiter = Arc::clone(&rate_limiter);
        std::thread::spawn(move || {
            if let Err(error) = handle(
                request,
                &dispatcher,
                &raw_dispatcher,
                &token,
                &allowed_origins,
                &rate_limiter,
                require_read_model,
            ) {
                eprintln!("synty mcp HTTP request failed: {error}");
            }
            in_flight.fetch_sub(1, Ordering::AcqRel);
        });
    }
    Ok(())
}

#[cfg(not(feature = "mcp-http"))]
pub fn run(_opts: Opts) -> Result<()> {
    bail!("HTTP MCP is not in this binary; rebuild with --features mcp-http")
}

#[cfg(feature = "mcp-http")]
fn handle(
    mut request: Request,
    dispatcher: &Dispatcher,
    raw_dispatcher: &Dispatcher,
    token: &str,
    allowed_origins: &[String],
    rate_limiter: &Mutex<RateLimiter>,
    require_read_model: bool,
) -> Result<()> {
    let path = request.url().split('?').next().unwrap_or(request.url());
    if !origin_allowed(&request, allowed_origins) {
        return respond(request, StatusCode(403), "origin forbidden".into(), false);
    }
    if request.method() == &Method::Get && path == "/health" {
        let ready = !require_read_model || crate::readmodel::current().is_some();
        let body = json!({
            "status": if ready { "ok" } else { "starting" },
            "name": "synty",
            "version": env!("CARGO_PKG_VERSION"),
        })
        .to_string();
        return respond(request, StatusCode(health_code(ready)), body, true);
    }
    if path != "/mcp" {
        return respond(request, StatusCode(404), "not found".into(), false);
    }
    if request.method() != &Method::Post {
        return respond(request, StatusCode(405), "method not allowed".into(), false);
    }
    let client = request
        .remote_addr()
        .map(|address| address.ip().to_string())
        .unwrap_or_else(|| "unknown".into());
    if !rate_limiter
        .lock()
        .map_err(|_| anyhow::anyhow!("MCP rate limiter lock poisoned"))?
        .allow(&client, Instant::now())
    {
        return respond(request, StatusCode(429), "rate limit exceeded".into(), false);
    }
    if !authorized(&request, token) {
        return respond(request, StatusCode(401), "unauthorized".into(), false);
    }
    if !accepts_mcp(header(&request, "Accept").unwrap_or("")) {
        return respond(request, StatusCode(406), "Accept must include application/json and text/event-stream".into(), false);
    }
    if !is_json_content(header(&request, "Content-Type").unwrap_or("")) {
        return respond(request, StatusCode(415), "Content-Type must be application/json".into(), false);
    }
    if header(&request, "Content-Length")
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|size| size > MAX_BODY_BYTES)
    {
        return respond(request, StatusCode(413), "request body too large".into(), false);
    }
    let mut body = Vec::new();
    request.as_reader().take((MAX_BODY_BYTES + 1) as u64).read_to_end(&mut body)?;
    if body.len() > MAX_BODY_BYTES {
        return respond(request, StatusCode(413), "request body too large".into(), false);
    }
    let value: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(error) => {
            return respond(
                request,
                StatusCode(400),
                json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {"code": -32700, "message": format!("parse error: {error}")}
                }).to_string(),
                true,
            );
        }
    };
    if value["method"].as_str() != Some("initialize")
        && !protocol_version_supported(header(&request, "MCP-Protocol-Version"))
    {
        return respond(request, StatusCode(400), "missing or unsupported MCP-Protocol-Version".into(), false);
    }
    let dispatcher = if uses_raw_history(&value) {
        raw_dispatcher
    } else {
        dispatcher
    };
    let response = match dispatcher.call(value) {
        Ok(response) => response,
        Err(DispatchError::Busy) => {
            return respond(request, StatusCode(503), "dispatcher queue full".into(), false);
        }
        Err(DispatchError::Timeout) => {
            return respond(request, StatusCode(504), "tool call timed out".into(), false);
        }
        Err(DispatchError::Stopped) => {
            return respond(request, StatusCode(503), "dispatcher unavailable".into(), false);
        }
    };
    match response {
        Some(response) => respond(request, StatusCode(200), response.to_string(), true),
        None => respond(request, StatusCode(202), String::new(), false),
    }
}

#[cfg(any(feature = "mcp-http", test))]
fn health_code(ready: bool) -> u16 {
    if ready { 200 } else { 503 }
}

#[cfg(any(feature = "mcp-http", test))]
fn uses_raw_history(request: &serde_json::Value) -> bool {
    request["method"].as_str() == Some("tools/call")
        && matches!(
            request["params"]["name"].as_str(),
            Some(
                "synty_topics"
                    | "synty_recent"
                    | "synty_status"
                    | "synty_stats"
                    | "synty_tool"
                    | "synty_show"
                    | "synty_trace_list"
                    | "synty_trace_show"
                    | "synty_trace_search"
                    | "synty_trace_compare"
            )
        )
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
fn origin_allowed(request: &Request, allowed: &[String]) -> bool {
    header(request, "Origin").is_none_or(|origin| origin_value_allowed(origin, allowed))
}

#[cfg(any(feature = "mcp-http", test))]
fn origin_value_allowed(origin: &str, allowed: &[String]) -> bool {
    allowed.iter().any(|candidate| candidate == origin)
}

#[cfg(any(feature = "mcp-http", test))]
fn accepts_mcp(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    let types: Vec<&str> = value
        .split(',')
        .filter_map(|part| part.trim().split(';').next())
        .collect();
    types.contains(&"application/json") && types.contains(&"text/event-stream")
}

#[cfg(any(feature = "mcp-http", test))]
fn is_json_content(value: &str) -> bool {
    value
        .trim()
        .split(';')
        .next()
        .is_some_and(|media_type| media_type.eq_ignore_ascii_case("application/json"))
}

#[cfg(any(feature = "mcp-http", test))]
fn protocol_version_supported(value: Option<&str>) -> bool {
    // The transport specification asks servers to assume the original
    // Streamable HTTP version when an older client omits this header.
    let value = value.unwrap_or("2025-03-26");
    crate::mcp::SUPPORTED_PROTOCOL_VERSIONS.contains(&value)
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

    #[test]
    fn browser_origins_are_exact_allowlist_matches() {
        let allowed = vec!["https://memory.example.com".to_string()];
        assert!(origin_value_allowed("https://memory.example.com", &allowed));
        assert!(!origin_value_allowed("https://evil.example.com", &allowed));
        assert!(!origin_value_allowed("null", &allowed));
    }

    #[test]
    fn streamable_http_accept_header_requires_both_media_types() {
        assert!(accepts_mcp("application/json, text/event-stream"));
        assert!(accepts_mcp("application/json; charset=utf-8, text/event-stream"));
        assert!(!accepts_mcp("application/json"));
        assert!(!accepts_mcp("*/*"));
        assert!(is_json_content("Application/JSON; charset=utf-8"));
        assert!(!is_json_content("text/plain"));
        assert!(protocol_version_supported(Some("2025-11-25")));
        assert!(protocol_version_supported(None));
        assert!(!protocol_version_supported(Some("2099-01-01")));
        assert_eq!(health_code(true), 200);
        assert_eq!(health_code(false), 503);
    }

    #[test]
    fn long_raw_analysis_cannot_take_the_search_dispatcher() {
        let call = |name| serde_json::json!({
            "method": "tools/call",
            "params": {"name": name, "arguments": {}}
        });
        assert!(uses_raw_history(&call("synty_status")));
        assert!(uses_raw_history(&call("synty_trace_search")));
        assert!(uses_raw_history(&call("synty_recent")));
        assert!(!uses_raw_history(&call("synty_search")));
        assert!(!uses_raw_history(&call("synty_related")));
        assert!(!uses_raw_history(&serde_json::json!({"method": "tools/list"})));
    }

    #[test]
    fn rate_windows_are_per_client_and_reset() {
        let mut limiter = RateLimiter::default();
        let now = Instant::now();
        for _ in 0..RATE_REQUESTS {
            assert!(limiter.allow("10.0.0.1", now));
        }
        assert!(!limiter.allow("10.0.0.1", now));
        assert!(limiter.allow("10.0.0.2", now), "one client cannot consume another's window");
        assert!(limiter.allow("10.0.0.1", now + RATE_WINDOW), "a completed window resets");
    }

    #[cfg(feature = "mcp-http")]
    #[test]
    fn bounded_dispatcher_roundtrips_protocol_calls() {
        let dispatcher = Dispatcher::start(
            "test",
            mcp::Server::new(
                "m".into(),
                crate::policy::McpRole::Operator,
                crate::policy::ReadScope::default(),
                crate::redact::Profile::Off,
                false,
                None,
            ),
        )
        .unwrap();
        let response = dispatcher
            .call(serde_json::json!({"jsonrpc":"2.0","id":7,"method":"ping"}))
            .unwrap()
            .unwrap();
        assert_eq!(response["id"], 7);
        assert!(response["result"].as_object().is_some());
    }
}
