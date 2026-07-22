// Optional authenticated Streamable HTTP transport for the same read-only MCP
// dispatcher as `synty mcp` stdio. The default bind is loopback; wider binds
// require an explicit flag because the corpus can contain source and tool data.

use anyhow::{Result, bail};

#[cfg(feature = "mcp-http")]
use crate::mcp;
#[cfg(feature = "mcp-http")]
use bytes::Bytes;
#[cfg(feature = "mcp-http")]
use http_body_util::{BodyExt, Full, LengthLimitError, Limited};
#[cfg(feature = "mcp-http")]
use hyper::{
    Method, Request, Response, StatusCode,
    body::{Body, Incoming},
    header::{self, HeaderValue},
    server::conn::http1,
    service::service_fn,
};
#[cfg(feature = "mcp-http")]
use hyper_util::rt::{TokioIo, TokioTimer};
#[cfg(feature = "mcp-http")]
use serde_json::{Value, json};
#[cfg(any(feature = "mcp-http", test))]
use std::collections::HashMap;
#[cfg(feature = "mcp-http")]
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, RecvTimeoutError, SyncSender, TrySendError},
};
#[cfg(any(feature = "mcp-http", test))]
use std::time::{Duration, Instant};
#[cfg(feature = "mcp-http")]
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::TcpListener,
    sync::Semaphore,
};
#[cfg(feature = "mcp-http")]
use tokio_rustls::TlsAcceptor;

#[cfg(feature = "mcp-http")]
const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;
#[cfg(feature = "mcp-http")]
const MAX_IN_FLIGHT: usize = 32;
#[cfg(feature = "mcp-http")]
const MAX_CONNECTIONS: usize = 64;
#[cfg(feature = "mcp-http")]
const DISPATCH_QUEUE: usize = 8;
#[cfg(feature = "mcp-http")]
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
#[cfg(feature = "mcp-http")]
const BODY_READ_TIMEOUT: Duration = Duration::from_secs(10);
#[cfg(feature = "mcp-http")]
const CONNECTION_READ_TIMEOUT: Duration = Duration::from_secs(10);
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
#[derive(Clone)]
struct AppState {
    dispatcher: Dispatcher,
    raw_dispatcher: Dispatcher,
    token: Arc<String>,
    allowed_origins: Arc<Vec<String>>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
    in_flight: Arc<Semaphore>,
    require_read_model: bool,
}

#[cfg(feature = "mcp-http")]
enum BodyRead {
    Complete(Bytes),
    TooLarge,
    Timeout,
    Failed(String),
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
    pub tls_cert: Option<String>,
    pub tls_key: Option<String>,
    pub role: crate::policy::McpRole,
    pub scope: crate::policy::ReadScope,
    pub redaction: crate::redact::Profile,
    pub allowed_origins: Vec<String>,
    pub bucket: Option<String>,
}

#[cfg(feature = "mcp-http")]
pub fn run(opts: Opts) -> Result<()> {
    anyhow::ensure!(opts.token.len() >= 32, "HTTP MCP bearer token must contain at least 32 bytes");
    let tls = tls_config(opts.tls_cert.as_deref(), opts.tls_key.as_deref())?;
    validate_listener(&opts.bind, opts.listen_public, tls.is_some())?;
    let scheme = if tls.is_some() { "https" } else { "http" };
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
    let state = AppState {
        dispatcher,
        raw_dispatcher,
        token: Arc::new(opts.token),
        allowed_origins: Arc::new(opts.allowed_origins),
        rate_limiter: Arc::new(Mutex::new(RateLimiter::default())),
        in_flight: Arc::new(Semaphore::new(MAX_IN_FLIGHT)),
        require_read_model,
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| anyhow::anyhow!("start MCP HTTP runtime: {error}"))?;
    runtime.block_on(serve(opts.bind, scheme, tls, state))
}

#[cfg(feature = "mcp-http")]
async fn serve(
    bind: String,
    scheme: &str,
    tls: Option<Arc<rustls::ServerConfig>>,
    state: AppState,
) -> Result<()> {
    let listener = TcpListener::bind(&bind)
        .await
        .map_err(|error| anyhow::anyhow!("bind {} {bind}: {error}", scheme.to_ascii_uppercase()))?;
    let connections = Arc::new(Semaphore::new(MAX_CONNECTIONS));
    eprintln!("synty mcp: serving authenticated HTTP at {scheme}://{bind}/mcp");
    loop {
        let (stream, client) = listener.accept().await?;
        let Ok(connection) = Arc::clone(&connections).try_acquire_owned() else {
            continue;
        };
        let state = state.clone();
        let tls = tls.clone();
        tokio::spawn(async move {
            let result = if let Some(config) = tls {
                match tokio::time::timeout(
                    CONNECTION_READ_TIMEOUT,
                    TlsAcceptor::from(config).accept(stream),
                )
                .await
                {
                    Ok(Ok(stream)) => serve_connection(stream, client, state).await,
                    Ok(Err(error)) => Err(anyhow::anyhow!("TLS handshake from {client}: {error}")),
                    Err(_) => Err(anyhow::anyhow!("TLS handshake from {client} timed out")),
                }
            } else {
                serve_connection(stream, client, state).await
            };
            if let Err(error) = result {
                eprintln!("synty mcp HTTP connection failed: {error}");
            }
            drop(connection);
        });
    }
}

#[cfg(feature = "mcp-http")]
async fn serve_connection<I>(
    stream: I,
    client: std::net::SocketAddr,
    state: AppState,
) -> Result<()>
where
    I: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let service = service_fn(move |request| {
        let state = state.clone();
        async move {
            let Ok(permit) = Arc::clone(&state.in_flight).try_acquire_owned() else {
                return Ok::<_, std::convert::Infallible>(response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "server busy".into(),
                    false,
                ));
            };
            let response = handle(request, client, state).await;
            drop(permit);
            Ok(response)
        }
    });
    let mut builder = http1::Builder::new();
    builder
        .timer(TokioTimer::new())
        .header_read_timeout(CONNECTION_READ_TIMEOUT);
    builder
        .serve_connection(TokioIo::new(stream), service)
        .await
        .map_err(|error| anyhow::anyhow!("serve {client}: {error}"))
}

#[cfg(feature = "mcp-http")]
fn tls_config(
    cert_path: Option<&str>,
    key_path: Option<&str>,
) -> Result<Option<Arc<rustls::ServerConfig>>> {
    match (cert_path, key_path) {
        (None, None) => Ok(None),
        (Some(cert_path), Some(key_path)) => {
            let cert_file = std::fs::File::open(cert_path)
                .map_err(|error| anyhow::anyhow!("read TLS certificate {cert_path}: {error}"))?;
            let certs = rustls_pemfile::certs(&mut std::io::BufReader::new(cert_file))
                .collect::<std::io::Result<Vec<_>>>()
                .map_err(|error| anyhow::anyhow!("parse TLS certificate {cert_path}: {error}"))?;
            anyhow::ensure!(!certs.is_empty(), "TLS certificate {cert_path} contains no certificates");
            let key_file = std::fs::File::open(key_path)
                .map_err(|error| anyhow::anyhow!("read TLS private key {key_path}: {error}"))?;
            let key = rustls_pemfile::private_key(&mut std::io::BufReader::new(key_file))
                .map_err(|error| anyhow::anyhow!("parse TLS private key {key_path}: {error}"))?
                .ok_or_else(|| anyhow::anyhow!("TLS private key {key_path} contains no key"))?;
            let config = rustls::ServerConfig::builder_with_provider(Arc::new(
                rustls::crypto::aws_lc_rs::default_provider(),
            ))
                .with_safe_default_protocol_versions()
                .map_err(|error| anyhow::anyhow!("configure TLS protocol versions: {error}"))?
                .with_no_client_auth()
                .with_single_cert(certs, key)
                .map_err(|error| anyhow::anyhow!("configure TLS certificate and key: {error}"))?;
            Ok(Some(Arc::new(config)))
        }
        _ => bail!("--tls-cert and --tls-key must be provided together"),
    }
}

#[cfg(any(feature = "mcp-http", test))]
fn validate_listener(bind: &str, listen_public: bool, tls: bool) -> Result<()> {
    if !is_loopback_bind(bind) && !listen_public {
        bail!("refusing non-loopback bind {bind}; pass --listen-public with TLS");
    }
    if !is_loopback_bind(bind) && !tls {
        bail!("refusing plaintext non-loopback bind {bind}; pass --tls-cert and --tls-key");
    }
    Ok(())
}

#[cfg(not(feature = "mcp-http"))]
pub fn run(_opts: Opts) -> Result<()> {
    bail!("HTTP MCP is not in this binary; rebuild with --features mcp-http")
}

#[cfg(feature = "mcp-http")]
async fn handle(
    request: Request<Incoming>,
    client: std::net::SocketAddr,
    state: AppState,
) -> Response<Full<Bytes>> {
    let path = request.uri().path();
    if !origin_allowed(&request, &state.allowed_origins) {
        return response(StatusCode::FORBIDDEN, "origin forbidden".into(), false);
    }
    if request.method() == Method::GET && path == "/health" {
        let ready = !state.require_read_model || crate::readmodel::current().is_some();
        let body = json!({
            "status": if ready { "ok" } else { "starting" },
            "name": "synty",
            "version": env!("CARGO_PKG_VERSION"),
        })
        .to_string();
        return response(StatusCode::from_u16(health_code(ready)).unwrap(), body, true);
    }
    if path != "/mcp" {
        return response(StatusCode::NOT_FOUND, "not found".into(), false);
    }
    if request.method() != Method::POST {
        return response(StatusCode::METHOD_NOT_ALLOWED, "method not allowed".into(), false);
    }
    let allowed = state
        .rate_limiter
        .lock()
        .map(|mut limiter| limiter.allow(&client.ip().to_string(), Instant::now()))
        .unwrap_or(false);
    if !allowed {
        return response(StatusCode::TOO_MANY_REQUESTS, "rate limit exceeded".into(), false);
    }
    if !authorized(&request, &state.token) {
        return response(StatusCode::UNAUTHORIZED, "unauthorized".into(), false);
    }
    if !accepts_mcp(header(&request, "Accept").unwrap_or("")) {
        return response(
            StatusCode::NOT_ACCEPTABLE,
            "Accept must include application/json and text/event-stream".into(),
            false,
        );
    }
    if !is_json_content(header(&request, "Content-Type").unwrap_or("")) {
        return response(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be application/json".into(),
            false,
        );
    }
    if header(&request, "Content-Length")
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|size| size > MAX_BODY_BYTES)
    {
        return response(StatusCode::PAYLOAD_TOO_LARGE, "request body too large".into(), false);
    }
    let protocol_version = header(&request, "MCP-Protocol-Version").map(str::to_owned);
    let body = match read_body(request.into_body(), MAX_BODY_BYTES, BODY_READ_TIMEOUT).await {
        BodyRead::Complete(body) => body,
        BodyRead::TooLarge => {
            return response(StatusCode::PAYLOAD_TOO_LARGE, "request body too large".into(), false);
        }
        BodyRead::Timeout => {
            return response(StatusCode::REQUEST_TIMEOUT, "request body timed out".into(), false);
        }
        BodyRead::Failed(error) => {
            eprintln!("synty mcp HTTP body from {client} failed: {error}");
            return response(StatusCode::BAD_REQUEST, "invalid request body".into(), false);
        }
    };
    let value: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(error) => {
            return response(
                StatusCode::BAD_REQUEST,
                json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {"code": -32700, "message": format!("parse error: {error}")}
                })
                .to_string(),
                true,
            );
        }
    };
    if value["method"].as_str() != Some("initialize")
        && !protocol_version_supported(protocol_version.as_deref())
    {
        return response(
            StatusCode::BAD_REQUEST,
            "missing or unsupported MCP-Protocol-Version".into(),
            false,
        );
    }
    let dispatcher = if uses_raw_history(&value) {
        state.raw_dispatcher
    } else {
        state.dispatcher
    };
    let dispatched = match tokio::task::spawn_blocking(move || dispatcher.call(value)).await {
        Ok(result) => result,
        Err(error) => {
            eprintln!("synty mcp HTTP dispatcher task failed: {error}");
            return response(StatusCode::SERVICE_UNAVAILABLE, "dispatcher unavailable".into(), false);
        }
    };
    let response_value = match dispatched {
        Ok(response) => response,
        Err(DispatchError::Busy) => {
            return response(StatusCode::SERVICE_UNAVAILABLE, "dispatcher queue full".into(), false);
        }
        Err(DispatchError::Timeout) => {
            return response(StatusCode::GATEWAY_TIMEOUT, "tool call timed out".into(), false);
        }
        Err(DispatchError::Stopped) => {
            return response(StatusCode::SERVICE_UNAVAILABLE, "dispatcher unavailable".into(), false);
        }
    };
    match response_value {
        Some(value) => response(StatusCode::OK, value.to_string(), true),
        None => response(StatusCode::ACCEPTED, String::new(), false),
    }
}

#[cfg(feature = "mcp-http")]
async fn read_body<B>(body: B, max_bytes: usize, timeout: Duration) -> BodyRead
where
    B: Body<Data = Bytes> + Unpin,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    match tokio::time::timeout(timeout, Limited::new(body, max_bytes).collect()).await {
        Err(_) => BodyRead::Timeout,
        Ok(Ok(body)) => BodyRead::Complete(body.to_bytes()),
        Ok(Err(error)) if error.downcast_ref::<LengthLimitError>().is_some() => BodyRead::TooLarge,
        Ok(Err(error)) => BodyRead::Failed(error.to_string()),
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
fn response(status: StatusCode, body: String, json_body: bool) -> Response<Full<Bytes>> {
    let mut response = Response::new(Full::new(Bytes::from(body)));
    *response.status_mut() = status;
    if json_body {
        response
            .headers_mut()
            .insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    }
    response
}

#[cfg(feature = "mcp-http")]
fn header<'a, B>(request: &'a Request<B>, name: &str) -> Option<&'a str> {
    request
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
}

#[cfg(feature = "mcp-http")]
fn authorized<B>(request: &Request<B>, expected: &str) -> bool {
    let Some(actual) = header(request, "Authorization").and_then(|value| value.strip_prefix("Bearer "))
    else {
        return false;
    };
    constant_time_eq(actual.as_bytes(), expected.as_bytes())
}

#[cfg(feature = "mcp-http")]
fn origin_allowed<B>(request: &Request<B>, allowed: &[String]) -> bool {
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
        assert!(validate_listener("127.0.0.1:8765", false, false).is_ok());
        assert!(validate_listener("0.0.0.0:8765", false, true).is_err());
        assert!(validate_listener("0.0.0.0:8765", true, false).is_err());
        assert!(validate_listener("0.0.0.0:8765", true, true).is_ok());
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
    fn body_reader_enforces_deadline_and_size() {
        struct PendingBody;
        impl Body for PendingBody {
            type Data = Bytes;
            type Error = std::convert::Infallible;

            fn poll_frame(
                self: std::pin::Pin<&mut Self>,
                _context: &mut std::task::Context<'_>,
            ) -> std::task::Poll<
                Option<std::result::Result<hyper::body::Frame<Self::Data>, Self::Error>>,
            > {
                std::task::Poll::Pending
            }
        }

        let runtime = tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
        let started = Instant::now();
        let timed_out = runtime.block_on(read_body(
            PendingBody,
            MAX_BODY_BYTES,
            Duration::from_millis(100),
        ));
        assert!(matches!(timed_out, BodyRead::Timeout));
        assert!(started.elapsed() < Duration::from_millis(400));
        let oversized = runtime.block_on(read_body(
            Full::new(Bytes::from_static(b"too large")),
            3,
            Duration::from_secs(1),
        ));
        assert!(matches!(oversized, BodyRead::TooLarge));
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
