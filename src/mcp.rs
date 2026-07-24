// `synty mcp` — an MCP server over stdio (JSON-RPC 2.0, newline-delimited)
// exposing synty's read surfaces as agent tools, so a coding agent can consult
// past work mid-session (synty_search / synty_topics / synty_recent /
// synty_status) instead of shelling out. The protocol slice MCP needs here is
// small (initialize, tools/list, tools/call, ping), so it's hand-rolled — no
// new dependencies, and stdout carries protocol JSON only (logs go to stderr).

use crate::{encode::Encoder, load_docs, readmodel, search, trace, units, view};
use anyhow::Result;
use next_plaid::{MmapIndex, SearchParameters};
use serde_json::{json, Value};
use std::io::{BufRead, Write};

pub(crate) const PROTOCOL_VERSION: &str = "2025-11-25";
pub(crate) const SUPPORTED_PROTOCOL_VERSIONS: &[&str] =
    &[PROTOCOL_VERSION, "2025-06-18", "2025-03-26"];

pub fn run(
    model_id: String,
    role: crate::policy::McpRole,
    scope: crate::policy::ReadScope,
    redaction: crate::redact::Profile,
    bucket: Option<String>,
    athena: Option<AthenaTraceOptions>,
) -> Result<()> {
    start_bucket_refresh(bucket.clone(), athena.is_none());
    let mut srv = Server::new(model_id, role, scope, redaction, true, bucket, athena);
    eprintln!("synty mcp: serving tools over stdio");
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(req) = serde_json::from_str::<Value>(&line) else { continue };
        if let Some(resp) = srv.handle(&req) {
            let mut out = std::io::stdout().lock();
            writeln!(out, "{resp}")?;
            out.flush()?;
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "athena"), allow(dead_code))]
pub(crate) struct AthenaTraceOptions {
    pub(crate) workgroup: String,
    pub(crate) database: String,
    pub(crate) table: String,
}

enum TraceBackend {
    Local,
    #[cfg(feature = "athena")]
    Athena(Box<crate::trace_athena::Backend>),
    Unavailable(String),
}

impl TraceBackend {
    fn new(bucket: Option<&str>, options: Option<AthenaTraceOptions>) -> Self {
        let Some(options) = options else { return Self::Local };
        let Some(bucket) = bucket else {
            return Self::Unavailable("--athena-workgroup requires --bucket s3://...".into());
        };
        #[cfg(feature = "athena")]
        {
            let result = crate::trace_athena::Config::new(
                bucket.to_string(),
                options.workgroup,
                options.database,
                options.table,
            )
            .and_then(crate::trace_athena::Backend::new);
            match result {
                Ok(backend) => Self::Athena(Box::new(backend)),
                Err(error) => Self::Unavailable(error.to_string()),
            }
        }
        #[cfg(not(feature = "athena"))]
        {
            let _ = (bucket, options);
            Self::Unavailable(
                "Athena trace is not in this binary; rebuild with --features athena".into(),
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn list(
        &mut self,
        entity: &str,
        repo: Option<&str>,
        machine: Option<&str>,
        source: Option<&str>,
        status: Option<&str>,
        operation: Option<&str>,
        has_errors: bool,
        since: Option<&str>,
        min_ms: Option<u64>,
        sort: &str,
        limit: usize,
        scope: &crate::policy::ReadScope,
    ) -> Result<String> {
        match self {
            Self::Local => trace::list_text(
                entity,
                repo,
                machine,
                source,
                status,
                operation,
                has_errors,
                since,
                min_ms,
                sort,
                limit,
                false,
                Some(scope),
            ),
            #[cfg(feature = "athena")]
            Self::Athena(backend) => backend.list(
                entity, repo, machine, source, status, operation, has_errors, since, min_ms, sort,
                limit, scope,
            ),
            Self::Unavailable(error) => Err(anyhow::anyhow!("{error}")),
        }
    }

    fn show(
        &mut self,
        id: &str,
        before: usize,
        after: usize,
        scope: &crate::policy::ReadScope,
    ) -> Result<String> {
        match self {
            Self::Local => trace::show_text(id, before, after, false, Some(scope)),
            #[cfg(feature = "athena")]
            Self::Athena(backend) => backend.show(id, before, after, scope),
            Self::Unavailable(error) => Err(anyhow::anyhow!("{error}")),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self,
        query: &str,
        repo: Option<&str>,
        machine: Option<&str>,
        source: Option<&str>,
        kind: Option<&str>,
        limit: usize,
        scope: &crate::policy::ReadScope,
    ) -> Result<String> {
        match self {
            Self::Local => trace::search_text(
                query,
                repo,
                machine,
                source,
                kind,
                limit,
                false,
                Some(scope),
            ),
            #[cfg(feature = "athena")]
            Self::Athena(backend) => {
                backend.search(query, repo, machine, source, kind, limit, scope)
            }
            Self::Unavailable(error) => Err(anyhow::anyhow!("{error}")),
        }
    }

    fn compare(
        &mut self,
        left: &str,
        right: &str,
        scope: &crate::policy::ReadScope,
    ) -> Result<String> {
        match self {
            Self::Local => trace::compare_text(left, right, false, Some(scope)),
            #[cfg(feature = "athena")]
            Self::Athena(backend) => backend.compare(left, right, scope),
            Self::Unavailable(error) => Err(anyhow::anyhow!("{error}")),
        }
    }
}

pub(crate) struct Server {
    model_id: String,
    role: crate::policy::McpRole,
    scope: crate::policy::ReadScope,
    redaction: crate::redact::Profile,
    allow_repo_paths: bool,
    trace: TraceBackend,
    /// The build the engine serves + encoder + index — loaded on the first
    /// search call, kept warm, reopened when the pointer moves.
    engine: Option<(readmodel::Current, Encoder, MmapIndex)>,
}

impl Server {
    pub(crate) fn new(
        model_id: String,
        role: crate::policy::McpRole,
        scope: crate::policy::ReadScope,
        redaction: crate::redact::Profile,
        allow_repo_paths: bool,
        bucket: Option<String>,
        athena: Option<AthenaTraceOptions>,
    ) -> Self {
        let trace = TraceBackend::new(bucket.as_deref(), athena);
        Self {
            model_id,
            role,
            scope,
            redaction,
            allow_repo_paths,
            trace,
            engine: None,
        }
    }

    /// One JSON-RPC message → one response, or None for notifications.
    pub(crate) fn handle(&mut self, req: &Value) -> Option<Value> {
        let id = match req.get("id") {
            Some(v) if !v.is_null() => v.clone(),
            _ => return None, // notification (e.g. notifications/initialized)
        };
        let method = req["method"].as_str().unwrap_or("");
        let result = match method {
            "initialize" => Ok(json!({
                "protocolVersion": negotiate_protocol(req["params"]["protocolVersion"].as_str()),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "synty", "version": env!("CARGO_PKG_VERSION")},
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({"tools": tool_defs(self.role, &self.scope, self.allow_repo_paths)})),
            "tools/call" => self.call(&req["params"]),
            other => Err(format!("method not found: {other}")),
        };
        Some(match result {
            Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
            Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": e}}),
        })
    }

    /// Dispatch a tools/call. An unknown tool is a protocol error; a failing
    /// tool returns isError content, per MCP.
    fn call(&mut self, p: &Value) -> std::result::Result<Value, String> {
        let name = p["name"].as_str().unwrap_or("");
        let a = &p["arguments"];
        if crate::policy::tool_scope(name).is_none() {
            return Err(format!("unknown tool: {name}"));
        }
        if !self.role.allows_tool(name) {
            return Err(format!("tool not allowed for {:?} role: {name}", self.role));
        }
        if !self.scope.exposes_tool(name) {
            return Err(format!("tool unavailable with a restricted read scope: {name}"));
        }
        let out = match name {
            "synty_search" => self.search(a),
            "synty_related" => self.related(a),
            "synty_topics" => topics_text(a, &self.scope),
            "synty_recent" => recent_text(a, &self.scope),
            "synty_status" => view::status().map(|s| view::status_md(&s)),
            "synty_stats" => view::stats(bounded_positive(a, "weeks", 4, 52)).map(|s| view::stats_md(&s)),
            "synty_tool" => {
                let name = a["name"].as_str().unwrap_or("");
                if name.is_empty() {
                    Err(anyhow::anyhow!("name is required (tool names appear in synty_status output)"))
                } else {
                    view::tool_report(name)
                }
            }
            "synty_show" => {
                let id = a["id"].as_str().unwrap_or("");
                if id.is_empty() {
                    Err(anyhow::anyhow!("id is required — ids appear inline in synty_search/synty_recent/synty_topics output"))
                } else {
                    view::show_report_scoped(id, &self.scope)
                }
            }
            "synty_trace_list" => {
                let entity = a["type"].as_str().unwrap_or("");
                if entity.is_empty() {
                    Err(anyhow::anyhow!("type is required (turns, spans, or jobs)"))
                } else {
                    self.trace.list(
                        entity,
                        string_arg(a, "repo"),
                        string_arg(a, "machine"),
                        string_arg(a, "source"),
                        string_arg(a, "status"),
                        string_arg(a, "operation"),
                        a["has_errors"].as_bool().unwrap_or(false),
                        string_arg(a, "since"),
                        a["min_ms"].as_u64(),
                        a["sort"].as_str().unwrap_or("recent"),
                        bounded_positive(a, "limit", 20, 100),
                        &self.scope,
                    )
                }
            }
            "synty_trace_show" => {
                let id = a["id"].as_str().unwrap_or("");
                if id.is_empty() {
                    Err(anyhow::anyhow!("id is required — trace ids appear in synty_trace_list"))
                } else {
                    self.trace.show(
                        id,
                        bounded(a, "before", 6, 100),
                        bounded(a, "after", 12, 100),
                        &self.scope,
                    )
                }
            }
            "synty_trace_search" => {
                let query = a["query"].as_str().unwrap_or("");
                if query.is_empty() {
                    Err(anyhow::anyhow!("query is required"))
                } else if query.chars().count() > 4096 {
                    Err(anyhow::anyhow!("query exceeds 4096 characters"))
                } else {
                    self.trace.search(
                        query,
                        string_arg(a, "repo"),
                        string_arg(a, "machine"),
                        string_arg(a, "source"),
                        string_arg(a, "kind"),
                        bounded_positive(a, "limit", 20, 100),
                        &self.scope,
                    )
                }
            }
            "synty_trace_compare" => {
                let left = a["left"].as_str().unwrap_or("");
                let right = a["right"].as_str().unwrap_or("");
                if left.is_empty() || right.is_empty() {
                    Err(anyhow::anyhow!("left and right trace ids are required"))
                } else {
                    self.trace.compare(left, right, &self.scope)
                }
            }
            other => return Err(format!("unknown tool: {other}")),
        };
        Ok(match out {
            Ok(text) => json!({
                "content": [{"type": "text", "text": crate::redact::text(
                    &text, self.redaction
                )}],
                "isError": false
            }),
            Err(e) => json!({
                "content": [{"type": "text", "text": crate::redact::text(
                    &e.to_string(), self.redaction
                )}],
                "isError": true
            }),
        })
    }

    fn search(&mut self, a: &Value) -> Result<String> {
        let query = a["query"].as_str().unwrap_or("");
        anyhow::ensure!(!query.is_empty(), "query is required");
        anyhow::ensure!(query.chars().count() <= 4096, "query exceeds 4096 characters");
        let k = bounded_positive(a, "k", 5, 100);
        let filter = a["filter"].as_str().filter(|f| !f.is_empty());
        self.search_query(query, k, filter)
    }

    /// `synty_related`: local stdio clients may derive context from a repo;
    /// remote clients must send context text and cannot make the server read an
    /// arbitrary filesystem path.
    fn related(&mut self, a: &Value) -> Result<String> {
        let query = if let Some(context) = a["context"].as_str().filter(|value| !value.is_empty()) {
            anyhow::ensure!(context.chars().count() <= 4096, "context exceeds 4096 characters");
            context.to_string()
        } else {
            anyhow::ensure!(self.allow_repo_paths, "remote synty_related requires `context`; repo paths are disabled over HTTP");
            let cwd = match a["repo"].as_str().filter(|r| !r.is_empty()) {
                Some(r) => std::path::PathBuf::from(r),
                None => std::env::current_dir()?,
            };
            crate::related::context_query(&cwd).ok_or_else(|| {
                anyhow::anyhow!(
                    "no git context at {} — pass `repo` as the path to a git repo you're working in, or use synty_search",
                    cwd.display()
                )
            })?
        };
        let k = bounded_positive(a, "k", 5, 100);
        self.search_query(&query, k, None)
    }

    fn search_query(&mut self, query: &str, k: usize, filter: Option<&str>) -> Result<String> {
        // Reopen the index when the pointer moved (a builder published a new
        // read-model while we were serving); the encoder stays loaded.
        let cur = readmodel::current();
        if self.engine.as_ref().is_none_or(|(b, _, _)| Some(b) != cur.as_ref()) {
            let cur = cur.ok_or_else(|| anyhow::anyhow!("no index yet (run `synty build` first)"))?;
            let idx = MmapIndex::load(&cur.dir().to_string_lossy())
                .map_err(|e| anyhow::anyhow!("load index: {e} (run `synty build` first)"))?;
            let enc = match self.engine.take() {
                Some((_, e, _)) => e,
                None => Encoder::load(&self.model_id)?,
            };
            self.engine = Some((cur, enc, idx));
        }
        let (_, enc, idx) = self.engine.as_mut().expect("engine loaded");
        let docs = load_docs(readmodel::docs_path())?;
        let q = enc.encode_query(query)?;
        let subset = search::subset_for_scope(filter, &self.scope)?;
        let params = SearchParameters { top_k: k, ..Default::default() };
        let res = idx.search(&q, &params, subset.as_deref()).map_err(|e| anyhow::anyhow!("search: {e}"))?;
        let mut out = search::render(&docs, query, filter, &res);
        if let Some(note) = view::stale_note() {
            out.push_str(&format!("\n_{note}_\n"));
        }
        Ok(out)
    }
}

/// Keep the selected published model fresh without holding the dispatcher.
/// Athena readers omit trace.json; neither mode mirrors the raw event lake.
pub(crate) fn start_bucket_refresh(bucket: Option<String>, include_trace: bool) {
    let Some(bucket) = bucket else { return };
    std::thread::spawn(move || loop {
        crate::sync::pull_read_model_for_mcp(&bucket, include_trace);
        std::thread::sleep(std::time::Duration::from_secs(30));
    });
}

fn string_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args[key].as_str().filter(|value| !value.is_empty())
}

fn bounded(args: &Value, key: &str, default: usize, max: usize) -> usize {
    args[key].as_u64().unwrap_or(default as u64).min(max as u64) as usize
}

fn bounded_positive(args: &Value, key: &str, default: usize, max: usize) -> usize {
    bounded(args, key, default, max).max(1)
}

fn negotiate_protocol(requested: Option<&str>) -> &str {
    requested
        .filter(|version| SUPPORTED_PROTOCOL_VERSIONS.contains(version))
        .unwrap_or(PROTOCOL_VERSION)
}

fn topics_text(a: &Value, scope: &crate::policy::ReadScope) -> Result<String> {
    let mut topics = units::topic_units_scoped(12, scope)?;
    if let Some(q) = a["query"].as_str().filter(|q| !q.is_empty()) {
        anyhow::ensure!(q.chars().count() <= 4096, "query exceeds 4096 characters");
        let ql = q.to_lowercase();
        topics.retain(|t| {
            t.label.to_lowercase().contains(&ql)
                || t.units.iter().any(|u| u.title.to_lowercase().contains(&ql))
        });
    }
    topics.truncate(bounded_positive(a, "limit", 12, 100));
    Ok(view::topics_md(&topics))
}

fn recent_text(a: &Value, scope: &crate::policy::ReadScope) -> Result<String> {
    let mut us = units::units()?;
    us.retain(|unit| scope.allows_unit(unit));
    if let Some(r) = a["repo"].as_str().filter(|r| !r.is_empty()) {
        us.retain(|u| u.repo == r);
    }
    us.truncate(bounded_positive(a, "limit", 20, 100));
    Ok(view::work_md(&us))
}

fn tool_defs(role: crate::policy::McpRole, scope: &crate::policy::ReadScope, allow_repo_paths: bool) -> Value {
    let obj = |props: Value, required: Value| json!({"type": "object", "properties": props, "required": required});
    let mut all = json!([
        {
            "name": "synty_search",
            "description": "Semantic search over this machine's coding-agent sessions and GitHub activity (PRs, issues, prompts). Use before starting work to find prior attempts, decisions, and related PRs. Full session ids and repo#123 references in the output feed synty_show.",
            "inputSchema": obj(json!({
                "query": {"type": "string", "maxLength": 4096, "description": "What to look for, in natural language"},
                "k": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Number of results (default 5)"},
                "filter": {"type": "string", "description": "Optional metadata filter, column=value (e.g. repo=sie-web, kind=pull_request, source=agent)"}
            }), json!(["query"])),
        },
        {
            "name": "synty_related",
            "description": "Prior work related to what you're doing now. Local clients may derive context from a repository; remote clients supply context text. Searches every repository the fleet has seen, and returned ids feed synty_show.",
            "inputSchema": obj(json!({
                "context": {"type": "string", "maxLength": 4096, "description": "Task/repository context text; required for remote HTTP clients"},
                "repo": {"type": "string", "description": "Absolute path to the git repo you're working in (defaults to the server's working directory)"},
                "k": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Number of results (default 5)"}
            }), json!([])),
        },
        {
            "name": "synty_topics",
            "description": "Emergent topics of recent work (clustered sessions, PRs, issues) with summaries and members. Topic keys and member ids in the output feed synty_show.",
            "inputSchema": obj(json!({
                "query": {"type": "string", "maxLength": 4096, "description": "Optional substring to filter topics"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Max topics (default 12)"}
            }), json!([])),
        },
        {
            "name": "synty_recent",
            "description": "Recent work units (sessions, PRs, issues), newest first. Ids in the output feed synty_show.",
            "inputSchema": obj(json!({
                "repo": {"type": "string", "description": "Optional repo name to filter"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Max units (default 20)"}
            }), json!([])),
        },
        {
            "name": "synty_status",
            "description": "Health: doc counts, repos, accounts, index freshness, and the fleet roster (machines, liveness, install rate, and who's active on GitHub but not tracked).",
            "inputSchema": obj(json!({}), json!([])),
        },
        {
            "name": "synty_stats",
            "description": "Usage and output over time: tokens, cache, tool calls, sessions, merged LOC, PRs, issues per Mon-aligned week, anchored to the most recent day with data.",
            "inputSchema": obj(json!({
                "weeks": {"type": "integer", "minimum": 1, "maximum": 52, "description": "Trailing weeks (default 4)"}
            }), json!([])),
        },
        {
            "name": "synty_tool",
            "description": "Fleet-wide profile of one tool: call/error volume, latency p50/p95, argument mix, recent invocations. Tool names appear in synty_status output.",
            "inputSchema": obj(json!({
                "name": {"type": "string", "description": "The tool name as the agent calls it (e.g. Bash, Edit, Read)"}
            }), json!(["name"])),
        },
        {
            "name": "synty_show",
            "description": "Detail for one id, as printed inline by synty_search/synty_recent/synty_topics: a session id (full or ≥4-char prefix), a PR/issue ref (repo#123 or gh:repo#123), or a topic key.",
            "inputSchema": obj(json!({
                "id": {"type": "string", "description": "The id to resolve"}
            }), json!(["id"])),
        },
        {
            "name": "synty_trace_list",
            "description": "List forensic agent turns, paired tool spans, or associated async jobs. Use to find slow waits, errors, and execution lifecycles; returned ids feed synty_trace_show and synty_trace_compare.",
            "inputSchema": obj(json!({
                "type": {"type": "string", "enum": ["turns", "spans", "jobs"]},
                "repo": {"type": "string"}, "machine": {"type": "string"},
                "source": {"type": "string"}, "status": {"type": "string"},
                "operation": {"type": "string"}, "has_errors": {"type": "boolean"},
                "since": {"type": "string"}, "min_ms": {"type": "integer", "minimum": 0},
                "sort": {"type": "string", "enum": ["recent", "duration", "wait"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            }), json!(["type"])),
        },
        {
            "name": "synty_trace_show",
            "description": "Show bounded execution evidence around one trace turn, span, job, event, or session id.",
            "inputSchema": obj(json!({
                "id": {"type": "string"}, "before": {"type": "integer", "minimum": 0, "maximum": 100},
                "after": {"type": "integer", "minimum": 0, "maximum": 100}
            }), json!(["id"])),
        },
        {
            "name": "synty_trace_search",
            "description": "Literal case-insensitive search over bounded execution evidence with optional repo, machine, source, and kind filters.",
            "inputSchema": obj(json!({
                "query": {"type": "string", "maxLength": 4096}, "repo": {"type": "string"},
                "machine": {"type": "string"}, "source": {"type": "string"},
                "kind": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            }), json!(["query"])),
        },
        {
            "name": "synty_trace_compare",
            "description": "Compare two trace turn, span, or job ids field by field.",
            "inputSchema": obj(json!({
                "left": {"type": "string"}, "right": {"type": "string"}
            }), json!(["left", "right"])),
        }
    ]);
    if !allow_repo_paths
        && let Some(related) = all.as_array_mut().and_then(|tools| tools.iter_mut().find(|tool| tool["name"] == "synty_related"))
    {
        related["inputSchema"]["properties"].as_object_mut().expect("related properties").remove("repo");
        related["inputSchema"]["required"] = json!(["context"]);
    }
    Value::Array(
        all.as_array()
            .expect("tool definitions are an array")
            .iter()
            .filter(|tool| {
                tool["name"].as_str().is_some_and(|name| {
                    role.allows_tool(name) && scope.exposes_tool(name)
                })
            })
            .cloned()
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn srv() -> Server {
        Server::new(
            "m".into(),
            crate::policy::McpRole::Operator,
            crate::policy::ReadScope::default(),
            crate::redact::Profile::Off,
            true,
            None,
            None,
        )
    }

    // The MCP handshake: initialize echoes the client's protocol version and
    // declares the tools capability.
    #[test]
    fn initialize_declares_tools() {
        let req = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "clientInfo": {"name": "x"}}});
        let resp = srv().handle(&req).expect("response");
        assert_eq!(resp["id"], 1);
        assert_eq!(resp["result"]["protocolVersion"], "2025-11-25");
        assert!(resp["result"]["capabilities"]["tools"].is_object());
        assert_eq!(resp["result"]["serverInfo"]["name"], "synty");
    }

    #[test]
    fn initialize_negotiates_unknown_protocol_to_the_latest_supported_version() {
        let req = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2099-01-01"}});
        let resp = srv().handle(&req).unwrap();
        assert_eq!(resp["result"]["protocolVersion"], PROTOCOL_VERSION);
    }

    #[test]
    fn restricted_scopes_do_not_advertise_or_dispatch_global_health_tools() {
        let mut server = Server::new(
            "m".into(),
            crate::policy::McpRole::Operator,
            crate::policy::ReadScope { repos: vec!["synty".into()], ..Default::default() },
            crate::redact::Profile::Off,
            true,
            None,
            None,
        );
        let listed = server.handle(&json!({"jsonrpc":"2.0","id":1,"method":"tools/list"})).unwrap();
        let names: Vec<&str> = listed["result"]["tools"].as_array().unwrap().iter()
            .filter_map(|tool| tool["name"].as_str()).collect();
        assert!(!names.contains(&"synty_status"));
        let called = server.handle(&json!({"jsonrpc":"2.0","id":2,"method":"tools/call",
            "params":{"name":"synty_status","arguments":{}}})).unwrap();
        assert!(called["error"]["message"].as_str().unwrap().contains("restricted"));
    }

    #[test]
    fn remote_related_schema_requires_context_and_hides_server_paths() {
        let tools = tool_defs(
            crate::policy::McpRole::Operator,
            &crate::policy::ReadScope::default(),
            false,
        );
        let related = tools.as_array().unwrap().iter()
            .find(|tool| tool["name"] == "synty_related").unwrap();
        assert_eq!(related["inputSchema"]["required"], json!(["context"]));
        assert!(related["inputSchema"]["properties"]["repo"].is_null());
    }

    // Notifications (no id) get no response — writing one would corrupt the stream.
    #[test]
    fn notifications_are_silent() {
        let req = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        assert!(srv().handle(&req).is_none());
    }

    // tools/list exposes the full read surface with schemas.
    #[test]
    fn tools_list_has_the_read_surfaces() {
        let req = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"});
        let resp = srv().handle(&req).expect("response");
        let names: Vec<&str> =
            resp["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(
            names,
            [
                "synty_search",
                "synty_related",
                "synty_topics",
                "synty_recent",
                "synty_status",
                "synty_stats",
                "synty_tool",
                "synty_show",
                "synty_trace_list",
                "synty_trace_show",
                "synty_trace_search",
                "synty_trace_compare",
            ]
        );
        assert_eq!(resp["result"]["tools"][0]["inputSchema"]["required"][0], "query");
    }

    // The drill tools declare their required argument, and calling without it
    // is an isError result (not a crash, not a silent empty answer) — before
    // any IO happens.
    #[test]
    fn new_tools_declare_and_enforce_required_args() {
        let req = json!({"jsonrpc": "2.0", "id": 7, "method": "tools/list"});
        let resp = srv().handle(&req).expect("response");
        let tools = resp["result"]["tools"].as_array().unwrap();
        let required = |n: &str| {
            tools.iter().find(|t| t["name"] == n).unwrap_or_else(|| panic!("{n} missing"))["inputSchema"]["required"][0]
                .clone()
        };
        assert_eq!(required("synty_tool"), "name");
        assert_eq!(required("synty_show"), "id");
        for (tool, hint) in [
            ("synty_tool", "name is required"),
            ("synty_show", "id is required"),
            ("synty_trace_list", "type is required"),
            ("synty_trace_show", "id is required"),
            ("synty_trace_search", "query is required"),
            ("synty_trace_compare", "left and right"),
        ] {
            let resp = srv()
                .handle(&json!({"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                    "params": {"name": tool, "arguments": {}}}))
                .unwrap();
            assert_eq!(resp["result"]["isError"], true, "{tool}");
            assert!(resp["result"]["content"][0]["text"].as_str().unwrap().contains(hint), "{tool}");
        }
    }

    // Unknown methods and unknown tools are JSON-RPC errors, not crashes.
    #[test]
    fn unknown_method_and_tool_error() {
        let resp = srv().handle(&json!({"jsonrpc": "2.0", "id": 3, "method": "bogus"})).unwrap();
        assert!(resp["error"]["message"].as_str().unwrap().contains("bogus"));
        let resp = srv()
            .handle(&json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                "params": {"name": "nope", "arguments": {}}}))
            .unwrap();
        assert!(resp["error"]["message"].as_str().unwrap().contains("nope"));
    }

    // A failing tool (missing query) reports through isError content, per MCP.
    #[test]
    fn tool_failure_is_iserror_content() {
        let resp = srv()
            .handle(&json!({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                "params": {"name": "synty_search", "arguments": {}}}))
            .unwrap();
        assert_eq!(resp["result"]["isError"], true);
        assert!(resp["result"]["content"][0]["text"].as_str().unwrap().contains("query"));
    }
}
