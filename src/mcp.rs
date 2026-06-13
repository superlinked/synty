// `synty mcp` — an MCP server over stdio (JSON-RPC 2.0, newline-delimited)
// exposing synty's read surfaces as agent tools, so a coding agent can consult
// past work mid-session (synty_search / synty_topics / synty_recent /
// synty_status) instead of shelling out. The protocol slice MCP needs here is
// small (initialize, tools/list, tools/call, ping), so it's hand-rolled — no
// new dependencies, and stdout carries protocol JSON only (logs go to stderr).

use crate::{encode::Encoder, load_docs, readmodel, search, units, view};
use anyhow::Result;
use next_plaid::{MmapIndex, SearchParameters};
use serde_json::{json, Value};
use std::io::{BufRead, Write};

pub fn run(model_id: String) -> Result<()> {
    let mut srv = Server { model_id, engine: None };
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

struct Server {
    model_id: String,
    /// The build the engine serves + encoder + index — loaded on the first
    /// search call, kept warm, reopened when the pointer moves.
    engine: Option<(readmodel::Current, Encoder, MmapIndex)>,
}

impl Server {
    /// One JSON-RPC message → one response, or None for notifications.
    fn handle(&mut self, req: &Value) -> Option<Value> {
        let id = match req.get("id") {
            Some(v) if !v.is_null() => v.clone(),
            _ => return None, // notification (e.g. notifications/initialized)
        };
        let method = req["method"].as_str().unwrap_or("");
        let result = match method {
            "initialize" => Ok(json!({
                "protocolVersion": req["params"]["protocolVersion"].as_str().unwrap_or("2025-03-26"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "synty", "version": env!("CARGO_PKG_VERSION")},
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({"tools": tool_defs()})),
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
        let out = match name {
            "synty_search" => self.search(a),
            "synty_topics" => topics_text(a),
            "synty_recent" => recent_text(a),
            "synty_status" => view::status().map(|s| view::status_md(&s)),
            "synty_stats" => view::stats(a["weeks"].as_u64().unwrap_or(4) as usize).map(|s| view::stats_md(&s)),
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
                    view::show_report(id)
                }
            }
            other => return Err(format!("unknown tool: {other}")),
        };
        Ok(match out {
            Ok(text) => json!({"content": [{"type": "text", "text": text}], "isError": false}),
            Err(e) => json!({"content": [{"type": "text", "text": e.to_string()}], "isError": true}),
        })
    }

    fn search(&mut self, a: &Value) -> Result<String> {
        let query = a["query"].as_str().unwrap_or("");
        anyhow::ensure!(!query.is_empty(), "query is required");
        let k = a["k"].as_u64().unwrap_or(5) as usize;
        let filter = a["filter"].as_str().filter(|f| !f.is_empty());
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
        let subset = search::subset_for(filter)?;
        let params = SearchParameters { top_k: k, ..Default::default() };
        let res = idx.search(&q, &params, subset.as_deref()).map_err(|e| anyhow::anyhow!("search: {e}"))?;
        let mut out = search::render(&docs, query, filter, &res);
        if let Some(note) = view::stale_note() {
            out.push_str(&format!("\n_{note}_\n"));
        }
        Ok(out)
    }
}

fn topics_text(a: &Value) -> Result<String> {
    let mut topics = units::topic_units(12)?;
    if let Some(q) = a["query"].as_str().filter(|q| !q.is_empty()) {
        let ql = q.to_lowercase();
        topics.retain(|t| {
            t.label.to_lowercase().contains(&ql)
                || t.units.iter().any(|u| u.title.to_lowercase().contains(&ql))
        });
    }
    Ok(view::topics_md(&topics))
}

fn recent_text(a: &Value) -> Result<String> {
    let mut us = units::units()?;
    if let Some(r) = a["repo"].as_str().filter(|r| !r.is_empty()) {
        us.retain(|u| u.repo == r);
    }
    us.truncate(a["limit"].as_u64().unwrap_or(20) as usize);
    Ok(view::work_md(&us))
}

fn tool_defs() -> Value {
    let obj = |props: Value, required: Value| json!({"type": "object", "properties": props, "required": required});
    json!([
        {
            "name": "synty_search",
            "description": "Semantic search over this machine's coding-agent sessions and GitHub activity (PRs, issues, prompts). Use before starting work to find prior attempts, decisions, and related PRs. Ids in the output ([a1b2c3d4], repo#123) feed synty_show.",
            "inputSchema": obj(json!({
                "query": {"type": "string", "description": "What to look for, in natural language"},
                "k": {"type": "integer", "description": "Number of results (default 5)"},
                "filter": {"type": "string", "description": "Optional metadata filter, column=value (e.g. repo=sie-web, kind=pull_request, source=agent)"}
            }), json!(["query"])),
        },
        {
            "name": "synty_topics",
            "description": "Emergent topics of recent work (clustered sessions, PRs, issues) with summaries and members. Topic keys and member ids in the output feed synty_show.",
            "inputSchema": obj(json!({
                "query": {"type": "string", "description": "Optional substring to filter topics"}
            }), json!([])),
        },
        {
            "name": "synty_recent",
            "description": "Recent work units (sessions, PRs, issues), newest first. Ids in the output feed synty_show.",
            "inputSchema": obj(json!({
                "repo": {"type": "string", "description": "Optional repo name to filter"},
                "limit": {"type": "integer", "description": "Max units (default 20)"}
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
                "weeks": {"type": "integer", "description": "Trailing weeks (default 4)"}
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
        }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn srv() -> Server {
        Server { model_id: "m".into(), engine: None }
    }

    // The MCP handshake: initialize echoes the client's protocol version and
    // declares the tools capability.
    #[test]
    fn initialize_declares_tools() {
        let req = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": "x"}}});
        let resp = srv().handle(&req).expect("response");
        assert_eq!(resp["id"], 1);
        assert_eq!(resp["result"]["protocolVersion"], "2025-06-18");
        assert!(resp["result"]["capabilities"]["tools"].is_object());
        assert_eq!(resp["result"]["serverInfo"]["name"], "synty");
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
            ["synty_search", "synty_topics", "synty_recent", "synty_status", "synty_stats", "synty_tool", "synty_show"]
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
        for (tool, hint) in [("synty_tool", "name is required"), ("synty_show", "id is required")] {
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
