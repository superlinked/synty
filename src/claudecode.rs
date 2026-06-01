// Claude Code session JSONL → canonical envelopes. Each
// ~/.claude/projects/<slug>/<uuid>.jsonl line is one JSON record discriminated
// by `type` (user, assistant, attachment, system, …).

use crate::event::{kind, source, Event};
use crate::tail::{resolve_ts, EmitCtx, FileParser, Source};
use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde_json::{json, Value};

pub struct ClaudeCode;

impl Source for ClaudeCode {
    fn id(&self) -> &'static str {
        "claudecode"
    }
    fn envelope_source(&self) -> &'static str {
        source::CLAUDE_CODE
    }
    /// Scan up to 50 lines for the first `version` stamp (the first record is
    /// often a metadata stub without one).
    fn detect_version(&self, head: &[u8]) -> String {
        for line in head.split(|&b| b == b'\n').take(50) {
            if line.is_empty() {
                continue;
            }
            #[derive(Deserialize)]
            struct Probe {
                version: Option<String>,
            }
            if let Ok(p) = serde_json::from_slice::<Probe>(line) {
                if let Some(v) = p.version {
                    if !v.is_empty() {
                        return v;
                    }
                }
            }
        }
        String::new()
    }
    fn new_parser(&self, version: &str) -> Option<Box<dyn FileParser>> {
        if version.starts_with("2.1.") || version == "2.1" {
            Some(Box::new(ParserV21::default()))
        } else {
            None
        }
    }
}

/// Per-file state: the first sidechain record emits one subagent_parent edge.
#[derive(Default)]
struct ParserV21 {
    subagent_edge_emitted: bool,
}

#[derive(Deserialize, Default)]
struct RawRecord {
    #[serde(rename = "type", default)]
    rtype: String,
    #[serde(rename = "sessionId", default)]
    session_id: String,
    #[serde(default)]
    uuid: String,
    #[serde(default)]
    timestamp: String,
    #[serde(default)]
    version: String,
    #[serde(default)]
    cwd: String,
    #[serde(rename = "gitBranch", default)]
    git_branch: String,
    #[serde(default)]
    entrypoint: String,
    #[serde(rename = "isMeta", default)]
    is_meta: bool,
    #[serde(rename = "isSidechain", default)]
    is_sidechain: bool,
    #[serde(rename = "agentId", default)]
    agent_id: String,
    #[serde(default)]
    message: Value,
    #[serde(default)]
    subtype: String,
    #[serde(rename = "permissionMode", default)]
    permission_mode: String,
    #[serde(rename = "lastPrompt", default)]
    last_prompt: String,
    #[serde(default)]
    attachment: Value,
    #[serde(rename = "prNumber", default)]
    pr_number: i64,
    #[serde(rename = "prRepository", default)]
    pr_repository: String,
    #[serde(rename = "prUrl", default)]
    pr_url: String,
    #[serde(rename = "messageCount", default)]
    message_count: i64,
    #[serde(rename = "durationMs", default)]
    duration_ms: i64,
    #[serde(rename = "messageId", default)]
    message_id: String,
    #[serde(rename = "isSnapshotUpdate", default)]
    is_snapshot_update: bool,
}

const FILE_TOOLS: &[&str] = &["Read", "Write", "Edit", "MultiEdit", "NotebookEdit"];

impl FileParser for ParserV21 {
    fn parse_line(&mut self, line: &[u8], ec: &mut EmitCtx) -> Result<Vec<Event>> {
        let mut r: RawRecord =
            serde_json::from_slice(line).map_err(|e| anyhow!("claudecode: unmarshal: {e}"))?;
        if r.rtype.is_empty() {
            return Err(anyhow!("claudecode: missing type"));
        }
        let (ts_ms, ts) = resolve_ts(&r.timestamp, ec.fallback_ms());
        let mut out = Vec::new();

        // Subagent transcripts: rewrite SessionID to a synthetic child id and
        // emit one subagent_parent edge per file.
        if r.is_sidechain && !r.agent_id.is_empty() {
            let parent = r.session_id.clone();
            let child = format!("agent-{}", r.agent_id);
            r.session_id = child.clone();
            if !self.subagent_edge_emitted {
                self.subagent_edge_emitted = true;
                out.push(ec.event(
                    ts_ms,
                    &ts,
                    kind::AGENT_META,
                    &child,
                    json!({
                        "subtype": "subagent_parent",
                        "parent_session_id": parent,
                        "child_session_id": child,
                        "agent_id": r.agent_id,
                    }),
                ));
            }
        }

        if !r.session_id.is_empty() && ec.first_seen(&r.session_id) {
            out.push(ec.event(
                ts_ms,
                &ts,
                kind::SESSION_START,
                &r.session_id,
                json!({
                    "session_id": r.session_id,
                    "version": r.version,
                    "cwd": r.cwd,
                    "git_branch": r.git_branch,
                    "entrypoint": r.entrypoint,
                }),
            ));
        }

        match r.rtype.as_str() {
            "user" => out.extend(handle_user(ts_ms, &ts, &r, ec)),
            "assistant" => out.extend(handle_assistant(ts_ms, &ts, &r, ec)),
            "attachment" => out.push(ec.event(
                ts_ms,
                &ts,
                kind::ATTACHMENT_REF,
                &r.session_id,
                json!({"attachment": r.attachment, "uuid": r.uuid, "cwd": r.cwd}),
            )),
            "permission-mode" => out.push(ec.event(
                ts_ms,
                &ts,
                kind::AGENT_META,
                &r.session_id,
                json!({"subtype": "permission_mode", "permission_mode": r.permission_mode}),
            )),
            "last-prompt" => out.push(ec.event(
                ts_ms,
                &ts,
                kind::AGENT_META,
                &r.session_id,
                json!({"subtype": "last_prompt", "last_prompt": r.last_prompt}),
            )),
            "file-history-snapshot" => out.push(ec.event(
                ts_ms,
                &ts,
                kind::AGENT_META,
                &r.session_id,
                json!({
                    "subtype": "file_history_snapshot",
                    "message_id": r.message_id,
                    "is_snapshot_update": r.is_snapshot_update,
                }),
            )),
            "system" => out.push(ec.event(
                ts_ms,
                &ts,
                kind::AGENT_META,
                &r.session_id,
                json!({"subtype": r.subtype, "message_count": r.message_count, "duration_ms": r.duration_ms}),
            )),
            "pr-link" => out.push(ec.event(
                ts_ms,
                &ts,
                kind::AGENT_META,
                &r.session_id,
                json!({
                    "subtype": "pr_link",
                    "pr_number": r.pr_number,
                    "pr_repository": r.pr_repository,
                    "pr_url": r.pr_url,
                }),
            )),
            other => out.push(ec.event(
                ts_ms,
                &ts,
                kind::AGENT_META,
                &r.session_id,
                json!({"subtype": other}),
            )),
        }
        Ok(out)
    }
}

fn handle_user(ts_ms: i64, ts: &str, r: &RawRecord, ec: &mut EmitCtx) -> Vec<Event> {
    let content = r.message.get("content").cloned().unwrap_or(Value::Null);
    let (ukind, meta) = if r.is_meta { (kind::AGENT_META, true) } else { (kind::USER_PROMPT, false) };

    // content is either a plain string (free-form prompt) …
    if let Some(s) = content.as_str() {
        let mut p = json!({"text": s, "preview": preview(s)});
        if meta {
            p["subtype"] = json!("user_meta");
        }
        return vec![ec.event(ts_ms, ts, ukind, &r.session_id, p)];
    }

    // … or an array of blocks (tool_result, or text).
    let mut out = Vec::new();
    if let Some(blocks) = content.as_array() {
        for b in blocks {
            match b.get("type").and_then(Value::as_str) {
                Some("tool_result") => out.push(ec.event(
                    ts_ms,
                    ts,
                    kind::TOOL_RESULT,
                    &r.session_id,
                    json!({
                        "tool_use_id": b.get("tool_use_id").and_then(Value::as_str).unwrap_or(""),
                        "content": b.get("content").cloned().unwrap_or(Value::Null),
                        "is_error": b.get("is_error").and_then(Value::as_bool).unwrap_or(false),
                    }),
                )),
                _ => {
                    let text = b.get("text").and_then(Value::as_str).unwrap_or("");
                    let mut p = json!({"text": text, "preview": preview(text)});
                    if meta {
                        p["subtype"] = json!("user_meta");
                    }
                    out.push(ec.event(ts_ms, ts, ukind, &r.session_id, p));
                }
            }
        }
    }
    out
}

fn handle_assistant(ts_ms: i64, ts: &str, r: &RawRecord, ec: &mut EmitCtx) -> Vec<Event> {
    let model = r.message.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    let blocks = r.message.get("content").and_then(Value::as_array).cloned().unwrap_or_default();

    let mut out = Vec::new();
    let mut text_chunks: Vec<String> = Vec::new();
    for b in &blocks {
        match b.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(t) = b.get("text").and_then(Value::as_str) {
                    text_chunks.push(t.to_string());
                }
            }
            Some("thinking") => {
                let t = b.get("thinking").and_then(Value::as_str).unwrap_or("");
                out.push(ec.event(ts_ms, ts, kind::THINKING, &r.session_id, json!({"text": t, "preview": preview(t)})));
            }
            Some("tool_use") => {
                let id = b.get("id").and_then(Value::as_str).unwrap_or("");
                let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                let input = b.get("input").cloned().unwrap_or(Value::Null);
                out.push(ec.event(
                    ts_ms,
                    ts,
                    kind::TOOL_CALL,
                    &r.session_id,
                    json!({"tool_use_id": id, "name": name, "input": input}),
                ));
                if FILE_TOOLS.contains(&name) {
                    if let Some(path) = input_file_path(&input) {
                        out.push(ec.event(
                            ts_ms,
                            ts,
                            kind::ATTACHMENT_REF,
                            &r.session_id,
                            json!({
                                "tool_use_id": id,
                                "tool_name": name,
                                "local_path": path,
                                "source_system": format!("claudecode_{}", name.to_lowercase()),
                            }),
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    if !text_chunks.is_empty() {
        let joined = text_chunks.join("\n\n");
        out.push(ec.event(
            ts_ms,
            ts,
            kind::ASSISTANT_MESSAGE,
            &r.session_id,
            json!({"text": joined, "preview": preview(&joined), "model": model}),
        ));
    }
    out
}

fn input_file_path(input: &Value) -> Option<String> {
    for key in ["file_path", "notebook_path", "path", "target_file"] {
        if let Some(v) = input.get(key).and_then(Value::as_str) {
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

fn preview(s: &str) -> String {
    if s.chars().count() <= 200 {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(200).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::Sequencer;
    use crate::tail::{drive, EmitCtx};
    use std::collections::HashSet;

    fn run(lines: &str) -> Vec<Event> {
        let src = ClaudeCode;
        let mut parser = src.new_parser("2.1.119").expect("parser");
        let mut seq = Sequencer::new();
        let mut started = HashSet::new();
        let mut ec = EmitCtx::new("edge-x-claudecode".into(), &src, &mut seq, &mut started);
        drive(&mut *parser, lines.as_bytes(), "f.jsonl", 0, 1_700_000_000_000, &mut ec).0
    }

    #[test]
    fn version_detection_skips_stub_first_line() {
        let head = b"{\"type\":\"permission-mode\",\"permissionMode\":\"default\"}\n{\"type\":\"user\",\"version\":\"2.1.119\"}";
        assert_eq!(ClaudeCode.detect_version(head), "2.1.119");
        assert!(ClaudeCode.new_parser("2.1.119").is_some());
        assert!(ClaudeCode.new_parser("1.0.0").is_none());
    }

    // A user prompt becomes one session_start + one user_prompt; a string
    // content turns into the text payload ingest reads.
    #[test]
    fn user_string_prompt_yields_session_start_then_prompt() {
        let evts = run(r#"{"type":"user","sessionId":"S1","timestamp":"2026-05-31T20:00:00Z","version":"2.1.119","cwd":"/c/sie-internal","message":{"role":"user","content":"fix the login redirect"}}"#);
        assert_eq!(evts.len(), 2);
        assert_eq!(evts[0].kind, "session_start");
        assert_eq!(evts[0].session_id, "S1");
        assert_eq!(evts[1].kind, "user_prompt");
        assert_eq!(evts[1].payload["text"], "fix the login redirect");
        assert_eq!(evts[1].source, "claude_code");
    }

    // Assistant text + a file-touching tool_use → assistant_message, tool_call,
    // and a sibling attachment_ref carrying the file path.
    #[test]
    fn assistant_text_and_file_tool_emit_message_call_and_attachment() {
        let line = r#"{"type":"assistant","sessionId":"S1","timestamp":"2026-05-31T20:01:00Z","message":{"role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"editing auth.ts"},{"type":"tool_use","id":"t1","name":"Edit","input":{"file_path":"/c/sie/auth.ts"}}]}}"#;
        let evts = run(&format!(
            "{}\n{}",
            r#"{"type":"user","sessionId":"S1","timestamp":"2026-05-31T20:00:00Z","message":{"content":"go"}}"#,
            line
        ));
        let kinds: Vec<&str> = evts.iter().map(|e| e.kind.as_str()).collect();
        assert!(kinds.contains(&"tool_call"));
        assert!(kinds.contains(&"attachment_ref"));
        assert!(kinds.contains(&"assistant_message"));
        let att = evts.iter().find(|e| e.kind == "attachment_ref").unwrap();
        assert_eq!(att.payload["local_path"], "/c/sie/auth.ts");
        assert_eq!(att.payload["source_system"], "claudecode_edit");
        let msg = evts.iter().find(|e| e.kind == "assistant_message").unwrap();
        assert_eq!(msg.payload["model"], "claude-opus-4-8");
    }

    // session_start is emitted once even across multiple records.
    #[test]
    fn session_start_emitted_once() {
        let evts = run(&format!(
            "{}\n{}",
            r#"{"type":"user","sessionId":"S2","timestamp":"2026-05-31T20:00:00Z","message":{"content":"a"}}"#,
            r#"{"type":"user","sessionId":"S2","timestamp":"2026-05-31T20:00:01Z","message":{"content":"b"}}"#
        ));
        assert_eq!(evts.iter().filter(|e| e.kind == "session_start").count(), 1);
    }

    // A sidechain record rewrites the session to a synthetic child and emits a
    // subagent_parent edge.
    #[test]
    fn sidechain_emits_subagent_parent_edge() {
        let evts = run(r#"{"type":"user","sessionId":"PARENT","isSidechain":true,"agentId":"a2","timestamp":"2026-05-31T20:00:00Z","message":{"content":"sub task"}}"#);
        let edge = evts.iter().find(|e| e.payload.get("subtype").and_then(|v| v.as_str()) == Some("subagent_parent")).unwrap();
        assert_eq!(edge.payload["parent_session_id"], "PARENT");
        assert_eq!(edge.payload["child_session_id"], "agent-a2");
        assert!(evts.iter().all(|e| e.session_id == "agent-a2"));
    }
}
