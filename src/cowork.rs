// Claude Cowork audit.jsonl → canonical envelopes. Cowork spawns Claude Code
// for the actual model calls and writes one tamper-evident `audit.jsonl` per
// session; presence of `_audit_hmac` is the v1 signature. The inner Claude Code
// session files nested under each session are intentionally not parsed here
// (they lack `_audit_hmac`, so they route to no-parser) to avoid double-counting.
//
// User/assistant content is the Claude message shape, decoded in `blocks`.

use crate::event::{kind, source, Event};
use crate::tail::{resolve_ts, EmitCtx, FileParser, Source};
use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

/// Lifecycle records buffered before a session proves real, after which it is
/// dropped as a heartbeat. Real sessions hit a user record within a few lines.
const HEARTBEAT_CAP: usize = 200;

pub struct Cowork;

impl Source for Cowork {
    fn id(&self) -> &'static str {
        "cowork"
    }
    fn envelope_source(&self) -> &'static str {
        source::COWORK
    }
    /// "v1" if any head record carries `_audit_hmac`; otherwise "" so non-audit
    /// files (e.g. the inner Claude Code transcripts) get no parser and skip.
    fn detect_version(&self, head: &[u8]) -> String {
        for line in head.split(|&b| b == b'\n').take(50) {
            if line.is_empty() {
                continue;
            }
            #[derive(Deserialize)]
            struct Probe {
                #[serde(rename = "_audit_hmac")]
                hmac: Option<String>,
            }
            if let Ok(p) = serde_json::from_slice::<Probe>(line) {
                if p.hmac.as_deref().is_some_and(|s| !s.is_empty()) {
                    return "v1".into();
                }
            }
        }
        String::new()
    }
    fn new_parser(&self, version: &str) -> Option<Box<dyn FileParser>> {
        (version == "v1").then(|| Box::new(CoworkParser::default()) as Box<dyn FileParser>)
    }
}

#[derive(Default)]
struct CoworkParser {
    real: HashSet<String>,
    dropped: HashSet<String>,
    prelude: HashMap<String, usize>,
}

// String fields use the null-tolerant deserializer: cowork records carry
// nullable fields (parent_tool_use_id, and others), and a single null would
// otherwise fail the whole record and drop the line.
#[derive(Deserialize, Default)]
struct RawRecord {
    #[serde(rename = "type", default, deserialize_with = "crate::tail::de_null_string")]
    rtype: String,
    #[serde(default, deserialize_with = "crate::tail::de_null_string")]
    subtype: String,
    #[serde(default, deserialize_with = "crate::tail::de_null_string")]
    session_id: String,
    #[serde(rename = "parent_tool_use_id", default, deserialize_with = "crate::tail::de_null_string")]
    parent_tool_use_id: String,
    #[serde(rename = "client_platform", default, deserialize_with = "crate::tail::de_null_string")]
    client_platform: String,
    #[serde(default, deserialize_with = "crate::tail::de_null_string")]
    timestamp: String,
    #[serde(rename = "_audit_timestamp", default, deserialize_with = "crate::tail::de_null_string")]
    audit_timestamp: String,
    #[serde(default, deserialize_with = "crate::tail::de_null_string")]
    cwd: String,
    #[serde(default)]
    message: Value,
    #[serde(default, deserialize_with = "crate::tail::de_null_string")]
    operation: String,
    #[serde(default, deserialize_with = "crate::tail::de_null_string")]
    content: String,
    #[serde(default, deserialize_with = "crate::tail::de_null_string")]
    status: String,
    #[serde(default, deserialize_with = "crate::tail::de_null_string")]
    model: String,
    #[serde(rename = "claude_code_version", default, deserialize_with = "crate::tail::de_null_string")]
    claude_code_version: String,
    #[serde(rename = "permissionMode", default, deserialize_with = "crate::tail::de_null_string")]
    permission_mode: String,
    #[serde(default)]
    result: Value,
}

impl FileParser for CoworkParser {
    fn parse_line(&mut self, line: &[u8], ec: &mut EmitCtx) -> Result<Vec<Event>> {
        let r: RawRecord =
            serde_json::from_slice(line).map_err(|e| anyhow!("cowork: unmarshal: {e}"))?;
        if r.rtype.is_empty() {
            return Err(anyhow!("cowork: missing type"));
        }

        let sid = &r.session_id;
        let is_real = r.rtype == "user" || r.rtype == "assistant";

        // Drop lifecycle preludes of sessions that never produce a real record.
        if !is_real && !sid.is_empty() {
            if self.dropped.contains(sid) {
                return Ok(vec![]);
            }
            if !self.real.contains(sid) {
                let c = self.prelude.entry(sid.clone()).or_insert(0);
                *c += 1;
                if *c > HEARTBEAT_CAP {
                    self.dropped.insert(sid.clone());
                    self.prelude.remove(sid);
                }
                return Ok(vec![]);
            }
        }
        if is_real && !sid.is_empty() {
            self.real.insert(sid.clone());
            self.prelude.remove(sid);
            self.dropped.remove(sid);
        }

        let raw_ts = if r.audit_timestamp.is_empty() { &r.timestamp } else { &r.audit_timestamp };
        let (ts_ms, ts) = resolve_ts(raw_ts, ec.fallback_ms());
        let mut out = Vec::new();

        if !sid.is_empty() && ec.first_seen(sid) {
            out.push(ec.event(
                ts_ms,
                &ts,
                kind::SESSION_START,
                sid,
                json!({
                    "session_id": sid,
                    "version": "v1",
                    "client_platform": r.client_platform,
                    "parent_tool_use_id": r.parent_tool_use_id,
                }),
            ));
        }

        match r.rtype.as_str() {
            "user" => {
                let content = r.message.get("content").cloned().unwrap_or(Value::Null);
                out.extend(crate::blocks::user_content(&content, false, ts_ms, &ts, sid, ec));
            }
            "assistant" => {
                let model = r.message.get("model").and_then(Value::as_str).unwrap_or("");
                let content = r.message.get("content").and_then(Value::as_array).cloned().unwrap_or_default();
                out.extend(crate::blocks::assistant_content(&content, model, "cowork", ts_ms, &ts, sid, ec));
            }
            "system" => out.push(ec.event(
                ts_ms,
                &ts,
                kind::AGENT_META,
                sid,
                json!({
                    "subtype": format!("cowork_{}", non_empty(&r.subtype, "system")),
                    "cwd": r.cwd,
                    "model": r.model,
                    "claude_code_version": r.claude_code_version,
                    "permission_mode": r.permission_mode,
                }),
            )),
            "result" => out.push(ec.event(
                ts_ms,
                &ts,
                kind::AGENT_META,
                sid,
                json!({"subtype": "cowork_result", "status": non_empty(&r.subtype, &r.status), "result": r.result}),
            )),
            "rate_limit_event" => out.push(ec.event(
                ts_ms,
                &ts,
                kind::AGENT_META,
                sid,
                json!({"subtype": "cowork_rate_limit"}),
            )),
            "queue-operation" => out.push(ec.event(
                ts_ms,
                &ts,
                kind::AGENT_META,
                sid,
                json!({"subtype": "cowork_queue", "operation": r.operation, "preview": crate::blocks::preview(&r.content)}),
            )),
            other => out.push(ec.event(ts_ms, &ts, kind::AGENT_META, sid, json!({"subtype": other}))),
        }
        Ok(out)
    }
}

fn non_empty<'a>(a: &'a str, b: &'a str) -> &'a str {
    if a.is_empty() {
        b
    } else {
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::Sequencer;
    use crate::tail::{drive, EmitCtx};
    use std::collections::HashSet;

    fn run(lines: &str) -> Vec<Event> {
        let src = Cowork;
        let mut parser = src.new_parser("v1").expect("parser");
        let mut seq = Sequencer::new();
        let mut started = HashSet::new();
        let mut ec = EmitCtx::new("edge-x-cowork".into(), &src, &mut seq, &mut started);
        drive(&mut *parser, lines.as_bytes(), "audit.jsonl", 0, 1_700_000_000_000, &mut ec).0
    }

    #[test]
    fn audit_hmac_is_the_v1_signature() {
        assert_eq!(Cowork.detect_version(br#"{"type":"system","_audit_hmac":"abc123"}"#), "v1");
        // No _audit_hmac (e.g. an inner Claude Code file) → not cowork, skip.
        assert_eq!(Cowork.detect_version(br#"{"type":"user","sessionId":"x"}"#), "");
        assert!(Cowork.new_parser("").is_none());
    }

    // A real user turn yields session_start + user_prompt, and the envelope
    // source is cowork. Audit timestamp is preferred over the record's own.
    #[test]
    fn user_turn_yields_session_start_and_prompt() {
        let evts = run(r#"{"type":"user","session_id":"S1","_audit_hmac":"h","_audit_timestamp":"2026-04-25T10:00:00Z","timestamp":"2026-04-25T09:59:00Z","message":{"role":"user","content":"build the thing"}}"#);
        assert_eq!(evts[0].kind, "session_start");
        assert_eq!(evts[0].source, "cowork");
        assert_eq!(evts[0].ts, "2026-04-25T10:00:00.000Z"); // audit ts wins
        assert!(evts.iter().any(|e| e.kind == "user_prompt" && e.payload["text"] == "build the thing"));
    }

    // A file-touching tool_use in an assistant turn tags the attachment with
    // the cowork source_system.
    #[test]
    fn assistant_file_tool_tagged_cowork() {
        let evts = run(&[
            r#"{"type":"user","session_id":"S1","_audit_hmac":"h","message":{"content":"go"}}"#,
            r#"{"type":"assistant","session_id":"S1","_audit_hmac":"h","message":{"model":"claude-opus-4-8","content":[{"type":"tool_use","id":"t1","name":"Write","input":{"file_path":"/c/x.rs"}}]}}"#,
        ].join("\n"));
        let att = evts.iter().find(|e| e.kind == "attachment_ref").unwrap();
        assert_eq!(att.payload["source_system"], "cowork_write");
        assert_eq!(att.payload["local_path"], "/c/x.rs");
    }

    // A heartbeat session — only lifecycle records, never a user/assistant —
    // emits nothing once it overflows the prelude cap.
    #[test]
    fn heartbeat_prelude_is_dropped_past_cap() {
        let mut lines = String::new();
        for _ in 0..(HEARTBEAT_CAP + 5) {
            lines.push_str(r#"{"type":"system","subtype":"status","session_id":"JUNK","_audit_hmac":"h"}"#);
            lines.push('\n');
        }
        let evts = run(&lines);
        assert!(evts.iter().all(|e| e.session_id != "JUNK"), "heartbeat session must not emit");
    }

    // A real record rescues a session whose prelude was still under the cap:
    // the prelude is abandoned but the conversation flows.
    #[test]
    fn real_record_after_short_prelude_flows() {
        let evts = run(&[
            r#"{"type":"system","subtype":"init","session_id":"S2","_audit_hmac":"h"}"#,
            r#"{"type":"user","session_id":"S2","_audit_hmac":"h","message":{"content":"real work"}}"#,
        ].join("\n"));
        // prelude system record abandoned; session_start + user_prompt present
        assert!(evts.iter().any(|e| e.kind == "session_start"));
        assert!(evts.iter().any(|e| e.kind == "user_prompt" && e.payload["text"] == "real work"));
    }
}
