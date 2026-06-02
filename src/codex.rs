// Codex CLI session JSONL → canonical envelopes. Each
// ~/.codex/sessions/<y>/<m>/<d>/rollout-*.jsonl line is `{type, timestamp?,
// payload}`. The first line is `session_meta` and carries the session id; later
// records reference it by file, so the parser remembers it.

use crate::event::{kind, source};
use crate::tail::{resolve_ts, EmitCtx, FileParser, Source};
use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde_json::{json, Value};

pub struct Codex;

impl Source for Codex {
    fn id(&self) -> &'static str {
        "codex"
    }
    fn envelope_source(&self) -> &'static str {
        source::CODEX_CLI
    }
    /// The `cli_version` from the first `session_meta` line.
    fn detect_version(&self, head: &[u8]) -> String {
        for line in head.split(|&b| b == b'\n').take(20) {
            if line.is_empty() {
                continue;
            }
            #[derive(Deserialize)]
            struct Probe {
                #[serde(rename = "type")]
                rtype: String,
                payload: Option<Value>,
            }
            if let Ok(p) = serde_json::from_slice::<Probe>(line) {
                if p.rtype == "session_meta" {
                    return p
                        .payload
                        .and_then(|v| v.get("cli_version").and_then(Value::as_str).map(str::to_owned))
                        .unwrap_or_default();
                }
            }
        }
        String::new()
    }
    /// The 0.x line shares one record vocabulary and payload shape, so one
    /// parser covers it; new payload subtypes fall through to agent_meta. An
    /// empty version (head-detection failed on a truncated file) still parses.
    fn new_parser(&self, version: &str) -> Option<Box<dyn FileParser>> {
        if version.is_empty() || version.starts_with("0.") {
            Some(Box::new(CodexParser::default()))
        } else {
            None
        }
    }
}

#[derive(Default)]
struct CodexParser {
    session_id: String,
}

#[derive(Deserialize, Default)]
struct RawRecord {
    #[serde(rename = "type", default)]
    rtype: String,
    #[serde(default)]
    timestamp: String,
    #[serde(default)]
    payload: Value,
}

const FILE_TOOLS: &[&str] = &["apply_patch", "read_file", "write_file", "edit_file"];

impl FileParser for CodexParser {
    fn parse_line(&mut self, line: &[u8], ec: &mut EmitCtx) -> Result<Vec<crate::event::Event>> {
        let r: RawRecord =
            serde_json::from_slice(line).map_err(|e| anyhow!("codex: unmarshal: {e}"))?;
        if r.rtype.is_empty() {
            return Err(anyhow!("codex: missing type"));
        }
        let (ts_ms, ts) = resolve_ts(&r.timestamp, ec.fallback_ms());
        let sid = self.session_id.clone();

        let evts = match r.rtype.as_str() {
            "session_meta" => {
                let id = r.payload.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                self.session_id = id.clone();
                if id.is_empty() || !ec.first_seen(&id) {
                    vec![]
                } else {
                    let g = |k: &str| r.payload.get(k).and_then(Value::as_str).unwrap_or("").to_string();
                    vec![ec.event(
                        ts_ms,
                        &ts,
                        kind::SESSION_START,
                        &id,
                        json!({
                            "session_id": id,
                            "cli_version": g("cli_version"),
                            "cwd": g("cwd"),
                            "originator": g("originator"),
                            "source": g("source"),
                        }),
                    )]
                }
            }
            "turn_context" => vec![ec.event(
                ts_ms,
                &ts,
                kind::AGENT_META,
                &sid,
                json!({"subtype": "turn_context", "payload": r.payload}),
            )],
            "response_item" => handle_response_item(ts_ms, &ts, &r.payload, &sid, ec),
            "event_msg" => {
                let ek = r.payload.get("type").and_then(Value::as_str).unwrap_or("");
                vec![ec.event(
                    ts_ms,
                    &ts,
                    kind::AGENT_META,
                    &sid,
                    json!({"subtype": "event_msg", "event_kind": ek, "payload": r.payload}),
                )]
            }
            "compacted" => vec![ec.event(
                ts_ms,
                &ts,
                kind::AGENT_META,
                &sid,
                json!({"subtype": "compacted", "payload": r.payload}),
            )],
            other => vec![ec.event(
                ts_ms,
                &ts,
                kind::AGENT_META,
                &sid,
                json!({"subtype": other, "payload": r.payload}),
            )],
        };
        Ok(evts)
    }
}

fn handle_response_item(
    ts_ms: i64,
    ts: &str,
    p: &Value,
    sid: &str,
    ec: &mut EmitCtx,
) -> Vec<crate::event::Event> {
    let ptype = p.get("type").and_then(Value::as_str).unwrap_or("");
    match ptype {
        "message" => {
            let role = p.get("role").and_then(Value::as_str).unwrap_or("");
            let text = message_text(p);
            match role {
                "user" => vec![ec.event(ts_ms, ts, kind::USER_PROMPT, sid, json!({"text": text, "preview": preview(&text), "role": role}))],
                "assistant" => vec![ec.event(ts_ms, ts, kind::ASSISTANT_MESSAGE, sid, json!({"text": text, "preview": preview(&text), "role": role}))],
                _ => vec![ec.event(ts_ms, ts, kind::AGENT_META, sid, json!({"subtype": format!("message_{role}"), "text": text, "preview": preview(&text)}))],
            }
        }
        "reasoning" => {
            // The full reasoning is encrypted; the human-readable `summary[]` is
            // what survives (on some items). `content[]` is empty in practice.
            let text = join_texts(p.get("summary")).or_else(|| {
                let c = message_text(p);
                (!c.is_empty()).then_some(c)
            });
            let text = text.unwrap_or_default();
            vec![ec.event(ts_ms, ts, kind::THINKING, sid, json!({"text": text, "preview": preview(&text)}))]
        }
        "function_call" => {
            let name = p.get("name").and_then(Value::as_str).unwrap_or("");
            let call_id = p.get("call_id").and_then(Value::as_str).unwrap_or("");
            let args = p.get("arguments").and_then(Value::as_str).unwrap_or("");
            let mut out = vec![ec.event(ts_ms, ts, kind::TOOL_CALL, sid, json!({"name": name, "call_id": call_id, "arguments": args}))];
            if FILE_TOOLS.contains(&name) {
                if let Some(path) = path_from_args(args) {
                    out.push(ec.event(ts_ms, ts, kind::ATTACHMENT_REF, sid, json!({
                        "call_id": call_id, "tool_name": name, "local_path": path,
                        "source_system": format!("codex_{name}"),
                    })));
                }
            }
            out
        }
        "function_call_output" => {
            let call_id = p.get("call_id").and_then(Value::as_str).unwrap_or("");
            let output = p.get("output").and_then(Value::as_str).unwrap_or("");
            vec![ec.event(ts_ms, ts, kind::TOOL_RESULT, sid, json!({"call_id": call_id, "output": output}))]
        }
        "custom_tool_call" => {
            let name = p.get("name").and_then(Value::as_str).unwrap_or("");
            let call_id = p.get("call_id").and_then(Value::as_str).unwrap_or("");
            vec![ec.event(ts_ms, ts, kind::TOOL_CALL, sid, json!({
                "name": name, "call_id": call_id,
                "input": p.get("input").cloned().unwrap_or(Value::Null),
                "status": p.get("status").and_then(Value::as_str).unwrap_or(""),
                "variant": "custom",
            }))]
        }
        "custom_tool_call_output" => {
            let call_id = p.get("call_id").and_then(Value::as_str).unwrap_or("");
            let output = p.get("output").and_then(Value::as_str).unwrap_or("");
            vec![ec.event(ts_ms, ts, kind::TOOL_RESULT, sid, json!({"call_id": call_id, "output": output, "variant": "custom"}))]
        }
        "web_search_call" => vec![ec.event(ts_ms, ts, kind::TOOL_CALL, sid, json!({
            "name": "web_search",
            "action": p.get("action").cloned().unwrap_or(Value::Null),
            "status": p.get("status").and_then(Value::as_str).unwrap_or(""),
        }))],
        other => vec![ec.event(ts_ms, ts, kind::AGENT_META, sid, json!({"subtype": format!("response_item:{other}"), "payload": p}))],
    }
}

/// Join the non-empty `content[].text` parts of a message payload.
fn message_text(p: &Value) -> String {
    join_texts(p.get("content")).unwrap_or_default()
}

/// Join the non-empty `text` fields of a `[{..., text}]` array; None if the
/// array is absent or has no text.
fn join_texts(v: Option<&Value>) -> Option<String> {
    let arr = v?.as_array()?;
    let parts: Vec<&str> = arr
        .iter()
        .filter_map(|c| c.get("text").and_then(Value::as_str))
        .filter(|t| !t.is_empty())
        .collect();
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

/// A file path from a function_call's JSON-encoded arguments string.
fn path_from_args(args: &str) -> Option<String> {
    let v: Value = serde_json::from_str(args.trim()).ok()?;
    for key in ["path", "file_path", "target_file", "filename"] {
        if let Some(s) = v.get(key).and_then(Value::as_str) {
            if !s.is_empty() {
                return Some(s.to_string());
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
    use crate::event::{Event, Sequencer};
    use crate::tail::{drive, EmitCtx};
    use std::collections::HashSet;

    fn run(lines: &str, version: &str) -> Vec<Event> {
        let src = Codex;
        let mut parser = src.new_parser(version).expect("parser");
        let mut seq = Sequencer::new();
        let mut started = HashSet::new();
        let mut ec = EmitCtx::new("edge-x-codex".into(), &src, &mut seq, &mut started);
        drive(&mut *parser, lines.as_bytes(), "r.jsonl", 0, 1_700_000_000_000, &mut ec).0
    }

    // The current 0.133 version (and the whole 0.x line) is accepted — the gap
    // that left codex sessions uncaptured.
    #[test]
    fn current_version_is_supported() {
        let head = br#"{"type":"session_meta","payload":{"id":"S","cli_version":"0.133.0"}}"#;
        assert_eq!(Codex.detect_version(head), "0.133.0");
        assert!(Codex.new_parser("0.133.0").is_some());
        assert!(Codex.new_parser("0.120.5").is_some());
        assert!(Codex.new_parser("2.0.0").is_none());
    }

    // session_meta → one session_start; the id threads into later records.
    #[test]
    fn session_meta_then_user_and_assistant() {
        let evts = run(
            &[
                r#"{"type":"session_meta","timestamp":"2026-05-27T19:32:00Z","payload":{"id":"S1","cli_version":"0.133.0","cwd":"/c/sie"}}"#,
                r#"{"type":"response_item","timestamp":"2026-05-27T19:32:01Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"add a test"}]}}"#,
                r#"{"type":"response_item","timestamp":"2026-05-27T19:32:02Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}}"#,
            ]
            .join("\n"),
            "0.133.0",
        );
        assert_eq!(evts[0].kind, "session_start");
        assert_eq!(evts[0].session_id, "S1");
        assert_eq!(evts[0].source, "codex_cli");
        let user = evts.iter().find(|e| e.kind == "user_prompt").unwrap();
        assert_eq!(user.payload["text"], "add a test");
        assert_eq!(user.session_id, "S1"); // threaded from session_meta
        assert!(evts.iter().any(|e| e.kind == "assistant_message" && e.payload["text"] == "done"));
    }

    // apply_patch function_call → tool_call + attachment_ref with the path.
    #[test]
    fn function_call_apply_patch_emits_attachment() {
        let evts = run(
            &[
                r#"{"type":"session_meta","payload":{"id":"S2"}}"#,
                r#"{"type":"response_item","payload":{"type":"function_call","name":"apply_patch","call_id":"c1","arguments":"{\"path\":\"/c/sie/x.rs\"}"}}"#,
            ]
            .join("\n"),
            "0.133.0",
        );
        let att = evts.iter().find(|e| e.kind == "attachment_ref").unwrap();
        assert_eq!(att.payload["local_path"], "/c/sie/x.rs");
        assert_eq!(att.payload["source_system"], "codex_apply_patch");
        assert!(evts.iter().any(|e| e.kind == "tool_call" && e.payload["name"] == "apply_patch"));
    }

    // reasoning → thinking; event_msg → agent_meta carrying event_kind.
    #[test]
    fn reasoning_and_event_msg() {
        let evts = run(
            &[
                r#"{"type":"session_meta","payload":{"id":"S3"}}"#,
                r#"{"type":"response_item","payload":{"type":"reasoning","content":[{"type":"reasoning_text","text":"think"}]}}"#,
                r#"{"type":"event_msg","payload":{"type":"token_count","total":5}}"#,
            ]
            .join("\n"),
            "0.133.0",
        );
        assert!(evts.iter().any(|e| e.kind == "thinking" && e.payload["text"] == "think"));
        let em = evts.iter().find(|e| e.payload.get("subtype").and_then(|v| v.as_str()) == Some("event_msg")).unwrap();
        assert_eq!(em.payload["event_kind"], "token_count");
    }
}
