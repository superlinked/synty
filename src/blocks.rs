// Claude message-content decoding shared by the Claude Code and Cowork tailers:
// both carry the same `message` shape — content is either a plain string or an
// array of text / thinking / tool_use / tool_result blocks.

use crate::event::{kind, Event};
use crate::tail::EmitCtx;
use serde_json::{json, Value};

const FILE_TOOLS: &[&str] = &["Read", "Write", "Edit", "MultiEdit", "NotebookEdit"];

/// Decode a user record's `content`. A string is one prompt; an array yields a
/// tool_result per tool_result block and a user_prompt per other block. `meta`
/// routes a synthetic command-bracket record (Claude Code's isMeta) to
/// agent_meta instead of user_prompt.
pub fn user_content(content: &Value, meta: bool, ts_ms: i64, ts: &str, sid: &str, ec: &mut EmitCtx) -> Vec<Event> {
    let ukind = if meta { kind::AGENT_META } else { kind::USER_PROMPT };
    let tag = |p: &mut Value| {
        if meta {
            p["subtype"] = json!("user_meta");
        }
    };

    if let Some(s) = content.as_str() {
        let mut p = json!({"text": s, "preview": preview(s)});
        tag(&mut p);
        return vec![ec.event(ts_ms, ts, ukind, sid, p)];
    }

    let mut out = Vec::new();
    if let Some(blocks) = content.as_array() {
        for b in blocks {
            if b.get("type").and_then(Value::as_str) == Some("tool_result") {
                out.push(ec.event(ts_ms, ts, kind::TOOL_RESULT, sid, json!({
                    "tool_use_id": b.get("tool_use_id").and_then(Value::as_str).unwrap_or(""),
                    "content": b.get("content").cloned().unwrap_or(Value::Null),
                    "is_error": b.get("is_error").and_then(Value::as_bool).unwrap_or(false),
                })));
            } else {
                let text = b.get("text").and_then(Value::as_str).unwrap_or("");
                let mut p = json!({"text": text, "preview": preview(text)});
                tag(&mut p);
                out.push(ec.event(ts_ms, ts, ukind, sid, p));
            }
        }
    }
    out
}

/// Decode an assistant record's content array: text blocks join into one
/// assistant_message; thinking and tool_use each become their own envelope; a
/// file-touching tool also emits a sibling attachment_ref tagged `<prefix>_<tool>`.
pub fn assistant_content(content: &[Value], model: &str, prefix: &str, ts_ms: i64, ts: &str, sid: &str, ec: &mut EmitCtx) -> Vec<Event> {
    let mut out = Vec::new();
    let mut chunks: Vec<String> = Vec::new();
    for b in content {
        match b.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(t) = b.get("text").and_then(Value::as_str) {
                    chunks.push(t.to_string());
                }
            }
            Some("thinking") => {
                let t = b.get("thinking").and_then(Value::as_str).unwrap_or("");
                out.push(ec.event(ts_ms, ts, kind::THINKING, sid, json!({"text": t, "preview": preview(t)})));
            }
            Some("tool_use") => {
                let id = b.get("id").and_then(Value::as_str).unwrap_or("");
                let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                let input = b.get("input").cloned().unwrap_or(Value::Null);
                out.push(ec.event(ts_ms, ts, kind::TOOL_CALL, sid, json!({"tool_use_id": id, "name": name, "input": input})));
                if FILE_TOOLS.contains(&name) {
                    if let Some(path) = input_file_path(&input) {
                        out.push(ec.event(ts_ms, ts, kind::ATTACHMENT_REF, sid, json!({
                            "tool_use_id": id,
                            "tool_name": name,
                            "local_path": path,
                            "source_system": format!("{prefix}_{}", name.to_lowercase()),
                        })));
                    }
                }
            }
            _ => {}
        }
    }
    if !chunks.is_empty() {
        let joined = chunks.join("\n\n");
        out.push(ec.event(ts_ms, ts, kind::ASSISTANT_MESSAGE, sid, json!({"text": joined, "preview": preview(&joined), "model": model})));
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

pub fn preview(s: &str) -> String {
    if s.chars().count() <= 200 {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(200).collect::<String>())
    }
}
