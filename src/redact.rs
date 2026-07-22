// Built-in redaction profiles for data crossing trust boundaries. Local raw
// capture remains available; team uploads and remote MCP responses can remove
// common credential shapes and bound untrusted payload sizes.

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::str::FromStr;
use std::sync::LazyLock;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    Off,
    #[default]
    Standard,
    McpSafe,
}

impl Profile {
    pub fn max_chars(self) -> Option<usize> {
        match self {
            Self::Off => None,
            Self::Standard => Some(32 * 1024),
            Self::McpSafe => Some(4 * 1024),
        }
    }
}

impl FromStr for Profile {
    type Err = anyhow::Error;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "standard" => Ok(Self::Standard),
            "mcp_safe" | "mcp-safe" => Ok(Self::McpSafe),
            _ => anyhow::bail!("redaction profile must be off, standard, or mcp_safe"),
        }
    }
}

static SECRET_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}",
        r"\b(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{12,}",
        r"\bsk-[A-Za-z0-9_-]{16,}",
        r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
        r"(?i)\b(?:AWS_SECRET_ACCESS_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|GITHUB_TOKEN)\s*=\s*[^\s]+",
    ]
    .into_iter()
    .map(|pattern| Regex::new(pattern).expect("static redaction regex"))
    .collect()
});

pub fn text(input: &str, profile: Profile) -> String {
    if profile == Profile::Off {
        return input.to_string();
    }
    let mut output = input.to_string();
    for pattern in SECRET_PATTERNS.iter() {
        output = pattern.replace_all(&output, "[REDACTED]").into_owned();
    }
    if let Some(max_chars) = profile.max_chars() {
        if output.chars().count() > max_chars {
            output = output.chars().take(max_chars).collect();
            output.push_str("\n[TRUNCATED]");
        }
    }
    output
}

pub fn value(node: &mut Value, profile: Profile) {
    match node {
        Value::String(text_value) => *text_value = text(text_value, profile),
        Value::Array(values) => {
            for item in values {
                value(item, profile);
            }
        }
        Value::Object(values) => {
            for item in values.values_mut() {
                value(item, profile);
            }
        }
        _ => {}
    }
}

/// Redact one canonical envelope line without changing its identity fields.
pub fn event_line(line: &[u8], profile: Profile) -> Option<Vec<u8>> {
    let mut event: Value = serde_json::from_slice(line).ok()?;
    if let Some(payload) = event.get_mut("payload") {
        value(payload, profile);
    }
    let mut bytes = serde_json::to_vec(&event).ok()?;
    bytes.push(b'\n');
    Some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_redacts_tokens_and_preserves_event_identity() {
        let line = br#"{"event_id":"E","kind":"tool_result","payload":{"text":"Authorization: Bearer abcdefghijklmnop and ghp_abcdefghijklmnop"}}"#;
        let redacted = event_line(line, Profile::Standard).unwrap();
        let value: Value = serde_json::from_slice(&redacted).unwrap();
        assert_eq!(value["event_id"], "E");
        assert!(!value["payload"]["text"].as_str().unwrap().contains("abcdefghijklmnop"));
        assert!(value["payload"]["text"].as_str().unwrap().contains("[REDACTED]"));
    }

    #[test]
    fn mcp_safe_bounds_output() {
        let output = text(&"x".repeat(8 * 1024), Profile::McpSafe);
        assert!(output.ends_with("[TRUNCATED]"));
        assert!(output.len() < 5 * 1024);
    }
}
