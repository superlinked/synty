// Read-side policy for mediated agent access. Operators with raw bucket access
// remain in synty's documented high-trust tier; MCP clients receive only tools
// and records allowed by this scope.

use crate::event::Event;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::str::FromStr;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ReadScope {
    #[serde(default)]
    pub repos: Vec<String>,
    #[serde(default)]
    pub campaigns: Vec<String>,
    #[serde(default)]
    pub roles: Vec<String>,
    #[serde(default)]
    pub sources: Vec<String>,
}

impl ReadScope {
    pub fn load(path: Option<&str>) -> Result<Self> {
        let Some(path) = path else { return Ok(Self::default()) };
        let bytes = std::fs::read(path).with_context(|| format!("read MCP scope {path}"))?;
        serde_json::from_slice(&bytes).with_context(|| format!("parse MCP scope {path}"))
    }

    pub fn allows_repo(&self, repo: &str) -> bool {
        allowed(&self.repos, repo)
    }

    pub fn allows_source(&self, source: &str) -> bool {
        allowed(&self.sources, source)
    }

    pub fn allows_event(&self, event: &Event, repo: &str) -> bool {
        self.allows_repo(repo)
            && self.allows_source(&event.source)
            && allowed(&self.campaigns, campaign(event))
            && allowed(&self.roles, role(event))
    }

    pub fn restricted(&self) -> bool {
        !self.repos.is_empty()
            || !self.campaigns.is_empty()
            || !self.roles.is_empty()
            || !self.sources.is_empty()
    }
}

fn allowed(values: &[String], candidate: &str) -> bool {
    values.is_empty() || values.iter().any(|value| value == candidate)
}

fn campaign(event: &Event) -> &str {
    if !event.rollup_dim.is_empty() {
        return &event.rollup_dim;
    }
    event.payload["campaign_id"].as_str().unwrap_or("")
}

fn role(event: &Event) -> &str {
    event.payload["campaign_role"].as_str().unwrap_or("")
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum McpRole {
    #[default]
    Primary,
    Investigator,
    Validator,
    Operator,
}

impl FromStr for McpRole {
    type Err = anyhow::Error;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "primary" => Ok(Self::Primary),
            "investigator" => Ok(Self::Investigator),
            "validator" => Ok(Self::Validator),
            "operator" => Ok(Self::Operator),
            _ => anyhow::bail!("MCP role must be primary, investigator, validator, or operator"),
        }
    }
}

impl McpRole {
    pub fn allows_tool(self, tool: &str) -> bool {
        match self {
            Self::Operator => true,
            Self::Primary => matches!(
                tool,
                "synty_search"
                    | "synty_related"
                    | "synty_topics"
                    | "synty_recent"
                    | "synty_status"
                    | "synty_show"
            ),
            Self::Investigator => true,
            Self::Validator => matches!(
                tool,
                "synty_search"
                    | "synty_status"
                    | "synty_show"
                    | "synty_trace_list"
                    | "synty_trace_show"
                    | "synty_trace_search"
                    | "synty_trace_compare"
            ),
        }
    }
}

pub fn validate_scope_path(path: Option<&str>) -> Result<()> {
    if let Some(path) = path {
        anyhow::ensure!(Path::new(path).is_file(), "MCP scope file does not exist: {path}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_have_narrow_default_tool_surfaces() {
        assert!(McpRole::Primary.allows_tool("synty_related"));
        assert!(!McpRole::Primary.allows_tool("synty_trace_search"));
        assert!(McpRole::Investigator.allows_tool("synty_trace_search"));
        assert!(!McpRole::Validator.allows_tool("synty_related"));
        assert!(McpRole::Operator.allows_tool("synty_stats"));
    }

    #[test]
    fn event_scope_reads_campaign_from_rollup_or_payload() {
        let mut event: Event = serde_json::from_value(serde_json::json!({
            "v": 1, "event_id": "e", "stream": "s", "seq": 0,
            "ts": "2026-07-22T00:00:00Z", "source": "harness",
            "session_id": "s", "kind": "agent_meta",
            "payload": {"campaign_role": "validator"}, "rollup_dim": "campaign-1"
        }))
        .unwrap();
        let scope = ReadScope {
            campaigns: vec!["campaign-1".into()],
            roles: vec!["validator".into()],
            sources: vec!["harness".into()],
            ..Default::default()
        };
        assert!(scope.allows_event(&event, "sie-internal"));
        event.rollup_dim = "other".into();
        assert!(!scope.allows_event(&event, "sie-internal"));
    }
}
