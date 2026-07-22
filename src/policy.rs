// Read-side policy for mediated agent access. Operators with raw bucket access
// remain in synty's documented high-trust tier; MCP clients receive only tools
// and records allowed by this scope.

use crate::{Meta, event::Event, units::{Session, Unit}};
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

    /// Source scopes name the capture producer (`harness`, `codex_cli`,
    /// `github`), while indexed agent documents retain the broad `agent`
    /// source used by search filters. `capture_source` carries the producer;
    /// `backend` independently identifies the campaign execution backend.
    pub fn allows_fields(&self, repo: &str, campaign: &str, role: &str, source: &str) -> bool {
        self.allows_repo(repo)
            && self.allows_source(source)
            && allowed(&self.campaigns, campaign)
            && allowed(&self.roles, role)
    }

    pub fn allows_doc(&self, meta: &Meta) -> bool {
        let source = if meta.capture_source.is_empty() { &meta.source } else { &meta.capture_source };
        self.allows_fields(&meta.repo, &meta.campaign_id, &meta.campaign_role, source)
    }

    pub fn allows_session(&self, session: &Session) -> bool {
        self.allows_fields(
            &session.repo,
            &session.campaign_id,
            &session.campaign_role,
            &session.source,
        )
    }

    pub fn allows_unit(&self, unit: &Unit) -> bool {
        self.allows_fields(
            &unit.repo,
            &unit.campaign_id,
            &unit.campaign_role,
            &unit.source,
        )
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

    pub fn exposes_tool(&self, tool: &str) -> bool {
        match tool_scope(tool) {
            Some(ToolScope::Filtered) => true,
            Some(ToolScope::Global) => !self.restricted(),
            None => false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ToolScope {
    /// The implementation receives the scope and filters before rendering.
    Filtered,
    /// Fleet aggregates cannot be safely reconstructed for a partial corpus.
    Global,
}

pub(crate) fn tool_scope(tool: &str) -> Option<ToolScope> {
    match tool {
        "synty_search"
        | "synty_related"
        | "synty_topics"
        | "synty_recent"
        | "synty_show"
        | "synty_trace_list"
        | "synty_trace_show"
        | "synty_trace_search"
        | "synty_trace_compare" => Some(ToolScope::Filtered),
        "synty_status" | "synty_stats" | "synty_tool" => Some(ToolScope::Global),
        _ => None,
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
            Self::Investigator => matches!(
                tool,
                "synty_search"
                    | "synty_topics"
                    | "synty_recent"
                    | "synty_status"
                    | "synty_show"
                    | "synty_trace_list"
                    | "synty_trace_show"
                    | "synty_trace_search"
                    | "synty_trace_compare"
            ),
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
        let tools = [
            "synty_search", "synty_related", "synty_topics", "synty_recent",
            "synty_status", "synty_stats", "synty_tool", "synty_show",
            "synty_trace_list", "synty_trace_show", "synty_trace_search",
            "synty_trace_compare",
        ];
        let matrix = [
            (McpRole::Primary, &[
                "synty_search", "synty_related", "synty_topics", "synty_recent",
                "synty_status", "synty_show",
            ][..]),
            (McpRole::Investigator, &[
                "synty_search", "synty_topics", "synty_recent", "synty_status",
                "synty_show", "synty_trace_list", "synty_trace_show",
                "synty_trace_search", "synty_trace_compare",
            ][..]),
            (McpRole::Validator, &[
                "synty_search", "synty_status", "synty_show", "synty_trace_list",
                "synty_trace_show", "synty_trace_search", "synty_trace_compare",
            ][..]),
            (McpRole::Operator, &tools[..]),
        ];
        for (role, allowed) in matrix {
            for tool in tools {
                assert_eq!(role.allows_tool(tool), allowed.contains(&tool), "{role:?} × {tool}");
            }
        }
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

    #[test]
    fn indexed_agent_docs_use_their_capture_source_for_source_scope() {
        let scope = ReadScope { sources: vec!["harness".into()], ..Default::default() };
        let meta = Meta {
            source: "agent".into(),
            kind: "user_prompt".into(),
            repo: "synty".into(),
            author: String::new(),
            session_id: "s".into(),
            campaign_id: String::new(),
            campaign_role: String::new(),
            backend: "codex".into(),
            capture_source: "harness".into(),
            ts: String::new(),
            number: None,
            url: None,
            state: None,
            labels: vec![],
            agent_attr: None,
        };
        assert!(scope.allows_doc(&meta));
    }

    #[test]
    fn every_tool_has_a_policy_for_every_scope_dimension() {
        let tools = [
            "synty_search", "synty_related", "synty_topics", "synty_recent",
            "synty_status", "synty_stats", "synty_tool", "synty_show",
            "synty_trace_list", "synty_trace_show", "synty_trace_search",
            "synty_trace_compare",
        ];
        let scopes = [
            ReadScope { repos: vec!["repo".into()], ..Default::default() },
            ReadScope { campaigns: vec!["campaign".into()], ..Default::default() },
            ReadScope { roles: vec!["role".into()], ..Default::default() },
            ReadScope { sources: vec!["source".into()], ..Default::default() },
        ];
        for scope in scopes {
            for tool in tools {
                let expected = !matches!(tool_scope(tool), Some(ToolScope::Global));
                assert_eq!(scope.exposes_tool(tool), expected, "{tool} with {scope:?}");
            }
        }
        assert!(!ReadScope::default().exposes_tool("synty_future_tool"));
    }

    #[test]
    fn all_record_dimensions_are_conjunctive() {
        let matching = ("repo", "campaign", "role", "source");
        let dimensions = [
            ReadScope { repos: vec![matching.0.into()], ..Default::default() },
            ReadScope { campaigns: vec![matching.1.into()], ..Default::default() },
            ReadScope { roles: vec![matching.2.into()], ..Default::default() },
            ReadScope { sources: vec![matching.3.into()], ..Default::default() },
        ];
        for scope in dimensions {
            assert!(scope.allows_fields(matching.0, matching.1, matching.2, matching.3));
        }
        assert!(!ReadScope { repos: vec!["other".into()], ..Default::default() }
            .allows_fields(matching.0, matching.1, matching.2, matching.3));
        assert!(!ReadScope { campaigns: vec!["other".into()], ..Default::default() }
            .allows_fields(matching.0, matching.1, matching.2, matching.3));
        assert!(!ReadScope { roles: vec!["other".into()], ..Default::default() }
            .allows_fields(matching.0, matching.1, matching.2, matching.3));
        assert!(!ReadScope { sources: vec!["other".into()], ..Default::default() }
            .allows_fields(matching.0, matching.1, matching.2, matching.3));
    }
}
