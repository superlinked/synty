// Shared view-models for the CLI and the TUI, computed once from the derived
// corpus + clusters so both surfaces show the same thing. The struct-returning
// functions feed the TUI; the `*_md` renderers print Markdown for the CLI.

use crate::{excerpt, first_line, load_docs, short, Doc, DOCS_PATH};
use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;

pub struct Topic {
    pub id: i64,
    pub label: String,
    pub size: usize,
    pub repos: String,
    pub kinds: String,
    pub members: Vec<String>,
}

pub struct RecentItem {
    pub ts: String,
    pub kind: String,
    pub repo: String,
    pub title: String,
    pub url: Option<String>,
}

pub struct Status {
    pub docs: usize,
    pub github: usize,
    pub sessions: usize,
    pub by_kind: Vec<(String, usize)>,
    pub by_repo: Vec<(String, usize)>,
    pub newest_ts: String,
    pub last_indexed: Option<String>,
    pub last_tracked: Option<String>,
}

/// Emergent topics from the last `cluster` run, joined with the docs they hold.
pub fn topics() -> Result<Vec<Topic>> {
    let docs = load_docs(DOCS_PATH)?;
    let by_id: HashMap<i64, &Doc> = docs.iter().map(|d| (d.id, d)).collect();
    let raw = std::fs::read_to_string("clusters.json")
        .map_err(|_| anyhow::anyhow!("no clusters.json; run `cluster` first"))?;
    let arr: Vec<Value> = serde_json::from_str(&raw)?;

    // cluster index → (label, member doc ids)
    let mut groups: HashMap<i64, (String, Vec<i64>)> = HashMap::new();
    for it in &arr {
        let c = it["cluster"].as_i64().unwrap_or(-1);
        let label = it["label"].as_str().unwrap_or("").to_string();
        let id = it["id"].as_i64().unwrap_or(-1);
        let e = groups.entry(c).or_insert_with(|| (label, Vec::new()));
        e.1.push(id);
    }
    let mut topics: Vec<Topic> = groups
        .into_iter()
        .map(|(c, (label, ids))| {
            let members: Vec<&Doc> = ids.iter().filter_map(|i| by_id.get(i).copied()).collect();
            Topic {
                id: c,
                label,
                size: members.len(),
                repos: top_counts(members.iter().map(|d| d.meta.repo.clone()), 3),
                kinds: top_counts(members.iter().map(|d| d.meta.kind.clone()), 3),
                members: members.iter().take(6).map(|d| title_of(d)).collect(),
            }
        })
        .collect();
    topics.sort_by(|a, b| b.size.cmp(&a.size));
    Ok(topics)
}

/// The most recent human-initiated activity: PRs, issues, and user prompts.
pub fn recent(repo: Option<&str>, limit: usize) -> Result<Vec<RecentItem>> {
    let docs = load_docs(DOCS_PATH)?;
    let mut items: Vec<&Doc> = docs
        .iter()
        .filter(|d| matches!(d.meta.kind.as_str(), "pull_request" | "issue" | "user_prompt"))
        .filter(|d| repo.is_none_or(|r| d.meta.repo == r))
        .collect();
    items.sort_by(|a, b| b.meta.ts.cmp(&a.meta.ts));
    Ok(items
        .into_iter()
        .take(limit)
        .map(|d| RecentItem {
            ts: d.meta.ts.split('T').next().unwrap_or("").to_string(),
            kind: d.meta.kind.clone(),
            repo: if d.meta.repo.is_empty() { "local".into() } else { d.meta.repo.clone() },
            title: title_of(d),
            url: d.meta.url.clone(),
        })
        .collect())
}

/// What synty currently holds and how fresh it is.
pub fn status() -> Result<Status> {
    let docs = load_docs(DOCS_PATH).unwrap_or_default();
    let github = docs.iter().filter(|d| d.meta.source == "github").count();
    let sessions = docs.len() - github;
    let newest_ts = docs.iter().map(|d| d.meta.ts.as_str()).max().unwrap_or("").to_string();
    Ok(Status {
        docs: docs.len(),
        github,
        sessions,
        by_kind: counts(docs.iter().map(|d| d.meta.kind.clone())),
        by_repo: counts(docs.iter().filter(|d| d.meta.source == "github").map(|d| d.meta.repo.clone())),
        newest_ts: newest_ts.split('T').next().unwrap_or("").to_string(),
        last_indexed: mtime_str("index/doc_hashes.json"),
        last_tracked: mtime_str(".synty/cursors.json"),
    })
}

// ── Markdown renderers (CLI) ──────────────────────────────────────────────

pub fn topics_md(topics: &[Topic]) -> String {
    let mut o = format!("# topics ({})\n\n", topics.len());
    for t in topics {
        o.push_str(&format!("## {} — {}  ({} docs)\n", t.id, t.label, t.size));
        o.push_str(&format!("repos: {} · kinds: {}\n", t.repos, t.kinds));
        for m in &t.members {
            o.push_str(&format!("- {m}\n"));
        }
        o.push('\n');
    }
    o
}

pub fn recent_md(items: &[RecentItem]) -> String {
    let mut o = String::from("# recent\n\n");
    for it in items {
        let url = it.url.as_deref().map(|u| format!(" · {u}")).unwrap_or_default();
        o.push_str(&format!("- `{}` **{}** _{}_ — {}{}\n", it.ts, it.kind, it.repo, it.title, url));
    }
    o
}

pub fn status_md(s: &Status) -> String {
    let mut o = String::from("# status\n\n");
    o.push_str(&format!(
        "{} docs · {} GitHub · {} session\nnewest: {}\nlast indexed: {} · last tracked: {}\n\n",
        s.docs,
        s.github,
        s.sessions,
        if s.newest_ts.is_empty() { "—" } else { &s.newest_ts },
        s.last_indexed.as_deref().unwrap_or("never"),
        s.last_tracked.as_deref().unwrap_or("never"),
    ));
    o.push_str("kinds: ");
    o.push_str(&s.by_kind.iter().take(8).map(|(k, n)| format!("{k}({n})")).collect::<Vec<_>>().join(", "));
    o.push_str("\nrepos: ");
    o.push_str(&s.by_repo.iter().take(8).map(|(k, n)| format!("{k}({n})")).collect::<Vec<_>>().join(", "));
    o.push('\n');
    o
}

// ── helpers ───────────────────────────────────────────────────────────────

pub fn title_of(d: &Doc) -> String {
    match d.meta.kind.as_str() {
        "pull_request" | "issue" => {
            format!("{} {}#{} {}", d.meta.kind, d.meta.repo, d.meta.number.unwrap_or(0), first_line(&d.text))
        }
        _ => format!("{} {} \"{}\"", d.meta.kind, short(&d.meta.session_id), excerpt(&d.text, 80)),
    }
}

fn counts(it: impl Iterator<Item = String>) -> Vec<(String, usize)> {
    let mut m: HashMap<String, usize> = HashMap::new();
    for x in it {
        *m.entry(x).or_default() += 1;
    }
    let mut v: Vec<_> = m.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

fn top_counts(it: impl Iterator<Item = String>, k: usize) -> String {
    counts(it).into_iter().take(k).map(|(s, n)| format!("{s}({n})")).collect::<Vec<_>>().join(", ")
}

fn mtime_str(path: &str) -> Option<String> {
    let m = std::fs::metadata(path).ok()?.modified().ok()?;
    let dt: chrono::DateTime<chrono::Utc> = m.into();
    Some(dt.format("%Y-%m-%d %H:%M").to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_sorts_desc_then_name() {
        let c = counts(["b", "a", "b", "c", "a", "b"].into_iter().map(String::from));
        assert_eq!(c, vec![("b".into(), 3), ("a".into(), 2), ("c".into(), 1)]);
    }

    #[test]
    fn top_counts_caps_and_formats() {
        let s = top_counts(["x", "x", "y", "z"].into_iter().map(String::from), 2);
        assert_eq!(s, "x(2), y(1)");
    }
}
