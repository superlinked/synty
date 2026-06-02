// CLI rendering of the shared view-models from `units` (work units, topics) plus
// index/tracker status. The TUI renders the same view-models as widgets, so the
// two surfaces stay at parity.

use crate::units::{Kind, TopicUnits, Unit};
use crate::{load_docs, DOCS_PATH};
use anyhow::Result;
use std::collections::HashMap;

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

/// What synty holds and how fresh it is.
pub fn status() -> Result<Status> {
    let docs = load_docs(DOCS_PATH).unwrap_or_default();
    let github = docs.iter().filter(|d| d.meta.source == "github").count();
    Ok(Status {
        docs: docs.len(),
        github,
        sessions: docs.len() - github,
        by_kind: counts(docs.iter().map(|d| d.meta.kind.clone())),
        by_repo: counts(docs.iter().filter(|d| d.meta.source == "github").map(|d| d.meta.repo.clone())),
        newest_ts: docs.iter().map(|d| d.meta.ts.as_str()).max().unwrap_or("").split('T').next().unwrap_or("").to_string(),
        last_indexed: mtime_str("index/doc_hashes.json"),
        last_tracked: mtime_str(".synty/cursors.json"),
    })
}

// ── Markdown renderers (CLI) ──────────────────────────────────────────────

/// Work feed: sessions, PRs, and issues, newest first.
pub fn work_md(units: &[Unit]) -> String {
    let mut o = format!("# work ({})\n\n", units.len());
    for u in units {
        let tag = match u.kind {
            Kind::Session => "session",
            Kind::Pr => "pr",
            Kind::Issue => "issue",
        };
        let st = if matches!(u.kind, Kind::Session) { format!(" {}", meter(u.struggle)) } else { String::new() };
        let what = u.summary.as_deref().unwrap_or(&u.title);
        o.push_str(&format!("- `{}` {:<7} _{}_ — {} · {}{}\n", u.when, tag, u.repo, what, u.outcome, st));
    }
    o
}

/// Topics with activity, type mix, and their member units.
pub fn topics_md(topics: &[TopicUnits]) -> String {
    let mut o = format!("# topics ({})\n\n", topics.len());
    for t in topics {
        let (gh, asst, prompt) = t.mix;
        let n = t.activity.len();
        let wk = |i: usize| n.checked_sub(i).and_then(|x| t.activity.get(x)).copied().unwrap_or(0);
        o.push_str(&format!(
            "## {} — {}\n{} units · last active {} · activity this/last/prior wk: {}/{}/{} · gh {gh} / asst {asst} / prompt {prompt}\n",
            t.id,
            t.label,
            t.units.len(),
            t.last_active,
            wk(1),
            wk(2),
            wk(3),
        ));
        if let Some(s) = &t.summary {
            o.push_str(&format!("{s}\n"));
        }
        if !t.repos.is_empty() {
            o.push_str(&format!("repos: {}\n", t.repos.iter().take(6).cloned().collect::<Vec<_>>().join(", ")));
        }
        if !t.authors.is_empty() {
            o.push_str(&format!("authors: {}\n", t.authors.iter().take(6).cloned().collect::<Vec<_>>().join(", ")));
        }
        for u in t.units.iter().take(6) {
            let tag = match u.kind {
                Kind::Session => "session",
                Kind::Pr => "pr",
                Kind::Issue => "issue",
            };
            o.push_str(&format!("- `{}` {tag}: {}\n", u.when, u.title));
        }
        o.push('\n');
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

/// A 5-dot struggle meter from a 0..1 score.
pub fn meter(score: f32) -> String {
    let filled = (score * 5.0).round().clamp(0.0, 5.0) as usize;
    format!("[{}{}]", "●".repeat(filled), "○".repeat(5 - filled))
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
    fn meter_maps_score_to_dots() {
        assert_eq!(meter(0.0), "[○○○○○]");
        assert_eq!(meter(1.0), "[●●●●●]");
        assert_eq!(meter(0.5), "[●●●○○]");
    }
}
