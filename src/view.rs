// CLI rendering of the shared view-models from `units` (work units, topics) plus
// index/tracker status. The TUI renders the same view-models as widgets, so the
// two surfaces stay at parity.

use crate::units::{Kind, TopicUnits, Unit};
use crate::{load_docs, Doc, DOCS_PATH};
use anyhow::Result;
use std::collections::HashMap;

/// A per-dimension tally — one repo, or one account: how many docs mention it,
/// how many of those are GitHub objects, and how many distinct agent sessions.
pub struct Tally {
    pub name: String,
    pub docs: usize,
    pub github: usize,
    pub sessions: usize,
}

pub struct Status {
    pub docs: usize,
    pub github: usize,
    pub sessions: usize,
    pub by_kind: Vec<(String, usize)>,
    pub by_repo: Vec<Tally>,
    pub by_user: Vec<Tally>,
    pub newest_ts: String,
    pub last_indexed: Option<String>,
    pub last_tracked: Option<String>,
    pub autostart: bool,
}

/// What synty holds and how fresh it is.
pub fn status() -> Result<Status> {
    use std::collections::HashSet;
    let docs = load_docs(DOCS_PATH).unwrap_or_default();
    let github = docs.iter().filter(|d| d.meta.source == "github").count();
    let sessions = docs
        .iter()
        .filter(|d| d.meta.source != "github" && !d.meta.session_id.is_empty())
        .map(|d| d.meta.session_id.as_str())
        .collect::<HashSet<_>>()
        .len();
    Ok(Status {
        docs: docs.len(),
        github,
        sessions,
        by_kind: counts(docs.iter().map(|d| d.meta.kind.clone())),
        by_repo: tally(&docs, true),
        by_user: tally(&docs, false),
        newest_ts: docs.iter().map(|d| d.meta.ts.as_str()).max().unwrap_or("").split('T').next().unwrap_or("").to_string(),
        last_indexed: mtime_str("index/doc_hashes.json"),
        last_tracked: mtime_str(".synty/cursors.json"),
        autostart: crate::track::autostart_enabled(),
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
        let (sess, prs, issues) = t.mix;
        let n = t.activity.len();
        let wk = |i: usize| n.checked_sub(i).and_then(|x| t.activity.get(x)).copied().unwrap_or(0);
        o.push_str(&format!(
            "## {} — {}\n{} units · last active {} · activity this/last/prior wk: {}/{}/{} · {sess} sessions / {prs} PRs / {issues} issues\n",
            t.id,
            t.title(),
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
        "{} docs · {} GitHub · {} sessions · autostart {}\nnewest: {}\nlast indexed: {} · last tracked: {}\n\nkinds: ",
        s.docs,
        s.github,
        s.sessions,
        if s.autostart { "on" } else { "off" },
        if s.newest_ts.is_empty() { "—" } else { &s.newest_ts },
        s.last_indexed.as_deref().unwrap_or("never"),
        s.last_tracked.as_deref().unwrap_or("never"),
    ));
    o.push_str(&s.by_kind.iter().take(8).map(|(k, n)| format!("{k}({n})")).collect::<Vec<_>>().join(", "));
    let block = |o: &mut String, label: &str, rows: &[Tally]| {
        o.push_str(&format!("\n\n{label} (sessions · github · docs):\n"));
        for f in rows.iter().take(12) {
            o.push_str(&format!("  {:<24} {:>4} {:>5} {:>6}\n", f.name, f.sessions, f.github, f.docs));
        }
    };
    block(&mut o, "repos", &s.by_repo);
    block(&mut o, "accounts", &s.by_user);
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

/// Tally docs by repo (`by_repo = true`) or author, counting total docs, GitHub
/// objects, and distinct sessions for each. Skips the empty key; most docs first.
fn tally(docs: &[Doc], by_repo: bool) -> Vec<Tally> {
    use std::collections::HashSet;
    let mut m: HashMap<&str, (usize, usize, HashSet<&str>)> = HashMap::new();
    for d in docs {
        let key = if by_repo { d.meta.repo.as_str() } else { d.meta.author.as_str() };
        if key.is_empty() {
            continue;
        }
        let e = m.entry(key).or_default();
        e.0 += 1;
        if d.meta.source == "github" {
            e.1 += 1;
        } else if !d.meta.session_id.is_empty() {
            e.2.insert(d.meta.session_id.as_str());
        }
    }
    let mut v: Vec<Tally> = m
        .into_iter()
        .map(|(name, (docs, github, s))| Tally { name: name.to_string(), docs, github, sessions: s.len() })
        .collect();
    v.sort_by(|a, b| b.docs.cmp(&a.docs).then(a.name.cmp(&b.name)));
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

    fn doc(source: &str, kind: &str, repo: &str, author: &str, sid: &str) -> Doc {
        Doc {
            id: 0,
            text: String::new(),
            meta: crate::Meta {
                source: source.into(),
                kind: kind.into(),
                repo: repo.into(),
                author: author.into(),
                session_id: sid.into(),
                ts: "2026-01-01T00:00:00Z".into(),
                number: None,
                url: None,
                state: None,
                labels: vec![],
            },
        }
    }

    // Per-repo tally splits docs into GitHub objects and distinct sessions.
    #[test]
    fn tally_counts_docs_github_and_sessions() {
        let docs = vec![
            doc("github", "pull_request", "sie", "alice", ""),
            doc("agent", "user_prompt", "sie", "", "S1"),
            doc("agent", "assistant_message", "sie", "", "S1"),
            doc("agent", "user_prompt", "sie", "", "S2"),
            doc("github", "issue", "web", "bob", ""),
        ];
        let by_repo = tally(&docs, true);
        assert_eq!(by_repo[0].name, "sie", "most-active repo first");
        let sie = &by_repo[0];
        assert_eq!((sie.docs, sie.github, sie.sessions), (4, 1, 2), "4 docs · 1 github · 2 sessions");
        // by author: only GitHub objects carry one; sessions aren't attributed.
        let by_user = tally(&docs, false);
        let alice = by_user.iter().find(|t| t.name == "alice").unwrap();
        assert_eq!((alice.github, alice.sessions), (1, 0));
    }
}
