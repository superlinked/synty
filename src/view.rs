// CLI rendering of the shared view-models from `units` (work units, topics) plus
// index/tracker status. The TUI renders the same view-models as widgets, so the
// two surfaces stay at parity.

use crate::units::{Kind, TopicUnits, Unit};
use crate::{load_docs, readmodel, Doc};
use anyhow::Result;
use std::collections::HashMap;

/// A per-dimension tally — one repo, or one account: how many docs mention it,
/// how many of those are GitHub objects, how many distinct agent sessions, and
/// the sessions' token/tool spend (out tokens as the headline; the full class
/// split lives in the session detail and the stats panel).
pub struct Tally {
    pub name: String,
    pub docs: usize,
    pub github: usize,
    pub sessions: usize,
    pub tok_out: u64,
    pub tools: u64,
}

/// One tool's fleet-wide tally: which agent calls it, how often, how many
/// calls errored (attributed where the source reports errors).
pub struct ToolTally {
    pub name: String,
    pub agent: String,
    pub calls: u64,
    pub errs: u64,
    /// Args + result payload volume, chars. Tokens are NOT metered per tool
    /// by any agent (usage is per API turn), so views show this as a marked
    /// ~chars/4 estimate of the context the tool's calls consumed.
    pub chars: u64,
}

impl ToolTally {
    /// The marked estimate: ~4 chars per token over the measured payloads.
    pub fn est_tokens(&self) -> u64 {
        self.chars / 4
    }
}

pub struct Status {
    pub docs: usize,
    pub github: usize,
    pub sessions: usize,
    pub by_kind: Vec<(String, usize)>,
    pub by_repo: Vec<Tally>,
    pub by_user: Vec<Tally>,
    /// Fleet-wide tool mix, estimated context (~tok) desc — the expensive
    /// tools first, which is the question the table answers.
    pub by_tool: Vec<ToolTally>,
    /// Fleet-wide per-model token split, out tokens desc.
    pub by_model: Vec<crate::units::ModelUsage>,
    pub newest_ts: String,
    pub last_indexed: Option<String>,
    pub last_tracked: Option<String>,
    pub autostart: bool,
    /// Tracked events are newer than the index — answers may miss recent work.
    pub stale: bool,
    /// Per-machine liveness and the actor↔GitHub-author join (M8 coverage).
    pub fleet: crate::fleet::Roster,
}

/// What synty holds and how fresh it is.
pub fn status() -> Result<Status> {
    use std::collections::HashSet;
    let docs = load_docs(readmodel::docs_path()).unwrap_or_default();
    let github = docs.iter().filter(|d| d.meta.source == "github").count();
    let sessions = docs
        .iter()
        .filter(|d| d.meta.source != "github" && !d.meta.session_id.is_empty())
        .map(|d| d.meta.session_id.as_str())
        .collect::<HashSet<_>>()
        .len();
    // Segment the sessions' token/tool spend onto the repo and account rows,
    // and tally the fleet-wide tool mix — sessions carry repo, author, and
    // the per-name tool counts already.
    let sess = crate::units::sessions().unwrap_or_default();
    let local_actor = crate::identity::actor();
    let mut by_repo = tally(&docs, true);
    let mut by_user = tally(&docs, false);
    let mut tools: std::collections::HashMap<String, (std::collections::BTreeSet<String>, u64, u64, u64)> =
        std::collections::HashMap::new();
    let mut models: std::collections::HashMap<String, crate::units::ModelUsage> = std::collections::HashMap::new();
    for s in &sess {
        fold_spend(&mut by_repo, &s.repo, s.tok_out, s.tools as u64);
        let who = if s.author.is_empty() { local_actor.as_str() } else { s.author.as_str() };
        fold_spend(&mut by_user, who, s.tok_out, s.tools as u64);
        let agent = crate::units::agent_label(&s.source).to_string();
        for (name, calls, errs, chars) in &s.tools_by_name {
            let e = tools.entry(name.clone()).or_default();
            e.0.insert(agent.clone());
            e.1 += *calls as u64;
            e.2 += *errs as u64;
            e.3 += chars;
        }
        for mu in &s.by_model {
            let m = models.entry(mu.model.clone()).or_default();
            m.model = mu.model.clone();
            m.tok_in += mu.tok_in;
            m.tok_out += mu.tok_out;
            m.cache_read += mu.cache_read;
            m.cache_create += mu.cache_create;
            m.turns += mu.turns;
        }
    }
    let mut by_tool: Vec<ToolTally> = tools
        .into_iter()
        .map(|(name, (agents, calls, errs, chars))| ToolTally {
            name,
            agent: agents.into_iter().collect::<Vec<_>>().join("+"),
            calls,
            errs,
            chars,
        })
        .collect();
    by_tool.sort_by(|a, b| b.chars.cmp(&a.chars).then(b.calls.cmp(&a.calls)).then(a.name.cmp(&b.name)));
    let mut by_model: Vec<crate::units::ModelUsage> = models.into_values().collect();
    by_model.sort_by(|a, b| b.tok_out.cmp(&a.tok_out).then(a.model.cmp(&b.model)));
    Ok(Status {
        docs: docs.len(),
        github,
        sessions,
        by_kind: counts(docs.iter().map(|d| d.meta.kind.clone())),
        by_repo,
        by_user,
        by_tool,
        by_model,
        newest_ts: docs.iter().map(|d| d.meta.ts.as_str()).max().unwrap_or("").split('T').next().unwrap_or("").to_string(),
        last_indexed: readmodel::last_updated().map(fmt_time),
        last_tracked: mtime_str(".synty/cursors.json"),
        autostart: crate::track::autostart_enabled(),
        stale: stale_note().is_some(),
        fleet: crate::fleet::roster(&docs, std::path::Path::new(crate::units::LOCAL_DIR)),
    })
}

/// The fleet section of `status`: every machine with its liveness, who it
/// attributes to, and the install-rate join — empty string when no streams
/// have ever been seen (a fresh viewer before its first build).
fn fleet_md(f: &crate::fleet::Roster) -> String {
    if f.machines.is_empty() {
        return String::new();
    }
    let mut o = format!(
        "\n\nfleet ({}d window · quiet after {}d): {} machines ({} active, {} quiet) · {} actors tracked · {} active on GitHub · install rate {}%\n",
        crate::fleet::GH_WINDOW_DAYS,
        f.quiet_days,
        f.machines.len(),
        f.active(),
        f.machines.len() - f.active(),
        f.actors_tracked.len(),
        f.gh_active,
        f.install_rate_pct,
    );
    for m in &f.machines {
        o.push_str(&format!(
            "  {:<26} {:<14} {:<14} {:>10} {:>8} {:>6}{}\n",
            m.machine,
            m.sources.join("+"),
            if m.actors.is_empty() { "—".to_string() } else { m.actors.join("+") },
            m.last_ts.split('T').next().unwrap_or("—"),
            if m.version.is_empty() { "—".to_string() } else { format!("v{}", m.version) },
            m.events,
            if m.quiet { "  QUIET" } else { "" },
        ));
    }
    if !f.untracked_attributed.is_empty() {
        o.push_str(&format!(
            "  runs agents, untracked: {}\n",
            f.untracked_attributed.iter().map(|(a, l)| format!("{a} ({l})")).collect::<Vec<_>>().join(", "),
        ));
    }
    o
}

/// Add a session's token/tool spend to its facet row, creating the row when
/// the facet has no docs yet (an all-session repo or account still counts).
fn fold_spend(rows: &mut Vec<Tally>, name: &str, tok_out: u64, tools: u64) {
    if name.is_empty() {
        return;
    }
    match rows.iter_mut().find(|t| t.name == name) {
        Some(t) => {
            t.tok_out += tok_out;
            t.tools += tools;
        }
        None => rows.push(Tally { name: name.into(), docs: 0, github: 0, sessions: 0, tok_out, tools }),
    }
}

/// A warning when tracked events are newer than the index — surfaces stale
/// answers instead of silently serving them. None when fresh (or unbuilt).
pub fn stale_note() -> Option<String> {
    let indexed = readmodel::last_updated()?;
    let newest_event = walkdir::WalkDir::new("corpus/local")
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("jsonl"))
        .filter_map(|e| e.metadata().ok()?.modified().ok())
        .max()?;
    // A minute of slack: `up` ticks rewrite event files moments before indexing.
    (newest_event > indexed + std::time::Duration::from_secs(60)).then(|| {
        "note: tracked events are newer than the index — run `synty up` or `synty build` to refresh".to_string()
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
    if s.stale {
        o.push_str("⚠ index is stale — tracked events are newer; run `synty up` or `synty build`\n\n");
    }
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
    // 4-week usage totals — the CLI's one-line parity with the TUI stats panel
    // (anchored to the most recent day with data, not the wall clock).
    let days = crate::units::day_stats();
    if let Some(gmax) = days.keys().max().cloned() {
        if let Ok(end) = chrono::NaiveDate::parse_from_str(&gmax, "%Y-%m-%d") {
            let mut t = crate::units::DayStat::default();
            for (d, v) in &days {
                if let Ok(d) = chrono::NaiveDate::parse_from_str(d, "%Y-%m-%d") {
                    if (end - d).num_days() < 28 {
                        t.tok_in += v.tok_in;
                        t.tok_out += v.tok_out;
                        t.cache_read += v.cache_read;
                        t.cache_create += v.cache_create;
                        t.tools += v.tools;
                    }
                }
            }
            o.push_str(&format!(
                "\ntokens 4w: {} in · {} out · {} cache-read · {} cache-write · {} tool calls",
                fmt_tokens(t.tok_in),
                fmt_tokens(t.tok_out),
                fmt_tokens(t.cache_read),
                fmt_tokens(t.cache_create),
                fmt_tokens(t.tools),
            ));
        }
    }
    o.push_str(&fleet_md(&s.fleet));
    let block = |o: &mut String, label: &str, rows: &[Tally]| {
        o.push_str(&format!("\n\n{label} (docs · sessions · github · tok-out · tools):\n"));
        for f in rows.iter().take(12) {
            o.push_str(&format!(
                "  {:<24} {:>6} {:>4} {:>5} {:>8} {:>6}\n",
                f.name,
                f.docs,
                f.sessions,
                f.github,
                fmt_tokens(f.tok_out),
                fmt_tokens(f.tools),
            ));
        }
    };
    block(&mut o, "repos", &s.by_repo);
    block(&mut o, "accounts", &s.by_user);
    if !s.by_model.is_empty() {
        o.push_str("\n\nmodels (out · in · cache-r · cache-w · turns):\n");
        for m in s.by_model.iter().take(8) {
            o.push_str(&format!(
                "  {:<28} {:>7} {:>7} {:>7} {:>7} {:>6}\n",
                m.model,
                fmt_tokens(m.tok_out),
                fmt_tokens(m.tok_in),
                fmt_tokens(m.cache_read),
                fmt_tokens(m.cache_create),
                m.turns,
            ));
        }
    }
    if !s.by_tool.is_empty() {
        o.push_str("\n\ntools (~tok · calls · errors · agent):\n");
        for t in s.by_tool.iter().take(16) {
            o.push_str(&format!(
                "  {:<24} {:>8} {:>6} {:>6}  {}\n",
                t.name,
                fmt_tokens(t.est_tokens()),
                t.calls,
                t.errs,
                t.agent,
            ));
        }
    }
    o
}

// ── JSON renderers (CLI --json, for scripts and agents) ──────────────────

fn unit_json(u: &Unit) -> serde_json::Value {
    serde_json::json!({
        "kind": match u.kind { Kind::Session => "session", Kind::Pr => "pr", Kind::Issue => "issue" },
        "when": u.when,
        "repo": u.repo,
        "title": u.title,
        "outcome": u.outcome,
        "summary": u.summary,
        "topic": u.topic,
        "struggle": u.struggle,
        "author": u.author,
        "doc_id": u.doc_id,
        "session_id": u.session_id,
    })
}

pub fn work_json(units: &[Unit]) -> String {
    serde_json::Value::Array(units.iter().map(unit_json).collect()).to_string()
}

pub fn topics_json(topics: &[TopicUnits]) -> String {
    let arr: Vec<serde_json::Value> = topics
        .iter()
        .map(|t| {
            serde_json::json!({
                "id": t.id,
                "key": t.cache_key,
                "title": t.title(),
                "label": t.label,
                "summary": t.summary,
                "name": t.name,
                "last_active": t.last_active,
                "mix": {"sessions": t.mix.0, "prs": t.mix.1, "issues": t.mix.2},
                "repos": t.repos,
                "authors": t.authors,
                "span": t.span,
                "units": t.units.iter().map(unit_json).collect::<Vec<_>>(),
            })
        })
        .collect();
    serde_json::Value::Array(arr).to_string()
}

pub fn status_json(s: &Status) -> String {
    serde_json::json!({
        "docs": s.docs,
        "github": s.github,
        "sessions": s.sessions,
        "by_kind": s.by_kind,
        "by_repo": s.by_repo.iter().map(|t| serde_json::json!({"name": t.name, "docs": t.docs, "github": t.github, "sessions": t.sessions})).collect::<Vec<_>>(),
        "by_user": s.by_user.iter().map(|t| serde_json::json!({"name": t.name, "docs": t.docs, "github": t.github, "sessions": t.sessions})).collect::<Vec<_>>(),
        "newest_ts": s.newest_ts,
        "last_indexed": s.last_indexed,
        "last_tracked": s.last_tracked,
        "autostart": s.autostart,
        "stale": s.stale,
        "fleet": {
            "machines": s.fleet.machines.iter().map(|m| serde_json::json!({
                "machine": m.machine,
                "sources": m.sources,
                "actors": m.actors,
                "last_ts": m.last_ts,
                "tracker_version": m.version,
                "events": m.events,
                "quiet": m.quiet,
            })).collect::<Vec<_>>(),
            "actors_tracked": s.fleet.actors_tracked,
            "gh_active": s.fleet.gh_active,
            "untracked": s.fleet.untracked,
            "untracked_attributed": s.fleet.untracked_attributed,
            "install_rate_pct": s.fleet.install_rate_pct,
            "quiet_days": s.fleet.quiet_days,
        },
    })
    .to_string()
}

// ── helpers ───────────────────────────────────────────────────────────────

/// A 5-dot struggle meter from a 0..1 score.
pub fn meter(score: f32) -> String {
    let filled = (score * 5.0).round().clamp(0.0, 5.0) as usize;
    format!("[{}{}]", "●".repeat(filled), "○".repeat(5 - filled))
}

/// Humanize a token count: 999, 46.1k, 1.2M, 2.5B. One decimal, ".0" trimmed.
pub fn fmt_tokens(n: u64) -> String {
    let scaled = |v: f64, suffix: &str| {
        let s = format!("{v:.1}");
        format!("{}{suffix}", s.strip_suffix(".0").unwrap_or(&s))
    };
    match n {
        0..=999 => n.to_string(),
        1_000..=999_999 => scaled(n as f64 / 1_000.0, "k"),
        1_000_000..=999_999_999 => scaled(n as f64 / 1_000_000.0, "M"),
        _ => scaled(n as f64 / 1_000_000_000.0, "B"),
    }
}

/// One-line token accounting for a session — None when the source recorded no
/// usage (no number beats a fake 0; cowork and pre-capture envelopes record
/// none). Cache classes stay separate: a cache read is not a fresh input.
pub fn usage_line(s: &crate::units::Session) -> Option<String> {
    if !s.has_usage() {
        return None;
    }
    let mut o = format!(
        "tokens: {} in · {} out · {} cache-read · {} cache-write",
        fmt_tokens(s.tok_in),
        fmt_tokens(s.tok_out),
        fmt_tokens(s.cache_read),
        fmt_tokens(s.cache_create),
    );
    if s.usage_turns > 0 {
        o.push_str(&format!(" · {} turns", s.usage_turns));
    }
    Some(o)
}

/// One-line tool mix for a session — the top names by call count, plus the
/// error count straight off tool_result. None when nothing tallied by name.
pub fn tools_line(s: &crate::units::Session) -> Option<String> {
    if s.tools_by_name.is_empty() {
        return None;
    }
    let shown: Vec<String> = s.tools_by_name.iter().take(6).map(|(n, c, _, _)| format!("{n}×{c}")).collect();
    let mut o = format!("tools: {}", shown.join(" · "));
    if s.tools_by_name.len() > 6 {
        o.push_str(&format!(" (+{} more)", s.tools_by_name.len() - 6));
    }
    if s.tool_err > 0 {
        o.push_str(&format!(" · {} errors", s.tool_err));
    }
    Some(o)
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
        .map(|(name, (docs, github, s))| Tally { name: name.to_string(), docs, github, sessions: s.len(), tok_out: 0, tools: 0 })
        .collect();
    v.sort_by(|a, b| b.docs.cmp(&a.docs).then(a.name.cmp(&b.name)));
    v
}

fn mtime_str(path: &str) -> Option<String> {
    Some(fmt_time(std::fs::metadata(path).ok()?.modified().ok()?))
}

fn fmt_time(m: std::time::SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Utc> = m.into();
    dt.format("%Y-%m-%d %H:%M").to_string()
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

    fn session() -> crate::units::Session {
        crate::units::Session {
            id: "s".into(),
            repo: String::new(),
            started: String::new(),
            ended: String::new(),
            ask: String::new(),
            prompts: 1,
            assistant: 1,
            thinking: 0,
            tools: 0,
            files: vec![],
            linked_pr: None,
            topic: None,
            struggle: 0.0,
            tok_in: 0,
            tok_out: 0,
            cache_read: 0,
            cache_create: 0,
            usage_turns: 0,
            tools_by_name: vec![],
            tool_err: 0,
            by_model: vec![],
            source: "claude_code".into(),
            summary: None,
            author: String::new(),
        }
    }

    #[test]
    fn fmt_tokens_humanizes() {
        assert_eq!(fmt_tokens(0), "0");
        assert_eq!(fmt_tokens(999), "999");
        assert_eq!(fmt_tokens(46_100), "46.1k");
        assert_eq!(fmt_tokens(2_000), "2k");
        assert_eq!(fmt_tokens(1_200_000), "1.2M");
        assert_eq!(fmt_tokens(2_543_100_000), "2.5B"); // a month of cache reads
    }

    // No usage recorded → no line at all; a fake "0 tokens" would read as a
    // measurement. With usage, the four cache classes stay separate.
    #[test]
    fn usage_line_absent_without_usage() {
        assert_eq!(usage_line(&session()), None);
        let mut s = session();
        (s.tok_in, s.tok_out, s.cache_read, s.cache_create, s.usage_turns) = (46_100, 12_300, 1_200_000, 96_000, 12);
        let line = usage_line(&s).unwrap();
        assert_eq!(line, "tokens: 46.1k in · 12.3k out · 1.2M cache-read · 96k cache-write · 12 turns");
        s.usage_turns = 0; // codex's cumulative path has no turn count
        assert!(!usage_line(&s).unwrap().contains("turns"));
    }

    #[test]
    fn tools_line_caps_and_counts_errors() {
        assert_eq!(tools_line(&session()), None);
        let mut s = session();
        s.tools_by_name = (0..8).map(|i| (format!("T{i}"), 8 - i, 0, 100)).collect();
        s.tool_err = 2;
        let line = tools_line(&s).unwrap();
        assert!(line.starts_with("tools: T0×8"));
        assert!(line.contains("(+2 more)") && line.ends_with("2 errors"), "{line}");
        assert!(!line.contains("T6×"), "capped at six names: {line}");
    }

    // The fleet section names every machine, marks the rotted tracker, and
    // calls out who runs agents unwatched — the numbers a team lead acts on.
    // (status_md inlines this section verbatim; the helper keeps the test off
    // the corpus-walking day_stats path.)
    #[test]
    fn fleet_section_names_machines_quiet_and_untracked() {
        let roster = crate::fleet::Roster {
            machines: vec![
                crate::fleet::Machine {
                    machine: "mac-3939".into(),
                    sources: vec!["claude".into(), "codex".into()],
                    actors: vec!["svonava".into()],
                    last_ts: "2026-06-12T08:00:00Z".into(),
                    version: "0.1.0".into(),
                    events: 41,
                    quiet: false,
                },
                crate::fleet::Machine {
                    machine: "ci-runner-7".into(),
                    sources: vec!["claude".into()],
                    actors: vec![],
                    last_ts: "2026-05-20T08:00:00Z".into(),
                    version: String::new(),
                    events: 0,
                    quiet: true,
                },
            ],
            actors_tracked: vec!["svonava".into()],
            gh_active: 2,
            untracked: vec!["bob".into()],
            untracked_attributed: vec![("bob".into(), "claude".into())],
            install_rate_pct: 50,
            quiet_days: 7,
        };
        let md = fleet_md(&roster);
        assert!(md.contains("install rate 50%"), "{md}");
        assert!(md.contains("mac-3939") && md.contains("claude+codex") && md.contains("v0.1.0"), "{md}");
        assert!(md.contains("ci-runner-7") && md.contains("QUIET"), "{md}");
        assert!(md.contains("runs agents, untracked: bob (claude)"), "{md}");
        assert!(fleet_md(&Default::default()).is_empty(), "no streams yet → no section");
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
                agent_attr: None,
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
