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
    /// The configured team bucket, or None when this machine is local-only (a
    /// trial). "Activated" — a real fleet member — is `bucket.is_some()` AND
    /// autostart on.
    pub bucket: Option<String>,
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
        bucket: crate::config::load().bucket,
        stale: stale_note().is_some(),
        fleet: crate::fleet::roster(&docs, std::path::Path::new(crate::units::LOCAL_DIR)),
    })
}

/// One Mon-aligned week of activity (`start` is the Monday, YYYY-MM-DD).
pub struct WeekRow {
    pub start: String,
    pub row: crate::units::DayRow,
}

/// `synty stats`: a fixed window of whole weeks anchored to the most recent
/// day WITH data (never the wall clock — same rule as the TUI charts), the
/// window total, and the raw day series for --json.
pub struct StatsView {
    pub anchor: String,
    pub weeks: Vec<WeekRow>,
    pub total: crate::units::DayRow,
    /// Ascending (day, row) pairs inside the window — day resolution lives
    /// in the JSON; the Markdown stays weekly.
    pub days: Vec<(String, crate::units::DayRow)>,
}

pub fn stats(weeks: usize) -> Result<StatsView> {
    let units = crate::units::units()?;
    Ok(stats_view(crate::units::activity_by_day(&units), weeks))
}

fn stats_view(activity: std::collections::HashMap<String, crate::units::DayRow>, weeks: usize) -> StatsView {
    use crate::units::{day_num, week_start_for, DayRow};
    let weeks = weeks.max(1);
    let Some(anchor) = activity.keys().max().cloned() else {
        return StatsView { anchor: String::new(), weeks: vec![], total: DayRow::default(), days: vec![] };
    };
    let Some(gmax) = day_num(&anchor) else {
        return StatsView { anchor: String::new(), weeks: vec![], total: DayRow::default(), days: vec![] };
    };
    let start = week_start_for(gmax, weeks);
    let mut buckets = vec![DayRow::default(); weeks];
    let mut total = DayRow::default();
    let mut days: Vec<(String, DayRow)> = Vec::new();
    for (d, r) in &activity {
        let Some(n) = day_num(d) else { continue };
        if n < start {
            continue;
        }
        buckets[(((n - start) / 7) as usize).min(weeks - 1)].add(r);
        total.add(r);
        days.push((d.clone(), *r));
    }
    days.sort_by(|a, b| a.0.cmp(&b.0));
    let week_rows = buckets
        .into_iter()
        .enumerate()
        .map(|(w, row)| WeekRow { start: day_str(start + 7 * w as i32), row })
        .collect();
    StatsView { anchor, weeks: week_rows, total, days }
}

/// YYYY-MM-DD for a num_days_from_ce (the inverse of units::day_num).
fn day_str(n: i32) -> String {
    chrono::NaiveDate::from_num_days_from_ce_opt(n).map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()
}

pub fn stats_md(s: &StatsView) -> String {
    if s.weeks.is_empty() {
        return "# stats\n\n_(no data yet — `synty up` tracks and indexes this machine)_\n".to_string();
    }
    let mut o = format!("# stats — {} weeks to {}\n", s.weeks.len(), s.anchor);
    // Two tables, mirroring the TUI's stacked charts: what the agents
    // consumed, then what the work produced.
    o.push_str("\nagents (wk of · tok-in · tok-out · cache-r · cache-w · tools · sessions):\n");
    let agents = |o: &mut String, label: &str, r: &crate::units::DayRow| {
        o.push_str(&format!(
            "  {:<10} {:>8} {:>8} {:>8} {:>8} {:>6} {:>8}\n",
            label,
            fmt_tokens(r.tok_in),
            fmt_tokens(r.tok_out),
            fmt_tokens(r.cache_read),
            fmt_tokens(r.cache_create),
            fmt_tokens(r.tools),
            r.sessions,
        ));
    };
    for w in &s.weeks {
        agents(&mut o, &w.start, &w.row);
    }
    agents(&mut o, "total", &s.total);
    o.push_str("\noutput (wk of · loc+ · loc- · prs · issues):\n");
    let output = |o: &mut String, label: &str, r: &crate::units::DayRow| {
        o.push_str(&format!(
            "  {:<10} {:>8} {:>8} {:>5} {:>6}\n",
            label,
            fmt_tokens(r.loc_add),
            fmt_tokens(r.loc_del),
            r.prs,
            r.issues,
        ));
    };
    for w in &s.weeks {
        output(&mut o, &w.start, &w.row);
    }
    output(&mut o, "total", &s.total);
    o
}

fn day_row_json(r: &crate::units::DayRow) -> serde_json::Value {
    serde_json::json!({
        "tok_in": r.tok_in, "tok_out": r.tok_out,
        "cache_read": r.cache_read, "cache_create": r.cache_create,
        "tools": r.tools, "sessions": r.sessions,
        "loc_add": r.loc_add, "loc_del": r.loc_del,
        "prs": r.prs, "issues": r.issues,
    })
}

pub fn stats_json(s: &StatsView) -> String {
    envelope(
        "stats",
        serde_json::json!({
            "anchor": s.anchor,
            "weeks": s.weeks.iter().map(|w| {
                let mut v = day_row_json(&w.row);
                v["start"] = serde_json::json!(w.start);
                v
            }).collect::<Vec<_>>(),
            "total": day_row_json(&s.total),
            "days": s.days.iter().map(|(d, r)| (d.clone(), day_row_json(r))).collect::<serde_json::Map<_, _>>(),
        }),
    )
}

/// `synty tool <name>` as Markdown — the same facts the TUI's tool overlay
/// shows: volume, latency, input sizes, context estimate, weekly call counts,
/// argument-key shares (with common values for the enum-ish keys), and the
/// most recent invocations.
pub fn tool_md(p: &crate::units::ToolProfile) -> String {
    let mut o = format!("# {} · {}\n\n", p.name, p.agent);
    let mut head = format!("{} calls · {} errors", fmt_tokens(p.calls), p.errs);
    if p.timed > 0 {
        head.push_str(&format!(" · result p50 {}ms · p95 {}ms ({} timed)", p.p50_ms, p.p95_ms, p.timed));
    }
    if p.input_p95 > 0 {
        head.push_str(&format!(" · input p50 {} / p95 {} chars", p.input_p50, p.input_p95));
    }
    if p.chars > 0 {
        head.push_str(&format!(" · context ~{} tok", fmt_tokens(p.chars / 4)));
    }
    o.push_str(&head);
    o.push('\n');
    // The Markdown stand-in for the overlay's day strip: per-week call counts
    // over the trailing four weeks of data.
    let by_num: Vec<(i32, u64)> =
        p.days.iter().filter_map(|(d, n)| Some((crate::units::day_num(d)?, *n))).collect();
    if let Some(gmax) = by_num.iter().map(|(d, _)| *d).max() {
        let start = crate::units::week_start_for(gmax, 4);
        let mut weeks = [0u64; 4];
        for (d, n) in &by_num {
            if *d >= start {
                weeks[(((d - start) / 7) as usize).min(3)] += n;
            }
        }
        o.push_str(&format!(
            "calls by week (oldest→newest): {}\n",
            weeks.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(" · "),
        ));
    }
    if !p.arg_keys.is_empty() {
        o.push_str("\nargs (share of calls):\n");
        for (key, n) in p.arg_keys.iter().take(10) {
            let pct = 100 * n / p.calls.max(1);
            o.push_str(&format!("  {:<22} {:>6} ({:>3}%)", key, fmt_tokens(*n), pct));
            if let Some((_, tops)) = p.arg_tops.iter().find(|(k, _)| k == key) {
                let vals: Vec<String> = tops.iter().map(|(v, c)| format!("{v}×{c}")).collect();
                o.push_str(&format!("  {}", vals.join(" · ")));
            }
            o.push('\n');
        }
    }
    if !p.samples.is_empty() {
        o.push_str("\nrecent:\n");
        for s in &p.samples {
            o.push_str(&format!("  {s}\n"));
        }
    }
    o
}

pub fn tool_json(p: &crate::units::ToolProfile) -> String {
    envelope(
        "tool",
        serde_json::json!({
            "name": p.name,
            "agent": p.agent,
            "calls": p.calls,
            "errors": p.errs,
            "context_chars": p.chars,
            "context_tokens_est": p.chars / 4,
            "p50_ms": p.p50_ms,
            "p95_ms": p.p95_ms,
            "timed": p.timed,
            "input_p50": p.input_p50,
            "input_p95": p.input_p95,
            "days": p.days,
            "arg_keys": p.arg_keys,
            "arg_tops": p.arg_tops,
            "samples": p.samples,
        }),
    )
}

/// The shared CLI/MCP entry for `tool <name>`: the profile as Markdown, or an
/// error that names close matches — misses behave identically on both
/// surfaces.
pub fn tool_report(name: &str) -> Result<String> {
    let p = crate::units::tool_profile(name);
    if p.calls > 0 {
        return Ok(tool_md(&p));
    }
    let sugg = tool_suggestions(name, &crate::units::tool_names());
    if sugg.is_empty() {
        anyhow::bail!("no calls recorded for tool `{name}` — names appear in `synty status`");
    }
    anyhow::bail!("no calls recorded for tool `{name}` — did you mean {}?", sugg.join(", "))
}

/// Case-insensitive exact and prefix matches, exact first.
fn tool_suggestions(name: &str, names: &[String]) -> Vec<String> {
    let q = name.to_lowercase();
    let mut out: Vec<String> = names.iter().filter(|n| n.to_lowercase() == q).cloned().collect();
    out.extend(names.iter().filter(|n| n.to_lowercase().starts_with(&q) && n.to_lowercase() != q).cloned());
    out.truncate(5);
    out
}

// ── show: detail for one stable id ────────────────────────────────────────

/// What a printed id resolves to. The Gh arm carries its topic membership
/// (key8 + title) because only the resolver has the topics in hand.
pub enum ShowTarget {
    Session(Box<crate::units::Session>),
    Gh { doc: Doc, topic: Option<(String, String)> },
    Topic(crate::units::TopicUnits),
}

/// Load the corpus once and resolve `id` against it — the shared substrate
/// for the Markdown and JSON paths.
fn show_load(id: &str) -> Result<(ShowTarget, Vec<Doc>)> {
    let docs = load_docs(readmodel::docs_path()).unwrap_or_default();
    let sessions = crate::units::sessions().unwrap_or_default();
    let topics = crate::units::topic_units(12).unwrap_or_default();
    let t = resolve_among(id, sessions, topics, &docs)?;
    Ok((t, docs))
}

/// The shared CLI/MCP entry: detail for one id as Markdown, or a miss/
/// ambiguity error that names its candidates — identical on both surfaces.
pub fn show_report(id: &str) -> Result<String> {
    let (t, docs) = show_load(id)?;
    Ok(match &t {
        ShowTarget::Session(s) => session_md(s, &session_prompts(&docs, &s.id)),
        ShowTarget::Gh { doc, topic } => unit_md(doc, topic.as_ref().map(|(k, t)| (k.as_str(), t.as_str()))),
        ShowTarget::Topic(t) => topic_md(t),
    })
}

pub fn show_json_report(id: &str) -> Result<String> {
    let (t, docs) = show_load(id)?;
    Ok(match &t {
        ShowTarget::Session(s) => envelope("session", session_json_obj(s, &session_prompts(&docs, &s.id))),
        ShowTarget::Gh { doc, topic } => envelope(
            "unit",
            serde_json::json!({
                "kind": doc.meta.kind, "repo": doc.meta.repo, "number": doc.meta.number,
                "state": doc.meta.state, "url": doc.meta.url, "author": doc.meta.author,
                "ts": doc.meta.ts, "labels": doc.meta.labels, "agent_attr": doc.meta.agent_attr,
                "topic": topic.as_ref().map(|(k, t)| serde_json::json!({"key": k, "title": t})),
                "text": crate::excerpt(&doc.text, 1500),
            }),
        ),
        ShowTarget::Topic(t) => envelope("topic", topic_json_obj(t)),
    })
}

/// Resolver rules: `repo#N` (with or without a `gh:` prefix) is a GitHub
/// ref; anything else needs ≥4 chars and matches session ids ∪ topic keys —
/// exact first, then unique prefix; ambiguity errors list the candidates so
/// the next try resolves.
fn resolve_among(
    id: &str,
    mut sessions: Vec<crate::units::Session>,
    mut topics: Vec<crate::units::TopicUnits>,
    docs: &[Doc],
) -> Result<ShowTarget> {
    let id = id.trim();
    let id = id.strip_prefix("gh:").unwrap_or(id);
    if let Some((repo, num)) = id.rsplit_once('#') {
        let n: i64 = num
            .parse()
            .map_err(|_| anyhow::anyhow!("`{id}` looks like a PR/issue ref, but `{num}` is not a number"))?;
        let doc = docs
            .iter()
            .find(|d| {
                matches!(d.meta.kind.as_str(), "pull_request" | "issue")
                    && d.meta.repo == repo
                    && d.meta.number == Some(n)
            })
            .cloned();
        return match doc {
            Some(doc) => {
                let topic = topics
                    .into_iter()
                    .find(|t| t.units.iter().any(|u| u.doc_id == Some(doc.id)))
                    .map(|t| (crate::short(&t.cache_key), t.title().to_string()));
                Ok(ShowTarget::Gh { doc, topic })
            }
            None => anyhow::bail!("no PR/issue {repo}#{n} in the corpus — `synty github` pulls the active window"),
        };
    }
    if id.len() < 4 {
        anyhow::bail!("id `{id}` is too short — use at least 4 characters of an id printed by search/recent/topic");
    }
    if let Some(i) = sessions.iter().position(|s| s.id == id) {
        return Ok(ShowTarget::Session(Box::new(sessions.swap_remove(i))));
    }
    if let Some(i) = topics.iter().position(|t| t.cache_key == id) {
        return Ok(ShowTarget::Topic(topics.swap_remove(i)));
    }
    let s_hits: Vec<usize> =
        (0..sessions.len()).filter(|&i| sessions[i].id.starts_with(id)).collect();
    let t_hits: Vec<usize> =
        (0..topics.len()).filter(|&i| topics[i].cache_key.starts_with(id)).collect();
    match (s_hits.as_slice(), t_hits.as_slice()) {
        ([i], []) => Ok(ShowTarget::Session(Box::new(sessions.swap_remove(*i)))),
        ([], [i]) => Ok(ShowTarget::Topic(topics.swap_remove(*i))),
        ([], []) => anyhow::bail!("nothing matches `{id}` — ids come from synty search/recent/topic output"),
        _ => {
            let mut cands: Vec<String> = s_hits
                .iter()
                .take(8)
                .map(|&i| format!("{} session — {}", crate::short(&sessions[i].id), crate::excerpt(&sessions[i].ask, 48)))
                .collect();
            cands.extend(
                t_hits
                    .iter()
                    .take(8usize.saturating_sub(cands.len()))
                    .map(|&i| format!("{} topic — {}", crate::short(&topics[i].cache_key), topics[i].title())),
            );
            anyhow::bail!("`{id}` is ambiguous — {} matches:\n  {}", s_hits.len() + t_hits.len(), cands.join("\n  "))
        }
    }
}

fn session_prompts<'a>(docs: &'a [Doc], sid: &str) -> Vec<&'a Doc> {
    docs.iter().filter(|d| d.meta.session_id == sid && d.meta.kind == "user_prompt").collect()
}

/// One session's detail — the same facts the TUI's detail pane shows (it
/// delegates here), plus the full id so prefixes have something to resolve
/// against.
pub fn session_md(s: &crate::units::Session, prompts: &[&Doc]) -> String {
    let day = |ts: &str| ts.split('T').next().unwrap_or("").to_string();
    let mut o = format!(
        "session {} · {}\nid: {}\n{} → {}\n\neffort {}\n{} prompts · {} assistant · {} thinking · {} tool calls\n",
        crate::short(&s.id),
        s.repo,
        s.id,
        day(&s.started),
        day(&s.ended),
        meter(s.struggle),
        s.prompts,
        s.assistant,
        s.thinking,
        s.tools,
    );
    for line in [usage_line(s), tools_line(s)].into_iter().flatten() {
        o.push_str(&line);
        o.push('\n');
    }
    if let Some(sum) = &s.summary {
        o.push_str(&format!("summary: {sum}\n"));
    }
    if let Some(pr) = &s.linked_pr {
        o.push_str(&format!("linked PR: {pr}\n"));
    }
    if !s.files.is_empty() {
        o.push_str(&format!("files: {}\n", s.files.iter().take(10).cloned().collect::<Vec<_>>().join(", ")));
    }
    o.push_str(&format!("\nask:\n{}\n", s.ask));
    // a short representative arc: the session's user prompts
    if prompts.len() > 1 {
        o.push_str("\nturns:\n");
        for d in prompts.iter().take(8) {
            o.push_str(&format!("· {}\n", crate::first_line(&d.text)));
        }
    }
    o
}

fn session_json_obj(s: &crate::units::Session, prompts: &[&Doc]) -> serde_json::Value {
    serde_json::json!({
        "id": s.id, "repo": s.repo, "started": s.started, "ended": s.ended,
        "ask": s.ask, "prompts": s.prompts, "assistant": s.assistant,
        "thinking": s.thinking, "tools": s.tools, "files": s.files,
        "linked_pr": s.linked_pr, "struggle": s.struggle,
        "tok_in": s.tok_in, "tok_out": s.tok_out,
        "cache_read": s.cache_read, "cache_create": s.cache_create,
        "tools_by_name": s.tools_by_name,
        "by_model": s.by_model.iter().map(|m| serde_json::json!({
            "model": m.model, "tok_in": m.tok_in, "tok_out": m.tok_out,
            "cache_read": m.cache_read, "cache_create": m.cache_create, "turns": m.turns,
        })).collect::<Vec<_>>(),
        "source": s.source, "summary": s.summary, "author": s.author,
        "turns": prompts.iter().take(8).map(|d| crate::first_line(&d.text)).collect::<Vec<_>>(),
    })
}

/// One PR/issue's detail: state, link, author, labels, topic membership, and
/// a body excerpt.
pub fn unit_md(d: &Doc, topic: Option<(&str, &str)>) -> String {
    let m = &d.meta;
    let nref = m.number.map(|n| format!("{}#{n}", m.repo)).unwrap_or_else(|| m.repo.clone());
    let mut o = format!("{} {} [{}]\n", m.kind, nref, m.state.as_deref().unwrap_or("—"));
    if let Some(u) = &m.url {
        o.push_str(&format!("{u}\n"));
    }
    o.push_str(&format!("author: {} · {}", if m.author.is_empty() { "—" } else { &m.author }, m.ts.split('T').next().unwrap_or("—")));
    if !m.labels.is_empty() {
        o.push_str(&format!(" · labels: {}", m.labels.join(", ")));
    }
    if let Some(a) = &m.agent_attr {
        o.push_str(&format!(" · agent: {a}"));
    }
    o.push('\n');
    if let Some((key, title)) = topic {
        o.push_str(&format!("topic: [{key}] {title}\n"));
    }
    o.push_str(&format!("\n{}\n", crate::excerpt(&d.text, 1500)));
    o
}

/// One topic's detail: the header block `topic` prints, plus the span and
/// the FULL member list (the list view caps at 6).
pub fn topic_md(t: &crate::units::TopicUnits) -> String {
    let (sess, prs, issues) = t.mix;
    let mut o = format!(
        "# [{}] {}\nkey: {}\n{} units · last active {} · {sess} sessions / {prs} PRs / {issues} issues\n",
        crate::short(&t.cache_key),
        t.title(),
        t.cache_key,
        t.units.len(),
        t.last_active,
    );
    if let Some((a, b)) = &t.span {
        o.push_str(&format!("span: {a} → {b}\n"));
    }
    if let Some(s) = &t.summary {
        o.push_str(&format!("{s}\n"));
    }
    if !t.repos.is_empty() {
        o.push_str(&format!("repos: {}\n", t.repos.iter().take(6).cloned().collect::<Vec<_>>().join(", ")));
    }
    if !t.authors.is_empty() {
        o.push_str(&format!("authors: {}\n", t.authors.iter().take(6).cloned().collect::<Vec<_>>().join(", ")));
    }
    o.push('\n');
    for u in &t.units {
        let tag = match u.kind {
            Kind::Session => "session",
            Kind::Pr => "pr",
            Kind::Issue => "issue",
        };
        let st = if matches!(u.kind, Kind::Session) { format!(" {}", meter(u.struggle)) } else { String::new() };
        o.push_str(&format!("- `{}` {tag}{}: {} · {}{}\n", u.when, unit_id_tag(u), u.title, u.outcome, st));
    }
    o
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
    if !f.untracked.is_empty() {
        let names: Vec<String> = f
            .untracked
            .iter()
            .map(|u| match &u.agent {
                Some(a) => format!("{} ({a})", u.login),
                None => u.login.clone(),
            })
            .collect();
        // Everyone active and uncovered, attributed marked — agent use is
        // wider than the artifacts, so don't gate the list on them.
        o.push_str(&format!("  active on GitHub, not tracked ({}): {}\n", names.len(), names.join(", ")));
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
        o.push_str(&format!("- `{}` {:<7}{} _{}_ — {} · {}{}\n", u.when, tag, unit_id_tag(u), u.repo, what, u.outcome, st));
    }
    o
}

/// The inline id a session row carries (` [a1b2c3d4]`) so a follow-up
/// `synty show <id>` needs nothing but this output. GitHub rows already lead
/// their title with `repo#N`, which `show` accepts directly.
fn unit_id_tag(u: &Unit) -> String {
    match (&u.kind, &u.session_id) {
        (Kind::Session, Some(s)) if !s.is_empty() => format!(" [{}]", crate::short(s)),
        _ => String::new(),
    }
}

/// Topics with activity, type mix, and their member units.
pub fn topics_md(topics: &[TopicUnits]) -> String {
    let mut o = format!("# topics ({})\n\n", topics.len());
    for t in topics {
        let (sess, prs, issues) = t.mix;
        let n = t.activity.len();
        let wk = |i: usize| n.checked_sub(i).and_then(|x| t.activity.get(x)).copied().unwrap_or(0);
        o.push_str(&format!(
            "## [{}] {}\n{} units · last active {} · activity this/last/prior wk: {}/{}/{} · {sess} sessions / {prs} PRs / {issues} issues\n",
            crate::short(&t.cache_key),
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
            o.push_str(&format!("- `{}` {tag}{}: {}\n", u.when, unit_id_tag(u), u.title));
        }
        o.push('\n');
    }
    o
}

/// The local→bucket activation state — the rollout ramp made legible. The bucket
/// is the *only* thing that moves it: a local-only machine is invisible to the
/// fleet (it pushes no events), and setting a bucket is exactly the moment it
/// becomes a tracked member. Autostart (tracking at login) is on by default and
/// reported separately — it's not a second gate, so it stays out of this line.
pub fn activation_line(bucket: Option<&str>) -> String {
    match bucket {
        Some(b) => format!("✓ on the team — {b}"),
        None => "◐ local — `synty init <bucket>` to join your team".to_string(),
    }
}

/// A bucket URI trimmed to a glanceable name for the footer: scheme dropped, last
/// path segment kept (`gs://acme/team-x` → `team-x`, `/srv/synty-bucket` →
/// `synty-bucket`). The full URI still shows on the status page.
pub fn bucket_short(uri: &str) -> &str {
    let no_scheme = uri.split_once("://").map(|(_, rest)| rest).unwrap_or(uri);
    no_scheme.trim_end_matches('/').rsplit('/').next().unwrap_or(no_scheme)
}

pub fn status_md(s: &Status) -> String {
    let mut o = String::from("# status\n\n");
    if s.stale {
        o.push_str("⚠ index is stale — tracked events are newer; run `synty up` or `synty build`\n\n");
    }
    o.push_str(&activation_line(s.bucket.as_deref()));
    o.push('\n');
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

/// Every --json output, one shape: `{"v": 1, "kind": "...", "data": ...}` —
/// a consumer checks `v` once and dispatches on `kind`. Pre-release the inner
/// shapes may still move; `v` bumps only for a break we intend to keep.
pub fn envelope(kind: &str, data: serde_json::Value) -> String {
    serde_json::json!({"v": 1, "kind": kind, "data": data}).to_string()
}

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
    envelope("work", serde_json::Value::Array(units.iter().map(unit_json).collect()))
}

fn topic_json_obj(t: &TopicUnits) -> serde_json::Value {
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
}

pub fn topics_json(topics: &[TopicUnits]) -> String {
    envelope("topics", serde_json::Value::Array(topics.iter().map(topic_json_obj).collect()))
}

pub fn status_json(s: &Status) -> String {
    envelope("status", serde_json::json!({
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
            "untracked": s.fleet.untracked.iter().map(|u| serde_json::json!({
                "login": u.login, "agent": u.agent,
            })).collect::<Vec<_>>(),
            "untracked_attributed": s.fleet.untracked_attributed(),
            "install_rate_pct": s.fleet.install_rate_pct,
            "quiet_days": s.fleet.quiet_days,
        },
    }))
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
            gh_active: 3,
            untracked: vec![
                crate::fleet::UntrackedAuthor { login: "bob".into(), agent: Some("claude".into()) },
                crate::fleet::UntrackedAuthor { login: "carol".into(), agent: None },
            ],
            install_rate_pct: 33,
            quiet_days: 7,
        };
        let md = fleet_md(&roster);
        assert!(md.contains("install rate 33%"), "{md}");
        assert!(md.contains("mac-3939") && md.contains("claude+codex") && md.contains("v0.1.0"), "{md}");
        assert!(md.contains("ci-runner-7") && md.contains("QUIET"), "{md}");
        // Everyone uncovered is listed — carol too, though she left no
        // artifact — with bob's agent marked.
        assert!(md.contains("active on GitHub, not tracked (2): bob (claude), carol"), "{md}");
        assert!(fleet_md(&Default::default()).is_empty(), "no streams yet → no section");
    }

    fn activity_fixture() -> std::collections::HashMap<String, crate::units::DayRow> {
        let mut m = std::collections::HashMap::new();
        m.insert(
            "2026-06-12".to_string(), // a Friday — the anchor
            crate::units::DayRow { tok_in: 46_100, tok_out: 12_300, cache_read: 1_200_000, cache_create: 96_000, tools: 412, sessions: 9, loc_add: 4_100, loc_del: 1_200, prs: 6, issues: 3 },
        );
        m.insert(
            "2026-05-26".to_string(), // a Tuesday two weeks earlier
            crate::units::DayRow { tok_out: 2_000, tools: 3, sessions: 1, prs: 1, ..Default::default() },
        );
        m.insert(
            "2026-04-01".to_string(), // far outside any 3-week window
            crate::units::DayRow { tok_out: 999_999, ..Default::default() },
        );
        m
    }

    // Weeks are whole Mon–Sun buckets ending with the week of the newest day
    // that HAS data; days outside the window don't leak into the totals.
    #[test]
    fn stats_weeks_are_monday_aligned_and_anchored_to_data() {
        let s = stats_view(activity_fixture(), 3);
        assert_eq!(s.anchor, "2026-06-12");
        let starts: Vec<&str> = s.weeks.iter().map(|w| w.start.as_str()).collect();
        assert_eq!(starts, ["2026-05-25", "2026-06-01", "2026-06-08"], "Mondays, oldest first");
        assert_eq!(s.weeks[0].row.prs, 1, "the Tuesday lands in its own week");
        assert_eq!(s.weeks[2].row.sessions, 9, "the anchor day lands in the last week");
        assert_eq!(s.total.tok_out, 14_300, "out-of-window days are excluded from the total");
        assert_eq!(s.days.len(), 2, "the day series carries only the window");
        assert!(s.days[0].0 < s.days[1].0, "days ascend");
        assert!(stats_view(Default::default(), 4).weeks.is_empty(), "no data → empty view");
    }

    // The Markdown mirrors the TUI's two charts: an agents table over an
    // output table, weekly rows plus a total, numbers humanized.
    #[test]
    fn stats_md_renders_agents_and_output_tables() {
        let s = stats_view(activity_fixture(), 3);
        let md = stats_md(&s);
        assert!(md.contains("# stats — 3 weeks to 2026-06-12"), "{md}");
        assert!(md.contains("agents (wk of · tok-in · tok-out · cache-r · cache-w · tools · sessions):"), "{md}");
        assert!(md.contains("output (wk of · loc+ · loc- · prs · issues):"), "{md}");
        assert!(md.contains("46.1k") && md.contains("1.2M"), "humanized numbers: {md}");
        assert_eq!(md.matches("total").count(), 2, "each table closes with a total row: {md}");
        assert!(stats_md(&stats_view(Default::default(), 4)).contains("no data yet"), "empty corpus stays friendly");
    }

    // --json is the day-resolution surface: an enveloped, day-keyed series.
    #[test]
    fn stats_json_is_day_keyed_under_envelope() {
        let s = stats_view(activity_fixture(), 3);
        let v: serde_json::Value = serde_json::from_str(&stats_json(&s)).unwrap();
        assert_eq!((v["v"].as_i64(), v["kind"].as_str()), (Some(1), Some("stats")));
        assert_eq!(v["data"]["anchor"], "2026-06-12");
        assert_eq!(v["data"]["days"]["2026-06-12"]["tok_in"], 46_100);
        assert_eq!(v["data"]["weeks"][0]["start"], "2026-05-25");
        assert_eq!(v["data"]["total"]["prs"], 7);
    }

    fn tool_profile_fixture() -> crate::units::ToolProfile {
        crate::units::ToolProfile {
            name: "Bash".into(),
            agent: "claude".into(),
            calls: 3665,
            errs: 106,
            chars: 5_600_000,
            days: [("2026-06-12".to_string(), 12u64), ("2026-05-26".to_string(), 30u64)].into_iter().collect(),
            arg_keys: vec![("command".into(), 3665), ("timeout".into(), 420)],
            arg_tops: vec![("timeout".into(), vec![("120000".into(), 300), ("600000".into(), 120)])],
            p50_ms: 740,
            p95_ms: 12_400,
            timed: 3600,
            input_p50: 180,
            input_p95: 900,
            samples: vec![r#"{"command":"cargo test"}"#.into()],
        }
    }

    // The CLI tool profile carries the same facts as the TUI overlay: volume,
    // latency, context estimate, weekly calls, argument shares, recent calls.
    #[test]
    fn tool_md_shows_volume_latency_args_and_samples() {
        let md = tool_md(&tool_profile_fixture());
        assert!(md.contains("# Bash · claude"), "{md}");
        assert!(md.contains("3.7k calls") && md.contains("106 errors"), "{md}");
        assert!(md.contains("result p50 740ms · p95 12400ms (3600 timed)"), "{md}");
        assert!(md.contains("input p50 180 / p95 900 chars"), "{md}");
        assert!(md.contains("context ~1.4M tok"), "{md}");
        assert!(md.contains("calls by week (oldest→newest): 0 · 30 · 0 · 12"), "{md}");
        assert!(md.contains("command") && md.contains("(100%)"), "{md}");
        assert!(md.contains("120000×300 · 600000×120"), "common values for enum-ish keys: {md}");
        assert!(md.contains("cargo test"), "{md}");
    }

    #[test]
    fn tool_json_envelope_kind_is_tool() {
        let v: serde_json::Value = serde_json::from_str(&tool_json(&tool_profile_fixture())).unwrap();
        assert_eq!((v["v"].as_i64(), v["kind"].as_str()), (Some(1), Some("tool")));
        assert_eq!(v["data"]["calls"], 3665);
        assert_eq!(v["data"]["context_tokens_est"], 1_400_000);
        assert_eq!(v["data"]["days"]["2026-06-12"], 12);
    }

    // A near-miss name gets pointed at the real one; rubbish gets nothing.
    #[test]
    fn tool_suggestions_match_case_and_prefix() {
        let names = vec!["Bash".to_string(), "Edit".to_string(), "Read".to_string()];
        assert_eq!(tool_suggestions("bash", &names), ["Bash"], "case-insensitive exact match");
        assert_eq!(tool_suggestions("re", &names), ["Read"], "prefix match");
        assert!(tool_suggestions("zzz", &names).is_empty());
    }

    fn show_session(id: &str, ask: &str) -> crate::units::Session {
        let mut s = session();
        s.id = id.into();
        s.ask = ask.into();
        s
    }

    fn show_topic(key: &str, title: &str) -> crate::units::TopicUnits {
        crate::units::TopicUnits {
            id: 0,
            cache_key: key.into(),
            label: title.into(),
            units: vec![],
            last_active: "2026-06-01".into(),
            activity: vec![],
            mix: (0, 0, 0),
            repos: vec![],
            authors: vec![],
            summary: None,
            name: None,
            span: None,
        }
    }

    /// unwrap_err without demanding Debug on the (heavy) Ok arm.
    fn show_err(r: Result<ShowTarget>) -> String {
        match r {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        }
    }

    fn pr_doc(repo: &str, n: i64) -> Doc {
        Doc {
            id: 7,
            text: "fix race\n\nSerialize the tailer so restarts can't double-emit.".into(),
            meta: crate::Meta {
                source: "github".into(),
                kind: "pull_request".into(),
                repo: repo.into(),
                author: "alice".into(),
                session_id: String::new(),
                ts: "2026-06-01T10:00:00Z".into(),
                number: Some(n),
                url: Some("https://gh/7".into()),
                state: Some("MERGED".into()),
                labels: vec!["bug".into()],
                agent_attr: None,
            },
        }
    }

    // The ids the views print resolve to the right kind of detail: repo#N to
    // the PR, a session id (full or unique prefix) to the session, a topic
    // key to the topic.
    #[test]
    fn show_resolves_gh_session_and_topic_ids() {
        let sess = || vec![show_session("a1b2c3d4-9999-4242-8888-777766665555", "fix login")];
        let tops = || vec![show_topic("72a778f8aabbccdd", "Auth work")];
        let docs = vec![pr_doc("sie", 7)];
        assert!(matches!(resolve_among("sie#7", sess(), tops(), &docs), Ok(ShowTarget::Gh { .. })));
        assert!(matches!(
            resolve_among("a1b2c3d4-9999-4242-8888-777766665555", sess(), tops(), &docs),
            Ok(ShowTarget::Session(_))
        ));
        assert!(matches!(resolve_among("a1b2c3d4", sess(), tops(), &docs), Ok(ShowTarget::Session(_))), "a printed id8 prefix resolves");
        assert!(matches!(resolve_among("72a778f8aabbccdd", sess(), tops(), &docs), Ok(ShowTarget::Topic(_))));
        assert!(matches!(resolve_among("72a778f8", sess(), tops(), &docs), Ok(ShowTarget::Topic(_))));
        let miss = show_err(resolve_among("ffff0000", sess(), tops(), &docs));
        assert!(miss.contains("nothing matches"), "{miss}");
    }

    // Both spellings of a GitHub ref work — `gh:` is what docs/JSON print,
    // bare repo#N is what humans type.
    #[test]
    fn show_accepts_gh_prefix_and_bare_repo_number() {
        let docs = vec![pr_doc("sie", 7)];
        for id in ["gh:sie#7", "sie#7"] {
            assert!(matches!(resolve_among(id, vec![], vec![], &docs), Ok(ShowTarget::Gh { .. })), "{id}");
        }
        let miss = show_err(resolve_among("sie#99", vec![], vec![], &docs));
        assert!(miss.contains("no PR/issue sie#99"), "{miss}");
    }

    // An ambiguous prefix errors with the candidates, so the next try lands.
    #[test]
    fn show_prefix_ambiguity_lists_candidates() {
        let sessions = vec![
            show_session("aaaa1111-0000-4000-8000-000000000001", "first ask"),
            show_session("aaaa2222-0000-4000-8000-000000000002", "second ask"),
        ];
        let e = show_err(resolve_among("aaaa", sessions, vec![], &[]));
        assert!(e.contains("ambiguous") && e.contains("2 matches"), "{e}");
        assert!(e.contains("aaaa1111") && e.contains("aaaa2222"), "{e}");
        assert!(e.contains("first ask"), "candidates carry their ask: {e}");
    }

    // A 1–3 char prefix would match half the corpus — refuse it with advice.
    #[test]
    fn show_rejects_short_prefixes() {
        let e = show_err(resolve_among("abc", vec![], vec![], &[]));
        assert!(e.contains("too short"), "{e}");
    }

    // The CLI session detail carries the same facts as the TUI pane (which
    // delegates here): header, effort, counts, files capped at 10, the ask,
    // the prompt arc — plus the full id for prefix resolution.
    #[test]
    fn session_md_mirrors_tui_detail() {
        let mut s = show_session("a1b2c3d4-9999-4242-8888-777766665555", "fix the login redirect");
        s.repo = "sie".into();
        s.started = "2026-06-01T08:00:00Z".into();
        s.ended = "2026-06-01T09:30:00Z".into();
        s.summary = Some("Fixed the redirect loop.".into());
        s.linked_pr = Some("sie#7".into());
        s.files = (0..12).map(|i| format!("file{i:02}.rs")).collect();
        let prompt = |t: &str| Doc {
            id: 0,
            text: t.into(),
            meta: crate::Meta {
                source: "agent".into(),
                kind: "user_prompt".into(),
                repo: "sie".into(),
                author: String::new(),
                session_id: s.id.clone(),
                ts: String::new(),
                number: None,
                url: None,
                state: None,
                labels: vec![],
                agent_attr: None,
            },
        };
        let (p1, p2) = (prompt("fix the login redirect"), prompt("now add a regression test"));
        let md = session_md(&s, &[&p1, &p2]);
        assert!(md.starts_with("session a1b2c3d4 · sie"), "{md}");
        assert!(md.contains("id: a1b2c3d4-9999-4242-8888-777766665555"), "{md}");
        assert!(md.contains("2026-06-01 → 2026-06-01"), "{md}");
        assert!(md.contains("effort [") && md.contains("1 prompts · 1 assistant"), "{md}");
        assert!(md.contains("summary: Fixed the redirect loop."), "{md}");
        assert!(md.contains("linked PR: sie#7"), "{md}");
        assert!(md.contains("file09.rs") && !md.contains("file10.rs"), "files cap at 10: {md}");
        assert!(md.contains("ask:\nfix the login redirect"), "{md}");
        assert!(md.contains("turns:") && md.contains("· now add a regression test"), "{md}");
    }

    // `show <topic>` lists every member (the topic list view caps at 6).
    #[test]
    fn topic_md_lists_all_members_with_ids() {
        let mut t = show_topic("72a778f8aabbccdd", "Auth work");
        t.units = (0..8)
            .map(|i| Unit {
                kind: Kind::Session,
                when: "2026-06-01".into(),
                repo: "sie".into(),
                title: format!("session number {i}"),
                outcome: "done".into(),
                summary: None,
                topic: Some(0),
                rank: 0,
                dup: false,
                struggle: 0.0,
                author: "alice".into(),
                doc_id: None,
                session_id: Some(format!("000000{i:02}-0000-4000-8000-000000000000")),
            })
            .collect();
        let md = topic_md(&t);
        assert!(md.contains("# [72a778f8] Auth work"), "{md}");
        assert!(md.contains("key: 72a778f8aabbccdd"), "the full key is there to copy: {md}");
        for i in 0..8 {
            assert!(md.contains(&format!("session number {i}")), "member {i} missing: {md}");
        }
        assert!(md.contains("[00000007]"), "member rows carry ids: {md}");
    }

    // PR detail names its state, link, author, topic, and excerpts the body.
    #[test]
    fn unit_md_carries_state_topic_and_body() {
        let md = unit_md(&pr_doc("sie", 7), Some(("72a778f8", "Auth work")));
        assert!(md.starts_with("pull_request sie#7 [MERGED]"), "{md}");
        assert!(md.contains("https://gh/7"), "{md}");
        assert!(md.contains("author: alice · 2026-06-01 · labels: bug"), "{md}");
        assert!(md.contains("topic: [72a778f8] Auth work"), "{md}");
        assert!(md.contains("Serialize the tailer"), "{md}");
    }

    // Session rows print their stable id inline, so an agent reading the
    // Markdown can `synty show <id>` without re-querying in JSON.
    #[test]
    fn work_md_inlines_session_ids() {
        let unit = Unit {
            kind: Kind::Session,
            when: "2026-06-01".into(),
            repo: "sie".into(),
            title: "fix login".into(),
            outcome: "done".into(),
            summary: None,
            topic: None,
            rank: 0,
            dup: false,
            struggle: 0.0,
            author: "alice".into(),
            doc_id: None,
            session_id: Some("a1b2c3d4-9999-4242-8888-777766665555".into()),
        };
        let md = work_md(&[unit]);
        assert!(md.contains("session [a1b2c3d4]"), "{md}");
    }

    // Topic headers lead with the stable content-addressed key — the
    // positional index is display-only and lies across re-clusters.
    #[test]
    fn topics_md_leads_with_the_stable_key() {
        let t = crate::units::TopicUnits {
            id: 3,
            cache_key: "72a778f8aabbccdd".into(),
            label: "auth work".into(),
            units: vec![Unit {
                kind: Kind::Session,
                when: "2026-06-01".into(),
                repo: "sie".into(),
                title: "fix login".into(),
                outcome: "done".into(),
                summary: None,
                topic: Some(3),
                rank: 0,
                dup: false,
                struggle: 0.0,
                author: "alice".into(),
                doc_id: None,
                session_id: Some("a1b2c3d4-9999-4242-8888-777766665555".into()),
            }],
            last_active: "2026-06-01".into(),
            activity: vec![1],
            mix: (1, 0, 0),
            repos: vec!["sie".into()],
            authors: vec!["alice".into()],
            summary: None,
            name: None,
            span: None,
        };
        let md = topics_md(&[t]);
        assert!(md.contains("## [72a778f8] auth work"), "{md}");
        assert!(!md.contains("## 3 —"), "the positional id is gone from MD: {md}");
        assert!(md.contains("session [a1b2c3d4]:"), "member rows carry ids too: {md}");
    }

    // Every --json output arrives in the same versioned envelope, so a script
    // checks `v` once and dispatches on `kind` — and the data keeps its shape.
    #[test]
    fn json_outputs_carry_versioned_envelope() {
        let unit = Unit {
            kind: Kind::Session,
            when: "2026-06-01".into(),
            repo: "sie".into(),
            title: "fix login".into(),
            outcome: "done".into(),
            summary: None,
            topic: None,
            rank: 0,
            dup: false,
            struggle: 0.0,
            author: "alice".into(),
            doc_id: None,
            session_id: Some("S1".into()),
        };
        let w: serde_json::Value = serde_json::from_str(&work_json(&[unit])).unwrap();
        assert_eq!((w["v"].as_i64(), w["kind"].as_str()), (Some(1), Some("work")));
        assert_eq!(w["data"][0]["repo"], "sie", "the data keeps the prior array shape");

        let t: serde_json::Value = serde_json::from_str(&topics_json(&[])).unwrap();
        assert_eq!((t["v"].as_i64(), t["kind"].as_str()), (Some(1), Some("topics")));
        assert!(t["data"].is_array());

        let s = Status {
            docs: 3,
            github: 1,
            sessions: 1,
            by_kind: vec![],
            by_repo: vec![],
            by_user: vec![],
            by_tool: vec![],
            by_model: vec![],
            newest_ts: String::new(),
            last_indexed: None,
            last_tracked: None,
            autostart: false,
            bucket: None,
            stale: false,
            fleet: Default::default(),
        };
        let s: serde_json::Value = serde_json::from_str(&status_json(&s)).unwrap();
        assert_eq!((s["v"].as_i64(), s["kind"].as_str()), (Some(1), Some("status")));
        assert_eq!(s["data"]["docs"], 3, "the data keeps the prior object shape");
        assert!(s["data"]["fleet"]["machines"].is_array());
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

    // The local→bucket ramp: the bucket is the only thing that moves the badge.
    // Pure — the rendered line is what tells a person where they stand.
    #[test]
    fn activation_local_invites_to_join() {
        let line = activation_line(None);
        assert!(line.contains("local"), "no bucket → local: {line}");
        assert!(line.contains("synty init"), "invites joining a team via init: {line}");
        assert!(!line.contains("trial"), "no '(trial)' framing: {line}");
    }

    #[test]
    fn activation_bucket_is_on_the_team() {
        let line = activation_line(Some("gs://my-team"));
        assert!(line.contains('✓'), "a bucket alone reads as activated: {line}");
        assert!(line.contains("gs://my-team"), "names the bucket: {line}");
        // Autostart is reported elsewhere; it never appears in this line and so
        // can't un-activate a machine that has a bucket.
        assert!(!line.to_lowercase().contains("autostart"), "autostart stays out of the line: {line}");
    }

    // The footer name is the bucket trimmed to one glanceable segment.
    #[test]
    fn bucket_short_drops_scheme_and_path() {
        assert_eq!(bucket_short("gs://acme-synty"), "acme-synty");
        assert_eq!(bucket_short("s3://acme/team-x"), "team-x");
        assert_eq!(bucket_short("/srv/synty-bucket/"), "synty-bucket");
        assert_eq!(bucket_short("local-dir"), "local-dir");
    }
}
