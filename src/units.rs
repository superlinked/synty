// Units of work — the human-facing objects (sessions, PRs, issues) the surfaces
// browse, each with a time axis and a derived "struggle" score. Built from the
// raw envelopes under corpus/local (for session structure) plus docs.jsonl and
// clusters.json (for PRs/issues and topic membership). Consumed by both the CLI
// and the TUI so they stay at parity.

use crate::{first_line, load_docs, readmodel};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

pub(crate) const LOCAL_DIR: &str = "corpus/local";
const FILE_TOOLS: &[&str] = &["Write", "Edit", "MultiEdit", "NotebookEdit"];

/// A coding session as one unit of work. The raw-derived fields are persisted
/// in the published analysis snapshot; topic and summary are refreshed when
/// the snapshot is read.
#[derive(Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub repo: String,
    pub started: String,
    pub ended: String,
    pub ask: String,
    pub prompts: usize,
    pub assistant: usize,
    pub thinking: usize,
    pub tools: usize,
    pub files: Vec<String>,
    pub linked_pr: Option<String>,
    pub topic: Option<i64>,
    pub struggle: f32, // 0..1, normalized across sessions
    // Token accounting from the agent's own usage records, cache classes kept
    // separate (see Agg). All zero when the source recorded no usage — the
    // views then show nothing rather than a fake 0.
    pub tok_in: u64,
    pub tok_out: u64,
    pub cache_read: u64,
    pub cache_create: u64,
    pub usage_turns: usize, // deduped API turns (0 for codex's cumulative path)
    pub tools_by_name: Vec<(String, usize, usize, u64)>, // (name, calls, attributed errors, payload chars), calls desc
    pub tool_err: usize,
    pub by_model: Vec<ModelUsage>, // per-model split of the same totals, tok_out desc
    pub source: String, // envelope source: claude_code | codex_cli | cowork
    pub campaign_id: String,
    pub campaign_role: String,
    pub summary: Option<String>, // abstractive one-liner (local LLM), if cached
    /// The actor the tracker stamped on session_start; empty for sessions
    /// tracked before stamping (callers fall back to the local actor).
    pub author: String,
}

impl Session {
    /// True when the source actually recorded token usage. The display rule:
    /// no usage → no number — a rendered "0 tokens" would read as a
    /// measurement, and cowork (plus pre-capture envelopes) record none.
    pub fn has_usage(&self) -> bool {
        self.usage_turns > 0 || self.tok_in + self.tok_out + self.cache_read + self.cache_create > 0
    }
}

/// One model's share of a session's (or the fleet's) token usage. Codex
/// reports no model on its cumulative snapshots, so its share carries the
/// agent name instead.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct ModelUsage {
    pub model: String,
    pub tok_in: u64,
    pub tok_out: u64,
    pub cache_read: u64,
    pub cache_create: u64,
    pub turns: usize,
}

/// The measurable volume of a tool event: the serialized arguments (call
/// side: Claude's input object, codex's arguments string, a web_search
/// action) or the result content (Claude's content, codex's output), in
/// chars. No agent meters tokens per tool call — usage is per API turn — so
/// payload volume is the honest per-tool basis, surfaced as a clearly-marked
/// ~chars/4 estimate.
fn payload_chars(p: &Value) -> u64 {
    let v = ["input", "arguments", "action", "content", "output"]
        .iter()
        .map(|k| &p[*k])
        .find(|v| !v.is_null());
    match v {
        Some(Value::String(s)) => s.chars().count() as u64,
        Some(other) => serde_json::to_string(other).map(|s| s.chars().count()).unwrap_or(0) as u64,
        None => 0,
    }
}

/// Short agent label from an envelope source ("claude_code" → "claude").
pub fn agent_label(source: &str) -> &str {
    match source {
        "claude_code" => "claude",
        "codex_cli" => "codex",
        s => s.split('_').next().unwrap_or(s),
    }
}

/// A session's cached LLM summary, keyed by a hash of its inputs so the
/// summarizer only regenerates when the underlying turns change.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CachedSummary {
    pub hash: String,
    pub summary: String,
}

pub type SummaryCache = HashMap<String, CachedSummary>;

/// The cache lives under .synty/, NOT index/ — a full index rebuild wipes
/// index/, and hours of local-LLM summarization must survive that.
const SUMMARIES_PATH: &str = ".synty/summaries.json";

/// Read the on-disk summary cache (empty if the summarizer hasn't run).
pub fn load_summary_cache() -> SummaryCache {
    std::fs::read_to_string(SUMMARIES_PATH).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

/// Persist the summary cache next to the index.
pub fn save_summary_cache(c: &SummaryCache) -> Result<()> {
    if let Some(parent) = Path::new(SUMMARIES_PATH).parent() {
        std::fs::create_dir_all(parent)?;
    }
    crate::write_atomic(SUMMARIES_PATH, serde_json::to_string_pretty(c)?.as_bytes())?;
    Ok(())
}

/// Per-session material for the summarizer: the ask and the few longest turns
/// (the substantive ones).
pub struct SessionInput {
    pub id: String,
    pub repo: String,
    pub ask: String,
    pub files: Vec<String>,
    pub turns: Vec<String>,
}

/// The kind of a unit in a unified list.
#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Session,
    Pr,
    Issue,
}

/// A row in the unified Work list / search results / topic membership.
#[derive(Clone)]
pub struct Unit {
    pub kind: Kind,
    pub when: String, // day
    pub repo: String,
    pub title: String,
    pub outcome: String, // PR state, or session file/PR summary
    pub summary: Option<String>, // one-line LLM summary, if cached
    pub topic: Option<i64>,
    pub rank: i64,  // centrality rank within its topic (0 = medoid; MAX unranked)
    pub dup: bool,  // a collapsed near-duplicate of another unit in its topic
    pub struggle: f32,
    pub author: String,    // PR/issue author, or the resolved actor for sessions
    pub doc_id: Option<i64>,    // for PR/issue → docs.jsonl
    pub session_id: Option<String>, // for sessions
    pub campaign_id: String,
    pub campaign_role: String,
    pub source: String, // native producer, suitable for read-scope checks
}

/// A topic with its work units, time span, activity over weeks, and type mix.
pub struct TopicUnits {
    pub id: i64, // positional cluster index — display/order only, NOT stable across runs
    pub cache_key: String, // stable content-addressed key for the summary/name cache
    pub label: String,
    pub units: Vec<Unit>,
    pub last_active: String,
    pub activity: Vec<u64>, // weekly buckets, oldest→newest
    pub mix: (usize, usize, usize), // (github, assistant, prompt) doc counts
    pub repos: Vec<String>,   // repos involved, most frequent first
    pub authors: Vec<String>, // authors involved, most frequent first
    pub summary: Option<String>, // map-reduced topic summary (local LLM), if cached
    pub name: Option<String>, // short LLM title (local LLM), if cached
    pub span: Option<(String, String)>, // first/last active day (YYYY-MM-DD)
}

impl TopicUnits {
    /// The display title: the short LLM name, else the (longer) summary, else the
    /// cluster's representative-member label — so a topic only falls back to the
    /// label when it has neither a name nor a summary.
    pub fn title(&self) -> &str {
        self.name.as_deref().or(self.summary.as_deref()).unwrap_or(&self.label)
    }
}

/// Cache key for a topic's reduced summary. `key` is the cluster's stable
/// content-addressed key (medoid hash), so the cache survives re-clustering's
/// renumbering — a renumber yields a cache miss (→ regenerate), never a stale
/// name attached to a different cluster.
pub fn topic_key(key: &str) -> String {
    format!("topic:{key}")
}

/// Cache key for a topic's short LLM name.
pub fn topic_name_key(key: &str) -> String {
    format!("topicname:{key}")
}

#[derive(Default)]
struct Agg {
    repo: String,
    actor: String,
    first: String,
    last: String,
    ask: String,
    prompts: usize,
    assistant: usize,
    thinking: usize,
    tools: usize,
    files: Vec<String>,
    seen_files: std::collections::HashSet<String>,
    linked_pr: Option<String>,
    texts: Vec<String>, // message texts, for the summarizer's turn selection
    // Token accounting from the agent's own usage records — never estimated.
    // The four classes stay separate end-to-end: a cache read prices ~10x
    // below fresh input, so one conflated "input" number would be a lie.
    tok_in: u64,
    tok_out: u64,
    cache_read: u64,
    cache_create: u64,
    usage_turns: usize,
    seen_msgs: std::collections::HashSet<String>, // streamed lines repeat usage per msg_id
    tool_counts: HashMap<String, usize>,
    tool_err: usize,
    tool_pending: HashMap<String, String>, // call id → tool name, for error attribution
    tool_errs: HashMap<String, usize>, // per-name errors (Claude only: codex results carry no error flag)
    tool_chars: HashMap<String, u64>, // per-name payload volume: args + result content, chars
    source: String, // envelope source, first seen wins (a session has one tailer)
    campaign_id: String,
    campaign_role: String,
    model_usage: HashMap<String, ModelUsage>, // per-model split, same msg_id dedup
    /// Codex reports cumulative (input_total, cached, output) snapshots; the
    /// last one is the session total. Kept raw here, normalized in sessions().
    codex_usage: Option<(u64, u64, u64)>,
}

/// Fold one envelope line into the per-session tallies. Split from
/// `aggregate()` so scenario tests can feed envelope strings directly; the
/// event-id `seen` set stays caller-owned because dedup spans files.
#[cfg(test)]
fn fold_line(
    line: &str,
    known: &std::collections::HashSet<String>,
    seen: &mut std::collections::HashSet<u64>,
    aggs: &mut HashMap<String, Agg>,
) {
    fold_line_since(line, known, seen, aggs, None, true)
}

fn fold_line_since(
    line: &str,
    known: &std::collections::HashSet<String>,
    seen: &mut std::collections::HashSet<u64>,
    aggs: &mut HashMap<String, Agg>,
    capture_since_ms: Option<i64>,
    collect_texts: bool,
) {
    if line.trim().is_empty() {
        return;
    }
    let Ok(ev) = serde_json::from_str::<Value>(line) else {
        return;
    };
    let ts = ev["ts"].as_str().unwrap_or("");
    let visible = crate::config::captured_at(ts, capture_since_ms);
    // A pre-boundary session_start carries repo/actor metadata required by a
    // retained later turn. Every other proven-old event stays out of summaries
    // and shared derived artifacts.
    if !visible && ev["kind"].as_str() != Some("session_start") {
        return;
    }
    if !crate::event::first_sighting(seen, ev["event_id"].as_str().unwrap_or("")) {
        return;
    }
    let sid = ev["session_id"].as_str().unwrap_or("");
    if sid.is_empty() {
        return;
    }
    let a = aggs.entry(sid.to_string()).or_default();
    if a.source.is_empty() {
        if let Some(src) = ev["source"].as_str().filter(|s| !s.is_empty()) {
            a.source = src.to_string();
        }
    }
    if a.campaign_id.is_empty() {
        a.campaign_id = ev["rollup_dim"]
            .as_str()
            .filter(|value| !value.is_empty())
            .or_else(|| ev["payload"]["campaign_id"].as_str())
            .unwrap_or("")
            .to_string();
    }
    if a.campaign_role.is_empty() {
        a.campaign_role = ev["payload"]["campaign_role"].as_str().unwrap_or("").to_string();
    }
    if visible && !ts.is_empty() {
        if a.first.is_empty() || ts < a.first.as_str() {
            a.first = ts.to_string();
        }
        if ts > a.last.as_str() {
            a.last = ts.to_string();
        }
    }
    match ev["kind"].as_str().unwrap_or("") {
        "session_start" => {
            if let Some(cwd) = ev["payload"]["cwd"].as_str() {
                a.repo = resolve_repo(cwd, known);
            }
            if let Some(actor) = ev["payload"]["actor"].as_str() {
                a.actor = actor.to_string();
            }
        }
        "user_prompt" => {
            // Skip slash-command echoes / hook output so the "ask" is the
            // real first human prompt and the count reflects real turns.
            if let Some(t) = ev["payload"]["text"].as_str() {
                let t = t.trim();
                if t.len() >= 12 && !crate::ingest::is_noise(t) {
                    a.prompts += 1;
                    if collect_texts {
                        a.texts.push(t.to_string());
                    }
                    if a.ask.is_empty() {
                        a.ask = crate::excerpt(t, 200);
                    }
                }
            }
        }
        "assistant_message" => {
            a.assistant += 1;
            if collect_texts && let Some(t) = ev["payload"]["text"].as_str() {
                if t.trim().len() >= 12 {
                    a.texts.push(t.trim().to_string());
                }
            }
        }
        "thinking" => a.thinking += 1,
        "tool_call" => {
            a.tools += 1;
            if let Some(name) = ev["payload"]["name"].as_str().filter(|n| !n.is_empty()) {
                *a.tool_counts.entry(name.to_string()).or_default() += 1;
                *a.tool_chars.entry(name.to_string()).or_default() += payload_chars(&ev["payload"]);
                // Both id schemes: Claude's tool_use_id, codex's call_id.
                let id = ev["payload"]["tool_use_id"].as_str().or_else(|| ev["payload"]["call_id"].as_str());
                if let Some(id) = id.filter(|i| !i.is_empty()) {
                    a.tool_pending.insert(id.to_string(), name.to_string());
                }
            }
        }
        "tool_result" => {
            let id = ev["payload"]["tool_use_id"].as_str().or_else(|| ev["payload"]["call_id"].as_str());
            let name = id.and_then(|i| a.tool_pending.get(i)).cloned();
            if let Some(name) = &name {
                *a.tool_chars.entry(name.clone()).or_default() += payload_chars(&ev["payload"]);
            }
            if ev["payload"]["is_error"].as_bool().unwrap_or(false) {
                a.tool_err += 1;
                if let Some(name) = name {
                    *a.tool_errs.entry(name).or_default() += 1;
                }
            }
        }
        "attachment_ref" => {
            if FILE_TOOLS.contains(&ev["payload"]["tool_name"].as_str().unwrap_or("")) {
                if let Some(tok) = ev["payload"]["local_path"].as_str().and_then(file_token) {
                    if a.seen_files.insert(tok.clone()) {
                        a.files.push(tok);
                    }
                }
            }
        }
        "agent_meta" => {
            let p = &ev["payload"];
            match p["subtype"].as_str().unwrap_or("") {
                // Claude Code: one usage record per raw assistant line; a
                // streamed turn repeats the IDENTICAL object on every line of
                // one msg_id (measured 2-4x), so count each msg_id once. A
                // record without msg_id can't be deduped — count it as-is.
                "usage" => {
                    let dedup = p["msg_id"].as_str().filter(|m| !m.is_empty()).map(String::from);
                    if dedup.map(|m| a.seen_msgs.insert(m)).unwrap_or(true) {
                        let n = |k: &str| p["usage"][k].as_u64().unwrap_or(0);
                        a.tok_in += n("in");
                        a.tok_out += n("out");
                        a.cache_read += n("cache_read");
                        a.cache_create += n("cache_create");
                        a.usage_turns += 1;
                        let model = p["model"].as_str().filter(|m| !m.is_empty()).unwrap_or("?");
                        let mu = a.model_usage.entry(model.to_string()).or_default();
                        mu.model = model.to_string();
                        mu.tok_in += n("in");
                        mu.tok_out += n("out");
                        mu.cache_read += n("cache_read");
                        mu.cache_create += n("cache_create");
                        mu.turns += 1;
                    }
                }
                // Codex: token_count snapshots are CUMULATIVE; one rollout
                // file per session in append order, so the last one seen is
                // the session total. The corpus keeps codex's raw semantics
                // (input includes cached; output includes reasoning), so this
                // policy is retroactively revisable from the same events.
                "event_msg" if p["event_kind"] == "token_count" => {
                    let u = &p["payload"]["info"]["total_token_usage"];
                    if u.is_object() {
                        a.codex_usage = Some((
                            u["input_tokens"].as_u64().unwrap_or(0),
                            u["cached_input_tokens"].as_u64().unwrap_or(0),
                            u["output_tokens"].as_u64().unwrap_or(0),
                        ));
                    }
                }
                _ => {}
            }
            if a.linked_pr.is_none() {
                if let Some(url) = p["pr_url"].as_str().filter(|u| !u.is_empty()) {
                    a.linked_pr = Some(url.to_string());
                } else if let (Some(repo), Some(num)) = (p["pr_repository"].as_str(), p["pr_number"].as_i64()) {
                    if num > 0 {
                        a.linked_pr = Some(format!("{repo}#{num}"));
                    }
                }
            }
        }
        _ => {}
    }
}

/// One day's usage across every session — the substrate for the Status stats
/// panel's time series. Token fields follow the same accounting as the
/// per-session aggregation (msg_id dedup, codex last-snapshot normalization);
/// `sessions` counts the distinct sessions active that day.
#[derive(Default, Clone, Copy, Serialize, Deserialize)]
pub struct DayStat {
    pub tok_in: u64,
    pub tok_out: u64,
    pub cache_read: u64,
    pub cache_create: u64,
    pub tools: u64,
    pub sessions: u64,
}

/// The day-series fold. Claude usage lands on the day of its (msg_id-deduped)
/// event — exact, even for sessions spanning days. A codex session's
/// cumulative last snapshot lands on that snapshot's own day — coarse, but
/// codex reports no per-turn usage.
#[derive(Default)]
struct DayFold {
    days: HashMap<String, DayStat>,
    seen: std::collections::HashSet<u64>,
    seen_msgs: std::collections::HashSet<String>,
    active: HashMap<String, std::collections::HashSet<String>>,
    codex: HashMap<String, ((u64, u64, u64), String)>, // sid -> (last snapshot, its day)
}

impl DayFold {
    fn fold(&mut self, line: &str) {
        if line.trim().is_empty() {
            return;
        }
        let Ok(ev) = serde_json::from_str::<Value>(line) else { return };
        if !crate::event::first_sighting(&mut self.seen, ev["event_id"].as_str().unwrap_or("")) {
            return;
        }
        let sid = ev["session_id"].as_str().unwrap_or("");
        let day = ev["ts"].as_str().unwrap_or("").split('T').next().unwrap_or("");
        if sid.is_empty() || day.len() != 10 {
            return;
        }
        self.active.entry(day.to_string()).or_default().insert(sid.to_string());
        match ev["kind"].as_str().unwrap_or("") {
            "tool_call" => self.days.entry(day.to_string()).or_default().tools += 1,
            "agent_meta" => {
                let p = &ev["payload"];
                match p["subtype"].as_str().unwrap_or("") {
                    "usage" => {
                        let dedup = p["msg_id"].as_str().filter(|m| !m.is_empty()).map(String::from);
                        if dedup.map(|m| self.seen_msgs.insert(m)).unwrap_or(true) {
                            let n = |k: &str| p["usage"][k].as_u64().unwrap_or(0);
                            let d = self.days.entry(day.to_string()).or_default();
                            d.tok_in += n("in");
                            d.tok_out += n("out");
                            d.cache_read += n("cache_read");
                            d.cache_create += n("cache_create");
                        }
                    }
                    "event_msg" if p["event_kind"] == "token_count" => {
                        let u = &p["payload"]["info"]["total_token_usage"];
                        if u.is_object() {
                            self.codex.insert(
                                sid.to_string(),
                                (
                                    (
                                        u["input_tokens"].as_u64().unwrap_or(0),
                                        u["cached_input_tokens"].as_u64().unwrap_or(0),
                                        u["output_tokens"].as_u64().unwrap_or(0),
                                    ),
                                    day.to_string(),
                                ),
                            );
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn finish(mut self) -> HashMap<String, DayStat> {
        for ((input, cached, output), day) in self.codex.into_values() {
            let d = self.days.entry(day).or_default();
            d.tok_in += input.saturating_sub(cached);
            d.cache_read += cached;
            d.tok_out += output;
        }
        for (day, sids) in self.active {
            self.days.entry(day).or_default().sessions = sids.len() as u64;
        }
        self.days
    }
}

const ANALYSIS_FORMAT: u32 = 2;

/// Compact raw-derived facts published with the immutable search build. MCP
/// readers use this instead of reparsing every event chunk for each request;
/// raw envelopes remain the source of truth and builders regenerate it.
#[derive(Clone, Serialize, Deserialize)]
struct AnalysisSnapshot {
    format: u32,
    sessions: Vec<Session>,
    days: BTreeMap<String, DayStat>,
    tools: BTreeMap<String, ToolProfile>,
    roster: crate::fleet::Roster,
}

type AnalysisCacheKey = (PathBuf, u64, u128);
type AnalysisCache = Option<(AnalysisCacheKey, std::sync::Arc<AnalysisSnapshot>)>;
static ANALYSIS_CACHE: std::sync::OnceLock<std::sync::Mutex<AnalysisCache>> =
    std::sync::OnceLock::new();

/// Cache identity changes whenever an atomically replaced projection changes.
fn analysis_key(path: &Path) -> Option<AnalysisCacheKey> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    Some((path.to_path_buf(), meta.len(), modified))
}

/// Load and share the current immutable analysis projection across requests.
fn load_analysis_snapshot() -> Option<std::sync::Arc<AnalysisSnapshot>> {
    let path = crate::readmodel::analysis_path();
    let key = analysis_key(&path)?;
    let cache = ANALYSIS_CACHE.get_or_init(|| std::sync::Mutex::new(None));
    let mut cache = cache.lock().ok()?;
    if let Some((cached_key, snapshot)) = cache.as_ref() {
        if cached_key == &key {
            return Some(std::sync::Arc::clone(snapshot));
        }
    }
    let file = std::fs::File::open(&path).ok()?;
    let snapshot: AnalysisSnapshot =
        serde_json::from_reader(std::io::BufReader::new(file)).ok()?;
    if snapshot.format != ANALYSIS_FORMAT {
        return None;
    }
    let snapshot = std::sync::Arc::new(snapshot);
    *cache = Some((key, std::sync::Arc::clone(&snapshot)));
    Some(snapshot)
}

/// One build-side pass shared by session and daily analytics. Keeping this on
/// the writer makes every mediated read proportional to the compact snapshot,
/// not the fleet's retained raw bytes.
pub(crate) struct AnalysisBuilder {
    known: std::collections::HashSet<String>,
    capture_since_ms: Option<i64>,
    seen: std::collections::HashSet<u64>,
    aggs: HashMap<String, Agg>,
    days: DayFold,
    tools: ToolProfilesFold,
}

impl AnalysisBuilder {
    /// Start one deterministic analytics pass for the configured boundary.
    pub(crate) fn new(
        known: std::collections::HashSet<String>,
        capture_since_ms: Option<i64>,
    ) -> Self {
        Self {
            known,
            capture_since_ms,
            seen: std::collections::HashSet::new(),
            aggs: HashMap::new(),
            days: DayFold::default(),
            tools: ToolProfilesFold::default(),
        }
    }

    /// Fold one raw event chunk into session, daily, and tool aggregates.
    pub(crate) fn fold_text(&mut self, text: &str) {
        for line in text.lines() {
            fold_line_since(
                line,
                &self.known,
                &mut self.seen,
                &mut self.aggs,
                self.capture_since_ms,
                false,
            );
            let visible = serde_json::from_str::<Value>(line)
                .ok()
                .is_none_or(|event| {
                    crate::config::captured_at(
                        event["ts"].as_str().unwrap_or(""),
                        self.capture_since_ms,
                    )
                });
            if visible {
                self.days.fold(line);
                self.tools.fold(line);
            }
        }
    }

    /// Atomically publish compact facts and emit their size/coverage metrics.
    pub(crate) fn write(mut self, path: &Path, roster: crate::fleet::Roster) -> Result<()> {
        self.aggs.retain(|_, agg| !agg.first.is_empty());
        let snapshot = AnalysisSnapshot {
            format: ANALYSIS_FORMAT,
            sessions: session_rows(self.aggs),
            days: self.days.finish().into_iter().collect(),
            tools: self.tools.finish().into_iter().collect(),
            roster,
        };
        let bytes = serde_json::to_vec(&snapshot)?;
        crate::write_atomic(&path.to_string_lossy(), &bytes)?;
        crate::metrics::Run::new("analysis")
            .set("sessions", snapshot.sessions.len())
            .set("days", snapshot.days.len())
            .set("tools", snapshot.tools.len())
            .set("bytes", bytes.len())
            .emit();
        Ok(())
    }
}

/// Read the fleet roster attached to the same immutable build as its documents.
pub(crate) fn analysis_roster() -> Option<crate::fleet::Roster> {
    load_analysis_snapshot().map(|snapshot| snapshot.roster.clone())
}

/// Per-day usage/tool tallies from the raw envelopes, for the time-series view.
pub fn day_stats() -> HashMap<String, DayStat> {
    if let Some(snapshot) = load_analysis_snapshot() {
        return snapshot.days.iter().map(|(day, stat)| (day.clone(), *stat)).collect();
    }
    use std::io::BufRead;

    let mut f = DayFold::default();
    for path in jsonl_files(Path::new(LOCAL_DIR)) {
        let Ok(file) = std::fs::File::open(&path) else { continue };
        for line in std::io::BufReader::new(file).lines().map_while(|line| line.ok()) {
            f.fold(&line);
        }
    }
    f.finish()
}

/// One day's full activity: what the agents consumed (tokens, cache, tools,
/// sessions) and what the work produced (merged LOC, PRs, issues). The one
/// substrate behind the TUI charts and `synty stats`.
#[derive(Default, Clone, Copy)]
pub struct DayRow {
    pub tok_in: u64,
    pub tok_out: u64,
    pub cache_read: u64,
    pub cache_create: u64,
    pub tools: u64,
    pub sessions: u64,
    pub loc_add: u64,
    pub loc_del: u64,
    pub prs: u64,
    pub issues: u64,
}

impl DayRow {
    /// Accumulate another day into this row (week buckets, window totals).
    pub fn add(&mut self, o: &DayRow) {
        self.tok_in += o.tok_in;
        self.tok_out += o.tok_out;
        self.cache_read += o.cache_read;
        self.cache_create += o.cache_create;
        self.tools += o.tools;
        self.sessions += o.sessions;
        self.loc_add += o.loc_add;
        self.loc_del += o.loc_del;
        self.prs += o.prs;
        self.issues += o.issues;
    }
}

/// day (YYYY-MM-DD) → combined activity: usage from the raw envelopes, LOC±
/// from merged PRs, PR/issue counts from the given unit list.
pub fn activity_by_day(units: &[Unit]) -> HashMap<String, DayRow> {
    combine_activity(day_stats(), gh_loc_by_day(), units)
}

fn combine_activity(
    day: HashMap<String, DayStat>,
    loc: HashMap<String, (u64, u64)>,
    units: &[Unit],
) -> HashMap<String, DayRow> {
    let mut out: HashMap<String, DayRow> = HashMap::new();
    for (d, s) in day {
        let r = out.entry(d).or_default();
        r.tok_in = s.tok_in;
        r.tok_out = s.tok_out;
        r.cache_read = s.cache_read;
        r.cache_create = s.cache_create;
        r.tools = s.tools;
        r.sessions = s.sessions;
    }
    for (d, (add, del)) in loc {
        let r = out.entry(d).or_default();
        r.loc_add = add;
        r.loc_del = del;
    }
    for u in units {
        if day_num(&u.when).is_none() {
            continue;
        }
        // Every dated unit claims its day (the charts anchor to the newest
        // one), but only GitHub kinds add to the output counters.
        let r = out.entry(u.when.clone()).or_default();
        match u.kind {
            Kind::Pr => r.prs += 1,
            Kind::Issue => r.issues += 1,
            Kind::Session => {}
        }
    }
    out
}

/// Every distinct tool name seen in the envelopes — the suggestion pool when
/// `synty tool <name>` misses. One light scan, called only on that miss.
pub fn tool_names() -> Vec<String> {
    use std::io::BufRead;

    if let Some(snapshot) = load_analysis_snapshot() {
        let mut names: Vec<String> = snapshot.tools.keys().cloned().collect();
        names.sort();
        return names;
    }
    let mut names = std::collections::BTreeSet::new();
    for path in jsonl_files(Path::new(LOCAL_DIR)) {
        let Ok(file) = std::fs::File::open(&path) else { continue };
        for line in std::io::BufReader::new(file).lines().map_while(|line| line.ok()) {
            let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
            if v["kind"].as_str() == Some("tool_call") {
                if let Some(n) = v["payload"]["name"].as_str() {
                    if !n.is_empty() {
                        names.insert(n.to_string());
                    }
                }
            }
        }
    }
    names.into_iter().collect()
}

/// Days from the CE epoch for a YYYY-MM-DD day (the time-series x scale).
pub fn day_num(day: &str) -> Option<i32> {
    use chrono::Datelike;
    chrono::NaiveDate::parse_from_str(day, "%Y-%m-%d").ok().map(|d| d.num_days_from_ce())
}

/// The oldest visible Monday (num_days_from_ce) for a window of `weeks`
/// Mon-Sun weeks ending with the week of `gmax`.
pub fn week_start_for(gmax: i32, weeks: usize) -> i32 {
    use chrono::Datelike;
    let dow = chrono::NaiveDate::from_num_days_from_ce_opt(gmax)
        .map(|d| d.weekday().num_days_from_monday() as i32)
        .unwrap_or(0);
    gmax - dow - 7 * (weeks as i32 - 1)
}

/// Everything gleanable about one tool across the corpus: call/error volume,
/// per-day usage, which argument keys appear — and the common values of the
/// low-cardinality ones — input sizes, and call→result latency where both
/// sides carry timestamps. Computed on demand for the Status drill-down.
#[derive(Clone, Serialize, Deserialize)]
pub struct ToolProfile {
    pub name: String,
    pub agent: String, // "claude" / "codex" / … / "a+b" when mixed
    pub calls: u64,
    pub errs: u64,
    pub chars: u64, // args + result payload volume — context, as a ~chars/4 estimate
    pub days: BTreeMap<String, u64>,
    pub arg_keys: Vec<(String, u64)>, // key → calls carrying it, desc
    pub arg_tops: Vec<(String, Vec<(String, u64)>)>, // low-cardinality keys → top values
    pub p50_ms: u64,
    pub p95_ms: u64,
    pub timed: usize, // calls with a paired, timestamped result
    pub input_p50: usize, // serialized input size, chars
    pub input_p95: usize,
    pub samples: Vec<String>, // the most recent invocations, excerpted
}

/// Beyond this many distinct values a key is free-form (paths, commands), not
/// an enum — its value breakdown would be noise.
const ARG_TOP_MAX: usize = 8;

#[derive(Default)]
struct ToolFold {
    target: String,
    seen: std::collections::HashSet<u64>,
    agents: std::collections::BTreeSet<String>,
    calls: u64,
    errs: u64,
    chars: u64,
    days: BTreeMap<String, u64>,
    key_counts: HashMap<String, u64>,
    val_counts: HashMap<String, HashMap<String, u64>>,
    sizes: Vec<usize>,
    pending_ts: HashMap<String, i64>, // call id → call ts (ms)
    durs: Vec<u64>,
    samples: std::collections::VecDeque<String>,
}

/// Builds every tool profile in one event pass. Calls are routed to their
/// named fold; results are offered to every known fold so only the owner of
/// the pending call id consumes it. Tool cardinality is small and bounded by
/// the actual event vocabulary, while the published profiles stay compact.
#[derive(Default)]
struct ToolProfilesFold {
    folds: HashMap<String, ToolFold>,
}

impl ToolProfilesFold {
    fn fold(&mut self, line: &str) {
        let Ok(ev) = serde_json::from_str::<Value>(line) else { return };
        match ev["kind"].as_str().unwrap_or("") {
            "tool_call" => {
                let Some(name) = ev["payload"]["name"].as_str().filter(|name| !name.is_empty())
                else {
                    return;
                };
                self.folds
                    .entry(name.to_string())
                    .or_insert_with(|| ToolFold {
                        target: name.to_string(),
                        ..Default::default()
                    })
                    .fold(line);
            }
            "tool_result" => {
                for fold in self.folds.values_mut() {
                    fold.fold(line);
                }
            }
            _ => {}
        }
    }

    fn finish(self) -> HashMap<String, ToolProfile> {
        self.folds
            .into_iter()
            .map(|(name, fold)| (name, fold.finish()))
            .collect()
    }
}

fn ts_ms(ts: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(ts).ok().map(|d| d.timestamp_millis())
}

impl ToolFold {
    fn fold(&mut self, line: &str) {
        if line.trim().is_empty() {
            return;
        }
        let Ok(ev) = serde_json::from_str::<Value>(line) else { return };
        if !crate::event::first_sighting(&mut self.seen, ev["event_id"].as_str().unwrap_or("")) {
            return;
        }
        let p = &ev["payload"];
        let id = p["tool_use_id"].as_str().or_else(|| p["call_id"].as_str()).unwrap_or("");
        match ev["kind"].as_str().unwrap_or("") {
            "tool_call" if p["name"].as_str() == Some(self.target.as_str()) => {
                self.calls += 1;
                self.chars += payload_chars(p);
                if let Some(src) = ev["source"].as_str().filter(|s| !s.is_empty()) {
                    self.agents.insert(agent_label(src).to_string());
                }
                if let Some(day) = ev["ts"].as_str().unwrap_or("").split('T').next().filter(|d| d.len() == 10) {
                    *self.days.entry(day.to_string()).or_default() += 1;
                }
                // The arguments, whichever shape the agent logs: Claude's
                // input object, codex's JSON-encoded arguments string, or a
                // web_search action object.
                let parsed; // keeps the codex parse alive past the borrow
                let input = if p["input"].is_object() {
                    &p["input"]
                } else if let Some(s) = p["arguments"].as_str() {
                    parsed = serde_json::from_str::<Value>(s).unwrap_or(Value::Null);
                    &parsed
                } else {
                    &p["action"]
                };
                if let Some(obj) = input.as_object() {
                    for (k, v) in obj {
                        *self.key_counts.entry(k.clone()).or_default() += 1;
                        // Scalars only, and only short ones — long strings are
                        // free-form content, not categorizable values.
                        let val = match v {
                            Value::Bool(b) => Some(b.to_string()),
                            Value::Number(n) => Some(n.to_string()),
                            Value::String(s) if s.chars().count() <= 32 => Some(s.clone()),
                            _ => None,
                        };
                        if let Some(val) = val {
                            let vals = self.val_counts.entry(k.clone()).or_default();
                            if vals.len() < 64 || vals.contains_key(&val) {
                                *vals.entry(val).or_default() += 1;
                            }
                        }
                    }
                    let ser = serde_json::to_string(obj).unwrap_or_default();
                    self.sizes.push(ser.chars().count());
                    self.samples.push_back(crate::excerpt(&ser, 110));
                    if self.samples.len() > 3 {
                        self.samples.pop_front();
                    }
                }
                if !id.is_empty() {
                    if let Some(ms) = ts_ms(ev["ts"].as_str().unwrap_or("")) {
                        self.pending_ts.insert(id.to_string(), ms);
                    }
                }
            }
            "tool_result" if !id.is_empty() => {
                if self.pending_ts.contains_key(id) {
                    self.chars += payload_chars(p);
                }
                if let Some(call_ms) = self.pending_ts.remove(id) {
                    if let Some(ms) = ts_ms(ev["ts"].as_str().unwrap_or("")) {
                        self.durs.push(ms.saturating_sub(call_ms).max(0) as u64);
                    }
                    if p["is_error"].as_bool().unwrap_or(false) {
                        self.errs += 1;
                    }
                }
            }
            _ => {}
        }
    }

    fn finish(mut self) -> ToolProfile {
        // Nearest-rank percentile: ceil(p·n)-1, clamped.
        fn pctl(sorted: &[u64], p: f64) -> u64 {
            if sorted.is_empty() {
                return 0;
            }
            let rank = (sorted.len() as f64 * p).ceil() as usize;
            sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
        }
        let mut arg_keys: Vec<(String, u64)> = self.key_counts.into_iter().collect();
        arg_keys.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut arg_tops: Vec<(String, Vec<(String, u64)>)> = Vec::new();
        for (key, _) in &arg_keys {
            if let Some(vals) = self.val_counts.get(key) {
                // Enum-ish = few distinct values AND values that repeat; when
                // every value is unique the key is free-form however few
                // there are.
                let repeats = vals.values().sum::<u64>() > vals.len() as u64;
                if !vals.is_empty() && vals.len() <= ARG_TOP_MAX && repeats {
                    let mut top: Vec<(String, u64)> = vals.clone().into_iter().collect();
                    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                    top.truncate(3);
                    arg_tops.push((key.clone(), top));
                }
            }
        }
        self.durs.sort_unstable();
        let mut sizes: Vec<u64> = self.sizes.iter().map(|&s| s as u64).collect();
        sizes.sort_unstable();
        ToolProfile {
            name: self.target,
            agent: self.agents.into_iter().collect::<Vec<_>>().join("+"),
            calls: self.calls,
            errs: self.errs,
            chars: self.chars,
            days: self.days,
            arg_keys,
            arg_tops,
            p50_ms: pctl(&self.durs, 0.5),
            p95_ms: pctl(&self.durs, 0.95),
            timed: self.durs.len(),
            input_p50: pctl(&sizes, 0.5) as usize,
            input_p95: pctl(&sizes, 0.95) as usize,
            samples: self.samples.into_iter().collect(),
        }
    }
}

/// Merged lines of code per day, straight from the scraped PR corpus:
/// day(mergedAt) → (additions, deletions). Only MERGED PRs count — LOC that
/// actually landed.
pub fn gh_loc_by_day() -> HashMap<String, (u64, u64)> {
    let mut out: HashMap<String, (u64, u64)> = HashMap::new();
    let Ok(entries) = std::fs::read_dir("corpus/github") else { return out };
    for e in entries.filter_map(|e| e.ok()) {
        let name = e.file_name().to_string_lossy().into_owned();
        if !name.starts_with("prs-") || !name.ends_with(".json") {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(e.path()) else { continue };
        let Ok(prs) = serde_json::from_str::<Vec<Value>>(&raw) else { continue };
        for p in &prs {
            if p["state"].as_str() != Some("MERGED") {
                continue;
            }
            let Some(day) = p["mergedAt"].as_str().and_then(|t| t.split('T').next()).filter(|d| d.len() == 10) else { continue };
            let d = out.entry(day.to_string()).or_default();
            d.0 += p["additions"].as_u64().unwrap_or(0);
            d.1 += p["deletions"].as_u64().unwrap_or(0);
        }
    }
    out
}

/// Profile one tool's use across every tracked session, on demand.
pub fn tool_profile(name: &str) -> ToolProfile {
    use std::io::BufRead;

    if let Some(snapshot) = load_analysis_snapshot() {
        return snapshot.tools.get(name).cloned().unwrap_or_else(|| ToolProfile {
            name: name.to_string(),
            agent: String::new(),
            calls: 0,
            errs: 0,
            chars: 0,
            days: BTreeMap::new(),
            arg_keys: Vec::new(),
            arg_tops: Vec::new(),
            p50_ms: 0,
            p95_ms: 0,
            timed: 0,
            input_p50: 0,
            input_p95: 0,
            samples: Vec::new(),
        });
    }
    let mut f = ToolFold { target: name.to_string(), ..Default::default() };
    for path in jsonl_files(Path::new(LOCAL_DIR)) {
        let Ok(file) = std::fs::File::open(&path) else { continue };
        for line in std::io::BufReader::new(file).lines().map_while(|line| line.ok()) {
            f.fold(&line);
        }
    }
    f.finish()
}

/// Aggregate the raw envelopes under corpus/local into per-session tallies.
fn aggregate(collect_texts: bool) -> HashMap<String, Agg> {
    use std::io::BufRead;

    // The org's repos (from the last back-fill) — fold worktree dirs onto them.
    let known: std::collections::HashSet<String> =
        crate::config::load().repos.into_iter().collect();
    let mut aggs: HashMap<String, Agg> = HashMap::new();
    // Overlapping trackers can append the same envelope twice; ids are
    // deterministic, so count each event once.
    let mut seen = std::collections::HashSet::new();
    let capture_since_ms = crate::config::capture_since_ms();
    for f in jsonl_files(Path::new(LOCAL_DIR)) {
        let Ok(file) = std::fs::File::open(&f) else {
            continue;
        };
        for line in std::io::BufReader::new(file).lines().map_while(|line| line.ok()) {
            fold_line_since(
                &line,
                &known,
                &mut seen,
                &mut aggs,
                capture_since_ms,
                collect_texts,
            );
        }
    }
    aggs.retain(|_, a| !a.first.is_empty());
    aggs
}

/// Build session units from the raw envelopes, with topic membership, a
/// normalized struggle score, and any cached LLM summary.
pub fn sessions() -> Result<Vec<Session>> {
    let mut out = if let Some(snapshot) = load_analysis_snapshot() {
        snapshot.sessions.clone()
    } else {
        session_rows(aggregate(false))
    };
    let cache = load_summary_cache();
    let topic_of = unit_topics();
    for session in &mut out {
        session.topic = topic_of.get(&session.id).map(|topic| topic.cluster);
        session.summary = cache
            .get(&session.id)
            .map(|cached| cached.summary.clone())
            .filter(|summary| !summary.is_empty());
    }
    out.sort_by(|x, y| y.ended.cmp(&x.ended).then(x.id.cmp(&y.id)));
    Ok(out)
}

fn session_rows(aggs: HashMap<String, Agg>) -> Vec<Session> {
    // Raw struggle signals → z-scores → summed → percentile-ranked to 0..1.
    let ids: Vec<String> = aggs.keys().cloned().collect();
    let sig = |a: &Agg| [a.thinking as f64, a.tools as f64, a.prompts as f64, duration_secs(a) as f64];
    let raw: Vec<[f64; 4]> = ids.iter().map(|id| sig(&aggs[id])).collect();
    let scores = struggle_scores(&raw);

    let mut out: Vec<Session> = ids
        .iter()
        .enumerate()
        .filter(|(_, id)| aggs[*id].prompts > 0) // real sessions only
        .map(|(i, id)| {
            let a = &aggs[id];
            // Claude path: the summed, msg_id-deduped per-turn usage. Codex
            // path: its last cumulative snapshot wins, normalized to the
            // shared classes — fresh in = input − cached, cache_read = cached
            // (codex exposes no cache creation), out = output (incl.
            // reasoning, comparable to Claude's output incl. thinking).
            let (tok_in, tok_out, cache_read, cache_create) = match a.codex_usage {
                Some((input, cached, output)) => (input.saturating_sub(cached), output, cached, 0),
                None => (a.tok_in, a.tok_out, a.cache_read, a.cache_create),
            };
            // Per-model split: codex snapshots name no model, so its share is
            // labeled by the agent; claude splits by the per-turn model field.
            let mut by_model: Vec<ModelUsage> = match a.codex_usage {
                Some(_) => vec![ModelUsage { model: "codex".into(), tok_in, tok_out, cache_read, cache_create, turns: 0 }],
                None => a.model_usage.values().cloned().collect(),
            };
            by_model.sort_by(|x, y| y.tok_out.cmp(&x.tok_out).then(x.model.cmp(&y.model)));
            let mut tools_by_name: Vec<(String, usize, usize, u64)> = a
                .tool_counts
                .iter()
                .map(|(n, c)| (n.clone(), *c, a.tool_errs.get(n).copied().unwrap_or(0), a.tool_chars.get(n).copied().unwrap_or(0)))
                .collect();
            tools_by_name.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
            Session {
                id: id.clone(),
                repo: a.repo.clone(),
                started: a.first.clone(),
                ended: a.last.clone(),
                ask: a.ask.clone(),
                prompts: a.prompts,
                assistant: a.assistant,
                thinking: a.thinking,
                tools: a.tools,
                files: a.files.clone(),
                linked_pr: a.linked_pr.clone(),
                tok_in,
                tok_out,
                cache_read,
                cache_create,
                usage_turns: a.usage_turns,
                tools_by_name,
                tool_err: a.tool_err,
                by_model,
                source: a.source.clone(),
                campaign_id: a.campaign_id.clone(),
                campaign_role: a.campaign_role.clone(),
                topic: None,
                struggle: scores[i],
                summary: None,
                author: a.actor.clone(),
            }
        })
        .collect();
    out.sort_by(|x, y| y.ended.cmp(&x.ended).then(x.id.cmp(&y.id)));
    out
}

/// All work units (sessions + PRs + issues), newest first.
pub fn units() -> Result<Vec<Unit>> {
    // Sessions carry the actor their tracker stamped on session_start (so a
    // fleet build credits each author); pre-stamp sessions fall back to this
    // machine's actor, matching what `ingest` stamps on session docs.
    let local_actor = crate::identity::actor();
    let topic_of = unit_topics();
    let mut out: Vec<Unit> = sessions()?
        .into_iter()
        .map(|s| Unit {
            kind: Kind::Session,
            when: day(&s.ended),
            repo: s.repo.clone(),
            title: if s.ask.is_empty() { format!("session {}", crate::short(&s.id)) } else { s.ask.clone() },
            outcome: session_outcome(&s),
            summary: s.summary.clone(),
            topic: s.topic,
            rank: topic_of.get(&s.id).map(|t| t.rank).unwrap_or(i64::MAX),
            dup: topic_of.get(&s.id).is_some_and(|t| t.dup.is_some()),
            struggle: s.struggle,
            author: if s.author.is_empty() { local_actor.clone() } else { s.author.clone() },
            doc_id: None,
            session_id: Some(s.id),
            campaign_id: s.campaign_id,
            campaign_role: s.campaign_role,
            source: s.source,
        })
        .collect();

    if let Ok(docs) = load_docs(readmodel::docs_path()) {
        let cache = load_summary_cache();
        for d in &docs {
            let kind = match d.meta.kind.as_str() {
                "pull_request" => Kind::Pr,
                "issue" => Kind::Issue,
                _ => continue,
            };
            let num = d.meta.number.unwrap_or(0);
            let key = gh_key(&d.meta.repo, num);
            out.push(Unit {
                kind,
                when: day(&d.meta.ts),
                repo: d.meta.repo.clone(),
                title: format!("{}#{} {}", d.meta.repo, num, first_line(&d.text)),
                outcome: d.meta.state.clone().unwrap_or_default(),
                summary: cache.get(&key).map(|c| c.summary.clone()).filter(|s| !s.is_empty()),
                topic: topic_of.get(&key).map(|t| t.cluster),
                rank: topic_of.get(&key).map(|t| t.rank).unwrap_or(i64::MAX),
                dup: topic_of.get(&key).is_some_and(|t| t.dup.is_some()),
                struggle: 0.0,
                author: d.meta.author.clone(),
                doc_id: Some(d.id),
                session_id: None,
                campaign_id: d.meta.campaign_id.clone(),
                campaign_role: d.meta.campaign_role.clone(),
                source: if d.meta.capture_source.is_empty() {
                    d.meta.source.clone()
                } else {
                    d.meta.capture_source.clone()
                },
            });
        }
    }
    out.sort_by(|a, b| b.when.cmp(&a.when));
    Ok(out)
}

/// Topics with their units, last-active, weekly activity, facets, and type mix.
/// Units are grouped by their single cluster (unit_clusters.json), so a topic is
/// exactly a set of units and every facet derives from those units.
pub fn topic_units(weeks: usize) -> Result<Vec<TopicUnits>> {
    let labels = cluster_labels();
    let cache_keys = cluster_cache_keys();
    let cache = load_summary_cache();
    let mut by_topic: HashMap<i64, Vec<Unit>> = HashMap::new();
    for u in units()? {
        if let Some(t) = u.topic {
            by_topic.entry(t).or_default().push(u);
        }
    }
    let mut out: Vec<TopicUnits> = by_topic
        .into_iter()
        .map(|(id, units)| {
            let last_active = units.iter().map(|u| u.when.clone()).max().unwrap_or_default();
            let days: Vec<String> = units.iter().map(|u| u.when.clone()).collect();
            let span = day_span(&days);
            let activity = weekly_buckets(&days, weeks);
            let mix = unit_mix(&units);
            let repos = by_count(units.iter().filter(|u| !u.repo.is_empty()).map(|u| u.repo.clone()));
            let authors = by_count(units.iter().filter(|u| !u.author.is_empty()).map(|u| u.author.clone()));
            let cache_key = cache_keys.get(&id).cloned().unwrap_or_else(|| id.to_string());
            let summary = cache.get(&topic_key(&cache_key)).map(|c| c.summary.clone()).filter(|s| !s.is_empty());
            let name = cache.get(&topic_name_key(&cache_key)).map(|c| c.summary.clone()).filter(|s| !s.is_empty());
            TopicUnits { id, cache_key, label: labels.get(&id).cloned().unwrap_or_default(), units, last_active, activity, mix, repos, authors, summary, name, span }
        })
        .collect();
    out.sort_by(|a, b| b.last_active.cmp(&a.last_active).then(b.units.len().cmp(&a.units.len())));
    Ok(out)
}

/// Apply every read-scope dimension before rebuilding topic facets. Cached
/// names and summaries describe the full cluster, so restricted views replace
/// them with an extractive label from an allowed member.
pub fn topic_units_scoped(weeks: usize, scope: &crate::policy::ReadScope) -> Result<Vec<TopicUnits>> {
    Ok(scope_topic_units(topic_units(weeks)?, weeks, scope))
}

fn scope_topic_units(
    mut topics: Vec<TopicUnits>,
    weeks: usize,
    scope: &crate::policy::ReadScope,
) -> Vec<TopicUnits> {
    if !scope.restricted() {
        return topics;
    }
    topics = topics
        .into_iter()
        .filter_map(|mut topic| {
            topic.units.retain(|unit| scope.allows_unit(unit));
            if topic.units.is_empty() {
                return None;
            }
            let days: Vec<String> = topic.units.iter().map(|unit| unit.when.clone()).collect();
            topic.last_active = days.iter().max().cloned().unwrap_or_default();
            topic.activity = weekly_buckets(&days, weeks);
            topic.mix = unit_mix(&topic.units);
            topic.repos = by_count(
                topic.units.iter().filter(|unit| !unit.repo.is_empty()).map(|unit| unit.repo.clone()),
            );
            topic.authors = by_count(
                topic.units.iter().filter(|unit| !unit.author.is_empty()).map(|unit| unit.author.clone()),
            );
            topic.span = day_span(&days);
            topic.label = topic.units.iter().min_by_key(|unit| unit.rank).map(|unit| unit.title.clone()).unwrap_or_default();
            topic.summary = None;
            topic.name = None;
            Some(topic)
        })
        .collect();
    topics.sort_by(|a, b| b.last_active.cmp(&a.last_active).then(b.units.len().cmp(&a.units.len())));
    topics
}

// ── struggle ────────────────────────────────────────────────────────────────

fn struggle_scores(raw: &[[f64; 4]]) -> Vec<f32> {
    if raw.is_empty() {
        return vec![];
    }
    let n = raw.len() as f64;
    let mut sums = [0.0; 4];
    for r in raw {
        for k in 0..4 {
            sums[k] += r[k];
        }
    }
    let mean: [f64; 4] = std::array::from_fn(|k| sums[k] / n);
    let mut var = [0.0; 4];
    for r in raw {
        for k in 0..4 {
            var[k] += (r[k] - mean[k]).powi(2);
        }
    }
    let sd: [f64; 4] = std::array::from_fn(|k| (var[k] / n).sqrt().max(1e-9));
    let z: Vec<f64> = raw.iter().map(|r| (0..4).map(|k| (r[k] - mean[k]) / sd[k]).sum()).collect();
    // Percentile-rank the summed z-scores rather than min-max: one outlier
    // session (e.g. left open for days) won't compress everyone else toward
    // zero, and the result reads directly as "harder than X% of sessions".
    let mut order: Vec<usize> = (0..z.len()).collect();
    order.sort_by(|&a, &b| z[a].partial_cmp(&z[b]).unwrap_or(std::cmp::Ordering::Equal));
    let denom = z.len().saturating_sub(1).max(1) as f32;
    let mut out = vec![0f32; z.len()];
    for (rank, &i) in order.iter().enumerate() {
        out[i] = rank as f32 / denom;
    }
    out
}

// ── topic membership (unit-level) ─────────────────────────────────────────

/// One unit's clustering row, parsed from unit_clusters.json (written by
/// `cluster`). Rank 0 = the medoid; `dup` names the representative when this
/// unit is a collapsed near-duplicate.
struct UnitTopic {
    cluster: i64,
    stable: String,
    label: String,
    rank: i64,
    dup: Option<String>,
}

/// unit key → its clustering row. Empty until clustering has run.
fn unit_topics() -> HashMap<String, UnitTopic> {
    let Ok(raw) = std::fs::read_to_string(readmodel::clusters_path()) else { return HashMap::new() };
    let arr: Vec<Value> = serde_json::from_str(&raw).unwrap_or_default();
    arr.iter()
        .filter_map(|it| {
            Some((it["key"].as_str()?.to_string(), UnitTopic {
                cluster: it["cluster"].as_i64()?,
                stable: it["topic"].as_str()?.to_string(),
                label: it["label"].as_str().unwrap_or("").to_string(),
                rank: it["rank"].as_i64()?,
                dup: it["dup"].as_str().map(String::from),
            }))
        })
        .collect()
}

/// cluster index → label.
fn cluster_labels() -> HashMap<i64, String> {
    let mut m = HashMap::new();
    for (_, t) in unit_topics() {
        m.entry(t.cluster).or_insert(t.label);
    }
    m
}

/// cluster index → stable cache key (medoid hash).
fn cluster_cache_keys() -> HashMap<i64, String> {
    let mut m = HashMap::new();
    for (_, t) in unit_topics() {
        m.entry(t.cluster).or_insert(t.stable);
    }
    m
}

/// Stable topic key → its members' embed-text hashes, centrality-ordered
/// (medoid first), collapsed duplicates excluded — one rerun's vectors are
/// enough. The embed texts are what `cluster` encoded into the shared store,
/// so the name-faithfulness gate can score names against members without
/// re-encoding anything but the names themselves.
pub fn topic_member_embed_hashes() -> Result<HashMap<String, Vec<u64>>> {
    let topic_of = unit_topics();
    let mut grouped: HashMap<String, Vec<(i64, u64)>> = HashMap::new();
    for u in cluster_units()? {
        if let Some(t) = topic_of.get(&u.key) {
            if t.dup.is_some() {
                continue;
            }
            grouped.entry(t.stable.clone()).or_default().push((t.rank, crate::index::fnv1a(u.embed.as_bytes())));
        }
    }
    Ok(grouped
        .into_iter()
        .map(|(k, mut v)| {
            v.sort_by_key(|(r, _)| *r);
            (k, v.into_iter().map(|(_, h)| h).collect())
        })
        .collect())
}

/// First and last day across day-or-timestamp strings (lexicographic works
/// because the zero-padded day prefix sorts chronologically).
fn day_span(ts: &[String]) -> Option<(String, String)> {
    let days: Vec<&str> = ts.iter().filter_map(|t| t.split('T').next()).filter(|d| d.len() == 10).collect();
    Some((days.iter().min()?.to_string(), days.iter().max()?.to_string()))
}

/// Distinct values, most frequent first.
fn by_count(it: impl Iterator<Item = String>) -> Vec<String> {
    let mut m: HashMap<String, usize> = HashMap::new();
    for x in it {
        *m.entry(x).or_default() += 1;
    }
    let mut v: Vec<(String, usize)> = m.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.into_iter().map(|(k, _)| k).collect()
}

/// (sessions, PRs, issues) counts among a topic's units.
fn unit_mix(units: &[Unit]) -> (usize, usize, usize) {
    let mut m = (0usize, 0usize, 0usize);
    for u in units {
        match u.kind {
            Kind::Session => m.0 += 1,
            Kind::Pr => m.1 += 1,
            Kind::Issue => m.2 += 1,
        }
    }
    m
}

// ── helpers ────────────────────────────────────────────────────────────────

/// Per-session inputs for the LLM summarizer (ask + the longest turns). Shares
/// aggregation with `sessions`.
pub fn session_inputs() -> Result<Vec<SessionInput>> {
    let aggs = aggregate(true);
    let ids: Vec<String> = aggs.keys().cloned().collect();
    Ok(ids
        .iter()
        .filter(|id| aggs[*id].prompts > 0)
        .map(|id| {
            let a = &aggs[id];
            SessionInput {
                id: id.clone(),
                repo: a.repo.clone(),
                ask: a.ask.clone(),
                files: a.files.iter().take(12).cloned().collect(),
                turns: top_turns(&a.texts, 8),
            }
        })
        .collect())
}

/// The repo a session belongs to, from its working directory. No path-shape
/// guessing: a cwd segment that is (or folds to) a known org repo wins — so org
/// repos resolve from the path string alone, machine-independently — and
/// anything else falls back to the cwd's own git remote (a local repo). Empty
/// only when neither applies (blank cwd, agent scratch dir, missing checkout).
pub fn resolve_repo(cwd: &str, known: &std::collections::HashSet<String>) -> String {
    let matched = repo_from_cwd(cwd, known);
    if matched.is_empty() { git_repo(cwd).unwrap_or_default() } else { matched }
}

/// The first cwd segment that exact-matches or folds to a known repo (empty if
/// none) — the pure, machine-independent half of `resolve_repo`.
pub fn repo_from_cwd(cwd: &str, known: &std::collections::HashSet<String>) -> String {
    cwd.split('/').filter(|s| !s.is_empty()).map(|seg| fold_repo(seg, known)).find(|r| known.contains(r)).unwrap_or_default()
}

/// Repo name from a cwd's `git remote origin` (basename, no `.git`), cached per
/// cwd; `None` if the checkout is gone or has no remote. Resolves local repos
/// that aren't in the tracked org, without a network lookup.
fn git_repo(cwd: &str) -> Option<String> {
    use std::sync::{LazyLock, Mutex};
    static CACHE: LazyLock<Mutex<HashMap<String, Option<String>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    if cwd.is_empty() {
        return None;
    }
    if let Some(hit) = CACHE.lock().ok()?.get(cwd) {
        return hit.clone();
    }
    let resolved = (|| {
        let out = std::process::Command::new("git").args(["-C", cwd, "config", "--get", "remote.origin.url"]).output().ok()?;
        out.status.success().then_some(())?;
        let url = String::from_utf8_lossy(&out.stdout);
        let name = url.trim().trim_end_matches(".git").rsplit(['/', ':']).next()?;
        (!name.is_empty()).then(|| name.to_string())
    })();
    if let Ok(mut c) = CACHE.lock() {
        c.insert(cwd.to_string(), resolved.clone());
    }
    resolved
}

/// Fold a raw repo-dir name to a known repo: an exact match wins, else the
/// longest known repo it extends with a `-` (a worktree or branch-aligned clone
/// like `sie-web-backbutton` → `sie-web`, `sie-internal-m5` → `sie-internal`).
/// An unknown name (a local-only repo) passes through unchanged.
pub fn fold_repo(dir: &str, known: &std::collections::HashSet<String>) -> String {
    if dir.is_empty() || known.contains(dir) {
        return dir.to_string();
    }
    known
        .iter()
        .filter(|r| dir.len() > r.len() + 1 && dir.as_bytes()[r.len()] == b'-' && dir.starts_with(r.as_str()))
        .max_by_key(|r| r.len())
        .cloned()
        .unwrap_or_else(|| dir.to_string())
}

/// Stable cache key for a GitHub PR/issue summary (repo#number is unique and
/// survives re-indexing, unlike doc ids).
pub fn gh_key(repo: &str, number: i64) -> String {
    format!("gh:{repo}#{number}")
}

/// Normalize a session's linked_pr (stored as a URL or "repo#num") to the gh:
/// unit key, so a session can be bridged to the PR it produced during clustering.
pub fn linked_pr_key(linked: &str) -> Option<String> {
    if let Some(rest) = linked.strip_prefix("https://github.com/").or_else(|| linked.strip_prefix("http://github.com/")) {
        let p: Vec<&str> = rest.split('/').collect();
        if p.len() >= 4 && (p[2] == "pull" || p[2] == "issues") {
            return Some(gh_key(p[1], p[3].parse().ok()?));
        }
        None
    } else {
        let (repo, num) = linked.split_once('#')?;
        Some(gh_key(repo, num.parse().ok()?))
    }
}

/// A PR/issue as summarizer input: title + body, capped.
pub struct DocInput {
    pub key: String,
    pub kind: &'static str, // "pull request" | "issue"
    pub repo: String,
    pub title: String,
    pub text: String,
}

/// PR/issue docs as summarizer inputs, so every work unit can get a summary.
pub fn doc_inputs() -> Result<Vec<DocInput>> {
    let docs = load_docs(readmodel::docs_path()).unwrap_or_default();
    Ok(docs
        .iter()
        .filter_map(|d| {
            let kind = match d.meta.kind.as_str() {
                "pull_request" => "pull request",
                "issue" => "issue",
                _ => return None,
            };
            Some(DocInput {
                key: gh_key(&d.meta.repo, d.meta.number.unwrap_or(0)),
                kind,
                repo: d.meta.repo.clone(),
                title: first_line(&d.text).to_string(),
                text: crate::excerpt(&d.text, 1500),
            })
        })
        .collect())
}

/// A unit (session/PR/issue) as clustering input: a stable key, its one-line
/// summary (the thing we cluster on), and repo. Units without a summary are
/// omitted — they can't be placed by summary.
pub struct UnitClusterInput {
    pub key: String,
    pub summary: String, // for c-TF-IDF labels
    pub embed: String,   // richer text actually embedded for clustering
    pub repo: String,    // dup collapsing never crosses repos
    pub linked: Option<String>, // for a session: the gh: key of the PR it produced
}

/// A session's embed text for clustering: a type prefix and the
/// project-identifying tokens (repo, touched files) lead, the summary follows
/// — sessions placed markedly worse than PRs with the summary leading (qdump
/// probe: ~3x the misplacement rate), and the leading tokens set the
/// contextual frame the encoder embeds the rest in. The head is capped
/// separately so a session with long file paths can never truncate its
/// summary, the highest-signal content.
fn session_embed(summary: &str, repo: &str, files: &str) -> String {
    let head = crate::excerpt(&format!("Session: {repo} {files}"), 320);
    crate::excerpt(&format!("{head} — {summary}"), 500)
}

/// All units that have a cached summary, for clustering. We embed more than the
/// one-line summary so the vectors separate (a lone 20-token summary is too thin
/// to cluster well). For sessions we add *project-identifying* tokens — the repo
/// and the touched file names — rather than the often-generic c-TF-IDF
/// keyphrases: a session that says "unit summaries and keyphrase labels" should
/// cluster with other `synty` work via the repo and `topics.rs`/`qwen.rs` paths,
/// not collide with vision-bench "unit labels" on incidental shared words.
pub fn cluster_units() -> Result<Vec<UnitClusterInput>> {
    let mut out = Vec::new();
    for s in sessions()? {
        if let Some(summary) = s.summary.filter(|x| !x.is_empty()) {
            // Cap files + total length so sessions stay comparable in length to the
            // 500-capped PR/issue embeds (MaxSim is length-biased — an embed that's
            // all file paths would otherwise dominate and over-group by repo).
            let files = s.files.iter().take(8).cloned().collect::<Vec<_>>().join(" ");
            let embed = session_embed(&summary, &s.repo, &files);
            let linked = s.linked_pr.as_deref().and_then(linked_pr_key);
            out.push(UnitClusterInput { key: s.id, summary, embed, repo: s.repo, linked });
        }
    }
    let cache = load_summary_cache();
    for d in load_docs(readmodel::docs_path()).unwrap_or_default() {
        if matches!(d.meta.kind.as_str(), "pull_request" | "issue") {
            let key = gh_key(&d.meta.repo, d.meta.number.unwrap_or(0));
            if let Some(c) = cache.get(&key).filter(|c| !c.summary.is_empty()) {
                // summary + title + body, capped so units stay comparable in length
                // (MaxSim is length-biased — long bodies would otherwise hub).
                let embed = crate::excerpt(&format!("{} {}", c.summary, d.text), 500);
                out.push(UnitClusterInput { key, summary: c.summary.clone(), embed, repo: d.meta.repo.clone(), linked: None });
            }
        }
    }
    Ok(out)
}

/// The k longest messages, returned in chronological order and each capped
/// generously — a keyphrase-free stand-in for "the on-topic turns": longer
/// messages carry more of the actual work than the generic opener. The material
/// a summarizer reads.
fn top_turns(texts: &[String], k: usize) -> Vec<String> {
    let mut idx: Vec<usize> = (0..texts.len()).collect();
    idx.sort_by_key(|&i| std::cmp::Reverse(texts[i].len()));
    idx.truncate(k);
    idx.sort_unstable(); // restore chronological order
    idx.into_iter().map(|i| crate::excerpt(&texts[i], 600)).collect()
}

/// A repo-qualified path tail for a touched file: the last up to 3 components
/// (e.g. "synty/src/topics.rs"), so generic basenames like main.rs separate by
/// repo in the clustering embed and the repo recurs once per file. Skips harness
/// artifacts (sub-agent task outputs under /tmp, .claude state) — not source.
fn file_token(path: &str) -> Option<String> {
    if path.contains("/tmp/") || path.contains("/.claude/") || path.ends_with(".output") {
        return None;
    }
    let mut parts: Vec<&str> = path.rsplit('/').filter(|s| !s.is_empty()).take(3).collect();
    parts.reverse();
    (!parts.is_empty()).then(|| parts.join("/"))
}

fn session_outcome(s: &Session) -> String {
    if let Some(pr) = &s.linked_pr {
        format!("→ PR {pr}")
    } else if !s.files.is_empty() {
        format!("{} files", s.files.len())
    } else {
        format!("{} prompts", s.prompts)
    }
}

/// Counts per week bucket (oldest→newest) over the last `weeks`, from ISO ts.
pub fn weekly_buckets(timestamps: &[String], weeks: usize) -> Vec<u64> {
    if weeks == 0 {
        return vec![];
    }
    let days: Vec<i64> = timestamps.iter().filter_map(|t| epoch_day(t)).collect();
    let Some(&max_day) = days.iter().max() else { return vec![0; weeks] };
    let mut out = vec![0u64; weeks];
    for d in days {
        let wk_ago = ((max_day - d) / 7) as usize;
        if wk_ago < weeks {
            out[weeks - 1 - wk_ago] += 1;
        }
    }
    out
}

/// Days since the Unix epoch from a day or full timestamp (only the date part
/// matters, so day-only "YYYY-MM-DD" and full RFC3339 both work, same scale).
fn epoch_day(ts: &str) -> Option<i64> {
    let day = ts.split('T').next().unwrap_or(ts);
    let d = chrono::NaiveDate::parse_from_str(day, "%Y-%m-%d").ok()?;
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)?;
    Some((d - epoch).num_days())
}

fn duration_secs(a: &Agg) -> i64 {
    match (chrono::DateTime::parse_from_rfc3339(&a.first), chrono::DateTime::parse_from_rfc3339(&a.last)) {
        (Ok(f), Ok(l)) => (l.timestamp() - f.timestamp()).max(0),
        _ => 0,
    }
}

fn day(ts: &str) -> String {
    ts.split('T').next().unwrap_or("").to_string()
}

pub(crate) fn jsonl_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if dir.is_dir() {
        for e in walkdir::WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
            if e.file_type().is_file() && e.path().extension().and_then(|s| s.to_str()) == Some("jsonl") {
                out.push(e.path().to_owned());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aggs_of(text: &str) -> HashMap<String, Agg> {
        let known = std::collections::HashSet::new();
        let mut seen = std::collections::HashSet::new();
        let mut aggs = HashMap::new();
        for line in text.lines() {
            fold_line(line, &known, &mut seen, &mut aggs);
        }
        aggs
    }

    #[test]
    fn restricted_topics_rebuild_facets_and_drop_full_corpus_summaries() {
        let unit = |repo: &str, title: &str, rank: i64| Unit {
            kind: Kind::Session,
            when: "2026-07-22".into(),
            repo: repo.into(),
            title: title.into(),
            outcome: String::new(),
            summary: None,
            topic: Some(0),
            rank,
            dup: false,
            struggle: 0.0,
            author: repo.into(),
            doc_id: None,
            session_id: Some(repo.into()),
            campaign_id: "campaign".into(),
            campaign_role: "primary".into(),
            source: "harness".into(),
        };
        let topics = vec![TopicUnits {
            id: 0,
            cache_key: "topic".into(),
            label: "secret full label".into(),
            units: vec![unit("allowed", "Allowed task", 1), unit("hidden", "Hidden task", 0)],
            last_active: "2026-07-22".into(),
            activity: vec![2],
            mix: (0, 2, 0),
            repos: vec!["allowed".into(), "hidden".into()],
            authors: vec!["allowed".into(), "hidden".into()],
            summary: Some("secret full summary".into()),
            name: Some("Secret name".into()),
            span: None,
        }];
        let scope = crate::policy::ReadScope { repos: vec!["allowed".into()], ..Default::default() };
        let scoped = scope_topic_units(topics, 1, &scope);
        assert_eq!(scoped[0].units.len(), 1);
        assert_eq!(scoped[0].repos, ["allowed"]);
        assert_eq!(scoped[0].label, "Allowed task");
        assert!(scoped[0].summary.is_none() && scoped[0].name.is_none());
    }

    fn aggs_since(text: &str, cutoff: i64) -> HashMap<String, Agg> {
        let known = std::collections::HashSet::new();
        let mut seen = std::collections::HashSet::new();
        let mut aggs = HashMap::new();
        for line in text.lines() {
            fold_line_since(line, &known, &mut seen, &mut aggs, Some(cutoff), true);
        }
        aggs.retain(|_, a| !a.first.is_empty());
        aggs
    }

    #[test]
    fn capture_boundary_keeps_metadata_but_summarizes_only_retained_work() {
        let line = |id: &str, kind: &str, ts: &str, text: &str| {
            serde_json::json!({
                "event_id": id, "session_id": "s", "source": "codex_cli", "kind": kind, "ts": ts,
                "payload": if kind == "session_start" {
                    serde_json::json!({"cwd": "/work/repo", "actor": "alice"})
                } else {
                    serde_json::json!({"text": text})
                }
            })
            .to_string()
        };
        let text = [
            line("a", "session_start", "2026-07-20T23:50:00Z", ""),
            line(
                "b",
                "user_prompt",
                "2026-07-20T23:55:00Z",
                "old secret prompt",
            ),
            line(
                "c",
                "user_prompt",
                "2026-07-21T00:05:00Z",
                "new retained prompt",
            ),
        ]
        .join("\n");
        let cutoff = chrono::DateTime::parse_from_rfc3339("2026-07-21T00:00:00Z")
            .unwrap()
            .timestamp_millis();
        let aggs = aggs_since(&text, cutoff);
        let a = &aggs["s"];
        assert_eq!(a.actor, "alice");
        assert_eq!(a.ask, "new retained prompt");
        assert_eq!(a.first, "2026-07-21T00:05:00Z");
        assert!(!a.texts.iter().any(|t| t.contains("old secret")));
    }

    fn day_stats_of(text: &str) -> HashMap<String, DayStat> {
        let mut f = DayFold::default();
        for line in text.lines() {
            f.fold(line);
        }
        f.finish()
    }

    // One activity row per day: usage copied through, LOC landing on its
    // merge day, PRs/issues counted by their day — and a session unit claims
    // its day without inventing output.
    #[test]
    fn activity_by_day_merges_usage_loc_and_unit_counts() {
        let unit = |kind: Kind, when: &str| Unit {
            kind,
            when: when.into(),
            repo: String::new(),
            title: String::new(),
            outcome: String::new(),
            summary: None,
            topic: None,
            rank: 0,
            dup: false,
            struggle: 0.0,
            author: String::new(),
            doc_id: None,
            session_id: None,
            campaign_id: String::new(),
            campaign_role: String::new(),
            source: String::new(),
        };
        let day: HashMap<String, DayStat> = [(
            "2026-06-01".to_string(),
            DayStat { tok_in: 100, tok_out: 50, cache_read: 7, cache_create: 3, tools: 4, sessions: 2 },
        )]
        .into_iter()
        .collect();
        let loc: HashMap<String, (u64, u64)> = [("2026-06-02".to_string(), (120u64, 40u64))].into_iter().collect();
        let units = vec![
            unit(Kind::Issue, "2026-06-01"),
            unit(Kind::Pr, "2026-06-02"),
            unit(Kind::Session, "2026-06-03"),
            unit(Kind::Pr, "not-a-date"),
        ];
        let act = combine_activity(day, loc, &units);
        let d1 = &act["2026-06-01"];
        assert_eq!((d1.tok_in, d1.tok_out, d1.tools, d1.sessions), (100, 50, 4, 2));
        assert_eq!((d1.issues, d1.prs), (1, 0));
        let d2 = &act["2026-06-02"];
        assert_eq!((d2.loc_add, d2.loc_del, d2.prs), (120, 40, 1));
        assert_eq!(d2.tok_in, 0, "a day present only in one source still appears");
        let d3 = &act["2026-06-03"];
        assert_eq!((d3.prs, d3.issues, d3.sessions), (0, 0, 0), "a session unit claims the day, nothing else");
        assert_eq!(act.len(), 3, "an undated unit lands nowhere");
    }

    // The day series attributes usage to the day of its event (exact even for
    // sessions spanning days), dedups streamed msg_ids across days, lands a
    // codex session's cumulative snapshot on the snapshot's day, and counts
    // distinct active sessions per day.
    #[test]
    fn day_stats_attribute_by_event_day() {
        let lines = [
            r#"{"event_id":"e1","session_id":"S","ts":"2026-06-11T23:00:00Z","kind":"agent_meta","payload":{"subtype":"usage","msg_id":"m1","usage":{"in":10,"out":5,"cache_read":100,"cache_create":0}}}"#,
            r#"{"event_id":"e2","session_id":"S","ts":"2026-06-12T01:00:00Z","kind":"agent_meta","payload":{"subtype":"usage","msg_id":"m1","usage":{"in":10,"out":5,"cache_read":100,"cache_create":0}}}"#,
            r#"{"event_id":"e3","session_id":"S","ts":"2026-06-12T01:05:00Z","kind":"agent_meta","payload":{"subtype":"usage","msg_id":"m2","usage":{"in":1,"out":2,"cache_read":3,"cache_create":4}}}"#,
            r#"{"event_id":"e4","session_id":"S","ts":"2026-06-12T01:06:00Z","kind":"tool_call","payload":{"name":"Bash"}}"#,
            r#"{"event_id":"e5","session_id":"T","ts":"2026-06-12T02:00:00Z","kind":"user_prompt","payload":{"text":"another session active today"}}"#,
            r#"{"event_id":"e6","session_id":"C","ts":"2026-06-10T09:00:00Z","kind":"agent_meta","payload":{"subtype":"event_msg","event_kind":"token_count","payload":{"info":{"total_token_usage":{"input_tokens":100,"cached_input_tokens":20,"output_tokens":50}}}}}"#,
        ]
        .join("\n");
        let d = day_stats_of(&lines);
        assert_eq!(d["2026-06-11"].tok_out, 5, "m1 lands on its first-seen day");
        assert_eq!(d["2026-06-12"].tok_out, 2, "the cross-day repeat of m1 does not double-count");
        assert_eq!(d["2026-06-12"].tools, 1);
        assert_eq!(d["2026-06-12"].sessions, 2, "S and T both active");
        assert_eq!(d["2026-06-10"].tok_in, 80, "codex snapshot normalized on its own day");
        assert_eq!(d["2026-06-10"].cache_read, 20);
    }

    // A streamed turn repeats the identical usage object on every raw line of
    // one msg_id (measured 2-4x in real logs) — it must count exactly once,
    // or token totals overcount ~3x.
    #[test]
    fn usage_dedups_by_msg_id() {
        let lines = [
            r#"{"event_id":"e1","session_id":"S","ts":"t1","kind":"agent_meta","payload":{"subtype":"usage","msg_id":"m1","usage":{"in":10,"out":5,"cache_read":100,"cache_create":7}}}"#,
            r#"{"event_id":"e2","session_id":"S","ts":"t2","kind":"agent_meta","payload":{"subtype":"usage","msg_id":"m1","usage":{"in":10,"out":5,"cache_read":100,"cache_create":7}}}"#,
            r#"{"event_id":"e3","session_id":"S","ts":"t3","kind":"agent_meta","payload":{"subtype":"usage","msg_id":"m2","usage":{"in":1,"out":2,"cache_read":3,"cache_create":4}}}"#,
        ]
        .join("\n");
        let a = &aggs_of(&lines)["S"];
        assert_eq!((a.tok_in, a.tok_out, a.cache_read, a.cache_create), (11, 7, 103, 11));
        assert_eq!(a.usage_turns, 2);
    }

    // Without a msg_id each event counts once; a literally duplicated
    // envelope (same event_id, e.g. overlapping trackers) still folds once.
    #[test]
    fn usage_without_msg_id_counts_each_once_per_event() {
        let dup = r#"{"event_id":"e1","session_id":"S","ts":"t1","kind":"agent_meta","payload":{"subtype":"usage","msg_id":"","usage":{"in":10,"out":0,"cache_read":0,"cache_create":0}}}"#;
        let other = r#"{"event_id":"e2","session_id":"S","ts":"t2","kind":"agent_meta","payload":{"subtype":"usage","msg_id":"","usage":{"in":10,"out":0,"cache_read":0,"cache_create":0}}}"#;
        let a = &aggs_of(&[dup, dup, other].join("\n"))["S"];
        assert_eq!(a.tok_in, 20, "two distinct events, the duplicated envelope folded once");
        assert_eq!(a.usage_turns, 2);
    }

    // Tool calls tally by name; errors come straight off tool_result.
    #[test]
    fn tool_calls_and_result_errors() {
        let lines = [
            r#"{"event_id":"e1","session_id":"S","ts":"t1","kind":"tool_call","payload":{"tool_use_id":"t1","name":"Bash","input":{"command":"ls -la"}}}"#,
            r#"{"event_id":"e2","session_id":"S","ts":"t2","kind":"tool_call","payload":{"tool_use_id":"t2","name":"Bash","input":{"command":"git status"}}}"#,
            r#"{"event_id":"e3","session_id":"S","ts":"t3","kind":"tool_result","payload":{"tool_use_id":"t1","is_error":true,"content":"command not found"}}"#,
            r#"{"event_id":"e4","session_id":"S","ts":"t4","kind":"tool_result","payload":{"tool_use_id":"t2","is_error":false,"content":"clean"}}"#,
        ]
        .join("\n");
        let a = &aggs_of(&lines)["S"];
        assert_eq!(a.tool_counts["Bash"], 2);
        assert_eq!(a.tool_err, 1);
        assert_eq!(a.tool_errs["Bash"], 1, "the error is attributed to its tool by id");
        assert_eq!(a.tools, 2);
        assert!(a.tool_chars["Bash"] > 0, "args + result volume accumulates per tool");
    }

    // Codex snapshots are cumulative: the last one is the session total, and
    // it normalizes to the shared classes (fresh in = input − cached).
    #[test]
    fn codex_token_count_last_wins_and_normalizes() {
        let lines = [
            r#"{"event_id":"e1","session_id":"C","ts":"t1","kind":"agent_meta","payload":{"subtype":"event_msg","event_kind":"token_count","payload":{"info":{"total_token_usage":{"input_tokens":100,"cached_input_tokens":20,"output_tokens":50}}}}}"#,
            r#"{"event_id":"e2","session_id":"C","ts":"t2","kind":"agent_meta","payload":{"subtype":"event_msg","event_kind":"token_count","payload":{"info":{"total_token_usage":{"input_tokens":200,"cached_input_tokens":80,"output_tokens":120}}}}}"#,
        ]
        .join("\n");
        let a = &aggs_of(&lines)["C"];
        assert_eq!(a.codex_usage, Some((200, 80, 120)), "last cumulative snapshot wins");
        // sessions() maps it to fresh-in 120 / cache_read 80 / out 120 — the
        // normalization itself is covered by the builder mapping above.
    }

    // A tool's profile gleans everything the envelopes hold: per-key argument
    // frequencies, common values for enum-ish keys, call→result latency from
    // the paired timestamps, error attribution, days, and the agent label —
    // across both id schemes (Claude tool_use_id, codex call_id shapes).
    #[test]
    fn tool_profile_collects_args_durations_errors() {
        let lines = [
            r#"{"event_id":"e1","session_id":"S","source":"claude_code","ts":"2026-06-12T01:00:00Z","kind":"tool_call","payload":{"tool_use_id":"t1","name":"Bash","input":{"command":"ls -la","timeout":5,"run_in_background":false}}}"#,
            r#"{"event_id":"e2","session_id":"S","source":"claude_code","ts":"2026-06-12T01:00:01.500Z","kind":"tool_result","payload":{"tool_use_id":"t1","is_error":true}}"#,
            r#"{"event_id":"e3","session_id":"S","source":"claude_code","ts":"2026-06-12T02:00:00Z","kind":"tool_call","payload":{"tool_use_id":"t2","name":"Bash","input":{"command":"git status","run_in_background":false}}}"#,
            r#"{"event_id":"e4","session_id":"S","source":"claude_code","ts":"2026-06-12T02:00:00.500Z","kind":"tool_result","payload":{"tool_use_id":"t2","is_error":false}}"#,
            r#"{"event_id":"e5","session_id":"C","source":"codex_cli","ts":"2026-06-12T03:00:00Z","kind":"tool_call","payload":{"call_id":"c1","name":"exec_command","arguments":"{\"cmd\":\"pwd\"}"}}"#,
        ]
        .join("\n");
        let mut f = ToolFold { target: "Bash".to_string(), ..Default::default() };
        for line in lines.lines() {
            f.fold(line);
        }
        let p = f.finish();
        assert_eq!((p.calls, p.errs, p.timed), (2, 1, 2));
        assert_eq!(p.agent, "claude");
        assert_eq!(p.arg_keys[0], ("command".to_string(), 2));
        let rib = p.arg_tops.iter().find(|(k, _)| k == "run_in_background").expect("low-cardinality key profiled");
        assert_eq!(rib.1[0], ("false".to_string(), 2));
        assert!(!p.arg_tops.iter().any(|(k, _)| k == "command"), "free-form keys get no value breakdown");
        assert_eq!(p.p50_ms, 500, "median of 1500ms and 500ms pairings");
        assert_eq!(p.p95_ms, 1500);
        assert_eq!(p.days["2026-06-12"], 2);
        assert_eq!(p.samples.len(), 2);
        // The codex-shaped arguments string parses for its own tool.
        let mut f = ToolFold { target: "exec_command".to_string(), ..Default::default() };
        for line in lines.lines() {
            f.fold(line);
        }
        let p = f.finish();
        assert_eq!(p.arg_keys[0], ("cmd".to_string(), 1));
        assert_eq!(p.agent, "codex");
    }

    // A remote reader receives the same session, usage, and tool facts without
    // any raw event files. This is the writer-side contract for the compact
    // analysis artifact published with a build.
    #[test]
    fn published_analysis_contains_session_day_and_tool_facts() {
        let lines = [
            r#"{"event_id":"old-tool","session_id":"S","source":"harness","ts":"2026-07-21T23:00:00Z","kind":"tool_call","payload":{"tool_use_id":"old","name":"SecretTool","input":{"secret":"must-not-publish"}}}"#,
            r#"{"event_id":"e0","session_id":"S","source":"harness","rollup_dim":"campaign-7","ts":"2026-07-21T23:30:00Z","kind":"session_start","payload":{"cwd":"/work/repo","campaign_role":"primary"}}"#,
            r#"{"event_id":"e1","session_id":"S","source":"harness","ts":"2026-07-22T01:00:01Z","kind":"user_prompt","payload":{"text":"investigate the deployment bottleneck"}}"#,
            r#"{"event_id":"e2","session_id":"S","source":"harness","ts":"2026-07-22T01:00:02Z","kind":"tool_call","payload":{"tool_use_id":"t1","name":"Bash","input":{"command":"kubectl get pods"}}}"#,
            r#"{"event_id":"e3","session_id":"S","source":"harness","ts":"2026-07-22T01:00:03Z","kind":"tool_result","payload":{"tool_use_id":"t1","is_error":false,"content":"ready"}}"#,
        ]
        .join("\n");
        let cutoff = chrono::DateTime::parse_from_rfc3339("2026-07-22T00:00:00Z")
            .unwrap()
            .timestamp_millis();
        let mut builder = AnalysisBuilder::new(
            std::collections::HashSet::from(["repo".to_string()]),
            Some(cutoff),
        );
        builder.fold_text(&lines);
        let path = std::env::temp_dir().join(format!(
            "synty-analysis-snapshot-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("scenario")
        ));
        builder.write(&path, crate::fleet::Roster::default()).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let snapshot: AnalysisSnapshot = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(snapshot.format, ANALYSIS_FORMAT);
        assert_eq!(snapshot.sessions.len(), 1);
        assert_eq!(snapshot.sessions[0].repo, "repo");
        assert_eq!(snapshot.sessions[0].campaign_id, "campaign-7");
        assert_eq!(snapshot.sessions[0].campaign_role, "primary");
        assert_eq!(snapshot.days["2026-07-22"].tools, 1);
        assert_eq!(snapshot.tools["Bash"].calls, 1);
        assert_eq!(snapshot.tools["Bash"].timed, 1);
        assert!(!snapshot.tools.contains_key("SecretTool"));
        let second = path.with_extension("again.json");
        let mut again = AnalysisBuilder::new(
            std::collections::HashSet::from(["repo".to_string()]),
            Some(cutoff),
        );
        again.fold_text(&lines);
        again.write(&second, crate::fleet::Roster::default()).unwrap();
        assert_eq!(
            bytes,
            std::fs::read(&second).unwrap(),
            "identical inputs publish an identical immutable projection"
        );
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(second).unwrap();
    }

    // A session with prompts but no usage records must read as unmeasured —
    // the coverage denominator, and the no-fake-zero display rule.
    #[test]
    fn has_usage_false_for_cowork_like_sessions() {
        let s = Session {
            id: "x".into(),
            repo: String::new(),
            started: String::new(),
            ended: String::new(),
            ask: "do things".into(),
            prompts: 3,
            assistant: 5,
            thinking: 0,
            tools: 2,
            files: vec![],
            linked_pr: None,
            topic: None,
            struggle: 0.0,
            tok_in: 0,
            tok_out: 0,
            cache_read: 0,
            cache_create: 0,
            usage_turns: 0,
            tools_by_name: vec![("Bash".into(), 2, 0, 640)],
            tool_err: 0,
            by_model: vec![],
            source: "cowork".into(),
            campaign_id: String::new(),
            campaign_role: String::new(),
            summary: None,
            author: String::new(),
        };
        assert!(!s.has_usage());
    }

    // The session embed leads with the project (type prefix, repo, files) and
    // appends the summary — which must survive the cap intact even when file
    // paths are long, since it carries the semantic signal.
    #[test]
    fn session_embed_keeps_summary_under_cap() {
        let files = (0..8).map(|i| format!("very/long/path/to/some/deeply/nested/module_{i}.rs")).collect::<Vec<_>>().join(" ");
        let summary = "S".repeat(170); // a typical one-liner's length
        let e = session_embed(&summary, "sie-internal", &files);
        assert!(e.starts_with("Session: sie-internal"));
        assert!(e.contains(" — "));
        assert!(e.ends_with(&summary), "the summary survives the cap intact");
        assert!(e.chars().count() <= 500);
    }

    // File tokens are repo-qualified path tails; harness/temp artifacts are dropped.
    // A cwd segment that is (or folds to) a known repo wins, anywhere in the
    // path; subdirs collapse to it; an unknown path matches nothing (the runtime
    // git fallback handles local repos), never a hardcoded workspace guess.
    #[test]
    fn repo_from_cwd_matches_known_repo_segment() {
        let known: std::collections::HashSet<String> =
            ["synty", "sie-web", "sie-internal"].into_iter().map(String::from).collect();
        assert_eq!(repo_from_cwd("/Users/svonava/c/synty/server", &known), "synty");
        assert_eq!(repo_from_cwd("/Users/svonava/c/sie-web/apps/site/public", &known), "sie-web");
        assert_eq!(repo_from_cwd("/Users/d/c/sie-internal/x", &known), "sie-internal");
        assert_eq!(repo_from_cwd("/Users/svonava/c/sie-web-backbutton/x", &known), "sie-web"); // folds
        // unknown to the org / no match → empty (git fallback resolves it at runtime)
        assert_eq!(repo_from_cwd("/Users/svonava/c/personal-thing", &known), "");
        assert_eq!(repo_from_cwd("/Users/svonava/Library/Application Support/Claude/x/outputs", &known), "");
        assert_eq!(repo_from_cwd("", &known), "");
    }

    // Worktree / branch-aligned dirs fold to the known repo; exact + longest win.
    #[test]
    fn fold_repo_collapses_worktrees_to_known_repo() {
        let known: std::collections::HashSet<String> =
            ["sie-web", "sie", "sie-internal", "synty"].into_iter().map(String::from).collect();
        assert_eq!(fold_repo("sie-web-backbutton", &known), "sie-web"); // longest prefix beats "sie"
        assert_eq!(fold_repo("sie-internal-m5", &known), "sie-internal");
        assert_eq!(fold_repo("synty", &known), "synty"); // exact
        assert_eq!(fold_repo("acme-tool", &known), "acme-tool"); // unknown passes through
        assert_eq!(fold_repo("", &known), "");
    }

    #[test]
    fn file_token_qualifies_and_filters() {
        assert_eq!(file_token("/Users/svonava/c/synty/src/topics.rs").as_deref(), Some("synty/src/topics.rs"));
        assert_eq!(file_token("/Users/svonava/c/sie-internal/src/main.rs").as_deref(), Some("sie-internal/src/main.rs"));
        assert_eq!(file_token("/Users/svonava/c/synty/Cargo.toml").as_deref(), Some("c/synty/Cargo.toml"));
        assert_eq!(file_token("/private/tmp/claude-501/x/tasks/be6b03qny.output"), None);
    }

    #[test]
    fn weekly_buckets_places_activity_by_age() {
        // newest at 2026-05-31; one item that week, one ~2 weeks earlier.
        let ts = vec!["2026-05-31T10:00:00Z".into(), "2026-05-17T10:00:00Z".into()];
        let b = weekly_buckets(&ts, 4);
        assert_eq!(b.len(), 4);
        assert_eq!(b[3], 1); // most recent week
        assert_eq!(b.iter().sum::<u64>(), 2);
    }

    #[test]
    fn struggle_scores_normalize_0_to_1() {
        let raw = vec![[0.0, 0.0, 0.0, 0.0], [10.0, 20.0, 5.0, 600.0], [3.0, 4.0, 2.0, 100.0]];
        let s = struggle_scores(&raw);
        assert_eq!(s.len(), 3);
        assert!((s[0] - 0.0).abs() < 1e-5, "lowest → 0");
        assert!((s[1] - 1.0).abs() < 1e-5, "highest → 1");
        assert!(s[2] > 0.0 && s[2] < 1.0);
    }
}
