// Units of work — the human-facing objects (sessions, PRs, issues) the surfaces
// browse, each with a time axis and a derived "struggle" score. Built from the
// raw envelopes under corpus/local (for session structure) plus docs.jsonl and
// clusters.json (for PRs/issues and topic membership). Consumed by both the CLI
// and the TUI so they stay at parity.

use crate::{first_line, load_docs, DOCS_PATH};
use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const LOCAL_DIR: &str = "corpus/local";
const FILE_TOOLS: &[&str] = &["Write", "Edit", "MultiEdit", "NotebookEdit"];

/// A coding session as one unit of work.
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
    pub summary: Option<String>, // abstractive one-liner (local LLM), if cached
}

/// A session's cached LLM summary, keyed by a hash of its inputs so the
/// summarizer only regenerates when the underlying turns change.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CachedSummary {
    pub hash: String,
    pub summary: String,
}

pub type SummaryCache = HashMap<String, CachedSummary>;

const SUMMARIES_PATH: &str = "index/summaries.json";

/// Read the on-disk summary cache (empty if the summarizer hasn't run).
pub fn load_summary_cache() -> SummaryCache {
    std::fs::read_to_string(SUMMARIES_PATH)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Persist the summary cache next to the index.
pub fn save_summary_cache(c: &SummaryCache) -> Result<()> {
    if let Some(parent) = Path::new(SUMMARIES_PATH).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(SUMMARIES_PATH, serde_json::to_string_pretty(c)?)?;
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
    pub struggle: f32,
    pub author: String,    // PR/issue author; empty for sessions
    pub doc_id: Option<i64>,    // for PR/issue → docs.jsonl
    pub session_id: Option<String>, // for sessions
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
}

/// Aggregate the raw envelopes under corpus/local into per-session tallies.
fn aggregate() -> HashMap<String, Agg> {
    let mut aggs: HashMap<String, Agg> = HashMap::new();
    for f in jsonl_files(Path::new(LOCAL_DIR)) {
        let Ok(data) = std::fs::read_to_string(&f) else { continue };
        for line in data.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let Ok(ev) = serde_json::from_str::<Value>(line) else { continue };
            let sid = ev["session_id"].as_str().unwrap_or("");
            if sid.is_empty() {
                continue;
            }
            let a = aggs.entry(sid.to_string()).or_default();
            let ts = ev["ts"].as_str().unwrap_or("");
            if !ts.is_empty() {
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
                        a.repo = cwd.rsplit('/').next().unwrap_or(cwd).to_string();
                    }
                }
                "user_prompt" => {
                    // Skip slash-command echoes / hook output so the "ask" is the
                    // real first human prompt and the count reflects real turns.
                    if let Some(t) = ev["payload"]["text"].as_str() {
                        let t = t.trim();
                        if t.len() >= 12 && !crate::ingest::is_noise(t) {
                            a.prompts += 1;
                            a.texts.push(t.to_string());
                            if a.ask.is_empty() {
                                a.ask = crate::excerpt(t, 200);
                            }
                        }
                    }
                }
                "assistant_message" => {
                    a.assistant += 1;
                    if let Some(t) = ev["payload"]["text"].as_str() {
                        if t.trim().len() >= 12 {
                            a.texts.push(t.trim().to_string());
                        }
                    }
                }
                "thinking" => a.thinking += 1,
                "tool_call" => a.tools += 1,
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
                    if a.linked_pr.is_none() {
                        let p = &ev["payload"];
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
    }
    aggs
}

/// Build session units from the raw envelopes, with topic membership, a
/// normalized struggle score, and any cached LLM summary.
pub fn sessions() -> Result<Vec<Session>> {
    let aggs = aggregate();
    let cache = load_summary_cache();
    let topic_of = unit_topics();

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
            Session {
                id: id.clone(),
                repo: if a.repo.is_empty() { "local".into() } else { a.repo.clone() },
                started: a.first.clone(),
                ended: a.last.clone(),
                ask: a.ask.clone(),
                prompts: a.prompts,
                assistant: a.assistant,
                thinking: a.thinking,
                tools: a.tools,
                files: a.files.clone(),
                linked_pr: a.linked_pr.clone(),
                topic: topic_of.get(id).map(|(c, _, _)| *c),
                struggle: scores[i],
                summary: cache.get(id).map(|c| c.summary.clone()).filter(|s| !s.is_empty()),
            }
        })
        .collect();
    out.sort_by(|x, y| y.ended.cmp(&x.ended));
    Ok(out)
}

/// All work units (sessions + PRs + issues), newest first.
pub fn units() -> Result<Vec<Unit>> {
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
            struggle: s.struggle,
            author: String::new(),
            doc_id: None,
            session_id: Some(s.id),
        })
        .collect();

    if let Ok(docs) = load_docs(DOCS_PATH) {
        let topic_of = unit_topics();
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
                topic: topic_of.get(&key).map(|(c, _, _)| *c),
                struggle: 0.0,
                author: d.meta.author.clone(),
                doc_id: Some(d.id),
                session_id: None,
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

/// unit key → (cluster index, stable cache key, label), from unit_clusters.json
/// (written by `cluster`). Empty until clustering has run. The stable key falls
/// back to the index string for pre-I1 files.
fn unit_topics() -> HashMap<String, (i64, String, String)> {
    let Ok(raw) = std::fs::read_to_string("unit_clusters.json") else { return HashMap::new() };
    let arr: Vec<Value> = serde_json::from_str(&raw).unwrap_or_default();
    arr.iter()
        .filter_map(|it| {
            let key = it["key"].as_str()?.to_string();
            let cluster = it["cluster"].as_i64()?;
            let cache_key = it["topic"].as_str().map(String::from).unwrap_or_else(|| cluster.to_string());
            Some((key, (cluster, cache_key, it["label"].as_str().unwrap_or("").to_string())))
        })
        .collect()
}

/// cluster index → label.
fn cluster_labels() -> HashMap<i64, String> {
    let mut m = HashMap::new();
    for (_, (c, _, l)) in unit_topics() {
        m.entry(c).or_insert(l);
    }
    m
}

/// cluster index → stable cache key (medoid hash).
fn cluster_cache_keys() -> HashMap<i64, String> {
    let mut m = HashMap::new();
    for (_, (c, k, _)) in unit_topics() {
        m.entry(c).or_insert(k);
    }
    m
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
    let aggs = aggregate();
    let ids: Vec<String> = aggs.keys().cloned().collect();
    Ok(ids
        .iter()
        .filter(|id| aggs[*id].prompts > 0)
        .map(|id| {
            let a = &aggs[id];
            SessionInput {
                id: id.clone(),
                repo: if a.repo.is_empty() { "local".into() } else { a.repo.clone() },
                ask: a.ask.clone(),
                files: a.files.iter().take(12).cloned().collect(),
                turns: top_turns(&a.texts, 8),
            }
        })
        .collect())
}

/// Stable cache key for a GitHub PR/issue summary (repo#number is unique and
/// survives re-indexing, unlike doc ids).
pub fn gh_key(repo: &str, number: i64) -> String {
    format!("gh:{repo}#{number}")
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
    let docs = load_docs(DOCS_PATH).unwrap_or_default();
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
            let embed = crate::excerpt(&format!("{summary} {} {}", s.repo, files), 500);
            out.push(UnitClusterInput { key: s.id, summary, embed });
        }
    }
    let cache = load_summary_cache();
    for d in load_docs(DOCS_PATH).unwrap_or_default() {
        if matches!(d.meta.kind.as_str(), "pull_request" | "issue") {
            let key = gh_key(&d.meta.repo, d.meta.number.unwrap_or(0));
            if let Some(c) = cache.get(&key).filter(|c| !c.summary.is_empty()) {
                // summary + title + body, capped so units stay comparable in length
                // (MaxSim is length-biased — long bodies would otherwise hub).
                let embed = crate::excerpt(&format!("{} {}", c.summary, d.text), 500);
                out.push(UnitClusterInput { key, summary: c.summary.clone(), embed });
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

fn jsonl_files(dir: &Path) -> Vec<PathBuf> {
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

    // File tokens are repo-qualified path tails; harness/temp artifacts are dropped.
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
