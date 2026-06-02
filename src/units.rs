// Units of work — the human-facing objects (sessions, PRs, issues) the surfaces
// browse, each with a time axis and a derived "struggle" score. Built from the
// raw envelopes under corpus/local (for session structure) plus docs.jsonl and
// clusters.json (for PRs/issues and topic membership). Consumed by both the CLI
// and the TUI so they stay at parity.

use crate::{first_line, load_docs, Doc, DOCS_PATH};
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
    pub keyphrases: Vec<String>,
    pub gist: String, // representative line (most keyphrase coverage)
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

/// Per-session material for the summarizer: the ask, keyphrases, and the few
/// messages with the most keyphrase coverage (the on-topic turns).
pub struct SessionInput {
    pub id: String,
    pub repo: String,
    pub ask: String,
    pub keyphrases: Vec<String>,
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
pub struct Unit {
    pub kind: Kind,
    pub when: String, // day
    pub repo: String,
    pub title: String,
    pub outcome: String, // PR state, or session file/PR summary
    pub summary: Option<String>, // session's one-line LLM summary, if cached
    pub topic: Option<i64>,
    pub struggle: f32,
    pub doc_id: Option<i64>,    // for PR/issue → docs.jsonl
    pub session_id: Option<String>, // for sessions
}

/// A topic with its work units, time span, activity over weeks, and type mix.
pub struct TopicUnits {
    pub id: i64,
    pub label: String,
    pub units: Vec<Unit>,
    pub last_active: String,
    pub activity: Vec<u64>, // weekly buckets, oldest→newest
    pub mix: (usize, usize, usize), // (github, assistant, prompt) doc counts
    pub repos: Vec<String>,   // repos involved, most frequent first
    pub authors: Vec<String>, // authors involved, most frequent first
    pub summary: Option<String>, // map-reduced topic summary (local LLM), if cached
}

/// Cache key for a topic's reduced summary.
pub fn topic_key(id: i64) -> String {
    format!("topic:{id}")
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
    texts: Vec<String>, // message texts for keyphrase extraction
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
                        if let Some(p) = ev["payload"]["local_path"].as_str() {
                            let base = p.rsplit('/').next().unwrap_or(p).to_string();
                            if a.seen_files.insert(base.clone()) {
                                a.files.push(base);
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
    let topic_of = session_topics().unwrap_or_default();

    // Raw struggle signals → z-scores → summed → percentile-ranked to 0..1.
    let ids: Vec<String> = aggs.keys().cloned().collect();
    let sig = |a: &Agg| [a.thinking as f64, a.tools as f64, a.prompts as f64, duration_secs(a) as f64];
    let raw: Vec<[f64; 4]> = ids.iter().map(|id| sig(&aggs[id])).collect();
    let scores = struggle_scores(&raw);

    // c-TF-IDF keyphrases per session, the same extractive labeling as topics.
    let text_refs: Vec<Vec<&str>> = ids.iter().map(|id| aggs[id].texts.iter().map(String::as_str).collect()).collect();
    let keyphrases = crate::keyphrase::labels(&text_refs, 4);

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
                topic: topic_of.get(id).copied(),
                struggle: scores[i],
                gist: best_line(&aggs[id].texts, keyphrases.get(i).map(Vec::as_slice).unwrap_or(&[])),
                keyphrases: keyphrases.get(i).cloned().unwrap_or_default(),
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
            doc_id: None,
            session_id: Some(s.id),
        })
        .collect();

    if let Ok(docs) = load_docs(DOCS_PATH) {
        let topic_of = doc_topics().unwrap_or_default();
        let cache = load_summary_cache();
        for d in &docs {
            let kind = match d.meta.kind.as_str() {
                "pull_request" => Kind::Pr,
                "issue" => Kind::Issue,
                _ => continue,
            };
            let num = d.meta.number.unwrap_or(0);
            out.push(Unit {
                kind,
                when: day(&d.meta.ts),
                repo: d.meta.repo.clone(),
                title: format!("{}#{} {}", d.meta.repo, num, first_line(&d.text)),
                outcome: d.meta.state.clone().unwrap_or_default(),
                summary: cache.get(&gh_key(&d.meta.repo, num)).map(|c| c.summary.clone()).filter(|s| !s.is_empty()),
                topic: topic_of.get(&d.id).copied(),
                struggle: 0.0,
                doc_id: Some(d.id),
                session_id: None,
            });
        }
    }
    out.sort_by(|a, b| b.when.cmp(&a.when));
    Ok(out)
}

/// Topics with their units, last-active, weekly activity, and type mix.
pub fn topic_units(weeks: usize) -> Result<Vec<TopicUnits>> {
    let all = units()?;
    let labels = topic_labels()?;
    let cache = load_summary_cache();
    let docs = load_docs(DOCS_PATH).unwrap_or_default();
    let by_id: HashMap<i64, &Doc> = docs.iter().map(|d| (d.id, d)).collect();

    let mut by_topic: HashMap<i64, Vec<Unit>> = HashMap::new();
    for u in all {
        if let Some(t) = u.topic {
            by_topic.entry(t).or_default().push(u);
        }
    }
    let mut out: Vec<TopicUnits> = by_topic
        .into_iter()
        .map(|(id, units)| {
            let last_active = units.iter().map(|u| u.when.clone()).max().unwrap_or_default();
            // activity from the constituent docs' timestamps (clusters are over docs)
            let ts: Vec<String> = cluster_doc_ts(id, &by_id);
            let activity = weekly_buckets(&ts, weeks);
            let mix = cluster_mix(id, &by_id);
            let (repos, authors) = cluster_facets(id, &by_id);
            let summary = cache.get(&topic_key(id)).map(|c| c.summary.clone()).filter(|s| !s.is_empty());
            TopicUnits { id, label: labels.get(&id).cloned().unwrap_or_default(), units, last_active, activity, mix, repos, authors, summary }
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

// ── topic membership ──────────────────────────────────────────────────────

/// doc id → topic cluster, from clusters.json.
fn doc_topics() -> Result<HashMap<i64, i64>> {
    let raw = std::fs::read_to_string("clusters.json")?;
    let arr: Vec<Value> = serde_json::from_str(&raw)?;
    Ok(arr.iter().filter_map(|it| Some((it["id"].as_i64()?, it["cluster"].as_i64()?))).collect())
}

/// topic cluster → label, from clusters.json.
fn topic_labels() -> Result<HashMap<i64, String>> {
    let raw = std::fs::read_to_string("clusters.json")?;
    let arr: Vec<Value> = serde_json::from_str(&raw)?;
    let mut m = HashMap::new();
    for it in &arr {
        if let (Some(c), Some(l)) = (it["cluster"].as_i64(), it["label"].as_str()) {
            m.entry(c).or_insert_with(|| l.to_string());
        }
    }
    Ok(m)
}

/// session id → majority topic of its messages.
fn session_topics() -> Result<HashMap<String, i64>> {
    let docs = load_docs(DOCS_PATH).unwrap_or_default();
    let dt = doc_topics().unwrap_or_default();
    let mut tally: HashMap<String, HashMap<i64, usize>> = HashMap::new();
    for d in &docs {
        if d.meta.session_id.is_empty() {
            continue;
        }
        if let Some(&t) = dt.get(&d.id) {
            *tally.entry(d.meta.session_id.clone()).or_default().entry(t).or_default() += 1;
        }
    }
    Ok(tally
        .into_iter()
        .filter_map(|(sid, m)| m.into_iter().max_by_key(|(_, n)| *n).map(|(t, _)| (sid, t)))
        .collect())
}

fn cluster_doc_ts(topic: i64, by_id: &HashMap<i64, &Doc>) -> Vec<String> {
    let dt = doc_topics().unwrap_or_default();
    dt.iter().filter(|(_, t)| **t == topic).filter_map(|(id, _)| by_id.get(id).map(|d| d.meta.ts.clone())).collect()
}

/// Distinct repos and authors in a cluster, most frequent first.
fn cluster_facets(topic: i64, by_id: &HashMap<i64, &Doc>) -> (Vec<String>, Vec<String>) {
    let dt = doc_topics().unwrap_or_default();
    let (mut repos, mut authors): (HashMap<String, usize>, HashMap<String, usize>) = Default::default();
    for (id, t) in &dt {
        if *t != topic {
            continue;
        }
        if let Some(d) = by_id.get(id) {
            if !d.meta.repo.is_empty() {
                *repos.entry(d.meta.repo.clone()).or_default() += 1;
            }
            if !d.meta.author.is_empty() {
                *authors.entry(d.meta.author.clone()).or_default() += 1;
            }
        }
    }
    (by_count(repos), by_count(authors))
}

fn by_count(m: HashMap<String, usize>) -> Vec<String> {
    let mut v: Vec<(String, usize)> = m.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.into_iter().map(|(k, _)| k).collect()
}

fn cluster_mix(topic: i64, by_id: &HashMap<i64, &Doc>) -> (usize, usize, usize) {
    let dt = doc_topics().unwrap_or_default();
    let (mut gh, mut asst, mut prompt) = (0, 0, 0);
    for (id, t) in &dt {
        if *t != topic {
            continue;
        }
        if let Some(d) = by_id.get(id) {
            match d.meta.kind.as_str() {
                "pull_request" | "issue" => gh += 1,
                "assistant_message" | "thinking" => asst += 1,
                "user_prompt" => prompt += 1,
                _ => {}
            }
        }
    }
    (gh, asst, prompt)
}

// ── helpers ────────────────────────────────────────────────────────────────

/// Per-session inputs for the LLM summarizer (ask + keyphrases + on-topic
/// turns). Shares aggregation and keyphrase extraction with `sessions`.
pub fn session_inputs() -> Result<Vec<SessionInput>> {
    let aggs = aggregate();
    let ids: Vec<String> = aggs.keys().cloned().collect();
    let text_refs: Vec<Vec<&str>> = ids.iter().map(|id| aggs[id].texts.iter().map(String::as_str).collect()).collect();
    let keyphrases = crate::keyphrase::labels(&text_refs, 4);
    Ok(ids
        .iter()
        .enumerate()
        .filter(|(_, id)| aggs[*id].prompts > 0)
        .map(|(i, id)| {
            let a = &aggs[id];
            let kps = keyphrases.get(i).cloned().unwrap_or_default();
            SessionInput {
                id: id.clone(),
                repo: if a.repo.is_empty() { "local".into() } else { a.repo.clone() },
                ask: a.ask.clone(),
                files: a.files.iter().take(12).cloned().collect(),
                turns: top_turns(&a.texts, &kps, 8),
                keyphrases: kps,
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

/// The k messages with the most keyphrase coverage, returned in chronological
/// order and each capped generously (so the substance after a preamble survives,
/// not just the generic opener). The material a summarizer reads.
fn top_turns(texts: &[String], kps: &[String], k: usize) -> Vec<String> {
    let lk: Vec<String> = kps.iter().map(|s| s.to_lowercase()).collect();
    let mut scored: Vec<(usize, usize, &String)> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let lt = t.to_lowercase();
            (lk.iter().filter(|k| lt.contains(k.as_str())).count(), i, t)
        })
        .collect();
    // most coverage first; ties toward earlier turns
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut picked: Vec<(usize, &String)> = scored.into_iter().take(k).map(|(_, i, t)| (i, t)).collect();
    // restore chronological order so the model reads the session in sequence
    picked.sort_by_key(|(i, _)| *i);
    picked.into_iter().map(|(_, t)| crate::excerpt(t, 600)).collect()
}

/// The session's most representative line: the message covering the most of its
/// own keyphrases (ties broken toward the more concise one), whitespace-collapsed
/// and capped. An extractive stand-in for an abstractive summary.
fn best_line(texts: &[String], kps: &[String]) -> String {
    if texts.is_empty() {
        return String::new();
    }
    if kps.is_empty() {
        return crate::excerpt(&texts[0], 160);
    }
    let lk: Vec<String> = kps.iter().map(|k| k.to_lowercase()).collect();
    let best = texts.iter().max_by_key(|t| {
        let lt = t.to_lowercase();
        let cov = lk.iter().filter(|k| lt.contains(k.as_str())).count();
        (cov, usize::MAX - t.len()) // most coverage, then shortest
    });
    best.map(|t| crate::excerpt(t, 160)).unwrap_or_default()
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

fn epoch_day(ts: &str) -> Option<i64> {
    let d = chrono::DateTime::parse_from_rfc3339(ts).ok()?;
    Some(d.timestamp() / 86_400)
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
