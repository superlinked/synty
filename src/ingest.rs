// Build corpus/docs.jsonl from two raw sources:
//   corpus/github/{prs,issues}-<repo>.json  (gh JSON arrays)
//   corpus/local/<stream>/**/*.jsonl         (local + pulled canonical envelopes)
// Sessions are chunked per user/assistant/thinking message. Capped by recency.

use crate::{Doc, Meta};
use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const CAP: usize = 12_000;
const MIN_TEXT: usize = 12;
const MAX_TEXT: usize = 6000;

pub fn run(corpus_dir: &str, out_path: &str, bucket: Option<&str>) -> Result<()> {
    run_with_config(corpus_dir, out_path, bucket, &crate::config::load())
}

fn run_with_config(
    corpus_dir: &str,
    out_path: &str,
    bucket: Option<&str>,
    cfg: &crate::config::Config,
) -> Result<()> {
    // Pull every device's events from the shared bucket so this build sees the
    // whole fleet's sessions, not just the local machine's.
    if let Some(b) = bucket {
        let local = Path::new(corpus_dir).join("local");
        match crate::sync::pull_events(b, &local.to_string_lossy()) {
            Ok(n) if n > 0 => eprintln!("ingest: pulled {n} event chunks from {b}/events/"),
            Ok(_) => {}
            Err(e) => eprintln!("ingest: event pull skipped ({e})"),
        }
    }

    // Gather the input file sets first: their (path, mtime, size) manifest is
    // the fast-path key — unchanged inputs mean docs.jsonl is already current,
    // so a polling `up` loop doesn't re-parse the whole corpus every tick.
    let gh_dir = Path::new(corpus_dir).join("github");
    let mut gh_files: Vec<(PathBuf, &'static str, String)> = Vec::new();
    if gh_dir.is_dir() {
        for entry in std::fs::read_dir(&gh_dir)? {
            let p = entry?.path();
            let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if let Some(r) = fname.strip_prefix("prs-").and_then(|s| s.strip_suffix(".json")) {
                gh_files.push((p.clone(), "pull_request", r.to_string()));
            } else if let Some(r) = fname.strip_prefix("issues-").and_then(|s| s.strip_suffix(".json")) {
                gh_files.push((p.clone(), "issue", r.to_string()));
            }
        }
        gh_files.sort();
    }
    let local_dir = Path::new(corpus_dir).join("local");
    let mut local_files: Vec<PathBuf> = Vec::new();
    if local_dir.is_dir() {
        collect_jsonl(&local_dir, &mut local_files)?;
        local_files.sort();
    }

    let manifest_path = format!("{out_path}.manifest");
    let manifest = input_manifest(gh_files.iter().map(|(p, _, _)| p).chain(&local_files), cfg);
    if Path::new(out_path).exists()
        && std::fs::read_to_string(&manifest_path).ok().as_deref() == Some(manifest.as_str())
    {
        eprintln!("ingest up to date ({} inputs unchanged) → {out_path}", gh_files.len() + local_files.len());
        return Ok(());
    }

    let mut docs: Vec<Doc> = Vec::new();
    let mut n_github = 0usize;
    let mut gh_failed = 0usize;
    for (p, kind, repo) in &gh_files {
        let data = std::fs::read_to_string(p)?;
        match github_docs(&data, kind, repo) {
            Ok(g) => {
                n_github += g.len();
                docs.extend(g);
            }
            Err(e) => {
                gh_failed += 1;
                eprintln!("ingest: skipping {}: {e}", p.display());
            }
        }
    }

    // Two passes over the event files — session_start maps first (a session can
    // span files), then docs — holding one file in memory at a time instead of
    // the whole corpus as one string.
    let known: std::collections::HashSet<String> = cfg.repos.iter().cloned().collect();
    let local_actor = crate::identity::actor();
    let mut maps = SessionMaps::default();
    for f in &local_files {
        if let Ok(d) = std::fs::read_to_string(f) {
            scan_starts(&d, &known, &mut maps);
        }
    }
    let mut n_session = 0usize;
    let mut seen = std::collections::HashSet::new();
    for f in &local_files {
        if let Ok(d) = std::fs::read_to_string(f) {
            let before = docs.len();
            docs_from_events(&d, &maps, &local_actor, &mut seen, &mut docs);
            n_session += docs.len() - before;
        }
    }
    let cutoff = cfg
        .capture_since
        .as_deref()
        .and_then(|raw| crate::config::capture_since_ms_from(raw).ok());
    if let Some(cutoff) = cutoff {
        docs.retain(|d| {
            d.meta.source != "agent" || crate::config::captured_at(&d.meta.ts, Some(cutoff))
        });
        n_session = docs.iter().filter(|d| d.meta.source == "agent").count();
    }
    let n_skipped = maps.skipped;

    let total = docs.len();
    let cap = cfg.max_docs.unwrap_or(CAP);
    let (docs, dropped) = cap_by_recency(docs, cap);
    let docs = stable_order(docs, out_path);
    if dropped > 0 {
        eprintln!(
            "ingest: WARNING — recency cap {cap} dropped the {dropped} oldest docs from the index.\n        Raise `max_docs` in .synty/config.json to keep more history."
        );
    }

    let mut out = String::new();
    for d in &docs {
        out.push_str(&serde_json::to_string(d)?);
        out.push('\n');
    }
    if let Some(parent) = Path::new(out_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    crate::write_atomic(out_path, out.as_bytes())?;
    std::fs::write(&manifest_path, &manifest)?;
    let mut m = crate::metrics::Run::new("ingest");
    m.set("github", n_github)
        .set("session", n_session)
        .set("kept", docs.len())
        .set("dropped_oldest", dropped)
        .set("lines_skipped", n_skipped)
        .set("github_files_failed", gh_failed);
    m.emit();
    // Fleet coverage rides on every full ingest: streams and docs are both in
    // hand here, and this is the moment the numbers can change.
    let r = crate::fleet::roster(&docs, &local_dir);
    let mut c = crate::metrics::Run::new("coverage");
    c.set("machines", r.machines.len())
        .set("machines_active", r.active())
        .set("machines_quiet", r.machines.len() - r.active())
        .set("actors_tracked", r.actors_tracked.len())
        .set("gh_active_authors", r.gh_active)
        .set("untracked", r.untracked.len())
        .set("untracked_attributed", r.untracked_attributed())
        .set("install_rate_pct", r.install_rate_pct)
        .set("window_days", crate::fleet::GH_WINDOW_DAYS)
        .set("quiet_days", r.quiet_days);
    c.emit();
    eprintln!(
        "ingest: {n_github} github + {n_session} session = {total} → kept {} (dropped {dropped} oldest) → {out_path}",
        docs.len()
    );
    Ok(())
}

/// Parse a gh JSON array (`gh pr/issue list --json ...`) into docs. A file
/// that fails to parse is an error the caller reports — not an empty vec —
/// so a truncated backfill can't silently drop a whole repo's PRs.
pub fn github_docs(json: &str, kind: &str, repo: &str) -> Result<Vec<Doc>> {
    let arr: Vec<Value> = serde_json::from_str(json)
        .map_err(|e| anyhow::anyhow!("github {kind} file for {repo}: {e}"))?;
    let mut out = Vec::new();
    for it in arr {
        let title = it["title"].as_str().unwrap_or("");
        let body = it["body"].as_str().unwrap_or("");
        let text = format!("{title}\n\n{body}");
        if text.trim().is_empty() {
            continue;
        }
        let labels = it["labels"]
            .as_array()
            .map(|a| a.iter().filter_map(|l| l["name"].as_str().map(String::from)).collect())
            .unwrap_or_default();
        let author = it["author"]["login"].as_str().unwrap_or("");
        out.push(Doc {
            id: 0,
            // Detect on the full text — trunc may cut a trailing footer.
            meta: Meta {
                source: "github".into(),
                kind: kind.into(),
                repo: repo.into(),
                author: author.into(),
                session_id: String::new(),
                campaign_id: String::new(),
                campaign_role: String::new(),
                backend: String::new(),
                capture_source: String::new(),
                ts: it["createdAt"].as_str().unwrap_or("").into(),
                number: it["number"].as_i64(),
                url: it["url"].as_str().map(String::from),
                state: it["state"].as_str().map(String::from),
                labels,
                agent_attr: crate::fleet::detect_agent(&text, author).map(String::from),
            },
            text: trunc(&text, MAX_TEXT),
        });
    }
    Ok(out)
}

/// Per-session context collected from session_start envelopes, plus the count
/// of malformed lines (surfaced in metrics — a corrupt envelope is data loss,
/// not noise).
#[derive(Default)]
pub struct SessionMaps {
    repo: HashMap<String, String>,
    actor: HashMap<String, String>,
    campaign: HashMap<String, String>,
    role: HashMap<String, String>,
    backend: HashMap<String, String>,
    capture_source: HashMap<String, String>,
    pub skipped: usize,
}

/// First pass: collect session→repo and session→actor from session_start
/// envelopes (a session's start and its messages may sit in different files).
pub fn scan_starts(text: &str, known: &std::collections::HashSet<String>, maps: &mut SessionMaps) {
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(line) {
            Ok(v) => {
                if v["kind"].as_str() == Some("session_start") {
                    if let (Some(sid), Some(cwd)) =
                        (v["session_id"].as_str(), v["payload"]["cwd"].as_str())
                    {
                        maps.repo.insert(sid.to_string(), crate::units::resolve_repo(cwd, known));
                    }
                    if let (Some(sid), Some(a)) =
                        (v["session_id"].as_str(), v["payload"]["actor"].as_str())
                    {
                        if !a.is_empty() {
                            maps.actor.insert(sid.to_string(), a.to_string());
                        }
                    }
                    if let Some(sid) = v["session_id"].as_str() {
                        let campaign = v["rollup_dim"]
                            .as_str()
                            .filter(|value| !value.is_empty())
                            .or_else(|| v["payload"]["campaign_id"].as_str());
                        if let Some(campaign) = campaign {
                            maps.campaign.insert(sid.to_string(), campaign.to_string());
                        }
                        if let Some(role) = v["payload"]["campaign_role"]
                            .as_str()
                            .filter(|value| !value.is_empty())
                        {
                            maps.role.insert(sid.to_string(), role.to_string());
                        }
                        if let Some(backend) =
                            v["payload"]["backend"].as_str().filter(|value| !value.is_empty())
                        {
                            maps.backend.insert(sid.to_string(), backend.to_string());
                        }
                        if let Some(source) = v["source"].as_str().filter(|value| !value.is_empty()) {
                            maps.capture_source.insert(sid.to_string(), source.to_string());
                        }
                    }
                }
            }
            Err(_) => maps.skipped += 1,
        }
    }
}

/// Second pass: one doc per user/assistant/thinking message, each event taken
/// once (`seen` dedups re-emissions by overlapping trackers across the whole
/// pass). Sessions are attributed to the actor their tracker stamped into
/// session_start — a build pulls many machines' streams from the bucket, and
/// each session belongs to its author, not to whoever runs the build. Events
/// from before actor stamping fall back to `local_actor`.
pub fn docs_from_events(
    text: &str,
    maps: &SessionMaps,
    local_actor: &str,
    seen: &mut std::collections::HashSet<u64>,
    out: &mut Vec<Doc>,
) {
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<Value>(line) else { continue };
        if !crate::event::first_sighting(seen, v["event_id"].as_str().unwrap_or("")) {
            continue;
        }
        let kind = v["kind"].as_str().unwrap_or("");
        if !matches!(kind, "user_prompt" | "assistant_message" | "thinking") {
            continue;
        }
        let text = v["payload"]["text"].as_str().unwrap_or("").trim();
        if text.len() < MIN_TEXT || is_noise(text) {
            continue;
        }
        let sid = v["session_id"].as_str().unwrap_or("").to_string();
        let repo = maps.repo.get(&sid).cloned().unwrap_or_default();
        let author = maps.actor.get(&sid).cloned().unwrap_or_else(|| local_actor.to_string());
        let campaign_id = maps.campaign.get(&sid).cloned().unwrap_or_default();
        let campaign_role = maps.role.get(&sid).cloned().unwrap_or_default();
        let backend = maps.backend.get(&sid).cloned().unwrap_or_default();
        let capture_source = maps.capture_source.get(&sid).cloned().unwrap_or_default();
        out.push(Doc {
            id: 0,
            text: trunc(text, MAX_TEXT),
            meta: Meta {
                source: "agent".into(),
                kind: kind.into(),
                repo,
                author,
                session_id: sid,
                campaign_id,
                campaign_role,
                backend,
                capture_source,
                ts: v["ts"].as_str().unwrap_or("").into(),
                number: None,
                url: None,
                state: None,
                labels: vec![],
                agent_attr: None,
            },
        });
    }
}

/// Parse newline-joined v1 envelopes into one doc per user/assistant/thinking
/// message (both passes over one buffer — the scenario-test surface).
#[cfg(test)]
pub fn session_docs(text: &str, known: &std::collections::HashSet<String>) -> (Vec<Doc>, usize) {
    let mut maps = SessionMaps::default();
    scan_starts(text, known, &mut maps);
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    docs_from_events(text, &maps, &crate::identity::actor(), &mut seen, &mut out);
    (out, maps.skipped)
}

/// The fast-path key: every input file's (path, mtime_ms, size), one per line,
/// plus the config that steers derivation. Equal manifest → equal docs.jsonl.
fn input_manifest<'a>(
    files: impl Iterator<Item = &'a PathBuf>,
    cfg: &crate::config::Config,
) -> String {
    let mut out = String::new();
    for p in files {
        let (mtime, size) = std::fs::metadata(p)
            .map(|m| {
                let ms = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_millis() as i64).unwrap_or(0);
                (ms, m.len())
            })
            .unwrap_or((0, 0));
        out.push_str(&format!("{}\t{mtime}\t{size}\n", p.display()));
    }
    out.push_str(&derivation_config(cfg));
    out
}

fn derivation_config(cfg: &crate::config::Config) -> String {
    let mut out = String::new();
    let mut repos = cfg.repos.clone();
    repos.sort();
    out.push_str(&format!("repos={}\n", repos.join(",")));
    out.push_str(&format!("cap={}\n", cfg.max_docs.unwrap_or(CAP)));
    out.push_str(&format!(
        "capture_since={}\n",
        cfg.capture_since.as_deref().unwrap_or("")
    ));
    // Doc-derivation format: bump whenever the derivation itself changes (new
    // Meta fields, new detectors) so a binary upgrade regenerates docs.jsonl
    // once even though the input files are unchanged.
    out.push_str("fmt=4\n");
    out
}

/// Keep the most recent docs (ISO8601 ts sorts lexicographically), ordered
/// oldest-first with sequential ids. The order is then made stable against the
/// previous output by `stable_order` — recency only decides what is KEPT.
/// Crossing the cap cuts 10% deeper than it has to: head-drops are patched
/// out of the index incrementally, but dropping to exactly `cap` would patch
/// on every single build once the corpus rides the cap — the hysteresis
/// amortizes that churn to once per ~cap/10 docs of new work.
/// Returns (kept, dropped_count).
pub fn cap_by_recency(mut docs: Vec<Doc>, cap: usize) -> (Vec<Doc>, usize) {
    docs.sort_by(|a, b| b.meta.ts.cmp(&a.meta.ts));
    let keep = if docs.len() > cap { cap - cap / 10 } else { docs.len() };
    let dropped = docs.len() - keep;
    docs.truncate(keep);
    docs.reverse();
    for (i, d) in docs.iter_mut().enumerate() {
        d.id = i as i64;
    }
    (docs, dropped)
}

/// Order docs so the previous docs.jsonl stays a strict prefix: a doc whose
/// text already appears there keeps that file's position (with its fresh
/// meta), and novel docs append at the tail, oldest first. Pure timestamp
/// order broke `index`'s append path on every active day — a doc routinely
/// ARRIVES later than its timestamp (a session tailed hours after its
/// messages, a backfilled PR), and one mid-corpus insertion forced minutes of
/// full PLAID re-quantization where appending the tail takes seconds. A
/// vanished text (edited PR body, recency-cap drop) still leaves a gap and
/// rebuilds in full — rare, and correct. Matching is a multiset on the text
/// hash, so identical one-liners ("ok") collapse to the right slots.
fn stable_order(docs: Vec<Doc>, prev_path: &str) -> Vec<Doc> {
    let prev = match crate::load_docs(prev_path) {
        Ok(p) if !p.is_empty() => p,
        _ => return docs,
    };
    let mut slots: Vec<Option<Doc>> = docs.into_iter().map(Some).collect();
    let mut by_hash: HashMap<u64, std::collections::VecDeque<usize>> = HashMap::new();
    for (i, d) in slots.iter().enumerate() {
        let h = crate::index::fnv1a(d.as_ref().expect("just filled").text.as_bytes());
        by_hash.entry(h).or_default().push_back(i);
    }
    let mut out: Vec<Doc> = Vec::with_capacity(slots.len());
    let (mut missed, mut sample) = (0usize, String::new());
    for p in &prev {
        match by_hash.get_mut(&crate::index::fnv1a(p.text.as_bytes())).and_then(|q| q.pop_front()) {
            Some(i) => out.push(slots[i].take().expect("each slot consumed once")),
            None => {
                // A previously-indexed doc left the corpus — the next `index`
                // patches it out of the cloned build (a full rebuild only if
                // the churn is heavy). Say which docs did it, so a recurring
                // shrink (edited bodies, vanished sources) stays diagnosable.
                missed += 1;
                if sample.is_empty() {
                    sample = crate::excerpt(&p.text, 60);
                }
            }
        }
    }
    if missed > 0 {
        eprintln!("ingest: {missed} previously-indexed doc(s) left the corpus → the next index patches them out (e.g. {sample:?})");
    }
    for s in &mut slots {
        if let Some(d) = s.take() {
            out.push(d); // novel docs, still oldest-first among themselves
        }
    }
    for (i, d) in out.iter_mut().enumerate() {
        d.id = i as i64;
    }
    out
}

/// System-injected pseudo-prompts (hook echoes, tool output, reminders) carry
/// no user intent and pollute retrieval/clustering.
pub(crate) fn is_noise(t: &str) -> bool {
    const MARKERS: &[&str] = &[
        "<task-notification", "<bash-input", "<bash-stdout", "<bash-stderr",
        "<command-", "<local-command", "<system-reminder", "<user-prompt-submit-hook",
    ];
    let t = t.trim_start();
    MARKERS.iter().any(|m| t.starts_with(m))
}

fn trunc(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect()
    }
}

fn collect_jsonl(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            collect_jsonl(&p, out)?;
        } else if p.extension().and_then(|s| s.to_str()) == Some("jsonl") {
            out.push(p);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // A reviewer pulls a PR; it should become exactly one doc carrying the
    // repo, number, author, labels, and title-first text.
    #[test]
    fn github_pr_becomes_one_doc_with_metadata() {
        let json = r#"[{"number":53,"title":"feat: replace docs PR","body":"Replaces #698 in wrong repo.","author":{"login":"alice"},"url":"https://gh/53","labels":[{"name":"P2"},{"name":"docs"}],"state":"OPEN","createdAt":"2026-05-01T10:00:00Z"}]"#;
        let docs = github_docs(json, "pull_request", "sie-web").unwrap();
        assert_eq!(docs.len(), 1);
        let d = &docs[0];
        assert!(d.text.starts_with("feat: replace docs PR"));
        assert_eq!(d.meta.repo, "sie-web");
        assert_eq!(d.meta.kind, "pull_request");
        assert_eq!(d.meta.number, Some(53));
        assert_eq!(d.meta.author, "alice");
        assert_eq!(d.meta.labels, vec!["P2", "docs"]);
        assert_eq!(d.meta.source, "github");
        assert_eq!(d.meta.agent_attr, None, "a clean human PR carries no attribution");
    }

    // A PR whose body credits an agent (trailer or footer) is flagged, so the
    // fleet roster can spot agent users no tracker has seen.
    #[test]
    fn github_doc_carries_agent_attribution() {
        let json = r#"[{"number":9,"title":"fix race","body":"Serialize the tailer.\n\nCo-authored-by: Claude <noreply@anthropic.com>","author":{"login":"bob"},"createdAt":"2026-06-01T10:00:00Z"}]"#;
        let docs = github_docs(json, "pull_request", "sie").unwrap();
        assert_eq!(docs[0].meta.agent_attr.as_deref(), Some("claude"));
    }

    // An empty PR (no title/body) carries no signal and should be skipped.
    #[test]
    fn empty_github_item_is_dropped() {
        let json = r#"[{"number":1,"title":"","body":"","author":{"login":"bot"}}]"#;
        assert!(github_docs(json, "issue", "sie").unwrap().is_empty());
    }

    // A session: cwd maps to a repo; prompts/messages become docs; an "ok"
    // ack below the length floor is dropped.
    #[test]
    fn session_messages_become_docs_with_repo_from_cwd() {
        let lines = [
            r#"{"kind":"session_start","source":"harness","session_id":"S1","rollup_dim":"campaign-42","payload":{"cwd":"/Users/d/c/sie-internal/x","campaign_role":"investigator","backend":"codex"}}"#,
            r#"{"kind":"user_prompt","session_id":"S1","ts":"2026-05-02T09:00:00Z","payload":{"text":"fix the login redirect bug"}}"#,
            r#"{"kind":"assistant_message","session_id":"S1","ts":"2026-05-02T09:01:00Z","payload":{"text":"I'll inspect auth.ts and the redirect handler"}}"#,
            r#"{"kind":"assistant_message","session_id":"S1","ts":"2026-05-02T09:02:00Z","payload":{"text":"ok"}}"#,
            r#"{"kind":"tool_call","session_id":"S1","payload":{"name":"Read"}}"#,
        ]
        .join("\n");
        let known: std::collections::HashSet<String> = ["sie-internal".to_string()].into_iter().collect();
        let (docs, _) = session_docs(&lines, &known);
        assert_eq!(docs.len(), 2); // two real messages; "ok" and tool_call excluded
        assert!(docs.iter().all(|d| d.meta.session_id == "S1"));
        assert!(docs.iter().all(|d| d.meta.repo == "sie-internal"));
        assert!(docs.iter().all(|d| d.meta.source == "agent"));
        assert!(docs.iter().all(|d| d.meta.campaign_id == "campaign-42"));
        assert!(docs.iter().all(|d| d.meta.campaign_role == "investigator"));
        assert!(docs.iter().all(|d| d.meta.backend == "codex"));
        assert!(docs.iter().all(|d| d.meta.capture_source == "harness"));
    }

    // Starting collection at an absolute boundary removes older local session
    // text from the user-visible corpus without applying that privacy gate to
    // the separately configured GitHub history.
    #[test]
    fn ingest_keeps_only_post_boundary_agent_docs() {
        let dir = std::env::temp_dir().join(format!("synty-ingest-cutoff-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let corpus = dir.join("corpus");
        let stream = corpus.join("local/edge-dev-codex");
        let github = corpus.join("github");
        std::fs::create_dir_all(&stream).unwrap();
        std::fs::create_dir_all(&github).unwrap();
        std::fs::write(stream.join("track.jsonl"), concat!(
            r#"{"event_id":"start","kind":"session_start","session_id":"S1","ts":"2026-07-20T23:50:00Z","payload":{"cwd":"/work/sie"}}"#, "\n",
            r#"{"event_id":"old","kind":"user_prompt","session_id":"S1","ts":"2026-07-20T23:55:00Z","payload":{"text":"old private agent prompt"}}"#, "\n",
            r#"{"event_id":"new","kind":"user_prompt","session_id":"S1","ts":"2026-07-21T00:05:00Z","payload":{"text":"new retained agent prompt"}}"#, "\n",
        )).unwrap();
        std::fs::write(
            github.join("prs-sie.json"),
            r#"[{"number":7,"title":"older GitHub work remains visible","body":"A team source, not a local capture.","author":{"login":"alice"},"createdAt":"2026-07-01T00:00:00Z"}]"#,
        ).unwrap();
        let cfg = crate::config::Config {
            capture_since: Some("2026-07-21T00:00:00Z".into()),
            ..Default::default()
        };
        let out = corpus.join("docs.jsonl");
        run_with_config(
            corpus.to_str().unwrap(),
            out.to_str().unwrap(),
            None,
            &cfg,
        ).unwrap();
        let body = std::fs::read_to_string(out).unwrap();
        assert!(body.contains("new retained agent prompt"), "{body}");
        assert!(!body.contains("old private agent prompt"), "{body}");
        assert!(body.contains("older GitHub work remains visible"), "{body}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Unchanged inputs skip the rebuild entirely (the `up` loop polls ingest;
    // it must not re-parse the corpus every tick), and any input change undoes
    // the skip.
    #[test]
    fn unchanged_inputs_skip_the_rebuild() {
        let dir = std::env::temp_dir().join(format!("synty-ingest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let stream = dir.join("corpus/local/edge-t-claudecode");
        std::fs::create_dir_all(&stream).unwrap();
        let log = stream.join("track.jsonl");
        std::fs::write(&log, concat!(
            r#"{"kind":"session_start","session_id":"S1","payload":{"cwd":"/x"}}"#, "\n",
            r#"{"kind":"user_prompt","session_id":"S1","ts":"2026-05-02T09:00:00Z","payload":{"text":"a real first prompt"}}"#, "\n",
        )).unwrap();
        let corpus = dir.join("corpus");
        let out = dir.join("corpus/docs.jsonl");
        let (corpus_s, out_s) = (corpus.to_string_lossy().into_owned(), out.to_string_lossy().into_owned());

        run(&corpus_s, &out_s, None).unwrap();
        assert!(std::fs::read_to_string(&out).unwrap().contains("a real first prompt"));

        // Second run with identical inputs must not rewrite docs.jsonl.
        std::fs::write(&out, "SENTINEL").unwrap();
        run(&corpus_s, &out_s, None).unwrap();
        assert_eq!(std::fs::read_to_string(&out).unwrap(), "SENTINEL", "unchanged inputs must skip");

        // An input change (the session grows) rebuilds.
        let mut f = std::fs::OpenOptions::new().append(true).open(&log).unwrap();
        std::io::Write::write_all(&mut f, format!("{}\n",
            r#"{"kind":"user_prompt","session_id":"S1","ts":"2026-05-02T09:05:00Z","payload":{"text":"a follow-up prompt arrives"}}"#).as_bytes()).unwrap();
        run(&corpus_s, &out_s, None).unwrap();
        assert!(std::fs::read_to_string(&out).unwrap().contains("a follow-up prompt arrives"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn capture_boundary_is_part_of_the_ingest_fast_path_key() {
        let before = crate::config::Config::default();
        let after = crate::config::Config {
            capture_since: Some("2026-07-21T00:00:00Z".into()),
            ..Default::default()
        };
        assert_ne!(derivation_config(&before), derivation_config(&after));
        assert!(derivation_config(&after).contains("capture_since=2026-07-21T00:00:00Z"));
    }

    // Overlapping trackers (autostart + a manual run) append the same envelope
    // twice under its deterministic id — readers must count it once.
    #[test]
    fn duplicated_envelopes_become_one_doc() {
        let line = r#"{"event_id":"01ABC","kind":"user_prompt","session_id":"S9","ts":"t","payload":{"text":"a prompt emitted twice by two trackers"}}"#;
        let (docs, _) = session_docs(&format!("{line}\n{line}"), &Default::default());
        assert_eq!(docs.len(), 1);
    }

    // A corrupt envelope line (truncated write, disk hiccup) is counted, not
    // silently dropped — and the lines around it still parse.
    #[test]
    fn malformed_envelope_lines_are_counted_not_fatal() {
        let lines = [
            r#"{"kind":"session_start","session_id":"S8","payload":{"cwd":"/x"}}"#,
            r#"{"kind":"user_prompt","session_id":"S8","ts":"t","pay"#, // truncated
            r#"{"kind":"user_prompt","session_id":"S8","ts":"t","payload":{"text":"still here after the corruption"}}"#,
        ]
        .join("\n");
        let (docs, skipped) = session_docs(&lines, &Default::default());
        assert_eq!(skipped, 1);
        assert_eq!(docs.len(), 1);
        assert!(docs[0].text.contains("still here"));
    }

    // A github backfill file that fails to parse is an error, not an empty vec.
    #[test]
    fn malformed_github_file_is_an_error() {
        assert!(github_docs("[{\"truncated\":", "pull_request", "sie").is_err());
    }

    // A session tracked on another machine carries its author in the
    // session_start actor stamp; the build must credit that author, not the
    // identity of whoever runs the build.
    #[test]
    fn session_actor_stamp_wins_over_build_identity() {
        let lines = [
            r#"{"kind":"session_start","session_id":"S7","payload":{"cwd":"/x","actor":"alice"}}"#,
            r#"{"kind":"user_prompt","session_id":"S7","ts":"t","payload":{"text":"a real prompt from alice"}}"#,
        ]
        .join("\n");
        let (docs, _) = session_docs(&lines, &Default::default());
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].meta.author, "alice");
    }

    // A cwd outside any workspace dir has no repo — empty, not a fake "local".
    #[test]
    fn session_unknown_cwd_has_no_repo() {
        let lines = [
            r#"{"kind":"session_start","session_id":"S2","payload":{"cwd":"/tmp/scratch"}}"#,
            r#"{"kind":"user_prompt","session_id":"S2","ts":"t","payload":{"text":"some real question here"}}"#,
        ]
        .join("\n");
        let (docs, _) = session_docs(&lines, &Default::default());
        assert_eq!(docs[0].meta.repo, "");
    }

    // Hook echoes and tool output injected as user turns are not real prompts.
    #[test]
    fn system_injected_prompts_are_dropped() {
        let lines = [
            r#"{"kind":"session_start","session_id":"S3","payload":{"cwd":"/x"}}"#,
            r#"{"kind":"user_prompt","session_id":"S3","ts":"t","payload":{"text":"<task-notification> bg done"}}"#,
            r#"{"kind":"user_prompt","session_id":"S3","ts":"t","payload":{"text":"<bash-stdout>ok</bash-stdout>"}}"#,
            r#"{"kind":"user_prompt","session_id":"S3","ts":"t","payload":{"text":"please refactor the auth module"}}"#,
        ]
        .join("\n");
        let (docs, _) = session_docs(&lines, &Default::default());
        assert_eq!(docs.len(), 1);
        assert!(docs[0].text.starts_with("please refactor"));
    }

    fn doc(ts: &str, text: &str) -> Doc {
        Doc {
            id: 0,
            text: text.into(),
            meta: Meta {
                source: "github".into(),
                kind: "issue".into(),
                repo: "r".into(),
                author: String::new(),
                session_id: String::new(),
                campaign_id: String::new(),
                campaign_role: String::new(),
                backend: String::new(),
                capture_source: String::new(),
                ts: ts.into(),
                number: None,
                url: None,
                state: None,
                labels: vec![],
                agent_attr: None,
            },
        }
    }

    fn write_docs(path: &Path, docs: &[Doc]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let out: String = docs.iter().map(|d| serde_json::to_string(d).unwrap() + "\n").collect();
        std::fs::write(path, out).unwrap();
    }

    // A doc that ARRIVES late (a session tailed hours after its timestamps, a
    // backfilled PR) must append at the tail, not insert mid-corpus by its
    // timestamp — one insertion forces `index` into a full rebuild.
    #[test]
    fn late_arrival_appends_instead_of_inserting() {
        let dir = std::env::temp_dir().join(format!("synty-order-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let prev_path = dir.join("docs.jsonl");
        write_docs(&prev_path, &[doc("2026-06-01", "alpha"), doc("2026-06-03", "gamma")]);
        // The new derivation knows a doc timestamped BETWEEN the existing two.
        let fresh = vec![doc("2026-06-01", "alpha"), doc("2026-06-02", "beta"), doc("2026-06-03", "gamma")];
        let out = stable_order(fresh, &prev_path.to_string_lossy());
        let texts: Vec<&str> = out.iter().map(|d| d.text.as_str()).collect();
        assert_eq!(texts, ["alpha", "gamma", "beta"], "previous order kept, late arrival at the tail");
        assert_eq!(out.iter().map(|d| d.id).collect::<Vec<_>>(), [0, 1, 2]);
        // …and identical texts match as a multiset, not first-wins-forever.
        write_docs(&prev_path, &[doc("2026-06-01", "ok"), doc("2026-06-02", "ok")]);
        let out = stable_order(vec![doc("2026-06-01", "ok"), doc("2026-06-02", "ok"), doc("2026-06-03", "ok")], &prev_path.to_string_lossy());
        assert_eq!(out.len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // An edited doc vacates its old slot (the gap correctly forces one full
    // rebuild) and re-enters at the tail; without a previous file the
    // timestamp order passes through untouched.
    #[test]
    fn edited_doc_moves_to_tail_and_no_prev_is_passthrough() {
        let dir = std::env::temp_dir().join(format!("synty-order2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let prev_path = dir.join("docs.jsonl");
        write_docs(&prev_path, &[doc("2026-06-01", "alpha"), doc("2026-06-02", "beta"), doc("2026-06-03", "gamma")]);
        let fresh = vec![doc("2026-06-01", "alpha"), doc("2026-06-02", "beta EDITED"), doc("2026-06-03", "gamma")];
        let out = stable_order(fresh, &prev_path.to_string_lossy());
        let texts: Vec<&str> = out.iter().map(|d| d.text.as_str()).collect();
        assert_eq!(texts, ["alpha", "gamma", "beta EDITED"]);
        let fresh = vec![doc("2026-06-01", "alpha"), doc("2026-06-02", "beta")];
        let out = stable_order(fresh, &dir.join("absent.jsonl").to_string_lossy());
        assert_eq!(out.iter().map(|d| d.text.as_str()).collect::<Vec<_>>(), ["alpha", "beta"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Capping keeps the newest and renumbers ids from 0.
    #[test]
    fn cap_keeps_most_recent_and_assigns_ids() {
        let mk = |ts: &str| Doc {
            id: 99,
            text: "t".into(),
            meta: Meta {
                source: "github".into(),
                kind: "issue".into(),
                repo: "r".into(),
                author: String::new(),
                session_id: String::new(),
                campaign_id: String::new(),
                campaign_role: String::new(),
                backend: String::new(),
                capture_source: String::new(),
                ts: ts.into(),
                number: None,
                url: None,
                state: None,
                labels: vec![],
                agent_attr: None,
            },
        };
        let (kept, dropped) = cap_by_recency(
            vec![mk("2026-01-01"), mk("2026-03-01"), mk("2026-02-01")],
            2,
        );
        assert_eq!(dropped, 1);
        assert_eq!(kept.len(), 2);
        // The most recent docs survive, ordered oldest-first so new work
        // appends at the tail (incremental index updates depend on this).
        assert_eq!(kept[0].meta.ts, "2026-02-01");
        assert_eq!(kept[1].meta.ts, "2026-03-01");
        assert_eq!(kept[0].id, 0);
        assert_eq!(kept[1].id, 1);
        // At scale, crossing the cap cuts 10% deeper (hysteresis): a head-drop
        // costs a full re-quantization, so it must not repeat every build.
        let many: Vec<Doc> = (0..120).map(|i| mk(&format!("2026-01-{:02}T{:02}:00:00Z", 1 + i / 24, i % 24))).collect();
        let (kept, dropped) = cap_by_recency(many, 100);
        assert_eq!((kept.len(), dropped), (90, 30));
    }
}
