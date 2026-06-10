// Build corpus/docs.jsonl from two raw sources:
//   corpus/github/{prs,issues}-<repo>.json  (gh JSON arrays)
//   corpus/local/<stream>/<hour>.jsonl      (v1 synty-agent --out envelopes)
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
    // Pull every device's events from the shared bucket so this build sees the
    // whole fleet's sessions, not just the local machine's.
    if let Some(b) = bucket {
        let local = Path::new(corpus_dir).join("local");
        match crate::sync::pull_events(b, &local.to_string_lossy()) {
            Ok(n) if n > 0 => eprintln!("ingest: pulled {n} event files from {b}/events/"),
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
    let manifest = input_manifest(gh_files.iter().map(|(p, _, _)| p).chain(&local_files));
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
    let known: std::collections::HashSet<String> = crate::config::load().repos.into_iter().collect();
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
    let n_skipped = maps.skipped;

    let total = docs.len();
    let cap = crate::config::load().max_docs.unwrap_or(CAP);
    let (docs, dropped) = cap_by_recency(docs, cap);
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
        out.push(Doc {
            id: 0,
            text: trunc(&text, MAX_TEXT),
            meta: Meta {
                source: "github".into(),
                kind: kind.into(),
                repo: repo.into(),
                author: it["author"]["login"].as_str().unwrap_or("").into(),
                session_id: String::new(),
                ts: it["createdAt"].as_str().unwrap_or("").into(),
                number: it["number"].as_i64(),
                url: it["url"].as_str().map(String::from),
                state: it["state"].as_str().map(String::from),
                labels,
            },
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
        out.push(Doc {
            id: 0,
            text: trunc(text, MAX_TEXT),
            meta: Meta {
                source: "agent".into(),
                kind: kind.into(),
                repo,
                author,
                session_id: sid,
                ts: v["ts"].as_str().unwrap_or("").into(),
                number: None,
                url: None,
                state: None,
                labels: vec![],
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
/// plus the config that steers repo folding. Equal manifest → equal docs.jsonl.
fn input_manifest<'a>(files: impl Iterator<Item = &'a PathBuf>) -> String {
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
    let cfg = crate::config::load();
    let mut repos = cfg.repos;
    repos.sort();
    out.push_str(&format!("repos={}\n", repos.join(",")));
    out.push_str(&format!("cap={}\n", cfg.max_docs.unwrap_or(CAP)));
    out
}

/// Keep the `cap` most recent docs (ISO8601 ts sorts lexicographically) and
/// assign sequential ids, ordered oldest-first — so new work lands at the tail
/// and `index` can append to the existing index instead of rebuilding.
/// Returns (kept, dropped_count).
pub fn cap_by_recency(mut docs: Vec<Doc>, cap: usize) -> (Vec<Doc>, usize) {
    docs.sort_by(|a, b| b.meta.ts.cmp(&a.meta.ts));
    let dropped = docs.len().saturating_sub(cap);
    docs.truncate(cap);
    docs.reverse();
    for (i, d) in docs.iter_mut().enumerate() {
        d.id = i as i64;
    }
    (docs, dropped)
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
            r#"{"kind":"session_start","session_id":"S1","payload":{"cwd":"/Users/d/c/sie-internal/x"}}"#,
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
                ts: ts.into(),
                number: None,
                url: None,
                state: None,
                labels: vec![],
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
    }
}
