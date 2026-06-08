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
const KNOWN_REPOS: &[&str] = &[
    "sie-internal", "sie-web", "sie-perf-lab", "sie-presentation", "sie",
    "infrastructure", "gtm-intel", "terraform-google-sie", "terraform-aws-sie",
    "VectorHub", "agents", "brave-new-demos",
];

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

    let mut docs: Vec<Doc> = Vec::new();
    let mut n_github = 0usize;

    let gh_dir = Path::new(corpus_dir).join("github");
    if gh_dir.is_dir() {
        for entry in std::fs::read_dir(&gh_dir)? {
            let p = entry?.path();
            let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let (kind, repo) = if let Some(r) = fname.strip_prefix("prs-").and_then(|s| s.strip_suffix(".json")) {
                ("pull_request", r.to_string())
            } else if let Some(r) = fname.strip_prefix("issues-").and_then(|s| s.strip_suffix(".json")) {
                ("issue", r.to_string())
            } else {
                continue;
            };
            let data = std::fs::read_to_string(&p)?;
            let g = github_docs(&data, kind, &repo);
            n_github += g.len();
            docs.extend(g);
        }
    }

    let local_dir = Path::new(corpus_dir).join("local");
    let mut n_session = 0usize;
    if local_dir.is_dir() {
        let mut files = Vec::new();
        collect_jsonl(&local_dir, &mut files)?;
        let mut all = String::new();
        for f in files {
            if let Ok(d) = std::fs::read_to_string(&f) {
                all.push_str(&d);
                all.push('\n');
            }
        }
        let s = session_docs(&all);
        n_session = s.len();
        docs.extend(s);
    }

    let total = docs.len();
    let (docs, dropped) = cap_by_recency(docs, CAP);

    let mut out = String::new();
    for d in &docs {
        out.push_str(&serde_json::to_string(d)?);
        out.push('\n');
    }
    if let Some(parent) = Path::new(out_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out_path, out)?;
    eprintln!(
        "ingest: {n_github} github + {n_session} session = {total} → kept {} (dropped {dropped} oldest) → {out_path}",
        docs.len()
    );
    Ok(())
}

/// Parse a gh JSON array (`gh pr/issue list --json ...`) into docs.
pub fn github_docs(json: &str, kind: &str, repo: &str) -> Vec<Doc> {
    let arr: Vec<Value> = serde_json::from_str(json).unwrap_or_default();
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
    out
}

/// Parse newline-joined v1 envelopes into one doc per user/assistant/thinking
/// message. `session_start.cwd` maps a session to a repo.
pub fn session_docs(text: &str) -> Vec<Doc> {
    let mut session_repo: HashMap<String, String> = HashMap::new();
    let mut evs: Vec<Value> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            if v["kind"].as_str() == Some("session_start") {
                if let (Some(sid), Some(cwd)) =
                    (v["session_id"].as_str(), v["payload"]["cwd"].as_str())
                {
                    session_repo.insert(sid.to_string(), repo_from_cwd(cwd));
                }
            }
            evs.push(v);
        }
    }
    // One identity per build — sessions on this machine are attributed to its
    // resolved actor (GitHub login if pinned, else git email / $USER), so they
    // merge with that person's GitHub PRs instead of a hardcoded name.
    let actor = crate::identity::actor();
    let mut out = Vec::new();
    for v in evs {
        let kind = v["kind"].as_str().unwrap_or("");
        if !matches!(kind, "user_prompt" | "assistant_message" | "thinking") {
            continue;
        }
        let text = v["payload"]["text"].as_str().unwrap_or("").trim();
        if text.len() < MIN_TEXT || is_noise(text) {
            continue;
        }
        let sid = v["session_id"].as_str().unwrap_or("").to_string();
        let repo = session_repo.get(&sid).cloned().unwrap_or_else(|| "local".into());
        out.push(Doc {
            id: 0,
            text: trunc(text, MAX_TEXT),
            meta: Meta {
                source: "agent".into(),
                kind: kind.into(),
                repo,
                author: actor.clone(),
                session_id: sid,
                ts: v["ts"].as_str().unwrap_or("").into(),
                number: None,
                url: None,
                state: None,
                labels: vec![],
            },
        });
    }
    out
}

/// Keep the `cap` most recent docs (ISO8601 ts sorts lexicographically) and
/// assign sequential ids. Returns (kept, dropped_count).
pub fn cap_by_recency(mut docs: Vec<Doc>, cap: usize) -> (Vec<Doc>, usize) {
    docs.sort_by(|a, b| b.meta.ts.cmp(&a.meta.ts));
    let dropped = docs.len().saturating_sub(cap);
    docs.truncate(cap);
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

fn repo_from_cwd(cwd: &str) -> String {
    KNOWN_REPOS
        .iter()
        .find(|r| cwd.contains(&format!("/{r}")))
        .map(|r| (*r).to_string())
        .unwrap_or_else(|| "local".into())
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
        let docs = github_docs(json, "pull_request", "sie-web");
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
        assert!(github_docs(json, "issue", "sie").is_empty());
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
        let docs = session_docs(&lines);
        assert_eq!(docs.len(), 2); // two real messages; "ok" and tool_call excluded
        assert!(docs.iter().all(|d| d.meta.session_id == "S1"));
        assert!(docs.iter().all(|d| d.meta.repo == "sie-internal"));
        assert!(docs.iter().all(|d| d.meta.source == "agent"));
    }

    // Unknown cwd falls back to "local" rather than guessing.
    #[test]
    fn session_unknown_cwd_is_local() {
        let lines = [
            r#"{"kind":"session_start","session_id":"S2","payload":{"cwd":"/tmp/scratch"}}"#,
            r#"{"kind":"user_prompt","session_id":"S2","ts":"t","payload":{"text":"some real question here"}}"#,
        ]
        .join("\n");
        let docs = session_docs(&lines);
        assert_eq!(docs[0].meta.repo, "local");
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
        let docs = session_docs(&lines);
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
        assert_eq!(kept[0].meta.ts, "2026-03-01"); // newest first
        assert_eq!(kept[1].meta.ts, "2026-02-01");
        assert_eq!(kept[0].id, 0);
        assert_eq!(kept[1].id, 1);
    }
}
