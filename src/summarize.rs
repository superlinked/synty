// Extractive summaries, no LLM.
//  - Sessions: re-parse corpus/local; per session report repo, prompt count,
//    files touched (Write/Edit attachment_refs), the opening ask, linked PRs.
//  - Topics: read clusters.json + docs.jsonl; per cluster report repos,
//    kind counts, and notable titles. Falls back to per-repo GitHub digests.

use crate::{excerpt, first_line, load_docs, short, Doc};
use anyhow::Result;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

const LOCAL_DIR: &str = "corpus/local";
const FILE_TOOLS: &[&str] = &["Write", "Edit", "MultiEdit", "NotebookEdit"];

pub fn run(sessions: usize, topics: usize) -> Result<()> {
    session_summaries(sessions)?;
    println!();
    topic_digests(topics)?;
    Ok(())
}

#[derive(Default)]
struct Session {
    repo: String,
    first_ts: String,
    last_ts: String,
    prompts: Vec<String>,
    files: Vec<String>,
    pr_links: Vec<String>,
}

fn session_summaries(n: usize) -> Result<()> {
    let mut files = Vec::new();
    if Path::new(LOCAL_DIR).is_dir() {
        collect_jsonl(Path::new(LOCAL_DIR), &mut files)?;
    }
    let mut seen_files: HashMap<String, HashSet<String>> = HashMap::new();
    let mut s: HashMap<String, Session> = HashMap::new();
    for f in files {
        let Ok(data) = std::fs::read_to_string(&f) else { continue };
        for line in data.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let Ok(ev) = serde_json::from_str::<Value>(line) else { continue };
            let sid = ev["session_id"].as_str().unwrap_or("").to_string();
            if sid.is_empty() {
                continue;
            }
            let kind = ev["kind"].as_str().unwrap_or("");
            let ts = ev["ts"].as_str().unwrap_or("").to_string();
            let e = s.entry(sid.clone()).or_default();
            if e.first_ts.is_empty() || ts < e.first_ts {
                e.first_ts = ts.clone();
            }
            if ts > e.last_ts {
                e.last_ts = ts.clone();
            }
            match kind {
                "session_start" => {
                    if let Some(cwd) = ev["payload"]["cwd"].as_str() {
                        e.repo = cwd.rsplit('/').next().unwrap_or(cwd).to_string();
                    }
                }
                "user_prompt" => {
                    if let Some(t) = ev["payload"]["text"].as_str() {
                        if t.trim().len() >= 12 {
                            e.prompts.push(t.trim().to_string());
                        }
                    }
                }
                "attachment_ref" => {
                    let tool = ev["payload"]["tool_name"].as_str().unwrap_or("");
                    if FILE_TOOLS.contains(&tool) {
                        if let Some(p) = ev["payload"]["local_path"].as_str() {
                            let base = p.rsplit('/').next().unwrap_or(p).to_string();
                            if seen_files.entry(sid.clone()).or_default().insert(base.clone()) {
                                e.files.push(base);
                            }
                        }
                    }
                }
                "agent_meta" => {
                    let blob = ev["payload"].to_string();
                    if let Some(pos) = blob.find("/pull/") {
                        let tail: String = blob[pos..].chars().take(24).collect();
                        e.pr_links.push(format!("…{tail}"));
                    }
                }
                _ => {}
            }
        }
    }

    let mut all: Vec<(String, Session)> = s.into_iter().filter(|(_, v)| !v.prompts.is_empty()).collect();
    all.sort_by(|a, b| b.1.last_ts.cmp(&a.1.last_ts));
    println!("# session summaries (most recent {n})\n");
    for (sid, sess) in all.into_iter().take(n) {
        let repo = if sess.repo.is_empty() { "?".into() } else { sess.repo };
        println!(
            "## {} · {} · {}",
            short(&sid),
            repo,
            sess.first_ts.split('T').next().unwrap_or("")
        );
        println!(
            "{} prompts · {} files touched",
            sess.prompts.len(),
            sess.files.len()
        );
        println!("ask: {}", excerpt(&sess.prompts[0], 200));
        if !sess.files.is_empty() {
            let shown: Vec<String> = sess.files.iter().take(8).cloned().collect();
            println!("files: {}", shown.join(", "));
        }
        if !sess.pr_links.is_empty() {
            println!("linked: {}", sess.pr_links.join(" "));
        }
        println!();
    }
    Ok(())
}

fn topic_digests(n: usize) -> Result<()> {
    let docs = load_docs(crate::DOCS_PATH)?;
    let by_id: HashMap<i64, &Doc> = docs.iter().map(|d| (d.id, d)).collect();

    println!("# topic digests (top {n})\n");
    if let Ok(raw) = std::fs::read_to_string("clusters.json") {
        let arr: Vec<Value> = serde_json::from_str(&raw).unwrap_or_default();
        let mut groups: BTreeMap<i64, (String, Vec<i64>)> = BTreeMap::new();
        for it in &arr {
            let c = it["cluster"].as_i64().unwrap_or(-1);
            let label = it["label"].as_str().unwrap_or("").to_string();
            let id = it["id"].as_i64().unwrap_or(-1);
            let e = groups.entry(c).or_insert_with(|| (label, Vec::new()));
            e.1.push(id);
        }
        let mut clusters: Vec<(i64, String, Vec<i64>)> =
            groups.into_iter().map(|(c, (l, ids))| (c, l, ids)).collect();
        clusters.sort_by(|a, b| b.2.len().cmp(&a.2.len()));
        for (_c, label, ids) in clusters.into_iter().take(n) {
            digest_one(&label, &ids, &by_id);
        }
        return Ok(());
    }

    // Fallback: per-repo GitHub digest.
    eprintln!("(no clusters.json — falling back to per-repo digests; run `cluster` first)");
    let mut by_repo: BTreeMap<String, Vec<i64>> = BTreeMap::new();
    for d in &docs {
        if d.meta.source == "github" {
            by_repo.entry(d.meta.repo.clone()).or_default().push(d.id);
        }
    }
    let mut repos: Vec<(String, Vec<i64>)> = by_repo.into_iter().collect();
    repos.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    for (repo, ids) in repos.into_iter().take(n) {
        digest_one(&repo, &ids, &by_id);
    }
    Ok(())
}

fn digest_one(label: &str, ids: &[i64], by_id: &HashMap<i64, &Doc>) {
    let docs: Vec<&Doc> = ids.iter().filter_map(|i| by_id.get(i).copied()).collect();
    let prs = docs.iter().filter(|d| d.meta.kind == "pull_request").count();
    let issues = docs.iter().filter(|d| d.meta.kind == "issue").count();
    let sess = docs.iter().filter(|d| d.meta.source == "agent").count();
    let mut repos: HashMap<&str, usize> = HashMap::new();
    for d in &docs {
        *repos.entry(d.meta.repo.as_str()).or_default() += 1;
    }
    let mut repo_v: Vec<_> = repos.into_iter().collect();
    repo_v.sort_by(|a, b| b.1.cmp(&a.1));
    let repo_s = repo_v
        .iter()
        .take(3)
        .map(|(r, c)| format!("{r}({c})"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("## {label}");
    println!("{} docs · {prs} PRs · {issues} issues · {sess} sessions · repos: {repo_s}", docs.len());
    let mut notable: Vec<&&Doc> = docs.iter().filter(|d| d.meta.source == "github").collect();
    notable.sort_by(|a, b| b.meta.ts.cmp(&a.meta.ts));
    for d in notable.into_iter().take(5) {
        println!(
            "- {}#{} {}",
            d.meta.repo,
            d.meta.number.unwrap_or(0),
            first_line(&d.text)
        );
    }
    println!();
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
