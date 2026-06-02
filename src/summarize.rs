// Extractive summaries, no LLM.
//  - Sessions: the `units` view-model — repo, counts, opening ask, c-TF-IDF
//    keyphrases, the representative line, files touched, effort, linked PR.
//  - Topics: read clusters.json + docs.jsonl; per cluster report repos,
//    kind counts, and notable titles. Falls back to per-repo GitHub digests.

use crate::{first_line, load_docs, short, Doc};
use anyhow::Result;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

pub fn run(sessions: usize, topics: usize) -> Result<()> {
    session_summaries(sessions)?;
    println!();
    topic_digests(topics)?;
    Ok(())
}

fn session_summaries(n: usize) -> Result<()> {
    let mut all = crate::units::sessions()?;
    all.sort_by(|a, b| b.ended.cmp(&a.ended));
    println!("# session summaries (most recent {n})\n");
    for s in all.into_iter().take(n) {
        println!(
            "## {} · {} · {}",
            short(&s.id),
            if s.repo.is_empty() { "?" } else { &s.repo },
            s.started.split('T').next().unwrap_or("")
        );
        println!(
            "{} prompts · {} assistant · {} tools · {} files · effort {}",
            s.prompts,
            s.assistant,
            s.tools,
            s.files.len(),
            crate::view::meter(s.struggle)
        );
        println!("ask: {}", s.ask);
        if !s.keyphrases.is_empty() {
            println!("about: {}", s.keyphrases.join(", "));
        }
        match &s.summary {
            Some(sum) => println!("summary: {sum}"),
            None if !s.gist.is_empty() => println!("gist: {}", s.gist),
            None => {}
        }
        if !s.files.is_empty() {
            println!("files: {}", s.files.iter().take(8).cloned().collect::<Vec<_>>().join(", "));
        }
        if let Some(pr) = &s.linked_pr {
            println!("linked: {pr}");
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
