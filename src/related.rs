// `synty related` — zero-effort recall. Instead of typing a query, derive one
// from what you're working on *right now* (recent commit subjects + recently
// touched files in the cwd) and surface prior sessions, PRs, and issues across
// every repo the fleet has seen. A shared work memory only pays off if you
// consult it before redoing something; this removes the one step of friction
// (thinking up a query) so an agent or a person can ask "has this been done?"
// for free at the start of a task.

use crate::search;
use anyhow::{anyhow, Result};
use std::path::Path;
use std::process::Command;

/// Recent commits repeating "fix" shouldn't drown the signal, and a 40-word
/// query keeps the ColBERT query encoder from truncating the tail. Cap chosen
/// well above a normal commit-subject vocabulary, low enough to stay on-topic.
const QUERY_WORD_CAP: usize = 40;

/// Turn raw git output into a retrieval query: commit subjects say *what* you've
/// been doing, changed-file basenames say *where*. Deduped, lowercased, ceremony
/// and pure-number tokens dropped. Pure on purpose — the git plumbing lives in
/// `context_query` and hands this the raw strings, so the interesting logic is
/// testable without a repo. Returns None when there's nothing to go on.
pub fn build_query(log_subjects: &str, changed_files: &str) -> Option<String> {
    let mut seen = std::collections::HashSet::new();
    let mut words: Vec<String> = Vec::new();
    add_words(log_subjects, &mut seen, &mut words);
    // File paths → basename without its first extension, then split like prose
    // ("src/auth/rate_limit.go" → rate, limit). Anchors the query on the area of
    // code, not the directory layout.
    for line in changed_files.lines() {
        let base = line.rsplit('/').next().unwrap_or(line);
        let stem = base.split_once('.').map(|(s, _)| s).unwrap_or(base);
        add_words(stem, &mut seen, &mut words);
    }
    if words.is_empty() {
        return None;
    }
    words.truncate(QUERY_WORD_CAP);
    Some(words.join(" "))
}

/// Split on non-alphanumerics (keeping hyphenated compounds, like the keyphrase
/// extractor), drop short/numeric/stopword tokens, and append the first sighting
/// of each — preserving order so the query and tests are deterministic.
fn add_words(s: &str, seen: &mut std::collections::HashSet<String>, words: &mut Vec<String>) {
    for raw in s.split(|c: char| !c.is_alphanumeric() && c != '-') {
        let w = raw.trim_matches('-').to_lowercase();
        if w.len() < 3 || w.chars().all(|c| c.is_ascii_digit()) || crate::qwen::STOPWORDS.contains(&w.as_str()) {
            continue;
        }
        if seen.insert(w.clone()) {
            words.push(w);
        }
    }
}

/// Derive a query from the git repo at `cwd`: the last 15 commit subjects and
/// the files touched in the last 10 commits. `git log --name-only` (not a
/// `HEAD~10` range) so shallow/young repos don't error. Returns None outside a
/// repo or with no history — the caller points at `synty search`.
pub fn context_query(cwd: &Path) -> Option<String> {
    let subjects = git(cwd, &["log", "-n", "15", "--pretty=%s"])?;
    let files = git(cwd, &["log", "-n", "10", "--name-only", "--pretty=format:"]).unwrap_or_default();
    build_query(&subjects, &files)
}

fn git(cwd: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git").arg("-C").arg(cwd).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

pub fn run(k: usize, model_id: &str, bucket: &str, json: bool) -> Result<()> {
    let cwd = std::env::current_dir().map_err(|e| anyhow!("cwd: {e}"))?;
    let query = context_query(&cwd).ok_or_else(|| {
        anyhow!(
            "no git context here — `synty related` reads recent commits and changed files to find prior work. \
             Run it inside a git repo with history, or use `synty search \"<what you're working on>\"`."
        )
    })?;
    // Cross-repo on purpose (filter None): related work lives anywhere the fleet
    // has seen, not just this repo. Reuse the search path verbatim — the header
    // shows the derived query, which doubles as "here's what I matched on".
    eprintln!("related: query derived from recent commits + changed files in {}", cwd.display());
    search::run(&query, None, k, model_id, bucket, json)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The query is what you've been doing, not how git formats it: subjects and
    // file names fold into deduped content words.
    #[test]
    fn build_query_folds_subjects_and_filenames() {
        let subjects = "M2: add codex session tailer\nfix codex token parsing\nM2: codex tailer error counts";
        let files = "src/codex.rs\nsrc/tail.rs\nsrc/codex.rs";
        let q = build_query(subjects, files).expect("a query");
        // Content survives; the repeated "codex" and the ceremony ("M2", "add",
        // "fix") collapse — "codex" appears once, the prefixes are gone.
        assert_eq!(q.matches("codex").count(), 1, "deduped: {q}");
        assert!(q.contains("session") && q.contains("tailer") && q.contains("parsing"), "keeps signal: {q}");
        assert!(!q.contains("add") && !q.contains("fix"), "drops commit ceremony: {q}");
        assert!(q.contains("tail"), "filename basename folded in: {q}");
    }

    // No commits / not a repo → no query, so the caller can fall back instead of
    // searching for an empty string.
    #[test]
    fn build_query_empty_is_none() {
        assert!(build_query("", "").is_none());
        // Subjects that are pure ceremony/numbers also yield nothing usable.
        assert!(build_query("123 456\nfix", "").is_none());
    }

    // The query stays within the cap even when history is long, so the encoder
    // doesn't silently truncate the tail.
    #[test]
    fn build_query_caps_length() {
        let many = (0..200).map(|i| format!("alpha{i} beta{i} gamma{i}")).collect::<Vec<_>>().join("\n");
        let q = build_query(&many, "").expect("a query");
        assert!(q.split_whitespace().count() <= QUERY_WORD_CAP, "capped: {} words", q.split_whitespace().count());
    }
}
