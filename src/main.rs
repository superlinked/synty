// synty — passive cross-agent + GitHub work memory. Local-first, no generative
// model: pylate-rs (ColBERT) encodes, next-plaid (PLAID) indexes; search prints
// Markdown to stdout for coding agents.
//
// Subcommands: ingest, index, search, cluster, summarize, eval.

mod cluster;
mod encode;
mod eval;
mod index;
mod ingest;
mod model;
mod search;
mod summarize;

use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

pub const CORPUS_DIR: &str = "corpus";
pub const DOCS_PATH: &str = "corpus/docs.jsonl";
pub const INDEX_PATH: &str = "index";

pub fn model_id() -> String {
    std::env::var("SYNTY_MODEL").unwrap_or_else(|_| "mixedbread-ai/mxbai-edge-colbert-v0-32m".into())
}

/// Document metadata. Serialized both as the per-doc JSON next-plaid stores for
/// WHERE-filtering and inside docs.jsonl for rendering.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Meta {
    pub source: String, // "github" | "agent"
    pub kind: String,   // pull_request | issue | user_prompt | assistant_message | thinking
    #[serde(default)]
    pub repo: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub ts: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub number: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Doc {
    pub id: i64,
    pub text: String,
    pub meta: Meta,
}

pub fn load_docs(path: &str) -> Result<Vec<Doc>> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read {path}: {e} (run `ingest` first)"))?;
    let mut v = Vec::new();
    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        v.push(serde_json::from_str::<Doc>(line)?);
    }
    Ok(v)
}

/// First non-empty line, trimmed — used as a card title.
pub fn first_line(s: &str) -> &str {
    s.lines().map(|l| l.trim()).find(|l| !l.is_empty()).unwrap_or("")
}

/// Whitespace-collapsed, length-capped excerpt.
pub fn excerpt(s: &str, n: usize) -> String {
    let one = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() <= n {
        one
    } else {
        format!("{}…", one.chars().take(n).collect::<String>())
    }
}

/// Short id prefix.
pub fn short(s: &str) -> String {
    s.chars().take(8).collect()
}

#[derive(Parser)]
#[command(name = "synty", about = "late-interaction work memory (experiment)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Parse corpus/{local,github} into corpus/docs.jsonl
    Ingest,
    /// Encode docs (pylate-rs) and build the next-plaid index
    Index,
    /// Semantic search; --filter is a SQL WHERE over metadata
    Search {
        query: String,
        #[arg(long)]
        filter: Option<String>,
        #[arg(long, default_value_t = 5)]
        k: usize,
    },
    /// Build clusters (kNN ∪ github links) and print labeled topics
    Cluster,
    /// Extractive session + topic summaries
    Summarize {
        #[arg(long, default_value_t = 10)]
        sessions: usize,
        #[arg(long, default_value_t = 5)]
        topics: usize,
    },
    /// Run the probe query set and write eval_report.md
    Eval,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Ingest => ingest::run(CORPUS_DIR, DOCS_PATH)?,
        Cmd::Index => index::run(DOCS_PATH, INDEX_PATH, &model_id())?,
        Cmd::Search { query, filter, k } => {
            search::run(&query, filter.as_deref(), k, &model_id())?
        }
        Cmd::Cluster => cluster::run(&model_id())?,
        Cmd::Summarize { sessions, topics } => summarize::run(sessions, topics)?,
        Cmd::Eval => eval::run(&model_id())?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{excerpt, first_line};

    #[test]
    fn first_line_skips_blanks_and_trims() {
        assert_eq!(first_line("\n  Hello world \nsecond"), "Hello world");
        assert_eq!(first_line(""), "");
    }

    #[test]
    fn excerpt_collapses_whitespace_and_caps() {
        assert_eq!(excerpt("a  b\n c", 100), "a b c");
        assert_eq!(excerpt("hello world", 5), "hello…");
    }
}
