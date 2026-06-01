// synty — passive cross-agent + GitHub work memory. Local-first, no generative
// model: pylate-rs (ColBERT) encodes, next-plaid (PLAID) indexes; search prints
// Markdown to stdout for coding agents.
//
// Subcommands: ingest, index, search, cluster, summarize, eval.

mod blocks;
mod bucket;
mod claudecode;
mod cluster;
mod codex;
mod cowork;
mod community;
mod encode;
mod eval;
mod event;
mod github;
mod keyphrase;
mod store;
mod sync;
mod tail;
mod track;
mod up;
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
    Ingest {
        /// Pull all devices' events from this bucket before ingesting
        #[arg(long)]
        bucket: Option<String>,
    },
    /// Encode docs (pylate-rs) and build the next-plaid index
    Index {
        /// Bucket holding the content-addressed embedding store (file path or
        /// s3://, gs:// with the matching feature). Shared across devices.
        #[arg(long, default_value = ".synty")]
        bucket: String,
    },
    /// Semantic search; --filter is a SQL WHERE over metadata
    Search {
        query: String,
        #[arg(long)]
        filter: Option<String>,
        #[arg(long, default_value_t = 5)]
        k: usize,
        /// Bucket to pull the published index from when local is stale/absent
        #[arg(long, default_value = ".synty")]
        bucket: String,
    },
    /// Louvain topics over a weighted kNN + github-link graph; print labeled
    /// clusters. --resolution >1 yields more/smaller topics, <1 fewer/larger.
    Cluster {
        #[arg(long, default_value_t = 1.0)]
        resolution: f64,
    },
    /// Extractive session + topic summaries
    Summarize {
        #[arg(long, default_value_t = 10)]
        sessions: usize,
        #[arg(long, default_value_t = 5)]
        topics: usize,
    },
    /// Run the probe query set and write eval_report.md
    Eval,
    /// Native tracker: parse local agent session files into canonical envelopes
    Track {
        /// Source to track: claudecode | codex | cowork | all
        #[arg(long, default_value = "all")]
        source: String,
        /// Output dir for envelope streams (where `ingest` reads)
        #[arg(long, default_value = "corpus/local")]
        out: String,
        /// Skip files whose mtime is older than this many days (0 = unbounded)
        #[arg(long, default_value_t = 90)]
        max_age_days: u64,
        /// Machine id used in the stream name
        #[arg(long, default_value = "local")]
        machine: String,
        /// Watch continuously instead of a single drain
        #[arg(long)]
        watch: bool,
        /// Poll interval in seconds for --watch
        #[arg(long, default_value_t = 30)]
        poll: u64,
        /// Write an autostart unit and exit: launchd | systemd
        #[arg(long)]
        install: Option<String>,
        /// Per-file cursor store
        #[arg(long, default_value = ".synty/cursors.json")]
        cursors: String,
        /// Push drained events to this bucket under events/ (for a fleet)
        #[arg(long)]
        bucket: Option<String>,
    },
    /// One-command solo mode: track + ingest + index on a loop, zero config
    Up {
        /// Bucket for the embedding store + published index
        #[arg(long, default_value = ".synty")]
        bucket: String,
        /// Machine id used in stream names
        #[arg(long, default_value = "local")]
        machine: String,
        /// Seconds between passes
        #[arg(long, default_value_t = 60)]
        poll: u64,
        /// Skip the startup GitHub pull
        #[arg(long)]
        no_github: bool,
    },
    /// Pull GitHub PRs/issues via GraphQL (token-based, no `gh` needed)
    Github {
        /// Repository owner / org
        #[arg(long, default_value = "superlinked")]
        owner: String,
        /// Comma-separated repo names (default: the known active set)
        #[arg(long)]
        repos: Option<String>,
        /// Trailing window in days
        #[arg(long, default_value_t = 90)]
        since_days: u64,
        /// Output dir for the per-repo JSON
        #[arg(long, default_value = "corpus/github")]
        out: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Ingest { bucket } => ingest::run(CORPUS_DIR, DOCS_PATH, bucket.as_deref())?,
        Cmd::Index { bucket } => index::run(DOCS_PATH, INDEX_PATH, &model_id(), &bucket)?,
        Cmd::Search { query, filter, k, bucket } => {
            search::run(&query, filter.as_deref(), k, &model_id(), &bucket)?
        }
        Cmd::Cluster { resolution } => cluster::run(resolution)?,
        Cmd::Summarize { sessions, topics } => summarize::run(sessions, topics)?,
        Cmd::Eval => eval::run(&model_id())?,
        Cmd::Track { source, out, max_age_days, machine, watch, poll, install, cursors, bucket } => {
            track::run(track::Opts {
                which: source,
                out,
                max_age_days,
                machine,
                watch,
                poll_secs: poll,
                install,
                cursors,
                bucket,
            })?
        }
        Cmd::Up { bucket, machine, poll, no_github } => {
            up::run(&bucket, &machine, poll, !no_github)?
        }
        Cmd::Github { owner, repos, since_days, out } => {
            github::run(&owner, repos, since_days, &out)?
        }
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
