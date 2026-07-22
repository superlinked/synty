// synty — passive cross-agent + GitHub work memory. Local-first: pylate-rs
// (ColBERT) encodes, next-plaid (PLAID) indexes, and a
// small local model (Qwen) writes the one-line summaries; search prints Markdown
// to stdout for coding agents. Team mode explicitly syncs through a bucket.
//
// Subcommands span the pipeline, the tracker and GitHub backfill, and the read
// surfaces (CLI + TUI + MCP).

mod blocks;
mod bucket;
mod claudecode;
mod codex;
mod cowork;
mod community;
mod encode;
mod eval;
mod event;
mod config;
mod fleet;
mod github;
mod identity;
mod lease;
mod mcp;
mod metrics;
mod progress;
mod readmodel;
mod init;
mod related;
mod release;
mod store;
mod sync;
mod tail;
mod track;
mod tui;
mod units;
mod up;
mod view;
mod index;
mod ingest;
mod model;
#[cfg(feature = "llm")]
mod qwen;
mod search;
mod summarize;
mod topics;
mod trace;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

pub const CORPUS_DIR: &str = "corpus";
pub const DOCS_PATH: &str = "corpus/docs.jsonl";
pub const INDEX_PATH: &str = "index";

/// The default encoder. Pinned by the stores: its vectors live in the
/// original, unprefixed namespace; any other model gets its own (see store.rs).
pub const DEFAULT_MODEL: &str = "mixedbread-ai/mxbai-edge-colbert-v0-32m";

pub fn model_id() -> String {
    std::env::var("SYNTY_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into())
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
    /// Agent a GitHub doc attributes its work to (Co-authored-by trailer,
    /// Generated-with footer, or bot author) — evidence agents ran for this
    /// work even when no tracker saw it. Always None on session docs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_attr: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Doc {
    pub id: i64,
    pub text: String,
    pub meta: Meta,
}

pub fn load_docs(path: impl AsRef<std::path::Path>) -> Result<Vec<Doc>> {
    let path = path.as_ref();
    let data = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read {}: {e} (run `ingest` first)", path.display()))?;
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

/// Write via a sibling tmp file + rename: readers never see a half-written
/// read-model, and concurrent writers can't interleave (last full write wins).
pub fn write_atomic(path: &str, data: &[u8]) -> Result<()> {
    let tmp = format!("{path}.tmp.{}", std::process::id());
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[derive(Parser)]
#[command(name = "synty", version, about = "late-interaction work memory (experiment)")]
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
        /// s3://, gs:// with the matching feature). Default: config, then .synty.
        #[arg(long)]
        bucket: Option<String>,
    },
    /// Semantic search; --filter is a SQL WHERE over metadata
    Search {
        query: String,
        #[arg(long)]
        filter: Option<String>,
        #[arg(long, default_value_t = 5)]
        k: usize,
        /// Bucket to pull the published index from when local is stale/absent
        #[arg(long)]
        bucket: Option<String>,
        /// Print results as JSON (for scripts and agents)
        #[arg(long)]
        json: bool,
    },
    /// Prior work related to what you're doing right now: derives a query from
    /// this repo's recent commits and changed files, then searches every repo
    /// the fleet has seen. Run it at the start of a task to build on past work.
    Related {
        #[arg(long, default_value_t = 5)]
        k: usize,
        /// Bucket to pull the published index from when local is stale/absent
        #[arg(long)]
        bucket: Option<String>,
        /// Print results as JSON (for scripts and agents)
        #[arg(long)]
        json: bool,
    },
    /// Cluster units (sessions, PRs, issues) by their summary embedding (MaxSim
    /// kNN + Louvain) into topics. --resolution >1 yields more/smaller topics.
    /// Run `summarize` first. Writes the clusters into the current build.
    Cluster {
        #[arg(long, default_value_t = 1.0)]
        resolution: f64,
        #[arg(long)]
        bucket: Option<String>,
    },
    /// List emergent topics (from the last `cluster` run); optional substring filter
    Topic {
        query: Option<String>,
        /// Print as JSON (for scripts and agents)
        #[arg(long)]
        json: bool,
    },
    /// Recent activity: latest PRs, issues, and prompts
    Recent {
        #[arg(long)]
        repo: Option<String>,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// Print as JSON (for scripts and agents)
        #[arg(long)]
        json: bool,
    },
    /// What synty holds and how fresh it is
    Status {
        /// Print as JSON (for scripts and agents)
        #[arg(long)]
        json: bool,
    },
    /// Usage and output over time: tokens, cache, tool calls, sessions,
    /// merged LOC±, PRs, issues — weekly tables here, a day-keyed series
    /// with --json
    Stats {
        /// Trailing Mon-aligned weeks, anchored to the most recent day with data
        #[arg(long, default_value_t = 4)]
        weeks: usize,
        /// Print as JSON (for scripts and agents)
        #[arg(long)]
        json: bool,
    },
    /// Fleet-wide profile of one tool: volume, latency, argument mix, recent
    /// invocations (tool names appear in `synty status`)
    Tool {
        /// The tool name as the agent calls it (e.g. Bash, Edit, Read)
        name: String,
        /// Print as JSON (for scripts and agents)
        #[arg(long)]
        json: bool,
    },
    /// Detail for one id printed by search/recent/topic: a session
    /// ([a1b2c3d4] or a full id), a PR/issue (repo#123 or gh:repo#123), or a
    /// topic key
    Show {
        /// The id to resolve (≥4 chars of a session id or topic key, or repo#N)
        id: String,
        /// Print as JSON (for scripts and agents)
        #[arg(long)]
        json: bool,
    },
    /// Inspect coding-agent execution as turns, paired tool spans, async job
    /// chains, and raw event evidence. This is a factual trace surface, not a
    /// bottleneck judge.
    Trace {
        #[command(subcommand)]
        command: TraceCmd,
    },
    /// Initialize synty on this machine in one step: set the bucket (omit for a
    /// local trial), pin the GitHub identity, enable the login-time tracker, and
    /// run the first build. Idempotent — re-run with a bucket to switch a local
    /// trial onto the team.
    Init {
        /// Team bucket (s3://… / gs://… / path). Omit for a local-only trial.
        bucket: Option<String>,
        /// AWS shared-config profile for S3. Prefer a rotating
        /// credential_process profile such as IAM Roles Anywhere.
        #[arg(long)]
        aws_profile: Option<String>,
        /// Do not collect or upload events before this boundary: `now`,
        /// YYYY-MM-DD, or RFC3339.
        #[arg(long)]
        capture_since: Option<String>,
        /// Seconds between batched event uploads.
        #[arg(long)]
        upload_interval: Option<u64>,
        /// Machine id used in this machine's stream names
        #[arg(long, default_value = "local")]
        machine: String,
        /// Configure + enable tracking but skip the first build
        #[arg(long)]
        no_build: bool,
    },
    /// Self-update from the latest GitHub Release: if a newer build is published
    /// for this platform, download it, verify its checksum, replace this binary,
    /// and restart the tracker. No-op when already current. Releases are cut by
    /// CI on a tag; override the source repo with $SYNTY_RELEASE_REPO.
    Upgrade,
    /// Interactive terminal UI: status + browse/drill over topics, recent, search.
    /// Pulls the fleet's read-model at startup and freshens in the background.
    Tui {
        /// Bucket to pull from and contribute builds to
        #[arg(long)]
        bucket: Option<String>,
    },
    /// MCP server over stdio: synty_search / synty_topics / synty_recent /
    /// synty_status as agent tools (add to a coding agent's MCP config)
    Mcp,
    /// Session + topic summaries. With the `llm` feature, session summaries are
    /// generated by a local Qwen3 model and cached; topics stay extractive.
    Summarize {
        #[arg(long, default_value_t = 10)]
        sessions: usize,
        #[arg(long, default_value_t = 5)]
        topics: usize,
        /// Bucket holding the fleet's shared write-once summary store
        #[arg(long)]
        bucket: Option<String>,
        /// Skip LLM generation; show cached/extractive summaries only.
        #[arg(long)]
        cached: bool,
        /// Print the raw summarizer inputs (ask/turns) and exit.
        #[arg(long)]
        dump: bool,
        /// Prompt-tuning dry run: generate (don't cache) N topic summaries and print them.
        #[arg(long)]
        sample: Option<usize>,
    },
    /// Run the probe query set and write evals/runs.md (retrieval). With
    /// --names, score topic names against their clusters → evals/names.{json,md}
    Eval {
        /// Score topic-name faithfulness instead of retrieval (the name eval)
        #[arg(long)]
        names: bool,
        /// Embedding store bucket (name eval; default: config, then .synty)
        #[arg(long)]
        bucket: Option<String>,
    },
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
        /// Absolute collection boundary: `now`, YYYY-MM-DD, or RFC3339. Takes
        /// precedence over --max-age-days and the persisted init setting.
        #[arg(long)]
        since: Option<String>,
        /// Machine id used in the stream name
        #[arg(long, default_value = "local")]
        machine: String,
        /// Watch continuously instead of a single drain
        #[arg(long)]
        watch: bool,
        /// Poll interval in seconds for --watch
        #[arg(long, default_value_t = 30)]
        poll: u64,
        /// Seconds between batched bucket uploads (default: config, then 60)
        #[arg(long)]
        upload_interval: Option<u64>,
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
    /// Run the whole pipeline once: track + ingest + index + summarize +
    /// cluster + topic summaries — no step order to remember
    Build {
        /// Bucket for the embedding store + published index
        #[arg(long)]
        bucket: Option<String>,
        /// Machine id used in stream names
        #[arg(long, default_value = "local")]
        machine: String,
        /// Louvain resolution for the cluster step (>1 = more/smaller topics)
        #[arg(long, default_value_t = 1.0)]
        resolution: f64,
        /// Skip the local tailer pass (when an autostart tracker already runs)
        #[arg(long)]
        no_track: bool,
    },
    /// One-command solo mode: track + ingest + index on a loop, zero config
    Up {
        /// Bucket for the embedding store + published index
        #[arg(long)]
        bucket: Option<String>,
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
        /// Repository owner / org (default: the org pinned by `synty init`)
        #[arg(long)]
        owner: Option<String>,
        /// Comma-separated repo names (default: the org's most-active set)
        #[arg(long)]
        repos: Option<String>,
        /// Trailing window in days
        #[arg(long, default_value_t = 90)]
        since_days: u64,
        /// Output dir for the per-repo JSON
        #[arg(long, default_value = "corpus/github")]
        out: String,
        /// Bucket to share the scraped corpus with (default: config, then .synty)
        #[arg(long)]
        bucket: Option<String>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum TraceEntity {
    Turns,
    Spans,
    Jobs,
}

impl TraceEntity {
    fn as_str(self) -> &'static str {
        match self {
            Self::Turns => "turns",
            Self::Spans => "spans",
            Self::Jobs => "jobs",
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum TraceSort {
    Recent,
    Duration,
    Wait,
}

impl TraceSort {
    fn as_str(self) -> &'static str {
        match self {
            Self::Recent => "recent",
            Self::Duration => "duration",
            Self::Wait => "wait",
        }
    }
}

#[derive(Subcommand)]
enum TraceCmd {
    /// List source-native turns, paired tool spans, or associated async jobs.
    List {
        /// Execution entity to list.
        #[arg(long = "type", value_enum, default_value_t = TraceEntity::Turns)]
        entity: TraceEntity,
        #[arg(long)]
        repo: Option<String>,
        #[arg(long)]
        machine: Option<String>,
        #[arg(long)]
        source: Option<String>,
        /// Exact factual status: ok, error, aborted, running, open, or unknown.
        #[arg(long)]
        status: Option<String>,
        /// Tool/command substring (spans, jobs, or turns with a matching span).
        #[arg(long)]
        operation: Option<String>,
        /// For turns or jobs, require at least one error-status child span.
        #[arg(long)]
        has_errors: bool,
        /// ISO timestamp or YYYY-MM-DD lower bound.
        #[arg(long)]
        since: Option<String>,
        /// Minimum turn/span duration, or associated-job lifecycle elapsed time.
        #[arg(long)]
        min_ms: Option<u64>,
        #[arg(long, value_enum, default_value_t = TraceSort::Recent)]
        sort: TraceSort,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Show a compact timeline for a turn/session or evidence around a span/event.
    Show {
        /// Full id or unique prefix printed by another trace command.
        id: String,
        /// Events before a span/event hit.
        #[arg(long, default_value_t = 6)]
        before: usize,
        /// Events after a span/event hit.
        #[arg(long, default_value_t = 12)]
        after: usize,
        #[arg(long)]
        json: bool,
    },
    /// Literal case-insensitive search over raw event envelopes, including
    /// prompts, commands, tool outputs, and metadata.
    Search {
        query: String,
        #[arg(long)]
        repo: Option<String>,
        #[arg(long)]
        machine: Option<String>,
        #[arg(long)]
        source: Option<String>,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Compare two turn, span, or async-job ids field-by-field.
    Compare {
        left: String,
        right: String,
        #[arg(long)]
        json: bool,
    },
}

/// Run from the synty home: an explicit $SYNTY_HOME wins; a cwd that already
/// holds synty state (.synty/) is its own home (the dev-checkout case); else
/// fall back to ~/.synty when the installer created it. Every state path in
/// the binary is home-relative, so this one chdir makes `synty tui` work from
/// any directory.
fn resolve_home() {
    if let Ok(h) = std::env::var("SYNTY_HOME") {
        if let Err(e) = std::env::set_current_dir(&h) {
            eprintln!("synty: cannot use SYNTY_HOME={h}: {e}");
        }
        return;
    }
    if std::path::Path::new(".synty").exists() {
        return;
    }
    if let Ok(home) = std::env::var("HOME") {
        let d = std::path::Path::new(&home).join(".synty");
        if d.is_dir() && std::env::set_current_dir(&d).is_ok() {
            eprintln!("synty: home {}", d.display());
        }
    }
}

/// Read commands without their own bucket flag use the configured fleet
/// bucket. Pulling here makes every CLI surface (not only semantic search) see
/// all machines and detect unpublished event deltas before rendering.
fn pull_configured_for_read() {
    if let Some(bucket) = config::resolve_bucket_opt(None) {
        sync::pull_for_read(&bucket);
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    resolve_home();
    match cli.cmd {
        Cmd::Ingest { bucket } => {
            ingest::run(CORPUS_DIR, DOCS_PATH, config::resolve_bucket_opt(bucket).as_deref())?
        }
        Cmd::Index { bucket } => index::run(DOCS_PATH, &model_id(), &config::resolve_bucket(bucket))?,
        Cmd::Search { query, filter, k, bucket, json } => {
            search::run(&query, filter.as_deref(), k, &model_id(), &config::resolve_bucket(bucket), json)?
        }
        Cmd::Related { k, bucket, json } => {
            related::run(k, &model_id(), &config::resolve_bucket(bucket), json)?
        }
        Cmd::Upgrade => release::upgrade()?,
        Cmd::Cluster { resolution, bucket } => {
            topics::run(resolution, &model_id(), &config::resolve_bucket(bucket))?
        }
        Cmd::Topic { query, json } => {
            pull_configured_for_read();
            let mut topics = units::topic_units(12)?;
            if let Some(q) = query {
                let ql = q.to_lowercase();
                topics.retain(|t| {
                    t.label.to_lowercase().contains(&ql)
                        || t.units.iter().any(|u| u.title.to_lowercase().contains(&ql))
                });
            }
            if json {
                println!("{}", view::topics_json(&topics));
            } else {
                print!("{}", view::topics_md(&topics));
            }
        }
        Cmd::Recent { repo, limit, json } => {
            pull_configured_for_read();
            let mut us = units::units()?;
            if let Some(r) = repo {
                us.retain(|u| u.repo == r);
            }
            us.truncate(limit);
            if json {
                println!("{}", view::work_json(&us));
            } else {
                print!("{}", view::work_md(&us));
            }
        }
        Cmd::Status { json } => {
            pull_configured_for_read();
            let s = view::status()?;
            if json {
                println!("{}", view::status_json(&s));
            } else {
                print!("{}", view::status_md(&s));
            }
        }
        Cmd::Stats { weeks, json } => {
            pull_configured_for_read();
            let s = view::stats(weeks)?;
            if json {
                println!("{}", view::stats_json(&s));
            } else {
                print!("{}", view::stats_md(&s));
            }
        }
        Cmd::Tool { name, json } => {
            pull_configured_for_read();
            if json {
                let p = units::tool_profile(&name);
                if p.calls == 0 {
                    // Same miss behavior as the Markdown path.
                    view::tool_report(&name)?;
                }
                println!("{}", view::tool_json(&p));
            } else {
                print!("{}", view::tool_report(&name)?);
            }
        }
        Cmd::Show { id, json } => {
            pull_configured_for_read();
            if json {
                println!("{}", view::show_json_report(&id)?);
            } else {
                print!("{}", view::show_report(&id)?);
            }
        }
        Cmd::Trace { command } => {
            pull_configured_for_read();
            match command {
                TraceCmd::List {
                    entity,
                    repo,
                    machine,
                    source,
                    status,
                    operation,
                    has_errors,
                    since,
                    min_ms,
                    sort,
                    limit,
                    json,
                } => trace::list(
                    entity.as_str(),
                    repo.as_deref(),
                    machine.as_deref(),
                    source.as_deref(),
                    status.as_deref(),
                    operation.as_deref(),
                    has_errors,
                    since.as_deref(),
                    min_ms,
                    sort.as_str(),
                    limit,
                    json,
                )?,
                TraceCmd::Show {
                    id,
                    before,
                    after,
                    json,
                } => trace::show(&id, before, after, json)?,
                TraceCmd::Search {
                    query,
                    repo,
                    machine,
                    source,
                    kind,
                    limit,
                    json,
                } => trace::search(
                    &query,
                    repo.as_deref(),
                    machine.as_deref(),
                    source.as_deref(),
                    kind.as_deref(),
                    limit,
                    json,
                )?,
                TraceCmd::Compare { left, right, json } => trace::compare(&left, &right, json)?,
            }
        }
        Cmd::Init {
            bucket,
            aws_profile,
            capture_since,
            upload_interval,
            machine,
            no_build,
        } => init::run(
            bucket,
            aws_profile,
            capture_since,
            upload_interval,
            &machine,
            no_build,
        )?,
        Cmd::Tui { bucket } => tui::run(model_id(), config::resolve_bucket(bucket))?,
        Cmd::Mcp => {
            pull_configured_for_read();
            mcp::run(model_id())?
        }
        Cmd::Summarize {
            sessions,
            topics,
            bucket,
            cached,
            dump,
            sample,
        } => {
            let bucket = config::resolve_bucket(bucket);
            let _ = (cached, sample, &bucket);
            if dump {
                summarize::dump_inputs()?;
            } else {
                #[cfg(feature = "llm")]
                match sample {
                    Some(n) => qwen::sample(n)?,
                    None => {
                        if !cached {
                            if let Err(e) = qwen::summarize_all(&bucket) {
                                eprintln!("llm summarize skipped: {e}");
                            }
                        }
                        summarize::run(sessions, topics)?;
                    }
                }
                #[cfg(not(feature = "llm"))]
                summarize::run(sessions, topics)?;
            }
        }
        Cmd::Eval { names, bucket } => {
            if names {
                eval::run_names(&config::resolve_bucket(bucket))?;
            } else {
                eval::run(&model_id())?;
            }
        }
        Cmd::Track {
            source,
            out,
            max_age_days,
            since,
            machine,
            watch,
            poll,
            upload_interval,
            install,
            cursors,
            bucket,
        } => {
            let capture_since_ms = match since {
                Some(raw) => Some(
                    chrono::DateTime::parse_from_rfc3339(&config::normalize_capture_since(&raw)?)?
                        .timestamp_millis(),
                ),
                None => config::capture_since_ms(),
            };
            track::run(track::Opts {
                which: source,
                out,
                max_age_days,
                capture_since_ms,
                machine,
                watch,
                poll_secs: poll,
                upload_interval_secs: upload_interval
                    .unwrap_or_else(config::upload_interval_secs)
                    .max(1),
                install,
                cursors,
                bucket: config::resolve_bucket_opt(bucket),
            })?
        }
        Cmd::Build {
            bucket,
            machine,
            resolution,
            no_track,
        } => up::build(
            &config::resolve_bucket(bucket),
            &machine,
            resolution,
            no_track,
        )?,
        Cmd::Up {
            bucket,
            machine,
            poll,
            no_github,
        } => up::run(&config::resolve_bucket(bucket), &machine, poll, !no_github)?,
        Cmd::Github {
            owner,
            repos,
            since_days,
            out,
            bucket,
        } => {
            let owner = owner.or_else(|| config::load().org).ok_or_else(|| {
                anyhow::anyhow!("no GitHub org: run `synty init` or pass --owner")
            })?;
            github::run(&owner, repos, since_days, &out)?;
            let n = sync::push_github(&config::resolve_bucket(bucket), &out)?;
            if n > 0 {
                eprintln!("github: pushed {n} corpus objects to the bucket");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{excerpt, first_line, Cli, Cmd, TraceCmd, TraceEntity, TraceSort};
    use clap::Parser;

    // `synty init` is the single onboarding command: the bucket is an optional
    // positional (omit = local trial), and the old `setup`/`join` names are gone.
    #[test]
    fn init_parses_optional_bucket_and_old_names_are_removed() {
        let local = Cli::try_parse_from(["synty", "init"]).expect("bare init parses");
        assert!(matches!(local.cmd, Cmd::Init { bucket: None, no_build: false, .. }));
        let team = Cli::try_parse_from(["synty", "init", "gs://team", "--no-build"]).expect("init w/ bucket");
        assert!(matches!(team.cmd, Cmd::Init { bucket: Some(b), no_build: true, .. } if b == "gs://team"));
        // The earlier names no longer exist — one onboarding path, no confusion.
        assert!(Cli::try_parse_from(["synty", "setup"]).is_err());
        assert!(Cli::try_parse_from(["synty", "join"]).is_err());
    }

    #[test]
    fn init_parses_durable_bucket_settings() {
        let cli = Cli::try_parse_from([
            "synty",
            "init",
            "s3://team",
            "--aws-profile",
            "synty-writer",
            "--capture-since",
            "2026-07-21",
            "--upload-interval",
            "120",
        ])
        .expect("durable init settings parse");
        assert!(matches!(
            cli.cmd,
            Cmd::Init {
                aws_profile: Some(p), capture_since: Some(s), upload_interval: Some(120), ..
            } if p == "synty-writer" && s == "2026-07-21"
        ));
    }

    #[test]
    fn track_parses_capture_and_batch_overrides() {
        let cli = Cli::try_parse_from([
            "synty",
            "track",
            "--since",
            "now",
            "--upload-interval",
            "180",
        ])
        .expect("tracker overrides parse");
        assert!(matches!(
            cli.cmd,
            Cmd::Track { since: Some(s), upload_interval: Some(180), .. } if s == "now"
        ));
    }

    // `upgrade` is argument-free: it self-updates from the latest GitHub Release.
    #[test]
    fn upgrade_parses_without_args() {
        assert!(matches!(Cli::try_parse_from(["synty", "upgrade"]).expect("upgrade").cmd, Cmd::Upgrade));
        // The bucket-publish path is gone — distribution is GitHub Releases.
        assert!(Cli::try_parse_from(["synty", "publish"]).is_err());
    }

    // Trace reads are grouped under one namespace and keep scriptable filters
    // on the list/search operations rather than adding more top-level commands.
    #[test]
    fn trace_parses_forensic_list_show_search_and_compare_queries() {
        let list = Cli::try_parse_from([
            "synty", "trace", "list", "--type", "spans", "--status", "error",
            "--sort", "duration", "--limit", "7", "--json",
        ]).expect("filtered span list");
        assert!(matches!(
            list.cmd,
            Cmd::Trace {
                command: TraceCmd::List {
                    entity: TraceEntity::Spans,
                    status: Some(s),
                    sort: TraceSort::Duration,
                    limit: 7,
                    json: true,
                    ..
                }
            } if s == "error"
        ));

        let jobs = Cli::try_parse_from([
            "synty", "trace", "list", "--type", "jobs", "--operation", "whisper",
            "--sort", "wait",
        ]).expect("associated job list");
        assert!(matches!(
            jobs.cmd,
            Cmd::Trace {
                command: TraceCmd::List {
                    entity: TraceEntity::Jobs,
                    operation: Some(q),
                    sort: TraceSort::Wait,
                    ..
                }
            } if q == "whisper"
        ));

        let search = Cli::try_parse_from([
            "synty", "trace", "search", "libxcb.so.1", "--kind", "tool_result",
        ]).expect("raw evidence search");
        assert!(matches!(
            search.cmd,
            Cmd::Trace {
                command: TraceCmd::Search { query, kind: Some(k), .. }
            } if query == "libxcb.so.1" && k == "tool_result"
        ));

        let show = Cli::try_parse_from(["synty", "trace", "show", "job:01ABC"])
            .expect("job detail with the default evidence window");
        assert!(matches!(
            show.cmd,
            Cmd::Trace {
                command: TraceCmd::Show {
                    id,
                    before: 6,
                    after: 12,
                    json: false,
                }
            } if id == "job:01ABC"
        ));

        let compare = Cli::try_parse_from([
            "synty", "trace", "compare", "job:left", "job:right", "--json",
        ]).expect("job comparison");
        assert!(matches!(
            compare.cmd,
            Cmd::Trace {
                command: TraceCmd::Compare { left, right, json: true }
            } if left == "job:left" && right == "job:right"
        ));
    }

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
