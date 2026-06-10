// `synty up` — one command for solo mode. It keeps the local index fresh as you
// work: drain the trackers, ingest, incrementally re-index, and publish, on a
// loop. `synty search` then always queries current data. Zero config — defaults
// to every source and a local-dir bucket. The first pass is a full build; later
// passes are cheap (cursors skip seen bytes, the embedding store skips seen
// text, an unchanged corpus skips the rebuild).

use crate::{index, ingest, track, CORPUS_DIR, DOCS_PATH, INDEX_PATH};
use anyhow::Result;
use std::time::Duration;

pub fn run(bucket: &str, machine: &str, poll_secs: u64, github: bool) -> Result<()> {
    eprintln!(
        "synty up: track + index every {poll_secs}s (bucket {bucket}). Ctrl-C to stop.",
    );

    // GitHub changes slowly; pull once at startup, best-effort (needs a token).
    if github {
        match crate::config::load().org {
            Some(org) => {
                if let Err(e) = crate::github::run(&org, None, 90, &format!("{CORPUS_DIR}/github")) {
                    eprintln!("up: github pull skipped ({e})");
                }
            }
            None => eprintln!("up: no GitHub org configured — run `synty setup` to add one"),
        }
    }

    let mut iteration = 0u64;
    loop {
        iteration += 1;
        if let Err(e) = tick(bucket, machine, poll_secs) {
            eprintln!("up: iteration {iteration} error: {e}");
        }
        std::thread::sleep(Duration::from_secs(poll_secs.max(1)));
    }
}

/// `synty build` — the whole pipeline, once: track + ingest + index (what `up`
/// loops), then unit summaries → cluster → topic summaries, so `topic`/`tui`
/// are fully populated without knowing the step order.
pub fn build(bucket: &str, machine: &str, resolution: f64) -> Result<()> {
    tick(bucket, machine, 60)?;
    #[cfg(feature = "llm")]
    if let Err(e) = crate::qwen::summarize_all() {
        eprintln!("build: unit summaries skipped ({e})");
    }
    crate::topics::run(resolution, &crate::model_id(), bucket)?;
    // Second summary pass: the topic reduce consumes the clusters just written.
    #[cfg(feature = "llm")]
    if let Err(e) = crate::qwen::summarize_all() {
        eprintln!("build: topic summaries skipped ({e})");
    }
    eprintln!("build: done — try `synty topic`, `synty search \"…\"`, or `synty tui`");
    Ok(())
}

fn tick(bucket: &str, machine: &str, poll_secs: u64) -> Result<()> {
    track::run(track::Opts {
        which: "all".into(),
        out: format!("{CORPUS_DIR}/local"),
        max_age_days: 90,
        machine: machine.into(),
        watch: false,
        poll_secs,
        install: None,
        cursors: ".synty/cursors.json".into(),
        bucket: None, // solo: events stay local; the bucket holds embeddings + index
    })?;
    ingest::run(CORPUS_DIR, DOCS_PATH, None)?;
    index::run(DOCS_PATH, INDEX_PATH, &crate::model_id(), bucket)?;
    Ok(())
}
