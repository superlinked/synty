// `synty up` — one command for solo mode. It keeps the local index fresh as you
// work: drain the trackers, ingest, incrementally re-index, and publish, on a
// loop. `synty search` then always queries current data. Zero config — defaults
// to every source and a local-dir bucket. The first pass is a full build; later
// passes are cheap (cursors skip seen bytes, the embedding store skips seen
// text, an unchanged corpus skips the rebuild).

use crate::{index, ingest, track, CORPUS_DIR, DOCS_PATH};
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

/// `synty build` — the whole pipeline, once, fleet-aware. Summaries are
/// write-once shared through the bucket, so every machine contributes that
/// work lease-free; the index-build + cluster + publish runs under a soft
/// lease so concurrent viewers don't duplicate it — the loser pulls the
/// winner's output instead. `no_track` skips the local tailer pass (the
/// autostart tracker is already the machine's writer; a second one would just
/// race it).
pub fn build(bucket: &str, machine: &str, resolution: f64, no_track: bool) -> Result<()> {
    let machine = crate::identity::resolve_machine(machine);
    // Start from the fleet's current read-model, so topic-key lineage continues
    // from what the last builder published, not this machine's stale copy.
    match crate::sync::pull_if_stale(bucket) {
        Ok(true) => eprintln!("build: pulled the fleet's current read-model from {bucket}"),
        Ok(false) => {}
        Err(e) => eprintln!("build: read-model pull skipped ({e})"),
    }

    if !no_track {
        track::run(track::Opts {
            which: "all".into(),
            out: format!("{CORPUS_DIR}/local"),
            max_age_days: 90,
            machine: machine.clone(),
            watch: false,
            poll_secs: 60,
            install: None,
            cursors: ".synty/cursors.json".into(),
            bucket: Some(bucket.to_string()),
        })?;
    } else {
        // The tailer is skipped, not the backplane: this machine's events must
        // still reach the bucket for other builders (push is idempotent).
        match crate::sync::push_events(bucket, &format!("{CORPUS_DIR}/local")) {
            Ok(n) if n > 0 => eprintln!("build: pushed {n} event files → {bucket}/events/"),
            Ok(_) => {}
            Err(e) => eprintln!("build: event push skipped ({e})"),
        }
    }
    crate::progress::phase("ingesting", 0, 1);
    ingest::run(CORPUS_DIR, DOCS_PATH, Some(bucket))?;

    // Unit summaries against this machine's current view — store-first, so
    // concurrent viewers split the list instead of repeating it.
    summarize(bucket, "unit summaries");

    let b = crate::bucket::open(bucket)?;
    let held = crate::lease::acquire(b.as_ref(), "build", &machine, now_ms(), crate::lease::TTL_MS)
        .unwrap_or(true); // a bucket without conditional-put support → behave solo
    if held {
        index::run(DOCS_PATH, &crate::model_id(), bucket)?;
        summarize(bucket, "delta unit summaries"); // units the new docs snapshot revealed
        crate::topics::run(resolution, &crate::model_id(), bucket)?;
        if crate::lease::refresh(b.as_ref(), "build", &machine, now_ms(), crate::lease::TTL_MS).unwrap_or(true) {
            crate::progress::phase("publishing", 0, 1);
            let n = crate::sync::publish(bucket)?;
            if n > 0 {
                eprintln!("build: published {n} read-model objects → {bucket}");
            }
        } else {
            eprintln!("build: lost the build lease mid-build — keeping local, pulling the fleet's");
            let _ = crate::sync::pull_if_stale(bucket);
        }
        summarize(bucket, "topic summaries"); // reduce + name the fresh clusters
        let _ = crate::lease::release(b.as_ref(), "build", &machine);
    } else {
        eprintln!("build: another machine holds the build lease — pulling its output instead");
        let _ = crate::sync::pull_if_stale(bucket);
        summarize(bucket, "topic summaries"); // store-first pass over the pulled clusters
    }
    eprintln!("build: done — try `synty topic`, `synty search \"…\"`, or `synty tui`");
    Ok(())
}

#[cfg(feature = "llm")]
fn summarize(bucket: &str, what: &str) {
    if let Err(e) = crate::qwen::summarize_all(bucket) {
        eprintln!("build: {what} skipped ({e})");
    }
}

#[cfg(not(feature = "llm"))]
fn summarize(_bucket: &str, _what: &str) {}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
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
        bucket: Some(bucket.to_string()), // push events so a fleet build sees them
    })?;
    ingest::run(CORPUS_DIR, DOCS_PATH, Some(bucket))?;
    index::run(DOCS_PATH, &crate::model_id(), bucket)?;
    Ok(())
}
