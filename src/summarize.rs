// CLI summaries over the shared `units` view-model (parity with the TUI):
//  - Sessions: repo, counts, opening ask, the one-line summary, files touched,
//    effort, linked PR.
//  - Topics: the unit-level topic digest, same as `topic`, top N by recency.

use crate::short;
use anyhow::Result;

pub fn run(sessions: usize, topics: usize) -> Result<()> {
    session_summaries(sessions)?;
    println!();
    topic_digests(topics)?;
    Ok(())
}

/// Print the exact summarizer inputs (ask, selected turns with their lengths)
/// for every session — to inspect input quality without running the model.
pub fn dump_inputs() -> Result<()> {
    for s in crate::units::session_inputs()? {
        println!("## {} · {}", short(&s.id), s.repo);
        println!("ask [{} chars]: {}", s.ask.len(), s.ask);
        for (i, t) in s.turns.iter().enumerate() {
            println!("  turn{i} [{} chars]: {}", t.len(), t);
        }
        println!();
    }
    Ok(())
}

fn session_summaries(n: usize) -> Result<()> {
    let mut all = crate::units::sessions()?;
    all.sort_by(|a, b| b.ended.cmp(&a.ended));
    // Capture coverage for the token accounting: the share of sessions whose
    // source actually recorded usage (cowork and pre-capture envelopes
    // don't). Emitted here, not in the LLM pass — this path always runs.
    let with = all.iter().filter(|s| s.has_usage()).count();
    crate::metrics::Run::new("stats")
        .set("sessions", all.len())
        .set("with_usage", with)
        .set("usage_coverage_pct", 100.0 * with as f64 / all.len().max(1) as f64)
        .emit();
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
        for line in [crate::view::usage_line(&s), crate::view::tools_line(&s)].into_iter().flatten() {
            println!("{line}");
        }
        println!("ask: {}", s.ask);
        if let Some(sum) = &s.summary {
            println!("summary: {sum}");
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
    let mut topics = crate::units::topic_units(12)?;
    if topics.is_empty() {
        eprintln!("(no topics — run `cluster` first)");
        return Ok(());
    }
    topics.truncate(n);
    print!("{}", crate::view::topics_md(&topics));
    Ok(())
}
