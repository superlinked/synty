// CLI summaries over the shared `units` view-model (parity with the TUI):
//  - Sessions: repo, counts, opening ask, keyphrases, the one-line summary (or
//    extractive gist), files touched, effort, linked PR.
//  - Topics: the unit-level topic digest, same as `topic`, top N by recency.

use crate::short;
use anyhow::Result;

pub fn run(sessions: usize, topics: usize) -> Result<()> {
    session_summaries(sessions)?;
    println!();
    topic_digests(topics)?;
    Ok(())
}

/// Print the exact summarizer inputs (ask, keyphrases, selected turns with their
/// lengths) for every session — to inspect input quality without running the model.
pub fn dump_inputs() -> Result<()> {
    for s in crate::units::session_inputs()? {
        println!("## {} · {}", short(&s.id), s.repo);
        println!("ask [{} chars]: {}", s.ask.len(), s.ask);
        println!("keyphrases: {}", s.keyphrases.join(", "));
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
    let mut topics = crate::units::topic_units(12)?;
    if topics.is_empty() {
        eprintln!("(no topics — run `cluster` first)");
        return Ok(());
    }
    topics.truncate(n);
    print!("{}", crate::view::topics_md(&topics));
    Ok(())
}
