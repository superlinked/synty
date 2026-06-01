// `synty track` — the native tracker. Discovers each source's session files,
// parses new content into canonical envelopes, and writes them out. This is the
// Rust replacement for the v1 Go agent.
//
// This first cut runs a one-shot drain (--once): walk the roots, parse every
// in-window file from the start, synthesize a session_end per session, write
// JSONL under <out>/<stream>/. The watch loop + persistent cursors land in a
// follow-up; the parsers and envelope output are the validated core.

use crate::claudecode::ClaudeCode;
use crate::event::{kind, Sequencer};
use crate::tail::{drive, ms_to_rfc3339, EmitCtx, Source};
use anyhow::{anyhow, bail, Result};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use walkdir::WalkDir;

const HEAD_BYTES: usize = 64 << 10;

/// Default watch roots per source on this machine.
fn default_roots(id: &str, home: &str) -> Vec<String> {
    match id {
        "claudecode" => vec![format!("{home}/.claude/projects")],
        _ => vec![],
    }
}

fn sources(which: &str) -> Result<Vec<Box<dyn Source>>> {
    let all: Vec<Box<dyn Source>> = vec![Box::new(ClaudeCode)];
    if which == "all" {
        return Ok(all);
    }
    let picked: Vec<Box<dyn Source>> = all.into_iter().filter(|s| s.id() == which).collect();
    if picked.is_empty() {
        bail!("unknown source {which} (have: claudecode)");
    }
    Ok(picked)
}

pub fn run(which: &str, out_dir: &str, max_age_days: u64, machine: &str) -> Result<()> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let now_ms = unix_ms(SystemTime::now());
    let cutoff_ms = if max_age_days == 0 { 0 } else { now_ms - (max_age_days as i64) * 86_400_000 };

    for src in sources(which)? {
        let stream = format!("edge-{machine}-{}", src.id());
        let mut seq = Sequencer::new();
        let mut started: HashSet<String> = HashSet::new();
        let mut ec = EmitCtx::new(stream.clone(), &*src, &mut seq, &mut started);

        let files = discover(&default_roots(src.id(), &home), cutoff_ms);
        let mut events = Vec::new();
        // last (ts_ms, ts) seen per session, to synthesize session_end on drain.
        let mut last_seen: HashMap<String, (i64, String)> = HashMap::new();
        let mut n_files = 0;

        for f in files {
            let Ok(content) = std::fs::read(&f) else { continue };
            let head = &content[..content.len().min(HEAD_BYTES)];
            let version = src.detect_version(head);
            let Some(mut parser) = src.new_parser(&version) else { continue };
            n_files += 1;
            let fallback_ms = file_mtime_ms(&f);
            let path = f.to_string_lossy();
            let (evts, _) = drive(&mut *parser, &content, &path, 0, fallback_ms, &mut ec);
            for e in &evts {
                if !e.session_id.is_empty() && e.kind != kind::SESSION_END {
                    last_seen.insert(e.session_id.clone(), (fallback_ms, e.ts.clone()));
                }
            }
            events.extend(evts);
        }

        // One synthesized session_end per session (the engine does this on idle
        // / shutdown; the drain does it once at the end).
        let mut ends: Vec<_> = last_seen.into_iter().collect();
        ends.sort_by(|a, b| a.0.cmp(&b.0));
        for (sid, (ts_ms, ts)) in ends {
            events.push(ec.event(ts_ms, &ts, kind::SESSION_END, &sid, json!({"reason": "drain"})));
        }

        write_stream(out_dir, &stream, &events, now_ms)?;
        eprintln!(
            "track {}: {} files → {} events ({} sessions) → {out_dir}/{stream}/",
            src.id(),
            n_files,
            events.len(),
            events.iter().filter(|e| e.kind == kind::SESSION_START).count(),
        );
    }
    Ok(())
}

/// Walk roots for *.jsonl files within the mtime window, oldest first.
fn discover(roots: &[String], cutoff_ms: i64) -> Vec<PathBuf> {
    let mut out: Vec<(i64, PathBuf)> = Vec::new();
    for root in roots {
        for e in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            let m = file_mtime_ms(p);
            if cutoff_ms > 0 && m < cutoff_ms {
                continue;
            }
            out.push((m, p.to_owned()));
        }
    }
    out.sort_by_key(|(m, _)| *m);
    out.into_iter().map(|(_, p)| p).collect()
}

fn write_stream(out_dir: &str, stream: &str, events: &[crate::event::Event], now_ms: i64) -> Result<()> {
    let dir = PathBuf::from(out_dir).join(stream);
    std::fs::create_dir_all(&dir)?;
    let fname = format!("drain-{}.jsonl", now_ms);
    let mut body = String::new();
    for e in events {
        body.push_str(&serde_json::to_string(e).map_err(|e| anyhow!("encode: {e}"))?);
        body.push('\n');
    }
    std::fs::write(dir.join(fname), body)?;
    Ok(())
}

fn file_mtime_ms(p: &std::path::Path) -> i64 {
    std::fs::metadata(p).and_then(|m| m.modified()).map(unix_ms).unwrap_or(0)
}

fn unix_ms(t: SystemTime) -> i64 {
    t.duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

#[allow(dead_code)]
fn rfc3339_now() -> String {
    ms_to_rfc3339(unix_ms(SystemTime::now()))
}
