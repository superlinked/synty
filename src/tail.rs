// Shared tailer machinery: a `Source` detects a file's format version and builds
// a stateful `FileParser`; `drive` feeds it complete JSONL lines and collects
// canonical envelopes.
//
// event_ids are deterministic in (source, file, line offset, sub-index): a
// re-parse after a lost cursor mints the same ids.

use crate::event::{deterministic_ulid, Event, Sequencer};
use anyhow::Result;
use chrono::{DateTime, SecondsFormat, TimeZone, Utc};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;

/// Deserialize a string field that may be JSON `null` (or absent) as "". Tools
/// emit nullable fields like `parent_tool_use_id`; serde rejects null for a
/// plain `String`, so a typed record needs this to avoid dropping the line.
pub fn de_null_string<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<String, D::Error> {
    Ok(Option::<String>::deserialize(d)?.unwrap_or_default())
}

/// A per-tool tailer: format detection + parser construction.
pub trait Source {
    /// Short id used in cursor keys and the ULID key ("claudecode").
    fn id(&self) -> &'static str;
    /// Canonical envelope source written on every event ("claude_code").
    fn envelope_source(&self) -> &'static str;
    /// Scan the head of a file and return its version stamp, or "" if unknown.
    fn detect_version(&self, head: &[u8]) -> String;
    /// Build a stateful parser for `version`, or None → format_unknown (skip).
    /// `head` is the same file prefix `detect_version` saw; parsers whose
    /// per-file state comes from the first lines (codex's session id) reseed
    /// from it, so a cursor resume mid-file doesn't lose that state.
    fn new_parser(&self, version: &str, head: &[u8]) -> Option<Box<dyn FileParser>>;
}

/// Parses one JSONL line into zero or more envelopes, owning whatever per-file
/// state it needs (subagent bookkeeping, heartbeat buffers, …) as struct fields.
pub trait FileParser {
    fn parse_line(&mut self, line: &[u8], ec: &mut EmitCtx) -> Result<Vec<Event>>;
}

/// Per-stream plumbing for minting envelopes, plus per-line context the driver
/// resets before each `parse_line`.
pub struct EmitCtx<'a> {
    pub stream: String,
    source_id: &'static str,
    envelope_source: &'static str,
    seq: &'a mut Sequencer,
    /// session_ids that already had their session_start emitted this process.
    started: &'a mut HashSet<String>,
    file: String,
    line_offset: i64,
    fallback_ms: i64,
    sub_index: i64,
}

impl<'a> EmitCtx<'a> {
    pub fn new(
        stream: String,
        src: &dyn Source,
        seq: &'a mut Sequencer,
        started: &'a mut HashSet<String>,
    ) -> Self {
        Self {
            stream,
            source_id: src.id(),
            envelope_source: src.envelope_source(),
            seq,
            started,
            file: String::new(),
            line_offset: 0,
            fallback_ms: 0,
            sub_index: 0,
        }
    }

    fn prepare_line(&mut self, file: &str, offset: i64, fallback_ms: i64) {
        self.file = file.to_string();
        self.line_offset = offset;
        self.fallback_ms = fallback_ms;
        self.sub_index = 0;
    }

    fn new_id(&mut self, ts_ms: i64) -> String {
        let key = format!("{}\0{}\0{}\0{}", self.source_id, self.file, self.line_offset, self.sub_index);
        self.sub_index += 1;
        deterministic_ulid(ts_ms.max(0) as u64, &key)
    }

    /// The fallback timestamp (file mtime) for records carrying no ts of their
    /// own. Parsers pass it into `resolve_ts`.
    pub fn fallback_ms(&self) -> i64 {
        self.fallback_ms
    }

    /// The byte offset of the line currently being parsed (for parsers that
    /// buffer lines to replay later).
    pub fn line_offset(&self) -> i64 {
        self.line_offset
    }

    /// Re-point minting at a buffered line's offset (resetting the sub-index),
    /// so a replayed line mints the same deterministic ids it would have live.
    pub fn replay_at(&mut self, offset: i64) {
        self.line_offset = offset;
        self.sub_index = 0;
    }

    /// Mint an envelope. `ts_ms` and `ts` come from `resolve_ts`.
    pub fn event(&mut self, ts_ms: i64, ts: &str, kind: &str, session_id: &str, payload: Value) -> Event {
        let event_id = self.new_id(ts_ms);
        let seq = self.seq.next(&self.stream);
        Event {
            event_id,
            stream: self.stream.clone(),
            seq,
            ts: ts.to_string(),
            source: self.envelope_source.to_string(),
            session_id: session_id.to_string(),
            kind: kind.to_string(),
            payload,
            rollup_dim: String::new(),
        }
    }

    /// True the first time a session_id is seen (so the caller emits exactly one
    /// session_start even when a session spans files).
    pub fn first_seen(&mut self, session_id: &str) -> bool {
        self.started.insert(session_id.to_string())
    }
}

/// Parse a record's RFC3339 timestamp into (epoch_ms, normalized RFC3339-Z).
/// Falls back to the file mtime so records without their own ts still get a
/// stable id and a sortable envelope ts.
pub fn resolve_ts(raw: &str, fallback_ms: i64) -> (i64, String) {
    if let Ok(dt) = DateTime::parse_from_rfc3339(raw) {
        let utc = dt.with_timezone(&Utc);
        return (utc.timestamp_millis(), utc.to_rfc3339_opts(SecondsFormat::Millis, true));
    }
    (fallback_ms, ms_to_rfc3339(fallback_ms))
}

pub fn ms_to_rfc3339(ms: i64) -> String {
    Utc.timestamp_millis_opt(ms)
        .single()
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().unwrap())
        .to_rfc3339_opts(SecondsFormat::Millis, true)
}

/// Drive a parser over a buffer of complete JSONL lines starting at byte
/// `start_offset`, collecting all emitted events. Empty lines advance the
/// offset without parsing; a malformed line is skipped and counted, so format
/// drift surfaces in metrics instead of silently thinning the data. Returns
/// (events, bytes_consumed, lines_skipped).
pub fn drive(
    parser: &mut dyn FileParser,
    content: &[u8],
    file: &str,
    start_offset: i64,
    fallback_ms: i64,
    ec: &mut EmitCtx,
) -> (Vec<Event>, i64, usize) {
    let mut out = Vec::new();
    let mut offset = start_offset;
    let mut skipped = 0usize;
    for raw in split_after_newline(content) {
        let line = trim_newline(raw);
        if line.is_empty() {
            offset += raw.len() as i64;
            continue;
        }
        ec.prepare_line(file, offset, fallback_ms);
        offset += raw.len() as i64;
        match parser.parse_line(line, ec) {
            Ok(evts) => out.extend(evts),
            Err(_) => skipped += 1,
        }
    }
    (out, offset - start_offset, skipped)
}

fn split_after_newline(b: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = 0;
    for i in 0..b.len() {
        if b[i] == b'\n' {
            out.push(&b[start..=i]);
            start = i + 1;
        }
    }
    if start < b.len() {
        out.push(&b[start..]);
    }
    out
}

fn trim_newline(b: &[u8]) -> &[u8] {
    let mut end = b.len();
    while end > 0 && (b[end - 1] == b'\n' || b[end - 1] == b'\r') {
        end -= 1;
    }
    &b[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_ts_parses_and_normalizes() {
        let (ms, s) = resolve_ts("2026-05-31T20:00:00.123Z", 0);
        assert_eq!(ms, 1780257600123);
        assert_eq!(s, "2026-05-31T20:00:00.123Z");
        // offset form normalizes to Z
        let (ms2, s2) = resolve_ts("2026-05-31T22:00:00+02:00", 0);
        assert_eq!(ms2, 1780257600000);
        assert_eq!(s2, "2026-05-31T20:00:00.000Z");
    }

    #[test]
    fn resolve_ts_falls_back_when_missing() {
        let (ms, s) = resolve_ts("", 1780257600123);
        assert_eq!(ms, 1780257600123);
        assert_eq!(s, "2026-05-31T20:00:00.123Z");
    }

    #[test]
    fn split_after_newline_keeps_partial_tail() {
        let parts = split_after_newline(b"a\nb\nc");
        assert_eq!(parts, vec![&b"a\n"[..], &b"b\n"[..], &b"c"[..]]);
    }
}
