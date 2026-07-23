// Canonical foreign-event importer for campaign harnesses and Devin exports.
// It normalizes NDJSON into synty's add-only envelope v1, mints deterministic
// ids, applies capture/redaction policy, and appends idempotently to one owned
// local stream.

use crate::event::{Event, deterministic_ulid, kind};
use anyhow::{Context, Result, bail};
use fs2::FileExt;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
pub enum Format {
    Envelope,
    Harness,
    Devin,
}

impl std::str::FromStr for Format {
    type Err = anyhow::Error;

    fn from_str(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "envelope" => Ok(Self::Envelope),
            "harness" | "harness_ndjson" => Ok(Self::Harness),
            "devin" | "devin_v3" => Ok(Self::Devin),
            _ => bail!("import format must be envelope, harness, or devin"),
        }
    }
}

pub struct Opts {
    pub input: String,
    pub format: Format,
    pub machine: String,
    pub campaign: Option<String>,
    pub role: Option<String>,
    pub repo: Option<String>,
    pub actor: Option<String>,
    pub since_ms: Option<i64>,
    pub dry_run: bool,
    pub quarantine: Option<PathBuf>,
    pub bucket: Option<String>,
    pub redaction: crate::redact::Profile,
}

#[derive(Default)]
pub struct Report {
    pub read: usize,
    pub imported: usize,
    pub duplicate: usize,
    pub rejected: usize,
    pub before_boundary: usize,
}

pub fn run(opts: Opts) -> Result<Report> {
    validate_component(&opts.machine, "machine")?;
    let source = match opts.format {
        Format::Envelope | Format::Harness => crate::event::source::HARNESS,
        Format::Devin => crate::event::source::DEVIN,
    };
    let stream = format!("edge-{}-{source}", opts.machine);
    let out_dir = Path::new(crate::units::LOCAL_DIR).join(&stream);
    let _lock = if opts.dry_run {
        None
    } else {
        std::fs::create_dir_all(&out_dir)?;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(out_dir.join(".import.lock"))?;
        file.lock_exclusive()?;
        Some(file)
    };
    let existing = existing_state(&out_dir)?;
    let mut seen = existing.ids;
    let mut started = existing.started;
    let mut next_seq = existing.max_seq.saturating_add(1);
    let mut per_day: BTreeMap<String, Vec<Event>> = BTreeMap::new();
    let mut report = Report::default();
    let mut quarantine = quarantine_writer(opts.quarantine.as_deref(), opts.dry_run)?;
    let capture_repos = crate::config::load().capture_repos;
    let reader: Box<dyn BufRead> = if opts.input == "-" {
        Box::new(std::io::BufReader::new(std::io::stdin()))
    } else {
        Box::new(std::io::BufReader::new(
            std::fs::File::open(&opts.input)
                .with_context(|| format!("open import {}", opts.input))?,
        ))
    };

    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        report.read += 1;
        let normalized = normalize(
            &line,
            line_index,
            source,
            &stream,
            &opts,
        );
        let mut event = match normalized {
            Ok(event) => event,
            Err(error) => {
                report.rejected += 1;
                write_reject(&mut quarantine, line_index, &error.to_string(), &line, opts.redaction)?;
                continue;
            }
        };
        if !crate::config::captured_at(&event.ts, opts.since_ms) {
            report.before_boundary += 1;
            continue;
        }
        let event_repo = event.payload["repo"].as_str().unwrap_or("");
        if let Some(expected) = opts.repo.as_deref() {
            if !event_repo.is_empty() && event_repo != expected {
                report.rejected += 1;
                write_reject(&mut quarantine, line_index, "repo_mismatch", &line, opts.redaction)?;
                continue;
            }
            ensure_payload_string(&mut event.payload, "repo", expected);
        }
        let event_repo = event.payload["repo"].as_str().unwrap_or("");
        if !capture_repos.is_empty() && !capture_repos.iter().any(|repo| repo == event_repo) {
            report.rejected += 1;
            write_reject(&mut quarantine, line_index, "repo_denied", &line, opts.redaction)?;
            continue;
        }
        crate::redact::value(&mut event.payload, opts.redaction);
        if !seen.insert(event.event_id.clone()) {
            report.duplicate += 1;
            continue;
        }
        if !event.session_id.is_empty()
            && event.kind != kind::SESSION_START
            && started.insert(event.session_id.clone())
        {
            let mut start = synthetic_start(&event, &opts, &stream);
            if seen.insert(start.event_id.clone()) {
                start.seq = next_seq;
                next_seq = next_seq.saturating_add(1);
                push_day(&mut per_day, start);
            }
        } else if event.kind == kind::SESSION_START {
            started.insert(event.session_id.clone());
        }
        event.seq = next_seq;
        next_seq = next_seq.saturating_add(1);
        push_day(&mut per_day, event);
        report.imported += 1;
    }

    if !opts.dry_run {
        for (day, events) in per_day {
            let path = out_dir.join(format!("track.{day}.jsonl"));
            let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
            for event in events {
                serde_json::to_writer(&mut file, &event)?;
                file.write_all(b"\n")?;
            }
        }
        if let Some(bucket) = opts.bucket.as_deref() {
            let owned = BTreeSet::from([stream]);
            crate::sync::push_events_for_streams(
                bucket,
                crate::units::LOCAL_DIR,
                ".synty/uploads.json",
                opts.since_ms,
                &owned,
            )?;
        }
    }
    crate::metrics::Run::new("import")
        .set("read", report.read)
        .set("imported", report.imported)
        .set("duplicate", report.duplicate)
        .set("rejected", report.rejected)
        .set("before_boundary", report.before_boundary)
        .emit();
    Ok(report)
}

fn normalize(
    line: &str,
    line_index: usize,
    source: &str,
    stream: &str,
    opts: &Opts,
) -> Result<Event> {
    match opts.format {
        Format::Envelope => {
            let mut event: Event = serde_json::from_str(line)?;
            anyhow::ensure!(event.v == crate::event::ENVELOPE_V, "unsupported envelope version");
            event.stream = stream.to_string();
            stamp(&mut event, opts);
            if event.event_id.is_empty() {
                event.event_id = stable_event_id(&event, "");
            }
            validate_timestamp(&event.ts)?;
            Ok(event)
        }
        Format::Harness => normalize_harness(line, line_index, source, stream, opts),
        Format::Devin => normalize_devin(line, line_index, source, stream, opts),
    }
}

#[derive(Deserialize)]
struct HarnessRow {
    #[serde(default)]
    event_id: String,
    #[serde(default)]
    run_id: String,
    #[serde(default)]
    campaign_id: String,
    #[serde(default)]
    role: String,
    #[serde(default)]
    backend: String,
    ts: String,
    session_id: String,
    kind: String,
    #[serde(default)]
    payload: Value,
    #[serde(default)]
    repo: String,
    #[serde(default)]
    candidate_sha: String,
}

fn normalize_harness(
    line: &str,
    line_index: usize,
    source: &str,
    stream: &str,
    opts: &Opts,
) -> Result<Event> {
    let row: HarnessRow = serde_json::from_str(line)?;
    anyhow::ensure!(!row.ts.is_empty(), "missing ts");
    anyhow::ensure!(!row.session_id.is_empty(), "missing session_id");
    anyhow::ensure!(!row.kind.is_empty(), "missing kind");
    let mut payload = object(row.payload);
    insert_string(&mut payload, "harness_run_id", row.run_id);
    insert_string(&mut payload, "campaign_id", row.campaign_id);
    insert_string(&mut payload, "campaign_role", row.role);
    insert_string(&mut payload, "backend", row.backend);
    insert_string(&mut payload, "repo", row.repo);
    insert_string(&mut payload, "candidate_sha", row.candidate_sha);
    let mut event = Event {
        v: crate::event::ENVELOPE_V,
        event_id: row.event_id,
        stream: stream.into(),
        seq: line_index as i64,
        ts: row.ts,
        source: source.into(),
        session_id: row.session_id,
        kind: row.kind,
        payload: Value::Object(payload),
        rollup_dim: String::new(),
    };
    stamp(&mut event, opts);
    if event.event_id.is_empty() {
        event.event_id = stable_event_id(&event, "");
    }
    validate_timestamp(&event.ts)?;
    Ok(event)
}

fn normalize_devin(
    line: &str,
    line_index: usize,
    source: &str,
    stream: &str,
    opts: &Opts,
) -> Result<Event> {
    let row: Value = serde_json::from_str(line)?;
    let ts = first_string(&row, &["ts", "timestamp", "created_at", "createdAt"])
        .context("missing Devin event timestamp")?;
    let session_id = first_string(&row, &["session_id", "sessionId"])
        .context("missing Devin session id")?;
    let native_kind = first_string(&row, &["kind", "type"]).unwrap_or("agent_meta");
    let kind = match native_kind {
        "user" | "user_message" | "prompt" => kind::USER_PROMPT,
        "assistant" | "assistant_message" | "message" => kind::ASSISTANT_MESSAGE,
        "tool_call" | "tool_start" => kind::TOOL_CALL,
        "tool_result" | "tool_end" => kind::TOOL_RESULT,
        "thinking" | "reasoning" => kind::THINKING,
        _ => kind::AGENT_META,
    };
    let mut payload = row["payload"].as_object().cloned().unwrap_or_default();
    if !payload.contains_key("text")
        && let Some(text) = first_string(&row, &["text", "content", "message"])
    {
        payload.insert("text".into(), Value::String(text.into()));
    }
    if let Some(id) = first_string(&row, &["event_id", "eventId", "id"]) {
        payload.insert("devin_event_id".into(), Value::String(id.into()));
    }
    if let Some(repo) = first_string(&row, &["repo", "repository"]) {
        payload.insert("repo".into(), Value::String(repo.into()));
    }
    let mut event = Event {
        v: crate::event::ENVELOPE_V,
        event_id: String::new(),
        stream: stream.into(),
        seq: line_index as i64,
        ts: ts.into(),
        source: source.into(),
        session_id: session_id.into(),
        kind: kind.into(),
        payload: Value::Object(payload),
        rollup_dim: String::new(),
    };
    stamp(&mut event, opts);
    let foreign = first_string(&row, &["event_id", "eventId", "id"]).unwrap_or("");
    event.event_id = stable_event_id(&event, foreign);
    validate_timestamp(&event.ts)?;
    Ok(event)
}

fn stamp(event: &mut Event, opts: &Opts) {
    if let Some(campaign) = opts.campaign.as_deref() {
        event.rollup_dim = campaign.into();
        ensure_payload_string(&mut event.payload, "campaign_id", campaign);
    } else if event.rollup_dim.is_empty() {
        event.rollup_dim = event.payload["campaign_id"].as_str().unwrap_or("").into();
    }
    if let Some(role) = opts.role.as_deref() {
        ensure_payload_string(&mut event.payload, "campaign_role", role);
    }
    if let Some(actor) = opts.actor.as_deref() {
        ensure_payload_string(&mut event.payload, "actor", actor);
    }
}

fn synthetic_start(event: &Event, opts: &Opts, stream: &str) -> Event {
    let mut payload = json!({
        "cwd": event.payload["repo"].as_str().unwrap_or(""),
        "repo": event.payload["repo"].as_str().unwrap_or(""),
        "actor": opts.actor.as_deref().unwrap_or(""),
        "campaign_id": event.rollup_dim,
        "campaign_role": event.payload["campaign_role"].as_str().unwrap_or(""),
        "backend": event.payload["backend"].as_str().unwrap_or(&event.source),
        "tracker_version": env!("CARGO_PKG_VERSION"),
    });
    crate::redact::value(&mut payload, opts.redaction);
    Event {
        v: crate::event::ENVELOPE_V,
        event_id: deterministic_id(&event.ts, &format!("{}\0{}\0session_start", event.source, event.session_id)),
        stream: stream.into(),
        seq: event.seq.saturating_sub(1),
        ts: event.ts.clone(),
        source: event.source.clone(),
        session_id: event.session_id.clone(),
        kind: kind::SESSION_START.into(),
        payload,
        rollup_dim: event.rollup_dim.clone(),
    }
}

fn stable_event_id(event: &Event, foreign_id: &str) -> String {
    let identity = if foreign_id.is_empty() {
        format!(
            "{}\0{}\0{}\0{}",
            event.source,
            event.session_id,
            event.kind,
            serde_json::to_string(&event.payload).unwrap_or_default(),
        )
    } else {
        format!("{}\0foreign\0{foreign_id}", event.source)
    };
    deterministic_id(&event.ts, &identity)
}

fn deterministic_id(ts: &str, identity: &str) -> String {
    let ts_ms = chrono::DateTime::parse_from_rfc3339(ts)
        .map(|value| value.timestamp_millis().max(0) as u64)
        .unwrap_or_default();
    deterministic_ulid(ts_ms, identity)
}

fn validate_timestamp(ts: &str) -> Result<()> {
    chrono::DateTime::parse_from_rfc3339(ts)
        .with_context(|| format!("timestamp is not RFC3339: {ts}"))?;
    Ok(())
}

#[derive(Default)]
struct ExistingState {
    ids: HashSet<String>,
    started: HashSet<String>,
    max_seq: i64,
}

fn existing_state(dir: &Path) -> Result<ExistingState> {
    let mut state = ExistingState { max_seq: -1, ..Default::default() };
    if !dir.is_dir() {
        return Ok(state);
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        for line in std::io::BufReader::new(std::fs::File::open(path)?).lines() {
            if let Ok(event) = serde_json::from_str::<Event>(&line?) {
                state.max_seq = state.max_seq.max(event.seq);
                state.ids.insert(event.event_id);
                if event.kind == kind::SESSION_START && !event.session_id.is_empty() {
                    state.started.insert(event.session_id);
                }
            }
        }
    }
    Ok(state)
}

fn push_day(days: &mut BTreeMap<String, Vec<Event>>, event: Event) {
    let day = event.ts.get(..10).unwrap_or("unknown").to_string();
    days.entry(day).or_default().push(event);
}

fn validate_component(value: &str, name: &str) -> Result<()> {
    anyhow::ensure!(
        !value.is_empty()
            && value
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_')),
        "{name} must contain only letters, digits, dash, or underscore"
    );
    Ok(())
}

fn first_string<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|key| value[*key].as_str().filter(|text| !text.is_empty()))
}

fn object(value: Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}

fn insert_string(payload: &mut Map<String, Value>, key: &str, value: String) {
    if !value.is_empty() {
        payload.insert(key.into(), Value::String(value));
    }
}

fn ensure_payload_string(payload: &mut Value, key: &str, value: &str) {
    if !payload.is_object() {
        *payload = json!({});
    }
    payload[key] = Value::String(value.into());
}

fn quarantine_writer(path: Option<&Path>, dry_run: bool) -> Result<Option<std::fs::File>> {
    if dry_run {
        return Ok(None);
    }
    path.map(|path| {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::OpenOptions::new().create(true).append(true).open(path)
    })
    .transpose()
    .map_err(Into::into)
}

fn write_reject(
    writer: &mut Option<std::fs::File>,
    line: usize,
    reason: &str,
    raw: &str,
    profile: crate::redact::Profile,
) -> Result<()> {
    if let Some(writer) = writer {
        serde_json::to_writer(
            &mut *writer,
            &json!({"line": line + 1, "reason": reason, "raw": crate::redact::text(raw, profile)}),
        )?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(format: Format) -> Opts {
        Opts {
            input: "fixture".into(),
            format,
            machine: "worker-1".into(),
            campaign: Some("campaign-1".into()),
            role: Some("investigator".into()),
            repo: Some("sie-internal".into()),
            actor: Some("alice".into()),
            since_ms: None,
            dry_run: true,
            quarantine: None,
            bucket: None,
            redaction: crate::redact::Profile::Standard,
        }
    }

    #[test]
    fn harness_rows_normalize_to_campaign_envelopes() {
        let row = json!({
            "run_id": "run-1", "campaign_id": "campaign-1", "role": "primary",
            "backend": "codex", "ts": "2026-07-22T12:00:00Z",
            "session_id": "session-1", "kind": "user_prompt",
            "payload": {"text": "optimize"}, "repo": "sie-internal"
        });
        let event =
            normalize_harness(&row.to_string(), 0, "harness", "edge-worker-1-harness", &opts(Format::Harness))
                .unwrap();
        assert_eq!(event.v, 1);
        assert_eq!(event.rollup_dim, "campaign-1");
        assert_eq!(event.payload["campaign_role"], "investigator");
        assert_eq!(event.payload["repo"], "sie-internal");
        assert!(!event.event_id.is_empty());
    }

    #[test]
    fn devin_rows_map_tool_lifecycle_and_keep_native_id() {
        let row = json!({
            "id": "devin-event-1", "timestamp": "2026-07-22T12:00:00Z",
            "sessionId": "session-1", "type": "tool_call",
            "payload": {"name": "shell"}, "repository": "sie-internal"
        });
        let event =
            normalize_devin(&row.to_string(), 0, "devin", "edge-worker-1-devin", &opts(Format::Devin))
                .unwrap();
        assert_eq!(event.kind, kind::TOOL_CALL);
        assert_eq!(event.payload["devin_event_id"], "devin-event-1");
    }

    #[test]
    fn content_ids_survive_reordering_without_colliding_at_the_same_line() {
        let first = json!({
            "ts": "2026-07-22T12:00:00Z", "session_id": "s", "kind": "user_prompt",
            "payload": {"text": "first"}
        });
        let second = json!({
            "ts": "2026-07-22T12:00:00Z", "session_id": "s", "kind": "user_prompt",
            "payload": {"text": "second"}
        });
        let a = normalize_harness(&first.to_string(), 0, "harness", "edge-w-harness", &opts(Format::Harness)).unwrap();
        let reordered = normalize_harness(&first.to_string(), 9, "harness", "edge-w-harness", &opts(Format::Harness)).unwrap();
        let b = normalize_harness(&second.to_string(), 0, "harness", "edge-w-harness", &opts(Format::Harness)).unwrap();
        assert_eq!(a.event_id, reordered.event_id);
        assert_ne!(a.event_id, b.event_id);
    }

    #[test]
    fn malformed_timestamps_are_quarantinable_instead_of_minting_epoch_ids() {
        let row = json!({"ts": "yesterday", "session_id": "s", "kind": "user_prompt"});
        assert!(normalize_harness(&row.to_string(), 0, "harness", "edge-w-harness", &opts(Format::Harness)).is_err());
    }

    #[test]
    fn dry_run_does_not_create_a_quarantine_file() {
        let path = std::env::temp_dir().join(format!("synty-import-dry-quarantine-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        assert!(quarantine_writer(Some(&path), true).unwrap().is_none());
        assert!(!path.exists());
    }

    #[test]
    fn existing_state_remembers_started_sessions_and_sequence_tail() {
        let root = std::env::temp_dir().join(format!("synty-import-existing-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let event = Event {
            v: 1,
            event_id: "start".into(),
            stream: "edge-w-harness".into(),
            seq: 41,
            ts: "2026-07-22T12:00:00Z".into(),
            source: "harness".into(),
            session_id: "session-1".into(),
            kind: kind::SESSION_START.into(),
            payload: json!({}),
            rollup_dim: String::new(),
        };
        std::fs::write(root.join("track.2026-07-22.jsonl"), format!("{}\n", serde_json::to_string(&event).unwrap())).unwrap();
        let state = existing_state(&root).unwrap();
        assert!(state.started.contains("session-1"));
        assert_eq!(state.max_seq, 41);
        let _ = std::fs::remove_dir_all(root);
    }
}
