// Canonical foreign-event importer for campaign harnesses and Devin exports.
// It normalizes NDJSON into synty's add-only envelope v1, mints deterministic
// ids, applies capture/redaction policy, and appends idempotently to one owned
// local stream.

use crate::event::{Event, deterministic_ulid, kind};
use anyhow::{Context, Result, bail};
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
    let mut seen = existing_ids(&out_dir)?;
    let mut started: HashSet<String> = HashSet::new();
    let mut per_day: BTreeMap<String, Vec<Event>> = BTreeMap::new();
    let mut report = Report::default();
    let mut quarantine = quarantine_writer(opts.quarantine.as_deref())?;
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
                write_reject(&mut quarantine, line_index, &error.to_string(), &line)?;
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
                write_reject(&mut quarantine, line_index, "repo_mismatch", &line)?;
                continue;
            }
            ensure_payload_string(&mut event.payload, "repo", expected);
        }
        let capture_repos = &crate::config::load().capture_repos;
        let event_repo = event.payload["repo"].as_str().unwrap_or("");
        if !capture_repos.is_empty() && !capture_repos.iter().any(|repo| repo == event_repo) {
            report.rejected += 1;
            write_reject(&mut quarantine, line_index, "repo_denied", &line)?;
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
            let start = synthetic_start(&event, &opts, &stream);
            if seen.insert(start.event_id.clone()) {
                push_day(&mut per_day, start);
            }
        } else if event.kind == kind::SESSION_START {
            started.insert(event.session_id.clone());
        }
        push_day(&mut per_day, event);
        report.imported += 1;
    }

    if !opts.dry_run {
        std::fs::create_dir_all(&out_dir)?;
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
    eprintln!(
        "[metrics import]\nread={}\nimported={}\nduplicate={}\nrejected={}\nbefore_boundary={}",
        report.read, report.imported, report.duplicate, report.rejected, report.before_boundary
    );
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
                event.event_id = stable_id(&opts.input, line_index, &event.ts, source);
            }
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
        event.event_id = stable_id(&opts.input, line_index, &event.ts, source);
    }
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
    if !payload.contains_key("text") {
        if let Some(text) = first_string(&row, &["text", "content", "message"]) {
            payload.insert("text".into(), Value::String(text.into()));
        }
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
    event.event_id = if foreign.is_empty() {
        stable_id(&opts.input, line_index, &event.ts, source)
    } else {
        stable_id(foreign, 0, &event.ts, source)
    };
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
        event_id: stable_id(&event.session_id, 0, &event.ts, "session_start"),
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

fn stable_id(key: &str, line: usize, ts: &str, source: &str) -> String {
    let ts_ms = chrono::DateTime::parse_from_rfc3339(ts)
        .map(|value| value.timestamp_millis().max(0) as u64)
        .unwrap_or(0);
    deterministic_ulid(ts_ms, &format!("{source}\0{key}\0{line}"))
}

fn existing_ids(dir: &Path) -> Result<HashSet<String>> {
    let mut ids = HashSet::new();
    if !dir.is_dir() {
        return Ok(ids);
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        for line in std::io::BufReader::new(std::fs::File::open(path)?).lines() {
            if let Ok(event) = serde_json::from_str::<Event>(&line?) {
                ids.insert(event.event_id);
            }
        }
    }
    Ok(ids)
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

fn quarantine_writer(path: Option<&Path>) -> Result<Option<std::fs::File>> {
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
) -> Result<()> {
    if let Some(writer) = writer {
        serde_json::to_writer(
            &mut *writer,
            &json!({"line": line + 1, "reason": reason, "raw": raw}),
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
    fn stable_ids_make_reimport_idempotent() {
        let a = stable_id("foreign", 7, "2026-07-22T12:00:00Z", "harness");
        let b = stable_id("foreign", 7, "2026-07-22T12:00:00Z", "harness");
        assert_eq!(a, b);
    }
}
