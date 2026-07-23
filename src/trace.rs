// Factual execution traces over the canonical event corpus. This projection
// turns source-native tasks/turns and paired tool calls into compact queryable
// rows; it deliberately stops short of judging bottlenecks or causes.

use crate::event::Event;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

const EXCERPT: usize = 360;
const TIMELINE_CAP: usize = 200;

#[derive(Clone, Default, Serialize, Deserialize)]
struct SessionContext {
    source: String,
    machine: String,
    cwd: String,
    repo: String,
    campaign: String,
    role: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct TraceEvent {
    id: String,
    ts: String,
    source: String,
    machine: String,
    session_id: String,
    kind: String,
    summary: String,
    /// Bounded normalized evidence used by literal search in the published
    /// trace projection. It is intentionally not returned to clients.
    #[serde(skip)]
    search_text: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    subtype: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    event_kind: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    turn_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u64>,
    #[serde(skip)]
    span: Option<usize>,
    #[serde(skip)]
    ordinal: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct Span {
    id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    result_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    turn_id: String,
    session_id: String,
    source: String,
    machine: String,
    repo: String,
    started: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    ended: String,
    operation: String,
    status: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    process_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exit_code: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    duration_source: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    workdir: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    input: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    output: String,
    #[serde(skip)]
    start_event: usize,
    #[serde(skip)]
    end_event: Option<usize>,
}

#[derive(Clone, Serialize, Deserialize)]
struct Job {
    id: String,
    process_id: String,
    association: String,
    initial_span_id: String,
    final_span_id: String,
    session_id: String,
    turn_ids: Vec<String>,
    source: String,
    machine: String,
    repo: String,
    started: String,
    ended: String,
    status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    elapsed_ms: Option<u64>,
    reported_wait_ms: u64,
    spans: usize,
    polls: usize,
    stdin_writes: usize,
    errors: usize,
    command: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    final_output: String,
    #[serde(skip)]
    span_indexes: Vec<usize>,
}

#[derive(Clone, Serialize, Deserialize)]
struct Turn {
    id: String,
    session_id: String,
    source: String,
    machine: String,
    repo: String,
    started: String,
    ended: String,
    status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u64>,
    duration_source: String,
    ask: String,
    events: usize,
    spans: usize,
    errors: usize,
    #[serde(skip)]
    event_indexes: Vec<usize>,
    #[serde(skip)]
    span_indexes: Vec<usize>,
}

#[derive(Default)]
struct TurnBuilder {
    id: String,
    ask: String,
    status: String,
    duration_ms: Option<u64>,
    duration_source: String,
    event_indexes: Vec<usize>,
    span_indexes: Vec<usize>,
}

#[derive(Default)]
struct TraceStore {
    events: Vec<TraceEvent>,
    spans: Vec<Span>,
    jobs: Vec<Job>,
    turns: Vec<Turn>,
    sessions: HashMap<String, SessionContext>,
    session_events: HashMap<String, Vec<usize>>,
    ids: HashSet<String>,
}

const TRACE_FORMAT: u32 = 1;
const SEARCH_EVIDENCE: usize = 512;

/// Serializable form keeps renderer-only indexes out of public tool JSON while
/// retaining them in parallel arrays for exact reconstruction.
#[derive(Serialize, Deserialize)]
struct TraceSnapshot {
    format: u32,
    events: Vec<TraceEvent>,
    event_search: Vec<String>,
    event_spans: Vec<Option<usize>>,
    event_ordinals: Vec<usize>,
    spans: Vec<Span>,
    span_events: Vec<(usize, Option<usize>)>,
    jobs: Vec<Job>,
    job_spans: Vec<Vec<usize>>,
    turns: Vec<Turn>,
    turn_events: Vec<Vec<usize>>,
    turn_spans: Vec<Vec<usize>>,
    sessions: BTreeMap<String, SessionContext>,
    session_events: BTreeMap<String, Vec<usize>>,
}

impl TraceSnapshot {
    /// Separate renderer indexes from client-visible records before encoding.
    fn from_store(store: TraceStore) -> Self {
        Self {
            format: TRACE_FORMAT,
            event_search: store.events.iter().map(|event| event.search_text.clone()).collect(),
            event_spans: store.events.iter().map(|event| event.span).collect(),
            event_ordinals: store.events.iter().map(|event| event.ordinal).collect(),
            span_events: store
                .spans
                .iter()
                .map(|span| (span.start_event, span.end_event))
                .collect(),
            job_spans: store.jobs.iter().map(|job| job.span_indexes.clone()).collect(),
            turn_events: store.turns.iter().map(|turn| turn.event_indexes.clone()).collect(),
            turn_spans: store.turns.iter().map(|turn| turn.span_indexes.clone()).collect(),
            events: store.events,
            spans: store.spans,
            jobs: store.jobs,
            turns: store.turns,
            sessions: store.sessions.into_iter().collect(),
            session_events: store.session_events.into_iter().collect(),
        }
    }

    /// Validate every parallel index before reconstructing a queryable store.
    fn into_store(mut self) -> Result<TraceStore> {
        anyhow::ensure!(self.format == TRACE_FORMAT, "unsupported trace snapshot format {}", self.format);
        anyhow::ensure!(
            self.events.len() == self.event_search.len()
                && self.events.len() == self.event_spans.len()
                && self.events.len() == self.event_ordinals.len()
                && self.spans.len() == self.span_events.len()
                && self.jobs.len() == self.job_spans.len()
                && self.turns.len() == self.turn_events.len()
                && self.turns.len() == self.turn_spans.len(),
            "trace snapshot index lengths do not match"
        );
        for (index, event) in self.events.iter_mut().enumerate() {
            event.search_text = std::mem::take(&mut self.event_search[index]);
            event.span = self.event_spans[index];
            event.ordinal = self.event_ordinals[index];
        }
        for (span, (start, end)) in self.spans.iter_mut().zip(self.span_events) {
            span.start_event = start;
            span.end_event = end;
        }
        for (job, spans) in self.jobs.iter_mut().zip(self.job_spans) {
            job.span_indexes = spans;
        }
        for ((turn, events), spans) in self
            .turns
            .iter_mut()
            .zip(self.turn_events)
            .zip(self.turn_spans)
        {
            turn.event_indexes = events;
            turn.span_indexes = spans;
        }
        let mut store = TraceStore {
            events: self.events,
            spans: self.spans,
            jobs: self.jobs,
            turns: self.turns,
            sessions: self.sessions.into_iter().collect(),
            session_events: self.session_events.into_iter().collect(),
            ids: HashSet::new(),
        };
        store.rebuild_ids();
        Ok(store)
    }
}

/// Build the mediated trace projection during ingest, applying the same
/// collection boundary as documents and analysis facts.
pub(crate) struct SnapshotBuilder {
    store: TraceStore,
    seen: HashSet<String>,
    pending: HashMap<(String, String), usize>,
    known: HashSet<String>,
    capture_since_ms: Option<i64>,
    ordinal: usize,
}

impl SnapshotBuilder {
    /// Start one deterministic projection pass for the configured repositories.
    pub(crate) fn new(known: HashSet<String>, capture_since_ms: Option<i64>) -> Self {
        Self {
            store: TraceStore::default(),
            seen: HashSet::new(),
            pending: HashMap::new(),
            known,
            capture_since_ms,
            ordinal: 0,
        }
    }

    /// Fold one event chunk, deduplicating envelope ids across chunks.
    pub(crate) fn fold_text(&mut self, text: &str) {
        for line in text.lines() {
            let Ok(event) = serde_json::from_str::<Event>(line) else { continue };
            if !event.event_id.is_empty() && !self.seen.insert(event.event_id.clone()) {
                continue;
            }
            if !crate::config::captured_at(&event.ts, self.capture_since_ms) {
                if event.kind == "session_start" {
                    self.store.fold_context(&event, &self.known);
                }
                continue;
            }
            self.store
                .fold(event, self.ordinal, &self.known, &mut self.pending);
            self.ordinal += 1;
        }
    }

    /// Complete joins after removing metadata-only, pre-boundary sessions.
    fn finish(mut self) -> TraceStore {
        let active: HashSet<String> = self.store.session_events.keys().cloned().collect();
        self.store.sessions.retain(|session, _| active.contains(session));
        self.store.finish(&self.known);
        self.store
    }

    /// Atomically publish the validated projection and emit its health metrics.
    pub(crate) fn write(self, path: &Path) -> Result<()> {
        use std::io::Write;

        let snapshot = TraceSnapshot::from_store(self.finish());
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = crate::write_unique_temp(path, &[])?;
        let result = (|| -> Result<()> {
            let file = std::fs::OpenOptions::new().write(true).truncate(true).open(&tmp)?;
            let mut writer = std::io::BufWriter::new(file);
            serde_json::to_writer(&mut writer, &snapshot)?;
            writer.flush()?;
            std::fs::rename(&tmp, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        result?;
        let bytes = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
        crate::metrics::Run::new("trace_index")
            .set("events", snapshot.events.len())
            .set("turns", snapshot.turns.len())
            .set("spans", snapshot.spans.len())
            .set("jobs", snapshot.jobs.len())
            .set("bytes", bytes)
            .emit();
        Ok(())
    }
}

type TraceCacheKey = (PathBuf, u64, u128);
type TraceCache = Option<(TraceCacheKey, std::sync::Arc<TraceStore>)>;
static TRACE_CACHE: std::sync::OnceLock<std::sync::Mutex<TraceCache>> =
    std::sync::OnceLock::new();

impl TraceStore {
    fn load() -> Result<std::sync::Arc<Self>> {
        let path = crate::readmodel::trace_path();
        if let Ok(meta) = std::fs::metadata(&path) {
            let modified = meta
                .modified()?
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let key = (path.clone(), meta.len(), modified);
            let cache = TRACE_CACHE.get_or_init(|| std::sync::Mutex::new(None));
            if let Ok(mut cache) = cache.lock() {
                if let Some((cached_key, store)) = cache.as_ref() {
                    if cached_key == &key {
                        return Ok(std::sync::Arc::clone(store));
                    }
                }
                let file = std::fs::File::open(&path)?;
                let snapshot: TraceSnapshot =
                    serde_json::from_reader(std::io::BufReader::new(file))?;
                let store = std::sync::Arc::new(snapshot.into_store()?);
                *cache = Some((key, std::sync::Arc::clone(&store)));
                return Ok(store);
            }
        }
        let paths = crate::units::jsonl_files(Path::new(crate::units::LOCAL_DIR));
        Ok(std::sync::Arc::new(Self::from_paths(&paths)?))
    }

    fn from_paths(paths: &[PathBuf]) -> Result<Self> {
        use std::io::BufRead;

        let known: HashSet<String> = crate::config::load().repos.into_iter().collect();
        let mut builder = SnapshotBuilder::new(known, crate::config::capture_since_ms());
        for path in paths {
            let file = std::fs::File::open(path)?;
            for line in std::io::BufReader::new(file).lines() {
                builder.fold_text(&line?);
            }
        }
        Ok(builder.finish())
    }

    #[cfg(test)]
    fn from_lines(lines: &[&str]) -> Self {
        let mut builder = SnapshotBuilder::new(HashSet::new(), None);
        builder.fold_text(&lines.join("\n"));
        builder.finish()
    }

    /// Preserve session-level scope and repository context even when the
    /// session_start predates the collection boundary. The old start itself is
    /// not published, and contexts with no retained events are removed.
    fn fold_context(&mut self, ev: &Event, known: &HashSet<String>) {
        let machine = machine_from_stream(&ev.stream);
        let ctx = self.sessions.entry(ev.session_id.clone()).or_default();
        if ctx.source.is_empty() {
            ctx.source = ev.source.clone();
        }
        if ctx.machine.is_empty() {
            ctx.machine = machine;
        }
        if ctx.campaign.is_empty() {
            ctx.campaign = crate::policy::campaign(ev).to_string();
        }
        if ctx.role.is_empty() {
            ctx.role = crate::policy::role(ev).to_string();
        }
        if let Some(cwd) = event_cwd(&ev.payload).filter(|cwd| !cwd.is_empty()) {
            if ctx.cwd.is_empty() {
                ctx.repo = crate::units::resolve_repo(&cwd, known);
                ctx.cwd = cwd;
            }
        }
    }

    fn fold(
        &mut self,
        ev: Event,
        ordinal: usize,
        known: &HashSet<String>,
        pending: &mut HashMap<(String, String), usize>,
    ) {
        let machine = machine_from_stream(&ev.stream);
        self.fold_context(&ev, known);

        let payload = &ev.payload;
        let subtype = payload["subtype"].as_str().unwrap_or("").to_string();
        let event_kind = payload["event_kind"].as_str().unwrap_or("").to_string();
        let turn_id = payload["turn_id"]
            .as_str()
            .or_else(|| payload["payload"]["turn_id"].as_str())
            .unwrap_or("")
            .to_string();
        let duration_ms = payload["duration_ms"]
            .as_u64()
            .or_else(|| payload["payload"]["duration_ms"].as_u64());
        let event_index = self.events.len();
        let mut rec = TraceEvent {
            id: ev.event_id.clone(),
            ts: ev.ts.clone(),
            source: ev.source.clone(),
            machine: machine.clone(),
            session_id: ev.session_id.clone(),
            kind: ev.kind.clone(),
            summary: event_summary(&ev),
            search_text: event_search_text(&ev),
            subtype,
            event_kind,
            turn_id,
            duration_ms,
            span: None,
            ordinal,
        };

        let cid = call_id(payload);
        match ev.kind.as_str() {
            "tool_call" => {
                let operation = payload["name"].as_str().unwrap_or("?").to_string();
                let (input, command, workdir) = tool_input(payload);
                let process_id = input_process_id(payload);
                let span_repo = if workdir.is_empty() {
                    String::new()
                } else {
                    crate::units::resolve_repo(&workdir, known)
                };
                let span_index = self.spans.len();
                self.spans.push(Span {
                    id: ev.event_id.clone(),
                    result_id: String::new(),
                    turn_id: String::new(),
                    session_id: ev.session_id.clone(),
                    source: ev.source.clone(),
                    machine,
                    repo: span_repo,
                    started: ev.ts.clone(),
                    ended: String::new(),
                    operation,
                    status: "open".into(),
                    process_id,
                    exit_code: None,
                    duration_ms: None,
                    duration_source: String::new(),
                    workdir,
                    input: if command.is_empty() { input } else { command },
                    output: String::new(),
                    start_event: event_index,
                    end_event: None,
                });
                rec.span = Some(span_index);
                if !cid.is_empty() {
                    pending.insert((ev.session_id.clone(), cid), span_index);
                }
            }
            "tool_result" if !cid.is_empty() => {
                if let Some(span_index) = pending.remove(&(ev.session_id.clone(), cid)) {
                    let output = result_output(payload);
                    let explicit_error = payload["is_error"].as_bool().unwrap_or(false);
                    let exit_code = parse_exit_code(&output);
                    let reported = payload["duration_ms"]
                        .as_u64()
                        .or_else(|| parse_wall_time_ms(&output));
                    let span = &mut self.spans[span_index];
                    span.result_id = ev.event_id.clone();
                    span.ended = ev.ts.clone();
                    span.exit_code = exit_code;
                    span.status = result_status(explicit_error, exit_code, &output).into();
                    if span.process_id.is_empty() {
                        span.process_id = output_process_id(&output);
                    }
                    span.output = excerpt(&output, EXCERPT);
                    if let Some(ms) = reported {
                        span.duration_ms = Some(ms);
                        span.duration_source = "reported".into();
                    } else if let Some(ms) = elapsed_ms(&span.started, &ev.ts) {
                        span.duration_ms = Some(ms);
                        span.duration_source = "event_gap".into();
                    }
                    span.end_event = Some(event_index);
                    rec.span = Some(span_index);
                    rec.summary = format!(
                        "{}{}{}",
                        span.status,
                        span.exit_code
                            .map(|n| format!(" exit={n}"))
                            .unwrap_or_default(),
                        if span.output.is_empty() {
                            String::new()
                        } else {
                            format!(" · {}", excerpt(&span.output, 160))
                        },
                    );
                }
            }
            _ => {}
        }
        self.session_events
            .entry(ev.session_id)
            .or_default()
            .push(event_index);
        self.events.push(rec);
    }

    fn finish(&mut self, known: &HashSet<String>) {
        for (sid, indexes) in &mut self.session_events {
            indexes.sort_by(|a, b| {
                self.events[*a]
                    .ts
                    .cmp(&self.events[*b].ts)
                    .then(self.events[*a].ordinal.cmp(&self.events[*b].ordinal))
            });
            let ctx = self.sessions.entry(sid.clone()).or_default();
            if ctx.repo.is_empty() && !ctx.cwd.is_empty() {
                ctx.repo = crate::units::resolve_repo(&ctx.cwd, known);
            }
        }
        for span in &mut self.spans {
            if let Some(ctx) = self.sessions.get(&span.session_id) {
                if span.repo.is_empty() {
                    span.repo = ctx.repo.clone();
                }
                if span.machine.is_empty() {
                    span.machine = ctx.machine.clone();
                }
            }
        }
        self.turns = build_turns(
            &self.events,
            &self.spans,
            &self.sessions,
            &self.session_events,
        );
        for (turn_index, turn) in self.turns.iter().enumerate() {
            for &span_index in &turn.span_indexes {
                if self.spans[span_index].turn_id.is_empty() {
                    self.spans[span_index].turn_id = self.turns[turn_index].id.clone();
                }
            }
        }
        self.jobs = build_jobs(&self.spans);
        self.rebuild_ids();
    }

    fn rebuild_ids(&mut self) {
        self.ids.clear();
        self.ids.extend(self.events.iter().map(|event| event.id.clone()));
        self.ids.extend(self.spans.iter().map(|span| span.id.clone()));
        self.ids.extend(self.jobs.iter().map(|job| job.id.clone()));
        self.ids.extend(self.turns.iter().map(|turn| turn.id.clone()));
        self.ids.extend(self.session_events.keys().cloned());
    }
}

fn build_turns(
    events: &[TraceEvent],
    spans: &[Span],
    sessions: &HashMap<String, SessionContext>,
    session_events: &HashMap<String, Vec<usize>>,
) -> Vec<Turn> {
    let mut out = Vec::new();
    for (sid, indexes) in session_events {
        let Some(ctx) = sessions.get(sid) else {
            continue;
        };
        if ctx.source == "codex_cli" {
            build_codex_turns(events, spans, ctx, sid, indexes, &mut out);
        } else {
            build_prompt_turns(events, spans, ctx, sid, indexes, &mut out);
        }
    }
    out.sort_by(|a, b| b.started.cmp(&a.started).then(a.id.cmp(&b.id)));
    out
}

fn build_jobs(spans: &[Span]) -> Vec<Job> {
    let mut order: Vec<usize> = (0..spans.len()).collect();
    order.sort_by(|&a, &b| {
        spans[a]
            .started
            .cmp(&spans[b].started)
            .then(spans[a].id.cmp(&spans[b].id))
    });
    let mut jobs = Vec::new();
    let mut active: HashMap<(String, String), usize> = HashMap::new();
    for span_index in order {
        let span = &spans[span_index];
        if span.source != "codex_cli" || span.process_id.is_empty() {
            continue;
        }
        let key = (span.session_id.clone(), span.process_id.clone());
        if span.operation == "exec_command" && span.status == "running" {
            let job_index = jobs.len();
            jobs.push(job_from_span(span, span_index, "exact_process_session_id"));
            active.insert(key, job_index);
            continue;
        }
        if span.operation != "write_stdin" {
            continue;
        }
        let job_index = if let Some(&index) = active.get(&key) {
            attach_job_span(&mut jobs[index], span, span_index);
            index
        } else {
            let index = jobs.len();
            jobs.push(job_from_span(span, span_index, "continuation_only"));
            index
        };
        if matches!(
            jobs[job_index].status.as_str(),
            "running" | "open" | "unknown"
        ) {
            active.insert(key, job_index);
        } else {
            active.remove(&key);
        }
    }
    jobs.sort_by(|a, b| b.started.cmp(&a.started).then(a.id.cmp(&b.id)));
    jobs
}

fn job_from_span(span: &Span, span_index: usize, association: &str) -> Job {
    let ended = span_end(span);
    Job {
        id: format!("job:{}", span.id),
        process_id: span.process_id.clone(),
        association: association.into(),
        initial_span_id: span.id.clone(),
        final_span_id: span.id.clone(),
        session_id: span.session_id.clone(),
        turn_ids: if span.turn_id.is_empty() {
            vec![]
        } else {
            vec![span.turn_id.clone()]
        },
        source: span.source.clone(),
        machine: span.machine.clone(),
        repo: span.repo.clone(),
        started: span.started.clone(),
        ended: ended.to_string(),
        status: job_status(span),
        elapsed_ms: elapsed_ms(&span.started, ended),
        reported_wait_ms: reported_wait(span),
        spans: 1,
        polls: usize::from(span.operation == "write_stdin"),
        stdin_writes: usize::from(span.operation == "write_stdin" && stdin_has_data(span)),
        errors: usize::from(span.status == "error"),
        command: span.input.clone(),
        final_output: span.output.clone(),
        span_indexes: vec![span_index],
    }
}

fn attach_job_span(job: &mut Job, span: &Span, span_index: usize) {
    if job.span_indexes.contains(&span_index) {
        return;
    }
    job.final_span_id = span.id.clone();
    job.ended = span_end(span).to_string();
    job.status = job_status(span);
    job.elapsed_ms = elapsed_ms(&job.started, &job.ended);
    job.reported_wait_ms = job.reported_wait_ms.saturating_add(reported_wait(span));
    job.spans += 1;
    if span.operation == "write_stdin" {
        job.polls += 1;
        job.stdin_writes += usize::from(stdin_has_data(span));
    }
    job.errors += usize::from(span.status == "error");
    if !span.turn_id.is_empty() && !job.turn_ids.contains(&span.turn_id) {
        job.turn_ids.push(span.turn_id.clone());
    }
    if job.repo.is_empty() && !span.repo.is_empty() {
        job.repo = span.repo.clone();
    }
    if !span.output.is_empty() {
        job.final_output = span.output.clone();
    }
    job.span_indexes.push(span_index);
}

fn span_end(span: &Span) -> &str {
    if span.ended.is_empty() {
        &span.started
    } else {
        &span.ended
    }
}

fn reported_wait(span: &Span) -> u64 {
    if span.duration_source == "reported" {
        span.duration_ms.unwrap_or(0)
    } else {
        0
    }
}

fn job_status(span: &Span) -> String {
    if span.operation == "write_stdin"
        && span.status == "error"
        && span.exit_code.is_none()
        && span.output.to_lowercase().contains("write_stdin failed:")
    {
        "unknown".into()
    } else {
        span.status.clone()
    }
}

fn stdin_has_data(span: &Span) -> bool {
    serde_json::from_str::<Value>(&span.input)
        .ok()
        .and_then(|v| v["chars"].as_str().map(|s| !s.is_empty()))
        .unwrap_or(false)
}

fn build_codex_turns(
    events: &[TraceEvent],
    spans: &[Span],
    ctx: &SessionContext,
    sid: &str,
    indexes: &[usize],
    out: &mut Vec<Turn>,
) {
    let mut cur: Option<TurnBuilder> = None;
    for (pos, &i) in indexes.iter().enumerate() {
        let e = &events[i];
        if e.event_kind == "task_started" {
            if let Some(mut prior) = cur.take() {
                if prior.duration_ms.is_none() {
                    prior.duration_ms = turn_gap_ms(events, &prior.event_indexes);
                    prior.duration_source = "event_gap".into();
                }
                if prior.status == "open" {
                    prior.status = "unknown".into();
                }
                push_turn(prior, events, spans, ctx, sid, out);
            }
            cur = Some(TurnBuilder {
                id: if e.turn_id.is_empty() {
                    e.id.clone()
                } else {
                    e.turn_id.clone()
                },
                status: "open".into(),
                event_indexes: vec![i],
                ..Default::default()
            });
            continue;
        }
        if cur.is_none() && is_real_prompt(e) {
            cur = Some(TurnBuilder {
                id: e.id.clone(),
                ask: prompt_text(e),
                status: "open".into(),
                event_indexes: vec![i],
                ..Default::default()
            });
            continue;
        }
        let Some(b) = cur.as_mut() else { continue };
        b.event_indexes.push(i);
        if b.ask.is_empty() && is_real_prompt(e) {
            b.ask = prompt_text(e);
        }
        if let Some(si) = e.span {
            if spans[si].start_event == i && !b.span_indexes.contains(&si) {
                b.span_indexes.push(si);
            }
        }
        if e.event_kind == "task_complete" || e.event_kind == "turn_aborted" {
            b.status = if e.event_kind == "task_complete" {
                "ok".into()
            } else {
                "aborted".into()
            };
            b.duration_ms = e
                .duration_ms
                .or_else(|| turn_gap_ms(events, &b.event_indexes));
            b.duration_source = if e.duration_ms.is_some() {
                "reported".into()
            } else {
                "event_gap".into()
            };
            if let Some(done) = cur.take() {
                push_turn(done, events, spans, ctx, sid, out);
            }
        } else if pos + 1 == indexes.len() {
            b.duration_ms = turn_gap_ms(events, &b.event_indexes);
            b.duration_source = "event_gap".into();
        }
    }
    if let Some(b) = cur {
        push_turn(b, events, spans, ctx, sid, out);
    }
}

fn build_prompt_turns(
    events: &[TraceEvent],
    spans: &[Span],
    ctx: &SessionContext,
    sid: &str,
    indexes: &[usize],
    out: &mut Vec<Turn>,
) {
    let mut cur: Option<TurnBuilder> = None;
    for &i in indexes {
        let e = &events[i];
        if is_real_prompt(e) {
            if let Some(mut prior) = cur.take() {
                prior.duration_ms = turn_gap_ms(events, &prior.event_indexes);
                prior.duration_source = "event_gap".into();
                prior.status = "unknown".into();
                push_turn(prior, events, spans, ctx, sid, out);
            }
            cur = Some(TurnBuilder {
                id: e.id.clone(),
                ask: prompt_text(e),
                status: "open".into(),
                event_indexes: vec![i],
                ..Default::default()
            });
            continue;
        }
        let Some(b) = cur.as_mut() else { continue };
        b.event_indexes.push(i);
        if let Some(si) = e.span {
            if spans[si].start_event == i && !b.span_indexes.contains(&si) {
                b.span_indexes.push(si);
            }
        }
        if e.subtype == "turn_duration" {
            b.status = "ok".into();
            b.duration_ms = e
                .duration_ms
                .or_else(|| turn_gap_ms(events, &b.event_indexes));
            b.duration_source = if e.duration_ms.is_some() {
                "reported".into()
            } else {
                "event_gap".into()
            };
            if let Some(done) = cur.take() {
                push_turn(done, events, spans, ctx, sid, out);
            }
        } else if e.kind == "session_end" {
            b.status = "unknown".into();
            b.duration_ms = turn_gap_ms(events, &b.event_indexes);
            b.duration_source = "event_gap".into();
            if let Some(done) = cur.take() {
                push_turn(done, events, spans, ctx, sid, out);
            }
        }
    }
    if let Some(mut b) = cur {
        b.duration_ms = turn_gap_ms(events, &b.event_indexes);
        b.duration_source = "event_gap".into();
        push_turn(b, events, spans, ctx, sid, out);
    }
}

fn push_turn(
    b: TurnBuilder,
    events: &[TraceEvent],
    spans: &[Span],
    ctx: &SessionContext,
    sid: &str,
    out: &mut Vec<Turn>,
) {
    if b.event_indexes.is_empty() {
        return;
    }
    let started = events[*b.event_indexes.first().expect("nonempty")]
        .ts
        .clone();
    let ended = events[*b.event_indexes.last().expect("nonempty")]
        .ts
        .clone();
    let repo = b
        .span_indexes
        .iter()
        .find_map(|&i| (!spans[i].repo.is_empty()).then(|| spans[i].repo.clone()))
        .unwrap_or_else(|| ctx.repo.clone());
    let errors = b
        .span_indexes
        .iter()
        .filter(|&&i| spans[i].status == "error")
        .count();
    out.push(Turn {
        id: b.id,
        session_id: sid.to_string(),
        source: ctx.source.clone(),
        machine: ctx.machine.clone(),
        repo,
        started,
        ended,
        status: if b.status.is_empty() {
            "unknown".into()
        } else {
            b.status
        },
        duration_ms: b.duration_ms,
        duration_source: b.duration_source,
        ask: b.ask,
        events: b.event_indexes.len(),
        spans: b.span_indexes.len(),
        errors,
        event_indexes: b.event_indexes,
        span_indexes: b.span_indexes,
    });
}

#[allow(clippy::too_many_arguments)]
pub fn list_text(
    entity: &str,
    repo: Option<&str>,
    machine: Option<&str>,
    source: Option<&str>,
    status: Option<&str>,
    operation: Option<&str>,
    has_errors: bool,
    since: Option<&str>,
    min_ms: Option<u64>,
    sort: &str,
    limit: usize,
    json_out: bool,
    scope: Option<&crate::policy::ReadScope>,
) -> Result<String> {
    let store = TraceStore::load()?;
    let out = match entity {
        "turns" => {
            let mut rows: Vec<&Turn> = store
                .turns
                .iter()
                .filter(|t| {
                    matches_common(
                        &t.repo,
                        &t.machine,
                        &t.source,
                        &t.status,
                        &t.started,
                        t.duration_ms,
                        repo,
                        machine,
                        source,
                        status,
                        since,
                        min_ms,
                    )
                })
                .filter(|t| !has_errors || t.errors > 0)
                .filter(|t| scope_allows(&store, &t.session_id, &t.repo, &t.source, scope))
                .filter(|t| {
                    operation.is_none_or(|q| {
                        t.span_indexes
                            .iter()
                            .any(|&i| span_operation_matches(&store.spans[i], q))
                    })
                })
                .collect();
            sort_turns(&mut rows, sort);
            rows.truncate(limit);
            if json_out {
                crate::view::envelope("trace_turns", json!(rows))
            } else {
                turns_md(&rows)
            }
        }
        "spans" => {
            let mut rows: Vec<&Span> = store
                .spans
                .iter()
                .filter(|s| {
                    matches_common(
                        &s.repo,
                        &s.machine,
                        &s.source,
                        &s.status,
                        &s.started,
                        s.duration_ms,
                        repo,
                        machine,
                        source,
                        status,
                        since,
                        min_ms,
                    )
                })
                .filter(|s| operation.is_none_or(|q| span_operation_matches(s, q)))
                .filter(|s| scope_allows(&store, &s.session_id, &s.repo, &s.source, scope))
                .collect();
            sort_spans(&mut rows, sort);
            rows.truncate(limit);
            if json_out {
                crate::view::envelope("trace_spans", json!(rows))
            } else {
                spans_md(&rows)
            }
        }
        "jobs" => {
            let mut rows: Vec<&Job> = store
                .jobs
                .iter()
                .filter(|j| {
                    matches_common(
                        &j.repo,
                        &j.machine,
                        &j.source,
                        &j.status,
                        &j.started,
                        j.elapsed_ms,
                        repo,
                        machine,
                        source,
                        status,
                        since,
                        min_ms,
                    )
                })
                .filter(|j| !has_errors || j.errors > 0)
                .filter(|j| scope_allows(&store, &j.session_id, &j.repo, &j.source, scope))
                .filter(|j| operation.is_none_or(|q| job_operation_matches(j, q)))
                .collect();
            sort_jobs(&mut rows, sort);
            rows.truncate(limit);
            if json_out {
                crate::view::envelope("trace_jobs", json!(rows))
            } else {
                jobs_md(&rows)
            }
        }
        _ => bail!("trace list: --type must be turns, spans, or jobs"),
    };
    Ok(out)
}

fn scope_allows(
    store: &TraceStore,
    session_id: &str,
    repo: &str,
    source: &str,
    scope: Option<&crate::policy::ReadScope>,
) -> bool {
    let Some(scope) = scope else { return true };
    let context = store.sessions.get(session_id);
    scope.allows_fields(
        repo,
        context.map(|ctx| ctx.campaign.as_str()).unwrap_or(""),
        context.map(|ctx| ctx.role.as_str()).unwrap_or(""),
        source,
    )
}

#[allow(clippy::too_many_arguments)]
fn matches_common(
    row_repo: &str,
    row_machine: &str,
    row_source: &str,
    row_status: &str,
    started: &str,
    duration_ms: Option<u64>,
    repo: Option<&str>,
    machine: Option<&str>,
    source: Option<&str>,
    status: Option<&str>,
    since: Option<&str>,
    min_ms: Option<u64>,
) -> bool {
    contains_opt(row_repo, repo)
        && contains_opt(row_machine, machine)
        && contains_opt(row_source, source)
        && status.is_none_or(|q| row_status.eq_ignore_ascii_case(q))
        && since.is_none_or(|q| started >= q)
        && min_ms.is_none_or(|n| duration_ms.unwrap_or(0) >= n)
}

fn contains_opt(value: &str, query: Option<&str>) -> bool {
    query.is_none_or(|q| value.to_lowercase().contains(&q.to_lowercase()))
}

fn span_operation_matches(span: &Span, query: &str) -> bool {
    let q = query.to_lowercase();
    span.operation.to_lowercase().contains(&q) || span.input.to_lowercase().contains(&q)
}

fn job_operation_matches(job: &Job, query: &str) -> bool {
    let q = query.to_lowercase();
    job.command.to_lowercase().contains(&q) || job.process_id.to_lowercase().contains(&q)
}

fn sort_turns(rows: &mut [&Turn], sort: &str) {
    if matches!(sort, "duration" | "wait") {
        rows.sort_by(|a, b| {
            b.duration_ms
                .unwrap_or(0)
                .cmp(&a.duration_ms.unwrap_or(0))
                .then(b.started.cmp(&a.started))
        });
    } else {
        rows.sort_by(|a, b| b.started.cmp(&a.started));
    }
}

fn sort_spans(rows: &mut [&Span], sort: &str) {
    if matches!(sort, "duration" | "wait") {
        rows.sort_by(|a, b| {
            b.duration_ms
                .unwrap_or(0)
                .cmp(&a.duration_ms.unwrap_or(0))
                .then(b.started.cmp(&a.started))
        });
    } else {
        rows.sort_by(|a, b| b.started.cmp(&a.started));
    }
}

fn sort_jobs(rows: &mut [&Job], sort: &str) {
    if sort == "wait" {
        rows.sort_by(|a, b| {
            b.reported_wait_ms
                .cmp(&a.reported_wait_ms)
                .then(b.started.cmp(&a.started))
        });
    } else if sort == "duration" {
        rows.sort_by(|a, b| {
            b.elapsed_ms
                .unwrap_or(0)
                .cmp(&a.elapsed_ms.unwrap_or(0))
                .then(b.started.cmp(&a.started))
        });
    } else {
        rows.sort_by(|a, b| b.started.cmp(&a.started));
    }
}

pub fn show(id: &str, before: usize, after: usize, json_out: bool) -> Result<()> {
    print!("{}", show_text(id, before, after, json_out, None)?);
    Ok(())
}

pub fn show_text(
    id: &str,
    before: usize,
    after: usize,
    json_out: bool,
    scope: Option<&crate::policy::ReadScope>,
) -> Result<String> {
    let store = TraceStore::load()?;
    let resolved = resolve(&store, id, scope)?;
    anyhow::ensure!(resolved_allowed(&store, &resolved, scope), "trace id is outside the read scope");
    match resolved {
        Resolved::Job(i) => show_job_text(&store, i, json_out),
        Resolved::Turn(i) => show_turn_text(&store, i, json_out),
        Resolved::Span(i) => show_span_text(&store, i, before, after, json_out),
        Resolved::Event(i) => show_event_text(&store, i, before, after, json_out),
        Resolved::Session(sid) => show_session_text(&store, &sid, json_out),
    }
}

pub fn compare(left: &str, right: &str, json_out: bool) -> Result<()> {
    print!("{}", compare_text(left, right, json_out, None)?);
    Ok(())
}

pub fn compare_text(
    left: &str,
    right: &str,
    json_out: bool,
    scope: Option<&crate::policy::ReadScope>,
) -> Result<String> {
    let store = TraceStore::load()?;
    let left_resolved = resolve(&store, left, scope)?;
    let right_resolved = resolve(&store, right, scope)?;
    anyhow::ensure!(
        resolved_allowed(&store, &left_resolved, scope)
            && resolved_allowed(&store, &right_resolved, scope),
        "trace id is outside the read scope"
    );
    let l = comparable(&store, left_resolved)?;
    let r = comparable(&store, right_resolved)?;
    let mut differences = Vec::new();
    if let (Some(lm), Some(rm)) = (l.as_object(), r.as_object()) {
        let mut keys: Vec<&String> = lm.keys().chain(rm.keys()).collect();
        keys.sort();
        keys.dedup();
        for key in keys {
            if lm.get(key) != rm.get(key) {
                differences.push(key.clone());
            }
        }
    }
    if json_out {
        Ok(crate::view::envelope(
            "trace_compare",
            json!({"left": l, "right": r, "differences": differences}),
        ))
    } else {
        let lm = l.as_object().expect("comparable object");
        let rm = r.as_object().expect("comparable object");
        let mut o = String::from("# trace compare\n\n| field | left | right |\n|---|---|---|\n");
        for key in differences {
            o.push_str(&format!(
                "| {} | {} | {} |\n",
                md_cell(key.as_str()),
                md_cell(&value_short(lm.get(&key))),
                md_cell(&value_short(rm.get(&key))),
            ));
        }
        Ok(o)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn search_text(
    query: &str,
    repo: Option<&str>,
    machine: Option<&str>,
    source: Option<&str>,
    kind: Option<&str>,
    limit: usize,
    json_out: bool,
    scope: Option<&crate::policy::ReadScope>,
) -> Result<String> {
    if query.trim().is_empty() {
        bail!("trace search: query cannot be empty");
    }
    let store = TraceStore::load()?;
    let hits = search_hits(&store, query, repo, machine, source, kind, limit, scope);
    if json_out {
        Ok(crate::view::envelope("trace_search", json!(hits)))
    } else {
        let mut o = format!("# trace search · {:?} ({})\n\n", query, hits.len());
        for h in &hits {
            o.push_str(&format!(
                "- `{}` [{}] {} · {} · {} · {}\n  {}\n",
                h.ts,
                h.id,
                h.kind,
                empty_dash(&h.repo),
                empty_dash(&h.machine),
                h.source,
                h.summary,
            ));
        }
        Ok(o)
    }
}

#[allow(clippy::too_many_arguments)]
fn search_hits(
    store: &TraceStore,
    query: &str,
    repo: Option<&str>,
    machine: Option<&str>,
    source: Option<&str>,
    kind: Option<&str>,
    limit: usize,
    scope: Option<&crate::policy::ReadScope>,
) -> Vec<SearchHit> {
    let q = query.to_lowercase();
    let mut hits: Vec<SearchHit> = store
        .events
        .iter()
        .filter(|event| event.search_text.to_lowercase().contains(&q))
        .filter(|event| contains_opt(&event.machine, machine))
        .filter(|event| contains_opt(&event.source, source))
        .filter(|event| contains_opt(&event.kind, kind))
        .filter_map(|event| {
            let context = store.sessions.get(&event.session_id);
            let event_repo = context.map(|ctx| ctx.repo.as_str()).unwrap_or("");
            if !contains_opt(event_repo, repo)
                || !scope_allows(store, &event.session_id, event_repo, &event.source, scope)
            {
                return None;
            }
            Some(SearchHit {
                id: event.id.clone(),
                ts: event.ts.clone(),
                source: event.source.clone(),
                machine: event.machine.clone(),
                session_id: event.session_id.clone(),
                repo: event_repo.to_string(),
                kind: event.kind.clone(),
                summary: event.summary.clone(),
            })
        })
        .collect();
    hits.sort_by(|a, b| b.ts.cmp(&a.ts));
    hits.truncate(limit);
    hits
}

#[derive(Serialize)]
struct SearchHit {
    id: String,
    ts: String,
    source: String,
    machine: String,
    session_id: String,
    repo: String,
    kind: String,
    summary: String,
}

enum Resolved {
    Job(usize),
    Turn(usize),
    Span(usize),
    Event(usize),
    Session(String),
}

fn resolved_allowed(
    store: &TraceStore,
    resolved: &Resolved,
    scope: Option<&crate::policy::ReadScope>,
) -> bool {
    match resolved {
        Resolved::Job(index) => {
            let row = &store.jobs[*index];
            scope_allows(store, &row.session_id, &row.repo, &row.source, scope)
        }
        Resolved::Turn(index) => {
            let row = &store.turns[*index];
            scope_allows(store, &row.session_id, &row.repo, &row.source, scope)
        }
        Resolved::Span(index) => {
            let row = &store.spans[*index];
            scope_allows(store, &row.session_id, &row.repo, &row.source, scope)
        }
        Resolved::Event(index) => {
            let row = &store.events[*index];
            let repo = store
                .sessions
                .get(&row.session_id)
                .map(|ctx| ctx.repo.as_str())
                .unwrap_or("");
            scope_allows(store, &row.session_id, repo, &row.source, scope)
        }
        Resolved::Session(session_id) => {
            let context = store.sessions.get(session_id);
            scope.is_none()
                || context.is_some_and(|ctx| {
                    scope_allows(store, session_id, &ctx.repo, &ctx.source, scope)
                })
        }
    }
}

fn resolve(
    store: &TraceStore,
    query: &str,
    scope: Option<&crate::policy::ReadScope>,
) -> Result<Resolved> {
    if query.chars().count() < 4 {
        bail!("trace id prefixes must be at least 4 characters");
    }
    let mut bases: HashMap<String, Resolved> = HashMap::new();
    for (i, j) in store.jobs.iter().enumerate() {
        let resolved = Resolved::Job(i);
        if id_matches(&j.id, query) && resolved_allowed(store, &resolved, scope) {
            bases.insert(j.id.clone(), resolved);
        }
    }
    for (i, t) in store.turns.iter().enumerate() {
        let resolved = Resolved::Turn(i);
        if id_matches(&t.id, query) && resolved_allowed(store, &resolved, scope) {
            bases.insert(t.id.clone(), resolved);
        }
    }
    for (i, s) in store.spans.iter().enumerate() {
        let resolved = Resolved::Span(i);
        if id_matches(&s.id, query) && resolved_allowed(store, &resolved, scope) {
            bases.entry(s.id.clone()).or_insert(resolved);
        }
    }
    for sid in store.session_events.keys() {
        let resolved = Resolved::Session(sid.clone());
        if id_matches(sid, query) && resolved_allowed(store, &resolved, scope) {
            bases.entry(sid.clone()).or_insert(resolved);
        }
    }
    for (i, e) in store.events.iter().enumerate() {
        let resolved = Resolved::Event(i);
        if id_matches(&e.id, query) && resolved_allowed(store, &resolved, scope) {
            bases.entry(e.id.clone()).or_insert(resolved);
        }
    }
    match bases.len() {
        0 => bail!("nothing matches trace id `{query}`"),
        1 => Ok(bases.into_values().next().expect("one")),
        n => {
            let mut ids: Vec<String> = bases.keys().cloned().collect();
            ids.sort();
            ids.truncate(8);
            bail!(
                "trace id `{query}` is ambiguous ({n} matches): {}",
                ids.join(", ")
            )
        }
    }
}

fn id_matches(id: &str, query: &str) -> bool {
    id == query || id.starts_with(query)
}

fn show_turn_text(store: &TraceStore, index: usize, json_out: bool) -> Result<String> {
    let turn = &store.turns[index];
    let timeline = turn_timeline(store, turn);
    if json_out {
        Ok(crate::view::envelope(
            "trace_turn",
            json!({"turn": turn, "timeline": timeline}),
        ))
    } else {
        let mut o = format!(
            "# turn [{}]\n\nid: {}\nsession: {}\n{} · {} · {} · {} · {} spans · {} errors\nask: {}\n\n",
            short_id(&turn.id),
            turn.id,
            turn.session_id,
            turn.status,
            duration_label(turn.duration_ms, &turn.duration_source),
            empty_dash(&turn.repo),
            empty_dash(&turn.machine),
            turn.spans,
            turn.errors,
            empty_dash(&turn.ask),
        );
        o.push_str("timeline:\n");
        for row in &timeline {
            o.push_str(&format!(
                "  +{:>7} [{}] {:<14} {}\n",
                offset(row.offset_ms),
                display_id(store, &row.id),
                row.kind,
                row.summary
            ));
        }
        if significant_event_count(store, turn) > timeline.len() {
            o.push_str(&format!(
                "  … {} more significant events omitted\n",
                significant_event_count(store, turn) - timeline.len()
            ));
        }
        Ok(o)
    }
}

fn show_job_text(store: &TraceStore, index: usize, json_out: bool) -> Result<String> {
    let job = &store.jobs[index];
    let children: Vec<&Span> = job.span_indexes.iter().map(|&i| &store.spans[i]).collect();
    if json_out {
        Ok(crate::view::envelope(
            "trace_job",
            json!({"job": job, "span_timeline": children}),
        ))
    } else {
        let mut o = format!(
            "# job [{}] · process {}\n\nid: {}\nassociation: {}\nsession: {}\n{} · {} elapsed · {} reported-wait · {} polls · {} stdin writes · {} child errors\n",
            short_id(&job.initial_span_id),
            job.process_id,
            job.id,
            job.association,
            job.session_id,
            job.status,
            duration(job.elapsed_ms),
            duration(Some(job.reported_wait_ms)),
            job.polls,
            job.stdin_writes,
            job.errors,
        );
        if !job.turn_ids.is_empty() {
            o.push_str(&format!("turns: {}\n", job.turn_ids.join(", ")));
        }
        if !job.command.is_empty() {
            o.push_str(&format!("\ncommand:\n{}\n", job.command));
        }
        if !job.final_output.is_empty() {
            o.push_str(&format!(
                "\nfinal output:\n{}\n",
                process_output(&job.final_output, EXCERPT)
            ));
        }
        o.push_str("\ntimeline:\n");
        for span in children {
            let offset_ms = elapsed_ms(&job.started, &span.started).unwrap_or(0);
            let action = if span.operation == "write_stdin" {
                if stdin_has_data(span) {
                    "stdin write"
                } else {
                    "poll"
                }
            } else {
                "start"
            };
            let evidence = process_output(&span.output, 140);
            o.push_str(&format!(
                "  +{:>7} [{}] {:<12} {:<11} {} · {}{}\n",
                offset(offset_ms),
                display_id(store, &span.id),
                span.operation,
                action,
                span.status,
                duration_label(span.duration_ms, &span.duration_source),
                if evidence.is_empty() {
                    String::new()
                } else {
                    format!(" · {evidence}")
                },
            ));
        }
        Ok(o)
    }
}

fn show_span_text(
    store: &TraceStore,
    index: usize,
    before: usize,
    after: usize,
    json_out: bool,
) -> Result<String> {
    let span = &store.spans[index];
    let around = event_window(store, &span.session_id, span.start_event, before, after);
    if json_out {
        Ok(crate::view::envelope(
            "trace_span",
            json!({"span": span, "around": around}),
        ))
    } else {
        let mut o = format!(
            "# span [{}] · {}\n\nid: {}\nturn: {}\nsession: {}\n{} · {} · {} · {} · {}\n",
            short_id(&span.id),
            span.operation,
            span.id,
            empty_dash(&span.turn_id),
            span.session_id,
            span.status,
            duration(span.duration_ms),
            empty_dash(&span.duration_source),
            empty_dash(&span.repo),
            empty_dash(&span.machine),
        );
        if let Some(code) = span.exit_code {
            o.push_str(&format!("exit: {code}\n"));
        }
        if !span.input.is_empty() {
            o.push_str(&format!("\ninput:\n{}\n", span.input));
        }
        if !span.output.is_empty() {
            o.push_str(&format!("\noutput:\n{}\n", span.output));
        }
        o.push_str("\naround:\n");
        for e in around {
            o.push_str(&format!(
                "  `{}` [{}] {:<14} {}\n",
                e.ts,
                display_id(store, &e.id),
                e.kind,
                e.summary
            ));
        }
        Ok(o)
    }
}

fn show_event_text(
    store: &TraceStore,
    index: usize,
    before: usize,
    after: usize,
    json_out: bool,
) -> Result<String> {
    let event = &store.events[index];
    let around = event_window(store, &event.session_id, index, before, after);
    if json_out {
        Ok(crate::view::envelope(
            "trace_event",
            json!({"event": event, "around": around}),
        ))
    } else {
        let mut o = format!(
            "# event [{}] · {}\n\nid: {}\nsession: {}\n{} · {} · {}\nsummary: {}\n\naround:\n",
            short_id(&event.id),
            event.kind,
            event.id,
            event.session_id,
            event.ts,
            event.source,
            empty_dash(&event.machine),
            event.summary,
        );
        for e in around {
            o.push_str(&format!(
                "  `{}` [{}] {:<14} {}\n",
                e.ts,
                display_id(store, &e.id),
                e.kind,
                e.summary
            ));
        }
        Ok(o)
    }
}

fn show_session_text(store: &TraceStore, sid: &str, json_out: bool) -> Result<String> {
    let mut turns: Vec<&Turn> = store.turns.iter().filter(|t| t.session_id == sid).collect();
    turns.sort_by(|a, b| a.started.cmp(&b.started));
    let ctx = store.sessions.get(sid).cloned().unwrap_or_default();
    if json_out {
        Ok(crate::view::envelope(
            "trace_session",
            json!({
                "session_id": sid, "source": ctx.source, "machine": ctx.machine,
                "repo": ctx.repo, "turns": turns,
            }),
        ))
    } else {
        let mut o = format!(
            "# trace session [{}]\n\nid: {}\n{} · {} · {} · {} turns\n\n",
            short_id(sid),
            sid,
            empty_dash(&ctx.source),
            empty_dash(&ctx.repo),
            empty_dash(&ctx.machine),
            turns.len()
        );
        o.push_str(&turns_md(&turns));
        Ok(o)
    }
}

#[derive(Serialize)]
struct TimelineRow {
    id: String,
    offset_ms: u64,
    kind: String,
    summary: String,
}

fn turn_timeline(store: &TraceStore, turn: &Turn) -> Vec<TimelineRow> {
    let (job_starts, collapsed_spans) = turn_job_projection(store, turn);
    let base = turn
        .event_indexes
        .first()
        .and_then(|&i| ts_ms(&store.events[i].ts))
        .unwrap_or(0);
    turn.event_indexes
        .iter()
        .filter_map(|&i| {
            let e = &store.events[i];
            if !significant(e) {
                return None;
            }
            let mut summary = e.summary.clone();
            let kind = if let Some(si) = e.span.filter(|&si| store.spans[si].start_event == i) {
                if collapsed_spans.contains(&si) {
                    return None;
                }
                if let Some(&job_index) = job_starts.get(&si) {
                    let job = &store.jobs[job_index];
                    summary = format!(
                        "{} → {} · {} elapsed · {} reported-wait · {} polls · {} errors",
                        excerpt(&job.command, 120),
                        job.status,
                        duration(job.elapsed_ms),
                        duration(Some(job.reported_wait_ms)),
                        job.polls,
                        job.errors,
                    );
                    "job".to_string()
                } else {
                    let s = &store.spans[si];
                    summary = format!(
                        "{} → {} · {}{}",
                        s.operation,
                        s.status,
                        duration_label(s.duration_ms, &s.duration_source),
                        if s.input.is_empty() {
                            String::new()
                        } else {
                            format!(" · {}", excerpt(&s.input, 140))
                        }
                    );
                    "span".to_string()
                }
            } else {
                e.kind.clone()
            };
            Some(TimelineRow {
                id: e
                    .span
                    .and_then(|si| job_starts.get(&si).map(|&ji| store.jobs[ji].id.clone()))
                    .unwrap_or_else(|| e.id.clone()),
                offset_ms: ts_ms(&e.ts).unwrap_or(base).saturating_sub(base).max(0) as u64,
                kind,
                summary,
            })
        })
        .take(TIMELINE_CAP)
        .collect()
}

fn significant_event_count(store: &TraceStore, turn: &Turn) -> usize {
    let (_, collapsed_spans) = turn_job_projection(store, turn);
    turn.event_indexes
        .iter()
        .filter(|&&i| {
            let event = &store.events[i];
            significant(event) && event.span.is_none_or(|si| !collapsed_spans.contains(&si))
        })
        .count()
}

fn turn_job_projection(store: &TraceStore, turn: &Turn) -> (HashMap<usize, usize>, HashSet<usize>) {
    let mut starts = HashMap::new();
    let mut collapsed = HashSet::new();
    for (job_index, job) in store.jobs.iter().enumerate() {
        let Some(&initial) = job.span_indexes.first() else {
            continue;
        };
        if store.spans[initial].turn_id != turn.id {
            continue;
        }
        starts.insert(initial, job_index);
        for &span_index in job.span_indexes.iter().skip(1) {
            if store.spans[span_index].turn_id == turn.id {
                collapsed.insert(span_index);
            }
        }
    }
    (starts, collapsed)
}

fn significant(e: &TraceEvent) -> bool {
    match e.kind.as_str() {
        "user_prompt" | "assistant_message" | "session_start" | "session_end" | "tool_call" => true,
        "agent_meta" => {
            matches!(e.subtype.as_str(), "turn_duration" | "compacted")
                || matches!(
                    e.event_kind.as_str(),
                    "task_started" | "task_complete" | "turn_aborted" | "context_compacted"
                )
        }
        _ => false,
    }
}

fn event_window(
    store: &TraceStore,
    sid: &str,
    target: usize,
    before: usize,
    after: usize,
) -> Vec<TraceEvent> {
    let Some(indexes) = store.session_events.get(sid) else {
        return vec![];
    };
    let Some(pos) = indexes.iter().position(|&i| i == target) else {
        return vec![];
    };
    let lo = pos.saturating_sub(before);
    let hi = (pos + after + 1).min(indexes.len());
    indexes[lo..hi]
        .iter()
        .map(|&i| store.events[i].clone())
        .collect()
}

fn comparable(store: &TraceStore, resolved: Resolved) -> Result<Value> {
    match resolved {
        Resolved::Job(i) => Ok(json!(store.jobs[i])),
        Resolved::Turn(i) => Ok(json!(store.turns[i])),
        Resolved::Span(i) => Ok(json!(store.spans[i])),
        _ => bail!("trace compare accepts turn, span, or job ids"),
    }
}

fn turns_md(rows: &[&Turn]) -> String {
    let mut o = format!("# trace turns ({})\n\n", rows.len());
    for t in rows {
        o.push_str(&format!(
            "- `{}` [{}] {} · {} · {} · {} · {} · {} spans · {} errors\n  {}\n",
            t.started,
            t.id,
            t.status,
            duration_label(t.duration_ms, &t.duration_source),
            empty_dash(&t.source),
            empty_dash(&t.repo),
            empty_dash(&t.machine),
            t.spans,
            t.errors,
            empty_dash(&t.ask),
        ));
    }
    o
}

fn spans_md(rows: &[&Span]) -> String {
    let mut o = format!("# trace spans ({})\n\n", rows.len());
    for s in rows {
        o.push_str(&format!(
            "- `{}` [{}] {} · {} · {} · {} · {} · {}\n  {}\n",
            s.started,
            s.id,
            s.status,
            duration_label(s.duration_ms, &s.duration_source),
            s.operation,
            empty_dash(&s.source),
            empty_dash(&s.repo),
            empty_dash(&s.machine),
            empty_dash(&s.input),
        ));
    }
    o
}

fn jobs_md(rows: &[&Job]) -> String {
    let mut o = format!("# trace jobs ({})\n\n", rows.len());
    for j in rows {
        o.push_str(&format!(
            "- `{}` [{}] {} · {} elapsed · {} reported-wait · {} polls · {} errors · pid={} · {} · {} · {}\n  {}\n",
            j.started,
            j.id,
            j.status,
            duration(j.elapsed_ms),
            duration(Some(j.reported_wait_ms)),
            j.polls,
            j.errors,
            j.process_id,
            empty_dash(&j.repo),
            empty_dash(&j.machine),
            j.association,
            empty_dash(&j.command),
        ));
    }
    o
}

fn event_summary(ev: &Event) -> String {
    let p = &ev.payload;
    match ev.kind.as_str() {
        "user_prompt" | "assistant_message" | "thinking" => {
            excerpt(p["text"].as_str().unwrap_or(""), 180)
        }
        "tool_call" => {
            let name = p["name"].as_str().unwrap_or("?");
            let (input, command, _) = tool_input(p);
            format!(
                "{name} · {}",
                excerpt(if command.is_empty() { &input } else { &command }, 180)
            )
        }
        "tool_result" => excerpt(&result_output(p), 180),
        "session_start" => format!(
            "cwd={} source={}",
            p["cwd"].as_str().unwrap_or(""),
            ev.source
        ),
        "session_end" => format!("reason={}", p["reason"].as_str().unwrap_or("")),
        "attachment_ref" => p["local_path"]
            .as_str()
            .or_else(|| p["cwd"].as_str())
            .unwrap_or("attachment")
            .to_string(),
        "agent_meta" => {
            let subtype = p["subtype"].as_str().unwrap_or("");
            let ek = p["event_kind"].as_str().unwrap_or("");
            let d = p["duration_ms"]
                .as_u64()
                .or_else(|| p["payload"]["duration_ms"].as_u64());
            format!(
                "{}{}",
                if ek.is_empty() { subtype } else { ek },
                d.map(|n| format!(" · {}", duration(Some(n))))
                    .unwrap_or_default()
            )
        }
        _ => excerpt(&serde_json::to_string(p).unwrap_or_default(), 180),
    }
}

/// Normalize a fixed-size literal-search record from one raw envelope.
fn event_search_text(ev: &Event) -> String {
    let payload = serde_json::to_string(&ev.payload).unwrap_or_default();
    excerpt(
        &format!(
            "{} {} {} {} {}",
            ev.kind,
            ev.source,
            ev.session_id,
            event_summary(ev),
            payload,
        ),
        SEARCH_EVIDENCE,
    )
}

fn tool_input(p: &Value) -> (String, String, String) {
    let parsed;
    let input = if p["input"].is_object() {
        &p["input"]
    } else if let Some(s) = p["arguments"].as_str() {
        parsed = serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({"arguments": s}));
        &parsed
    } else {
        &p["action"]
    };
    let command = input["cmd"]
        .as_str()
        .or_else(|| input["command"].as_str())
        .or_else(|| input["script"].as_str())
        .map(|command| excerpt(command, EXCERPT))
        .unwrap_or_default();
    let workdir = input["workdir"]
        .as_str()
        .or_else(|| input["cwd"].as_str())
        .unwrap_or("")
        .to_string();
    (
        excerpt(&serde_json::to_string(input).unwrap_or_default(), EXCERPT),
        command,
        workdir,
    )
}

fn input_process_id(p: &Value) -> String {
    let parsed;
    let input = if p["input"].is_object() {
        &p["input"]
    } else if let Some(s) = p["arguments"].as_str() {
        parsed = serde_json::from_str::<Value>(s).unwrap_or(Value::Null);
        &parsed
    } else {
        &p["action"]
    };
    match &input["session_id"] {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        _ => String::new(),
    }
}

fn output_process_id(text: &str) -> String {
    let lower = text.to_lowercase();
    for marker in [
        "process running with session id ",
        "script running with cell id ",
    ] {
        let Some(pos) = lower.find(marker) else {
            continue;
        };
        let rest = &text[pos + marker.len()..];
        let id: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
            .collect();
        if !id.is_empty() {
            return id;
        }
    }
    String::new()
}

fn result_output(p: &Value) -> String {
    let v = if !p["output"].is_null() {
        &p["output"]
    } else {
        &p["content"]
    };
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn process_output(text: &str, limit: usize) -> String {
    let body = text
        .split_once(" Output:")
        .map(|(_, output)| output)
        .unwrap_or(text)
        .trim();
    excerpt(body, limit)
}

fn call_id(p: &Value) -> String {
    p["tool_use_id"]
        .as_str()
        .or_else(|| p["call_id"].as_str())
        .unwrap_or("")
        .to_string()
}

fn event_cwd(p: &Value) -> Option<String> {
    if let Some(cwd) = p["cwd"].as_str() {
        return Some(cwd.to_string());
    }
    let parsed;
    let input = if p["input"].is_object() {
        &p["input"]
    } else if let Some(s) = p["arguments"].as_str() {
        parsed = serde_json::from_str::<Value>(s).ok()?;
        &parsed
    } else {
        return None;
    };
    input["workdir"]
        .as_str()
        .or_else(|| input["cwd"].as_str())
        .map(str::to_string)
}

fn machine_from_stream(stream: &str) -> String {
    let rest = stream.strip_prefix("edge-").unwrap_or(stream);
    for suffix in ["-claudecode", "-codex", "-cowork"] {
        if let Some(machine) = rest.strip_suffix(suffix) {
            return machine.to_string();
        }
    }
    rest.to_string()
}

fn parse_exit_code(text: &str) -> Option<i64> {
    for marker in [
        "Process exited with code ",
        "Exit code ",
        "exited with code ",
        "\"exit_code\":",
    ] {
        if let Some(n) = number_after(text, marker) {
            return Some(n);
        }
    }
    None
}

fn result_status(explicit_error: bool, exit_code: Option<i64>, output: &str) -> &'static str {
    if explicit_error || exit_code.is_some_and(|n| n != 0) {
        return "error";
    }
    let lower = output.to_lowercase();
    if lower.contains("write_stdin failed:")
        || lower.contains("unknown session id")
        || lower.contains("no active session")
    {
        "error"
    } else if lower.contains("aborted by user") || lower.contains("cancelled by user") {
        "aborted"
    } else if lower.contains("process running with session id")
        || lower.contains("script running with cell id")
    {
        "running"
    } else {
        "ok"
    }
}

fn parse_wall_time_ms(text: &str) -> Option<u64> {
    if let Some(pos) = text.find("Wall time:") {
        let rest = text[pos + "Wall time:".len()..].trim_start();
        let token = rest.split_whitespace().next()?;
        return token
            .parse::<f64>()
            .ok()
            .map(|s| (s * 1000.0).round().max(0.0) as u64);
    }
    if let Some(pos) = text.find("\"wall_time_seconds\":") {
        let rest = text[pos + "\"wall_time_seconds\":".len()..].trim_start();
        let token = rest
            .split(|c: char| c == ',' || c == '}' || c.is_whitespace())
            .find(|s| !s.is_empty())?;
        return token
            .parse::<f64>()
            .ok()
            .map(|s| (s * 1000.0).round().max(0.0) as u64);
    }
    None
}

fn number_after(text: &str, marker: &str) -> Option<i64> {
    let pos = text.find(marker)?;
    let rest = text[pos + marker.len()..].trim_start();
    let token: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect();
    (!token.is_empty()).then(|| token.parse().ok()).flatten()
}

fn is_real_prompt(e: &TraceEvent) -> bool {
    e.kind == "user_prompt" && !trace_noise(&e.summary) && !e.summary.trim().is_empty()
}

fn trace_noise(text: &str) -> bool {
    let t = text.trim_start().to_lowercase();
    crate::ingest::is_noise(text)
        || t.starts_with("# agents.md instructions")
        || t.starts_with("<environment_context>")
        || t.starts_with("<permissions instructions>")
        || t.starts_with("<collaboration_mode>")
}

fn prompt_text(e: &TraceEvent) -> String {
    excerpt(&e.summary, 220)
}

fn turn_gap_ms(events: &[TraceEvent], indexes: &[usize]) -> Option<u64> {
    elapsed_ms(&events[*indexes.first()?].ts, &events[*indexes.last()?].ts)
}

fn elapsed_ms(a: &str, b: &str) -> Option<u64> {
    let a = ts_ms(a)?;
    let b = ts_ms(b)?;
    Some(b.saturating_sub(a).max(0) as u64)
}

fn ts_ms(ts: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(ts)
        .ok()
        .map(|d| d.timestamp_millis())
}

fn duration(ms: Option<u64>) -> String {
    let Some(ms) = ms else { return "—".into() };
    if ms < 1_000 {
        format!("{ms}ms")
    } else if ms < 60_000 {
        format!("{:.1}s", ms as f64 / 1_000.0)
    } else if ms < 3_600_000 {
        format!("{:.1}m", ms as f64 / 60_000.0)
    } else {
        format!("{:.1}h", ms as f64 / 3_600_000.0)
    }
}

fn duration_label(ms: Option<u64>, source: &str) -> String {
    let value = duration(ms);
    match source {
        "reported" => format!("{value} reported"),
        "event_gap" => format!("{value} event-gap"),
        _ => value,
    }
}

fn offset(ms: u64) -> String {
    let sec = ms / 1_000;
    format!("{:02}:{:02}", sec / 60, sec % 60)
}

fn excerpt(s: &str, n: usize) -> String {
    crate::excerpt(s, n)
}

fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}

fn display_id(store: &TraceStore, id: &str) -> String {
    let len = id.chars().count();
    for width in 8.min(len)..=len {
        let prefix: String = id.chars().take(width).collect();
        if store
            .ids
            .iter()
            .all(|other| other == id || !other.starts_with(&prefix))
        {
            return prefix;
        }
    }
    id.to_string()
}

fn empty_dash(s: &str) -> &str {
    if s.is_empty() { "—" } else { s }
}

fn value_short(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => "—".into(),
        Some(Value::String(s)) => excerpt(s, 100),
        Some(other) => excerpt(&other.to_string(), 100),
    }
}

fn md_cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(id: &str, ts: &str, source: &str, sid: &str, kind: &str, payload: Value) -> String {
        json!({
            "v": 1, "event_id": id, "stream": format!("edge-m-{}", if source == "codex_cli" { "codex" } else { "claudecode" }),
            "seq": 0, "ts": ts, "source": source, "session_id": sid,
            "kind": kind, "payload": payload,
        }).to_string()
    }

    #[test]
    fn codex_task_becomes_turn_with_reported_tool_span() {
        let lines = [
            ev(
                "e001",
                "2026-06-01T10:00:00Z",
                "codex_cli",
                "S",
                "session_start",
                json!({"cwd":"/tmp/repo"}),
            ),
            ev(
                "e002",
                "2026-06-01T10:00:01Z",
                "codex_cli",
                "S",
                "agent_meta",
                json!({"subtype":"event_msg","event_kind":"task_started","payload":{"turn_id":"T1"}}),
            ),
            ev(
                "e003",
                "2026-06-01T10:00:02Z",
                "codex_cli",
                "S",
                "user_prompt",
                json!({"text":"build the release binary"}),
            ),
            ev(
                "e004",
                "2026-06-01T10:00:03Z",
                "codex_cli",
                "S",
                "tool_call",
                json!({"name":"exec_command","call_id":"c1","arguments":"{\"cmd\":\"cargo build\"}"}),
            ),
            ev(
                "e005",
                "2026-06-01T10:00:06Z",
                "codex_cli",
                "S",
                "tool_result",
                json!({"call_id":"c1","output":"Wall time: 2.5 seconds Process exited with code 1"}),
            ),
            ev(
                "e006",
                "2026-06-01T10:00:11Z",
                "codex_cli",
                "S",
                "agent_meta",
                json!({"subtype":"event_msg","event_kind":"task_complete","payload":{"turn_id":"T1","duration_ms":10000}}),
            ),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let s = TraceStore::from_lines(&refs);
        assert_eq!(s.turns.len(), 1);
        assert_eq!(s.turns[0].id, "T1");
        assert_eq!(s.turns[0].duration_ms, Some(10_000));
        assert_eq!(s.turns[0].ask, "build the release binary");
        assert_eq!(s.turns[0].errors, 1);
        assert_eq!(s.spans[0].status, "error");
        assert_eq!(s.spans[0].exit_code, Some(1));
        assert_eq!(s.spans[0].duration_ms, Some(2_500));
        assert_eq!(s.spans[0].duration_source, "reported");
    }

    #[test]
    fn new_task_finalizes_an_abandoned_prior_turn() {
        let lines = [
            ev(
                "a001",
                "2026-06-01T10:00:00Z",
                "codex_cli",
                "S",
                "agent_meta",
                json!({"event_kind":"task_started","payload":{"turn_id":"T1"}}),
            ),
            ev(
                "a002",
                "2026-06-01T10:00:01Z",
                "codex_cli",
                "S",
                "user_prompt",
                json!({"text":"first task"}),
            ),
            ev(
                "a003",
                "2026-06-01T10:00:05Z",
                "codex_cli",
                "S",
                "agent_meta",
                json!({"event_kind":"task_started","payload":{"turn_id":"T2"}}),
            ),
            ev(
                "a004",
                "2026-06-01T10:00:06Z",
                "codex_cli",
                "S",
                "user_prompt",
                json!({"text":"second task"}),
            ),
            ev(
                "a005",
                "2026-06-01T10:00:09Z",
                "codex_cli",
                "S",
                "agent_meta",
                json!({"event_kind":"task_complete","payload":{"turn_id":"T2","duration_ms":3000}}),
            ),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let s = TraceStore::from_lines(&refs);
        let first = s.turns.iter().find(|t| t.id == "T1").expect("first turn");
        let second = s.turns.iter().find(|t| t.id == "T2").expect("second turn");
        assert_eq!(first.status, "unknown");
        assert_eq!(first.duration_ms, Some(1_000));
        assert_eq!(first.duration_source, "event_gap");
        assert_eq!(second.status, "ok");
        assert_eq!(second.duration_ms, Some(3_000));
        assert_eq!(s.turns.len(), 2);
    }

    #[test]
    fn codex_process_session_associates_exec_and_polls_into_one_job() {
        let lines = [
            ev(
                "j001",
                "2026-06-01T10:00:00Z",
                "codex_cli",
                "S",
                "tool_call",
                json!({"name":"exec_command","call_id":"c1","arguments":"{\"cmd\":\"make image\"}"}),
            ),
            ev(
                "j002",
                "2026-06-01T10:00:30Z",
                "codex_cli",
                "S",
                "tool_result",
                json!({"call_id":"c1","output":"Wall time: 30.0 seconds Process running with session ID 42"}),
            ),
            ev(
                "j003",
                "2026-06-01T10:00:31Z",
                "codex_cli",
                "S",
                "tool_call",
                json!({"name":"write_stdin","call_id":"c2","arguments":"{\"session_id\":42,\"chars\":\"\"}"}),
            ),
            ev(
                "j004",
                "2026-06-01T10:00:51Z",
                "codex_cli",
                "S",
                "tool_result",
                json!({"call_id":"c2","output":"Wall time: 20.0 seconds Process running with session ID 42"}),
            ),
            ev(
                "j005",
                "2026-06-01T10:00:52Z",
                "codex_cli",
                "S",
                "tool_call",
                json!({"name":"write_stdin","call_id":"c3","arguments":"{\"session_id\":42,\"chars\":\"\\u0003\"}"}),
            ),
            ev(
                "j006",
                "2026-06-01T10:00:53Z",
                "codex_cli",
                "S",
                "tool_result",
                json!({"call_id":"c3","output":"write_stdin failed: stdin is closed"}),
            ),
            ev(
                "j007",
                "2026-06-01T10:00:55Z",
                "codex_cli",
                "S",
                "tool_call",
                json!({"name":"write_stdin","call_id":"c4","arguments":"{\"session_id\":42,\"chars\":\"\"}"}),
            ),
            ev(
                "j008",
                "2026-06-01T10:01:00Z",
                "codex_cli",
                "S",
                "tool_result",
                json!({"call_id":"c4","output":"Wall time: 5.0 seconds Process exited with code 0 Output: built"}),
            ),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let s = TraceStore::from_lines(&refs);
        assert_eq!(s.jobs.len(), 1);
        let job = &s.jobs[0];
        assert_eq!(job.id, "job:j001");
        assert_eq!(job.process_id, "42");
        assert_eq!(job.association, "exact_process_session_id");
        assert_eq!(job.status, "ok");
        assert_eq!(job.spans, 4);
        assert_eq!(job.polls, 3);
        assert_eq!(job.stdin_writes, 1);
        assert_eq!(job.errors, 1);
        assert_eq!(job.elapsed_ms, Some(60_000));
        assert_eq!(job.reported_wait_ms, 55_000);
        assert_eq!(
            job.final_output,
            "Wall time: 5.0 seconds Process exited with code 0 Output: built"
        );
    }

    #[test]
    fn turn_timeline_collapses_same_turn_job_polls() {
        let lines = [
            ev(
                "t1",
                "2026-06-01T10:00:00Z",
                "codex_cli",
                "S",
                "agent_meta",
                json!({"event_kind":"task_started","payload":{"turn_id":"T"}}),
            ),
            ev(
                "t2",
                "2026-06-01T10:00:01Z",
                "codex_cli",
                "S",
                "user_prompt",
                json!({"text":"run the build"}),
            ),
            ev(
                "t3",
                "2026-06-01T10:00:02Z",
                "codex_cli",
                "S",
                "tool_call",
                json!({"name":"exec_command","call_id":"a","arguments":"{\"cmd\":\"make\"}"}),
            ),
            ev(
                "t4",
                "2026-06-01T10:00:03Z",
                "codex_cli",
                "S",
                "tool_result",
                json!({"call_id":"a","output":"Process running with session ID 9"}),
            ),
            ev(
                "t5",
                "2026-06-01T10:00:04Z",
                "codex_cli",
                "S",
                "tool_call",
                json!({"name":"write_stdin","call_id":"b","arguments":"{\"session_id\":9}"}),
            ),
            ev(
                "t6",
                "2026-06-01T10:00:05Z",
                "codex_cli",
                "S",
                "tool_result",
                json!({"call_id":"b","output":"Process exited with code 0"}),
            ),
            ev(
                "t7",
                "2026-06-01T10:00:06Z",
                "codex_cli",
                "S",
                "agent_meta",
                json!({"event_kind":"task_complete","payload":{"turn_id":"T"}}),
            ),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let s = TraceStore::from_lines(&refs);
        let timeline = turn_timeline(&s, &s.turns[0]);
        let jobs: Vec<&TimelineRow> = timeline.iter().filter(|r| r.kind == "job").collect();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].id, "job:t3");
        assert!(!timeline.iter().any(|r| r.summary.contains("write_stdin")));
        assert_eq!(significant_event_count(&s, &s.turns[0]), timeline.len());
    }

    #[test]
    fn process_ids_are_scoped_to_agent_session() {
        let lines = [
            ev(
                "a1",
                "2026-06-01T10:00:00Z",
                "codex_cli",
                "A",
                "tool_call",
                json!({"name":"exec_command","call_id":"a","arguments":"{\"cmd\":\"one\"}"}),
            ),
            ev(
                "a2",
                "2026-06-01T10:00:01Z",
                "codex_cli",
                "A",
                "tool_result",
                json!({"call_id":"a","output":"Process running with session ID 42"}),
            ),
            ev(
                "b1",
                "2026-06-01T10:00:02Z",
                "codex_cli",
                "B",
                "tool_call",
                json!({"name":"exec_command","call_id":"b","arguments":"{\"cmd\":\"two\"}"}),
            ),
            ev(
                "b2",
                "2026-06-01T10:00:03Z",
                "codex_cli",
                "B",
                "tool_result",
                json!({"call_id":"b","output":"Process running with session ID 42"}),
            ),
            ev(
                "a3",
                "2026-06-01T10:00:04Z",
                "codex_cli",
                "A",
                "tool_call",
                json!({"name":"write_stdin","call_id":"ac","arguments":"{\"session_id\":42}"}),
            ),
            ev(
                "a4",
                "2026-06-01T10:00:05Z",
                "codex_cli",
                "A",
                "tool_result",
                json!({"call_id":"ac","output":"Process exited with code 0"}),
            ),
            ev(
                "b3",
                "2026-06-01T10:00:06Z",
                "codex_cli",
                "B",
                "tool_call",
                json!({"name":"write_stdin","call_id":"bc","arguments":"{\"session_id\":42}"}),
            ),
            ev(
                "b4",
                "2026-06-01T10:00:07Z",
                "codex_cli",
                "B",
                "tool_result",
                json!({"call_id":"bc","output":"Process exited with code 0"}),
            ),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let s = TraceStore::from_lines(&refs);
        assert_eq!(s.jobs.len(), 2);
        assert!(s.jobs.iter().all(|j| j.spans == 2 && j.status == "ok"));
        assert_ne!(s.jobs[0].session_id, s.jobs[1].session_id);
    }

    #[test]
    fn missing_initiator_is_preserved_as_continuation_only_job() {
        let lines = [
            ev(
                "o1",
                "2026-06-01T10:00:00Z",
                "codex_cli",
                "S",
                "tool_call",
                json!({"name":"write_stdin","call_id":"o","arguments":"{\"session_id\":77}"}),
            ),
            ev(
                "o2",
                "2026-06-01T10:00:01Z",
                "codex_cli",
                "S",
                "tool_result",
                json!({"call_id":"o","output":"write_stdin failed: stdin is closed"}),
            ),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let s = TraceStore::from_lines(&refs);
        assert_eq!(s.jobs.len(), 1);
        assert_eq!(s.jobs[0].association, "continuation_only");
        assert_eq!(s.jobs[0].process_id, "77");
        assert_eq!(s.jobs[0].polls, 1);
        assert_eq!(s.jobs[0].status, "unknown");
    }

    #[test]
    fn claude_turn_ignores_command_echo_and_uses_turn_duration() {
        let lines = [
            ev(
                "c001",
                "2026-06-01T10:00:00Z",
                "claude_code",
                "S",
                "session_start",
                json!({"cwd":""}),
            ),
            ev(
                "c002",
                "2026-06-01T10:00:01Z",
                "claude_code",
                "S",
                "user_prompt",
                json!({"text":"<command-name>/model</command-name>"}),
            ),
            ev(
                "c003",
                "2026-06-01T10:00:02Z",
                "claude_code",
                "S",
                "user_prompt",
                json!({"text":"diagnose the failing deploy"}),
            ),
            ev(
                "c004",
                "2026-06-01T10:00:03Z",
                "claude_code",
                "S",
                "tool_call",
                json!({"name":"Bash","tool_use_id":"x","input":{"command":"kubectl get pods"}}),
            ),
            ev(
                "c005",
                "2026-06-01T10:00:04Z",
                "claude_code",
                "S",
                "tool_result",
                json!({"tool_use_id":"x","is_error":false,"content":"ok"}),
            ),
            ev(
                "c006",
                "2026-06-01T10:00:07Z",
                "claude_code",
                "S",
                "agent_meta",
                json!({"subtype":"turn_duration","duration_ms":5000}),
            ),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let s = TraceStore::from_lines(&refs);
        assert_eq!(s.turns.len(), 1);
        assert_eq!(s.turns[0].ask, "diagnose the failing deploy");
        assert_eq!(s.turns[0].duration_ms, Some(5_000));
        assert_eq!(s.turns[0].spans, 1);
        assert_eq!(s.spans[0].status, "ok");
    }

    #[test]
    fn machine_exit_and_wall_time_parsers_cover_canonical_outputs() {
        assert_eq!(machine_from_stream("edge-laptop-7-codex"), "laptop-7");
        assert_eq!(parse_exit_code("Process exited with code 28"), Some(28));
        assert_eq!(parse_exit_code("Exit code 127\nnot found"), Some(127));
        assert_eq!(
            parse_wall_time_ms("Wall time: 14.1876 seconds"),
            Some(14_188)
        );
        assert_eq!(
            result_status(false, None, "Process running with session ID 42"),
            "running"
        );
        assert_eq!(
            result_status(false, None, "aborted by user after 57.8s"),
            "aborted"
        );
        assert_eq!(
            result_status(false, None, "write_stdin failed: stdin is closed"),
            "error"
        );
    }

    #[test]
    fn trace_noise_rejects_codex_context_but_keeps_human_ask() {
        assert!(trace_noise("# AGENTS.md instructions for /repo"));
        assert!(trace_noise(
            "<environment_context>cwd</environment_context>"
        ));
        assert!(!trace_noise("investigate why the image build failed"));
    }

    #[test]
    fn span_list_prints_copyable_id_and_distinguishes_event_gap() {
        let lines = [
            ev(
                "01JTRACEFULLCALL0000000000",
                "2026-06-01T10:00:03Z",
                "claude_code",
                "S",
                "tool_call",
                json!({"name":"Bash","tool_use_id":"x","input":{"command":"make image"}}),
            ),
            ev(
                "01JTRACEFULLRESULT00000000",
                "2026-06-01T10:00:04Z",
                "claude_code",
                "S",
                "tool_result",
                json!({"tool_use_id":"x","is_error":false,"content":"ok"}),
            ),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let s = TraceStore::from_lines(&refs);
        let md = spans_md(&[&s.spans[0]]);
        assert!(md.contains("[01JTRACEFULLCALL0000000000]"));
        assert!(md.contains("1.0s event-gap"));
        assert!(md.contains("claude_code"));
    }

    #[test]
    fn filtered_search_keeps_repo_context_from_nonmatching_session_start() {
        let lines = [
            ev(
                "start",
                "2026-06-01T10:00:00Z",
                "claude_code",
                "S",
                "session_start",
                json!({"cwd":"/workspace/repo"}),
            ),
            ev(
                "result",
                "2026-06-01T10:00:01Z",
                "claude_code",
                "S",
                "tool_result",
                json!({"tool_use_id":"x","content":"missing libneedle.so"}),
            ),
        ];
        let known = HashSet::from(["repo".to_string()]);
        let mut builder = SnapshotBuilder::new(known, None);
        builder.fold_text(&lines.join("\n"));
        let store = builder.finish();
        let hits = search_hits(
            &store,
            "libneedle.so",
            Some("repo"),
            None,
            None,
            Some("tool_result"),
            10,
            None,
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].repo, "repo");
    }

    #[test]
    fn published_trace_excludes_pre_boundary_evidence_but_keeps_context() {
        let lines = [
            ev(
                "start",
                "2026-06-01T23:00:00Z",
                "harness",
                "S",
                "session_start",
                json!({"cwd":"/workspace/repo"}),
            ),
            ev(
                "old",
                "2026-06-01T23:30:00Z",
                "harness",
                "S",
                "user_prompt",
                json!({"text":"old-secret-evidence"}),
            ),
            ev(
                "new",
                "2026-06-02T00:30:00Z",
                "harness",
                "S",
                "user_prompt",
                json!({"text":"retained-evidence"}),
            ),
        ];
        let cutoff = chrono::DateTime::parse_from_rfc3339("2026-06-02T00:00:00Z")
            .unwrap()
            .timestamp_millis();
        let mut builder = SnapshotBuilder::new(HashSet::from(["repo".to_string()]), Some(cutoff));
        builder.fold_text(&lines.join("\n"));
        let store = builder.finish();
        assert_eq!(store.events.len(), 1);
        assert!(
            search_hits(&store, "old-secret", None, None, None, None, 10, None).is_empty()
        );
        let hits = search_hits(&store, "retained-evidence", None, None, None, None, 10, None);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].repo, "repo", "pre-boundary start still supplies scope context");
    }

    #[test]
    fn published_trace_roundtrips_bounded_literal_evidence() {
        let long = format!("visible-needle {} hidden-tail-needle", "x".repeat(4_000));
        let lines = [
            ev(
                "start",
                "2026-06-01T10:00:00Z",
                "harness",
                "S",
                "session_start",
                json!({"cwd":"/workspace/repo"}),
            ),
            ev(
                "prompt",
                "2026-06-01T10:00:01Z",
                "harness",
                "S",
                "user_prompt",
                json!({"text":long}),
            ),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let snapshot = TraceSnapshot::from_store(TraceStore::from_lines(&refs));
        let bytes = serde_json::to_vec(&snapshot).unwrap();
        let again = serde_json::to_vec(&TraceSnapshot::from_store(TraceStore::from_lines(&refs)))
            .unwrap();
        assert_eq!(bytes, again, "identical events publish identical trace bytes");
        assert!(
            bytes.len() < 4_000,
            "one oversized raw payload cannot inflate the mediated projection"
        );
        let restored: TraceSnapshot = serde_json::from_slice(&bytes).unwrap();
        let store = restored.into_store().unwrap();
        assert_eq!(
            search_hits(&store, "visible-needle", None, None, None, None, 10, None).len(),
            1
        );
        assert!(
            search_hits(
                &store,
                "hidden-tail-needle",
                None,
                None,
                None,
                None,
                10,
                None
            )
            .is_empty(),
            "literal search is explicitly bounded to published evidence"
        );
    }

    #[test]
    fn evidence_timelines_print_unambiguous_compact_ids() {
        let left = "01KSDGX6YM5Z48J2GQC1GVJ1M0";
        let right = "01KSDGX6ZZZZZZZZZZZZZZZZZZ";
        let lines = [
            ev(
                left,
                "2026-06-01T10:00:00Z",
                "claude_code",
                "S",
                "assistant_message",
                json!({"text":"one"}),
            ),
            ev(
                right,
                "2026-06-01T10:00:01Z",
                "claude_code",
                "S",
                "assistant_message",
                json!({"text":"two"}),
            ),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let s = TraceStore::from_lines(&refs);
        let shown_left = display_id(&s, left);
        let shown_right = display_id(&s, right);
        assert_ne!(shown_left, shown_right);
        assert!(shown_left.len() > 8);
        assert!(resolve(&s, &shown_left, None).is_ok());
        assert!(resolve(&s, &shown_right, None).is_ok());
    }

    #[test]
    fn scoped_trace_prefixes_never_reveal_disallowed_candidates() {
        let lines = [
            ev(
                "01JSCOPEDALLOWED0000000000",
                "2026-06-01T10:00:00Z",
                "harness",
                "allowed-session",
                "assistant_message",
                json!({"text":"allowed evidence"}),
            ),
            ev(
                "01JSCOPEDBLOCKED0000000000",
                "2026-06-01T10:00:01Z",
                "codex_cli",
                "blocked-session",
                "assistant_message",
                json!({"text":"blocked evidence"}),
            ),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let store = TraceStore::from_lines(&refs);
        let scope = crate::policy::ReadScope {
            sources: vec!["harness".into()],
            ..Default::default()
        };

        let resolved = resolve(&store, "01JS", Some(&scope)).expect("one allowed candidate");
        let Resolved::Event(index) = resolved else { panic!("expected an event") };
        assert_eq!(store.events[index].id, "01JSCOPEDALLOWED0000000000");
        assert!(
            resolve(&store, "01JSCOPEDBLOCKED", Some(&scope)).is_err(),
            "an exact disallowed prefix must look absent"
        );
    }
}
