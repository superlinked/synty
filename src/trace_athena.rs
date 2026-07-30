// Read-only remote trace access over Athena. SQL prunes the immutable event
// lake by injected stream partitions and a bounded time window; the existing
// trace fold then reconstructs turns, spans, and jobs from only those rows.

use crate::{bucket, event_partitions, metrics, policy::ReadScope, trace};
use anyhow::{Context, Result, bail};
use aws_config::timeout::TimeoutConfig;
use aws_sdk_athena::types::{QueryExecutionContext, QueryExecutionState};
use chrono::{DateTime, Duration, NaiveDate, Utc};
use std::collections::BTreeSet;
use std::fmt;
use std::time::{Duration as StdDuration, Instant};

const DEFAULT_LIST_HOURS: i64 = 1;
const DEFAULT_LOOKUP_HOURS: i64 = 24 * 7;
const ID_LOOKUP_MINUTES: i64 = 5;
const MAX_LOOKBACK_HOURS: i64 = 24 * 7;
const MAX_EVENTS: usize = 50_000;
const MAX_RESULT_BYTES: usize = 64 * 1024 * 1024;
const MAX_SESSIONS: usize = 200;
const MAX_OBJECT_PATHS: usize = 1_000;
const MAX_PATH_PREDICATE_BYTES: usize = 200_000;
const REQUEST_QUERY_TIMEOUT: StdDuration = StdDuration::from_secs(45);
const AWS_CONNECT_TIMEOUT: StdDuration = StdDuration::from_secs(3);
const AWS_OPERATION_TIMEOUT: StdDuration = StdDuration::from_secs(10);

#[derive(Debug)]
enum TraceQueryError {
    Timeout(String),
    Limit(String),
}

impl fmt::Display for TraceQueryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timeout(message) | Self::Limit(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for TraceQueryError {}

fn timeout_error(message: impl Into<String>) -> anyhow::Error {
    TraceQueryError::Timeout(message.into()).into()
}

fn limit_error(message: impl Into<String>) -> anyhow::Error {
    TraceQueryError::Limit(message.into()).into()
}

#[derive(Clone, Debug)]
pub(crate) struct Config {
    pub bucket: String,
    pub workgroup: String,
    pub database: String,
    pub table: String,
}

impl Config {
    pub(crate) fn new(
        bucket: String,
        workgroup: String,
        database: String,
        table: String,
    ) -> Result<Self> {
        anyhow::ensure!(
            bucket.starts_with("s3://"),
            "Athena trace requires an s3:// bucket"
        );
        validate_name("Athena workgroup", &workgroup, false)?;
        validate_name("Glue database", &database, true)?;
        validate_name("Glue table", &table, true)?;
        Ok(Self {
            bucket,
            workgroup,
            database,
            table,
        })
    }
}

struct QueryRows {
    lines: Vec<String>,
}

struct ObjectSelection {
    days: Vec<String>,
    paths: Vec<String>,
}

trait EventQuery: Send {
    fn run(&mut self, sql: &str, deadline: Instant) -> Result<QueryRows>;
}

struct AwsEventQuery {
    client: aws_sdk_athena::Client,
    runtime: tokio::runtime::Runtime,
    workgroup: String,
    database: String,
}

impl AwsEventQuery {
    fn new(config: &Config) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("start Athena runtime")?;
        let profile = crate::config::load().aws_profile;
        let timeout = TimeoutConfig::builder()
            .connect_timeout(AWS_CONNECT_TIMEOUT)
            .operation_attempt_timeout(AWS_OPERATION_TIMEOUT)
            .operation_timeout(AWS_OPERATION_TIMEOUT)
            .build();
        let mut loader =
            aws_config::defaults(aws_config::BehaviorVersion::latest()).timeout_config(timeout);
        if let Some(profile) = profile {
            loader = loader.profile_name(profile);
        }
        let sdk = runtime.block_on(loader.load());
        Ok(Self {
            client: aws_sdk_athena::Client::new(&sdk),
            runtime,
            workgroup: config.workgroup.clone(),
            database: config.database.clone(),
        })
    }

    fn stop_execution(&self, id: &str) {
        let _ = self.runtime.block_on(
            self.client
                .stop_query_execution()
                .query_execution_id(id)
                .send(),
        );
    }
}

impl EventQuery for AwsEventQuery {
    fn run(&mut self, sql: &str, deadline: Instant) -> Result<QueryRows> {
        let started = Instant::now();
        let mut row_count = 0usize;
        let mut result_bytes = 0usize;
        let mut scanned_bytes = 0i64;
        let result = (|| {
            anyhow::ensure!(
                sql.trim_start().to_ascii_uppercase().starts_with("SELECT "),
                "Athena trace only permits SELECT statements"
            );
            if Instant::now() >= deadline {
                return Err(timeout_error(
                    "Athena trace request timed out before query execution",
                ));
            }
            let execution = self
                .runtime
                .block_on(
                    self.client
                        .start_query_execution()
                        .work_group(&self.workgroup)
                        .query_execution_context(
                            QueryExecutionContext::builder()
                                .database(&self.database)
                                .build(),
                        )
                        .query_string(sql)
                        .send(),
                )
                .context("start Athena trace query")?;
            let id = execution
                .query_execution_id()
                .context("Athena returned no query execution id")?
                .to_string();

            scanned_bytes = loop {
                if Instant::now() >= deadline {
                    self.stop_execution(&id);
                    return Err(timeout_error(
                        "Athena trace request timed out after its shared query budget",
                    ));
                }
                let poll = (|| {
                    let execution = self
                        .runtime
                        .block_on(
                            self.client
                                .get_query_execution()
                                .query_execution_id(&id)
                                .send(),
                        )
                        .context("poll Athena trace query")?;
                    let query = execution
                        .query_execution()
                        .context("Athena returned no query execution")?;
                    let status = query.status().context("Athena returned no query status")?;
                    match status.state() {
                        Some(QueryExecutionState::Succeeded) => Ok(Some(
                            query
                                .statistics()
                                .and_then(|statistics| statistics.data_scanned_in_bytes())
                                .unwrap_or(0),
                        )),
                        Some(QueryExecutionState::Failed | QueryExecutionState::Cancelled) => {
                            bail!(
                                "Athena trace query {}: {}",
                                status
                                    .state()
                                    .map(|state| state.as_str())
                                    .unwrap_or("failed"),
                                status.state_change_reason().unwrap_or("no reason returned")
                            )
                        }
                        _ => Ok(None),
                    }
                })();
                match poll {
                    Ok(Some(bytes)) => break bytes,
                    Ok(None) => std::thread::sleep(StdDuration::from_millis(250)),
                    Err(error) => {
                        self.stop_execution(&id);
                        return Err(error);
                    }
                }
            };

            let mut lines = Vec::new();
            let mut next_token = None;
            let mut first_page = true;
            loop {
                if Instant::now() >= deadline {
                    return Err(timeout_error(
                        "Athena trace result retrieval timed out after its shared query budget",
                    ));
                }
                let mut request = self
                    .client
                    .get_query_results()
                    .query_execution_id(&id)
                    .max_results(1000);
                if let Some(token) = next_token.as_deref() {
                    request = request.next_token(token);
                }
                let page = self
                    .runtime
                    .block_on(request.send())
                    .context("read Athena trace query results")?;
                if let Some(result_set) = page.result_set() {
                    for (index, row) in result_set.rows().iter().enumerate() {
                        if first_page && index == 0 {
                            continue;
                        }
                        let Some(line) =
                            row.data().first().and_then(|datum| datum.var_char_value())
                        else {
                            continue;
                        };
                        result_bytes = result_bytes.saturating_add(line.len());
                        if result_bytes > MAX_RESULT_BYTES {
                            return Err(limit_error(format!(
                                "Athena trace selection exceeds {} MiB; narrow the time, machine, source, or operation filter",
                                MAX_RESULT_BYTES / 1024 / 1024
                            )));
                        }
                        lines.push(line.to_string());
                        row_count = lines.len();
                        if lines.len() > MAX_EVENTS {
                            return Err(limit_error(format!(
                                "Athena trace selection exceeds {MAX_EVENTS} events; narrow the time, machine, source, or operation filter"
                            )));
                        }
                    }
                }
                first_page = false;
                next_token = page.next_token().map(str::to_string);
                if next_token.is_none() {
                    break;
                }
            }
            Ok(QueryRows { lines })
        })();
        let outcome = result
            .as_ref()
            .err()
            .map(query_outcome)
            .unwrap_or("success");
        metrics::Run::new("athena_trace")
            .set("outcome", outcome)
            .set("rows", row_count)
            .set("result_bytes", result_bytes)
            .set("scanned_bytes", scanned_bytes)
            .set("elapsed_ms", started.elapsed().as_millis() as u64)
            .emit();
        result
    }
}

pub(crate) struct Backend {
    config: Config,
    query: Box<dyn EventQuery>,
    // Tests inject a fixed registry; production re-lists the bounded registry
    // for every tool call so newly enrolled streams appear without a restart.
    streams: Option<Vec<String>>,
    // Tests may inject physical partitions without opening a real bucket.
    #[cfg(test)]
    days: Option<Vec<String>>,
    cached: Option<trace::TraceStore>,
}

impl Backend {
    pub(crate) fn new(config: Config) -> Result<Self> {
        let query = Box::new(AwsEventQuery::new(&config)?);
        Ok(Self {
            config,
            query,
            streams: None,
            #[cfg(test)]
            days: None,
            cached: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn list(
        &mut self,
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
        scope: &ReadScope,
    ) -> Result<String> {
        let deadline = Instant::now() + REQUEST_QUERY_TIMEOUT;
        let window = Window::parse(since, None, DEFAULT_LIST_HOURS)?;
        let store = self.load_store(
            window,
            machine,
            source,
            operation,
            None,
            &[],
            operation.is_some(),
            scope,
            deadline,
        )?;
        let out = trace::list_store_text(
            &store,
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
            false,
            Some(scope),
        )?;
        Ok(out)
    }

    pub(crate) fn show(
        &mut self,
        id: &str,
        before: usize,
        after: usize,
        scope: &ReadScope,
    ) -> Result<String> {
        if let Some(store) = &self.cached
            && let Ok(out) = trace::show_store_text(store, id, before, after, false, Some(scope))
        {
            return Ok(out);
        }
        let deadline = Instant::now() + REQUEST_QUERY_TIMEOUT;
        let window = id_lookup_window(id)
            .unwrap_or(Window::parse(None, None, DEFAULT_LOOKUP_HOURS)?);
        let store =
            self.load_store(window, None, None, None, None, &[id], true, scope, deadline)?;
        let out = trace::show_store_text(&store, id, before, after, false, Some(scope))?;
        self.cached = Some(store);
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search(
        &mut self,
        query: &str,
        repo: Option<&str>,
        machine: Option<&str>,
        source: Option<&str>,
        kind: Option<&str>,
        limit: usize,
        scope: &ReadScope,
    ) -> Result<String> {
        let deadline = Instant::now() + REQUEST_QUERY_TIMEOUT;
        let window = Window::parse(None, None, DEFAULT_LOOKUP_HOURS)?;
        let store = self.load_store(
            window,
            machine,
            source,
            Some(query),
            kind,
            &[],
            false,
            scope,
            deadline,
        )?;
        let out = trace::search_store_text(
            &store,
            query,
            repo,
            machine,
            source,
            kind,
            limit,
            false,
            Some(scope),
        )?;
        Ok(out)
    }

    pub(crate) fn compare(&mut self, left: &str, right: &str, scope: &ReadScope) -> Result<String> {
        if let Some(store) = &self.cached
            && let Ok(out) = trace::compare_store_text(store, left, right, false, Some(scope))
        {
            return Ok(out);
        }
        let deadline = Instant::now() + REQUEST_QUERY_TIMEOUT;
        let window = ids_lookup_window(&[left, right])
            .unwrap_or(Window::parse(None, None, DEFAULT_LOOKUP_HOURS)?);
        let store = self.load_store(
            window,
            None,
            None,
            None,
            None,
            &[left, right],
            true,
            scope,
            deadline,
        )?;
        let out = trace::compare_store_text(&store, left, right, false, Some(scope))?;
        self.cached = Some(store);
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_store(
        &mut self,
        window: Window,
        machine: Option<&str>,
        source: Option<&str>,
        needle: Option<&str>,
        kind: Option<&str>,
        ids: &[&str],
        expand_matching_sessions: bool,
        scope: &ReadScope,
        deadline: Instant,
    ) -> Result<trace::TraceStore> {
        let streams = self.selected_streams(machine, source, scope)?;
        let predicate = Predicate {
            source: source.map(str::to_string),
            scope_sources: scope.sources.clone(),
            needle: needle.map(str::to_string),
            kind_contains: kind.map(str::to_string),
            ids: ids.iter().map(|id| query_id(id).to_string()).collect(),
            ..Default::default()
        };
        let first = self.select(&streams, window, &predicate, deadline)?;
        let sessions = event_sessions(&first.lines);
        let matched_streams = event_streams(&first.lines);
        let context_streams = if matched_streams.is_empty() {
            streams.clone()
        } else {
            streams
                .iter()
                .filter(|stream| matched_streams.contains(*stream))
                .cloned()
                .collect()
        };
        if sessions.len() > MAX_SESSIONS {
            return Err(limit_error(format!(
                "Athena trace selection spans more than {MAX_SESSIONS} sessions; narrow the time, machine, source, or operation filter"
            )));
        }
        let context_window = Window {
            since: std::cmp::max(
                window.since - Duration::days(1),
                window.until - Duration::hours(MAX_LOOKBACK_HOURS),
            ),
            until: window.until,
        };
        let expands_sessions = (!ids.is_empty() || (needle.is_some() && expand_matching_sessions))
            && !sessions.is_empty();
        let mut lines = if expands_sessions {
            self.select(
                &context_streams,
                context_window,
                &Predicate {
                    sessions: sessions.iter().cloned().collect(),
                    scope_sources: scope.sources.clone(),
                    ..Default::default()
                },
                deadline,
            )?
            .lines
        } else {
            first.lines
        };
        if !sessions.is_empty() && !expands_sessions {
            let mut contexts = self
                .select(
                    &context_streams,
                    context_window,
                    &Predicate {
                        sessions: sessions.into_iter().collect(),
                        kinds: vec!["session_start".into(), "agent_meta".into()],
                        scope_sources: scope.sources.clone(),
                        dedupe_session_kinds: true,
                        ..Default::default()
                    },
                    deadline,
                )?
                .lines;
            contexts.append(&mut lines);
            lines = contexts;
        }
        Ok(trace::TraceStore::from_text(&lines.join("\n")))
    }

    fn select(
        &mut self,
        streams: &[String],
        window: Window,
        predicate: &Predicate,
        deadline: Instant,
    ) -> Result<QueryRows> {
        let permits_partition_scan =
            !predicate.ids.is_empty() || !predicate.sessions.is_empty();
        let (selection, partition_scan) = match self.selected_objects(streams, window) {
            Ok(selection) => (selection, false),
            Err(error)
                if permits_partition_scan
                    && matches!(
                        error.downcast_ref::<TraceQueryError>(),
                        Some(TraceQueryError::Limit(_))
                    ) =>
            {
                (
                    ObjectSelection {
                        days: window_days(window).into_iter().collect(),
                        paths: Vec::new(),
                    },
                    true,
                )
            }
            Err(error) => return Err(error),
        };
        if selection.days.is_empty() || (!partition_scan && selection.paths.is_empty()) {
            metrics::Run::new("athena_trace")
                .set("outcome", "empty")
                .set("rows", 0)
                .set("result_bytes", 0)
                .set("scanned_bytes", 0)
                .set("elapsed_ms", 0)
                .emit();
            return Ok(QueryRows { lines: Vec::new() });
        }
        let sql = select_sql(
            &self.config,
            streams,
            &selection.days,
            &selection.paths,
            window,
            predicate,
            MAX_EVENTS + 1,
        )?;
        self.query.run(&sql, deadline)
    }

    fn selected_streams(
        &mut self,
        machine: Option<&str>,
        source: Option<&str>,
        scope: &ReadScope,
    ) -> Result<Vec<String>> {
        let mut streams = if let Some(streams) = &self.streams {
            streams.clone()
        } else {
            let keys = bucket::open(&self.config.bucket)?.list("event-streams/")?;
            let streams: Vec<String> = keys
                .into_iter()
                .filter_map(|key| key.strip_prefix("event-streams/").map(str::to_string))
                .filter(|stream| !stream.is_empty() && !stream.contains('/'))
                .collect();
            anyhow::ensure!(
                !streams.is_empty(),
                "no event streams found in {}",
                self.config.bucket
            );
            streams
        };
        let machine = machine.map(str::to_ascii_lowercase);
        let requested_source = source.map(normalized_stream_source);
        let scope_sources = scope
            .sources
            .iter()
            .map(|source| normalized_stream_source(source))
            .collect::<BTreeSet<_>>();
        streams.retain(|stream| {
            let Some((stream_machine, stream_source)) = crate::identity::stream_parts(stream)
            else {
                return machine.is_none() && requested_source.is_none() && scope_sources.is_empty();
            };
            machine
                .as_deref()
                .is_none_or(|wanted| stream_machine.eq_ignore_ascii_case(wanted))
                && requested_source
                    .as_deref()
                    .is_none_or(|wanted| stream_source.eq_ignore_ascii_case(wanted))
                && (scope_sources.is_empty()
                    || scope_sources.contains(&normalized_stream_source(stream_source)))
        });
        anyhow::ensure!(
            !streams.is_empty(),
            "no event streams match the machine, source, and read-scope filters"
        );
        Ok(streams)
    }

    fn selected_objects(&self, streams: &[String], window: Window) -> Result<ObjectSelection> {
        #[cfg(test)]
        if let Some(days) = &self.days {
            let stream = streams.first().context("Athena trace needs a test stream")?;
            return Ok(ObjectSelection {
                days: days.clone(),
                paths: days
                    .iter()
                    .map(|day| {
                        format!(
                            "{}/events/{stream}/chunks/track.{day}/test.jsonl",
                            self.config.bucket.trim_end_matches('/')
                        )
                    })
                    .collect(),
            });
        }
        let bucket = bucket::open(&self.config.bucket)?;
        let mut selected = BTreeSet::new();
        for stream in streams {
            if let Some(index) = event_partitions::load(bucket.as_ref(), stream)? {
                for day in index.candidate_days(window.since, window.until) {
                    let physical_day = bucket
                        .list(&format!("events/{stream}/chunks/track.{day}/"))?
                        .into_iter()
                        .collect::<BTreeSet<_>>();
                    if let Some(objects) =
                        event_partitions::load_objects(bucket.as_ref(), stream, &day)?
                    {
                        let indexed_objects =
                            objects.objects.keys().cloned().collect::<BTreeSet<_>>();
                        selected.extend(
                            objects.candidate_objects(window.since, window.until),
                        );
                        selected.extend(
                            physical_day
                                .difference(&indexed_objects)
                                .cloned(),
                        );
                    } else {
                        selected.extend(physical_day);
                    }
                }
            } else {
                let days = window_days(window);
                selected.extend(
                    bucket
                        .list(&format!("events/{stream}/chunks/"))?
                        .into_iter()
                        .filter(|key| {
                            event_partitions::day_from_event_key(key)
                                .is_some_and(|day| days.contains(day))
                        }),
                );
            }
        }
        if selected.len() > MAX_OBJECT_PATHS {
            return Err(limit_error(format!(
                "Athena trace selection needs {} raw objects, exceeding the {MAX_OBJECT_PATHS}-object query limit; narrow the machine, source, or time filter, or backfill object-range metadata",
                selected.len()
            )));
        }
        let path_bytes = selected
            .iter()
            .map(|key| key.len() + self.config.bucket.len() + 4)
            .sum::<usize>();
        if path_bytes > MAX_PATH_PREDICATE_BYTES {
            return Err(limit_error(format!(
                "Athena trace object paths need {path_bytes} SQL bytes, exceeding the {MAX_PATH_PREDICATE_BYTES}-byte path-predicate limit; narrow the machine, source, or time filter"
            )));
        }
        let days = selected
            .iter()
            .filter_map(|key| event_partitions::day_from_event_key(key).map(str::to_string))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let paths = selected
            .into_iter()
            .map(|key| format!("{}/{}", self.config.bucket.trim_end_matches('/'), key))
            .collect();
        Ok(ObjectSelection { days, paths })
    }
}

#[derive(Default)]
struct Predicate {
    source: Option<String>,
    scope_sources: Vec<String>,
    needle: Option<String>,
    kind_contains: Option<String>,
    ids: Vec<String>,
    sessions: Vec<String>,
    kinds: Vec<String>,
    dedupe_session_kinds: bool,
}

#[derive(Clone, Copy)]
struct Window {
    since: DateTime<Utc>,
    until: DateTime<Utc>,
}

impl Window {
    fn parse(since: Option<&str>, until: Option<&str>, default_hours: i64) -> Result<Self> {
        let until = match until {
            Some(value) => parse_time(value)?,
            None => Utc::now(),
        };
        let since = match since {
            Some(value) => parse_time(value)?,
            None => until - Duration::hours(default_hours),
        };
        anyhow::ensure!(since < until, "trace since must be before until");
        if until - since > Duration::hours(MAX_LOOKBACK_HOURS) {
            return Err(limit_error(format!(
                "Athena trace windows are limited to {MAX_LOOKBACK_HOURS} hours"
            )));
        }
        Ok(Self { since, until })
    }
}

fn parse_time(value: &str) -> Result<DateTime<Utc>> {
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(value) {
        return Ok(timestamp.with_timezone(&Utc));
    }
    let date = NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .with_context(|| format!("trace time must be RFC3339 or YYYY-MM-DD: {value}"))?;
    Ok(date
        .and_hms_opt(0, 0, 0)
        .expect("midnight exists")
        .and_utc())
}

fn select_sql(
    config: &Config,
    streams: &[String],
    days: &[String],
    paths: &[String],
    window: Window,
    predicate: &Predicate,
    limit: usize,
) -> Result<String> {
    anyhow::ensure!(
        !streams.is_empty(),
        "Athena trace needs at least one stream"
    );
    anyhow::ensure!(!days.is_empty(), "Athena trace needs at least one day");
    let stream_values = streams
        .iter()
        .map(|stream| sql_string(stream))
        .collect::<Vec<_>>()
        .join(", ");
    let day_values = days
        .iter()
        .map(|day| sql_string(day))
        .collect::<Vec<_>>()
        .join(", ");
    let mut clauses = vec![
        format!("stream IN ({stream_values})"),
        format!("day IN ({day_values})"),
    ];
    if !paths.is_empty() {
        let path_values = paths
            .iter()
            .map(|path| sql_string(path))
            .collect::<Vec<_>>()
            .join(", ");
        clauses.push(format!("\"$path\" IN ({path_values})"));
    }
    clauses.extend([
        format!(
            "from_iso8601_timestamp(json_extract_scalar(line, '$.ts')) >= from_iso8601_timestamp({})",
            sql_string(&window.since.to_rfc3339())
        ),
        format!(
            "from_iso8601_timestamp(json_extract_scalar(line, '$.ts')) < from_iso8601_timestamp({})",
            sql_string(&window.until.to_rfc3339())
        ),
    ]);
    if let Some(source) = predicate.source.as_deref() {
        clauses.push(format!(
            "strpos(lower(coalesce(json_extract_scalar(line, '$.source'), '')), {}) > 0",
            sql_string(&source.to_ascii_lowercase())
        ));
    }
    if !predicate.scope_sources.is_empty() {
        clauses.push(format!(
            "json_extract_scalar(line, '$.source') IN ({})",
            predicate
                .scope_sources
                .iter()
                .map(|source| sql_string(source))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if let Some(needle) = predicate.needle.as_deref() {
        clauses.push(format!(
            "strpos(lower(line), {}) > 0",
            sql_string(&needle.to_ascii_lowercase())
        ));
    }
    if let Some(kind) = predicate.kind_contains.as_deref() {
        clauses.push(format!(
            "strpos(lower(coalesce(json_extract_scalar(line, '$.kind'), '')), {}) > 0",
            sql_string(&kind.to_ascii_lowercase())
        ));
    }
    if !predicate.ids.is_empty() {
        let matches = predicate
            .ids
            .iter()
            .map(|id| {
                format!(
                    "(starts_with(coalesce(json_extract_scalar(line, '$.event_id'), ''), {id}) OR \
                     starts_with(coalesce(json_extract_scalar(line, '$.session_id'), ''), {id}) OR \
                     strpos(line, {id}) > 0)",
                    id = sql_string(id)
                )
            })
            .collect::<Vec<_>>()
            .join(" OR ");
        clauses.push(format!("({matches})"));
    }
    if !predicate.sessions.is_empty() {
        clauses.push(format!(
            "json_extract_scalar(line, '$.session_id') IN ({})",
            predicate
                .sessions
                .iter()
                .map(|session| sql_string(session))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !predicate.kinds.is_empty() {
        clauses.push(format!(
            "json_extract_scalar(line, '$.kind') IN ({})",
            predicate
                .kinds
                .iter()
                .map(|kind| sql_string(kind))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let where_clause = clauses.join("\n  AND ");
    if predicate.dedupe_session_kinds {
        Ok(format!(
            "SELECT line FROM (\n  SELECT line, row_number() OVER (\n    \
             PARTITION BY json_extract_scalar(line, '$.session_id'), \
             json_extract_scalar(line, '$.kind')\n    \
             ORDER BY json_extract_scalar(line, '$.ts')\n  ) AS synty_context_rank\n  \
             FROM \"{}\".\"{}\"\n  WHERE {}\n)\n\
             WHERE synty_context_rank = 1\n\
             ORDER BY json_extract_scalar(line, '$.ts')\nLIMIT {limit}",
            config.database, config.table, where_clause,
        ))
    } else {
        Ok(format!(
            "SELECT line FROM \"{}\".\"{}\"\nWHERE {}\n\
             ORDER BY json_extract_scalar(line, '$.ts'), \
             try_cast(json_extract_scalar(line, '$.seq') AS bigint)\nLIMIT {limit}",
            config.database, config.table, where_clause,
        ))
    }
}

fn event_sessions(lines: &[String]) -> BTreeSet<String> {
    let mut sessions = BTreeSet::new();
    for line in lines {
        if let Ok(event) = serde_json::from_str::<crate::event::Event>(line)
            && !event.session_id.is_empty()
        {
            sessions.insert(event.session_id);
        }
    }
    sessions
}

fn event_streams(lines: &[String]) -> BTreeSet<String> {
    lines
        .iter()
        .filter_map(|line| serde_json::from_str::<crate::event::Event>(line).ok())
        .map(|event| event.stream)
        .filter(|stream| !stream.is_empty())
        .collect()
}

fn sql_string(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn query_id(id: &str) -> &str {
    id.strip_prefix("job:").unwrap_or(id)
}

fn id_lookup_window(id: &str) -> Option<Window> {
    ids_lookup_window(&[id])
}

fn ids_lookup_window(ids: &[&str]) -> Option<Window> {
    let mut timestamps = ids.iter().map(|id| {
        let timestamp = crate::event::ulid_timestamp_ms(query_id(id))?;
        DateTime::<Utc>::from_timestamp_millis(timestamp as i64)
    });
    let first = timestamps.next()??;
    let (mut since, mut until) = (first, first);
    for timestamp in timestamps {
        let timestamp = timestamp?;
        since = since.min(timestamp);
        until = until.max(timestamp);
    }
    let window = Window {
        since: since - Duration::minutes(ID_LOOKUP_MINUTES),
        until: until + Duration::minutes(ID_LOOKUP_MINUTES),
    };
    (window.until - window.since <= Duration::hours(MAX_LOOKBACK_HOURS))
        .then_some(window)
}

fn normalized_stream_source(source: &str) -> String {
    match source.to_ascii_lowercase().as_str() {
        "codex" | "codex_cli" | "codex-cli" => "codex".into(),
        "claude" | "claudecode" | "claude_code" | "claude-code" => "claudecode".into(),
        "cowork" => "cowork".into(),
        "harness" => "harness".into(),
        "devin" => "devin".into(),
        other => other.to_string(),
    }
}

fn window_days(window: Window) -> BTreeSet<String> {
    let mut days = BTreeSet::new();
    let mut day = window.since.date_naive();
    loop {
        let start = day
            .and_hms_opt(0, 0, 0)
            .expect("midnight exists")
            .and_utc();
        if start >= window.until {
            break;
        }
        days.insert(day.format("%Y-%m-%d").to_string());
        let Some(next) = day.succ_opt() else {
            break;
        };
        day = next;
    }
    days
}

fn query_outcome(error: &anyhow::Error) -> &'static str {
    match error.downcast_ref::<TraceQueryError>() {
        Some(TraceQueryError::Timeout(_)) => "timeout",
        Some(TraceQueryError::Limit(_)) => "limit",
        None => "error",
    }
}

fn validate_name(label: &str, value: &str, lowercase_only: bool) -> Result<()> {
    let valid = !value.is_empty()
        && value.len() <= 128
        && value.chars().all(|ch| {
            ch.is_ascii_lowercase()
                || ch.is_ascii_digit()
                || ch == '_'
                || (!lowercase_only && (ch.is_ascii_uppercase() || matches!(ch, '.' | '-')))
        });
    if !valid {
        bail!("{label} contains unsupported characters");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    struct FakeQuery {
        lines: Vec<String>,
        calls: Arc<Mutex<Vec<String>>>,
    }

    impl EventQuery for FakeQuery {
        fn run(&mut self, sql: &str, _deadline: Instant) -> Result<QueryRows> {
            self.calls.lock().unwrap().push(sql.to_string());
            Ok(QueryRows {
                lines: self.lines.clone(),
            })
        }
    }

    fn event(id: &str, ts: &str, kind: &str, payload: serde_json::Value) -> String {
        json!({
            "v": 1, "event_id": id, "stream": "edge-m-codex", "seq": 1,
            "ts": ts, "source": "codex_cli", "session_id": "session-1",
            "kind": kind, "payload": payload, "rollup_dim": ""
        })
        .to_string()
    }

    #[test]
    fn bounded_select_uses_both_physical_partitions_and_never_writes() {
        let config = Config::new(
            "s3://bucket".into(),
            "wg".into(),
            "synty".into(),
            "raw_events".into(),
        )
        .unwrap();
        let sql = select_sql(
            &config,
            &["edge-m-codex".into()],
            &["2026-07-20".into(), "2026-07-22".into()],
            &[
                "s3://bucket/events/edge-m-codex/chunks/track.2026-07-20/a.jsonl".into(),
                "s3://bucket/events/edge-m-codex/chunks/track.2026-07-22/b.jsonl".into(),
            ],
            Window {
                since: parse_time("2026-07-22T10:00:00Z").unwrap(),
                until: parse_time("2026-07-22T11:00:00Z").unwrap(),
            },
            &Predicate::default(),
            101,
        )
        .unwrap();
        assert!(sql.starts_with("SELECT line"));
        assert!(sql.contains("stream IN ('edge-m-codex')"));
        assert!(sql.contains("day IN ('2026-07-20', '2026-07-22')"));
        assert!(sql.contains("\"$path\" IN ('s3://bucket/events/edge-m-codex/"));
        for mutating in ["INSERT", "UPDATE", "DELETE", "CREATE", "UNLOAD", "CTAS"] {
            assert!(!sql.to_ascii_uppercase().contains(mutating), "{mutating}");
        }
    }

    #[test]
    fn exact_id_falls_back_to_stream_day_partitions_when_paths_exceed_the_guard() {
        use crate::bucket::Bucket;

        let root = std::env::temp_dir().join(format!(
            "synty-athena-id-partition-fallback-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let bucket = crate::bucket::LocalFs::new(&root);
        for index in 0..=MAX_OBJECT_PATHS {
            bucket
                .put(
                    &format!(
                        "events/edge-m-codex/chunks/track.2026-07-22/{index:04}.jsonl"
                    ),
                    b"{}\n",
                )
                .unwrap();
        }
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut backend = Backend {
            config: Config {
                bucket: root.to_string_lossy().into_owned(),
                workgroup: "wg".into(),
                database: "synty".into(),
                table: "raw_events".into(),
            },
            query: Box::new(FakeQuery {
                lines: Vec::new(),
                calls: Arc::clone(&calls),
            }),
            streams: Some(vec!["edge-m-codex".into()]),
            days: None,
            cached: None,
        };

        backend
            .select(
                &["edge-m-codex".into()],
                Window {
                    since: parse_time("2026-07-22T10:00:00Z").unwrap(),
                    until: parse_time("2026-07-22T10:10:00Z").unwrap(),
                },
                &Predicate {
                    ids: vec!["event-id".into()],
                    ..Default::default()
                },
                Instant::now() + StdDuration::from_secs(5),
            )
            .unwrap();

        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].contains("day IN ('2026-07-22')"));
        assert!(!calls[0].contains("\"$path\""));
        assert!(calls[0].contains("'event-id'"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn raw_athena_rows_reconstruct_the_existing_span_surface() {
        let started = Utc::now() - Duration::minutes(2);
        let called = started + Duration::seconds(1);
        let completed = started + Duration::seconds(3);
        let started = started.to_rfc3339();
        let called = called.to_rfc3339();
        let completed = completed.to_rfc3339();
        let lines = vec![
            event(
                "start",
                &started,
                "session_start",
                json!({"cwd":"/work/synty"}),
            ),
            event(
                "call-1",
                &called,
                "tool_call",
                json!({"name":"exec_command","call_id":"c1","arguments":"{\"cmd\":\"cargo test\"}"}),
            ),
            event(
                "result-1",
                &completed,
                "tool_result",
                json!({"call_id":"c1","output":"Process exited with code 0"}),
            ),
        ];
        let calls = Arc::new(Mutex::new(Vec::new()));
        let config = Config::new(
            "s3://bucket".into(),
            "wg".into(),
            "synty".into(),
            "raw_events".into(),
        )
        .unwrap();
        let mut backend = Backend {
            config,
            query: Box::new(FakeQuery {
                lines,
                calls: Arc::clone(&calls),
            }),
            streams: Some(vec!["edge-m-codex".into()]),
            days: Some(vec!["2026-07-22".into()]),
            cached: None,
        };
        let out = backend
            .list(
                "spans",
                None,
                None,
                None,
                None,
                None,
                false,
                Some(&started),
                None,
                "recent",
                20,
                &ReadScope::default(),
            )
            .unwrap();
        assert!(out.contains("exec_command"));
        assert!(out.contains("call-1"));
        assert!(
            backend.cached.is_none(),
            "a list slice must not satisfy a later show or compare lookup"
        );
        assert!(
            calls
                .lock()
                .unwrap()
                .iter()
                .all(|sql| sql.starts_with("SELECT "))
        );
    }

    #[test]
    fn literal_search_reads_matches_and_metadata_without_expanding_whole_sessions() {
        let lines = vec![
            event(
                "start",
                "2026-07-22T10:00:00Z",
                "session_start",
                json!({"cwd":"/work/synty"}),
            ),
            event(
                "result-1",
                "2026-07-22T10:00:03Z",
                "tool_result",
                json!({"call_id":"c1","output":"missing libxcb.so.1"}),
            ),
        ];
        let calls = Arc::new(Mutex::new(Vec::new()));
        let config = Config::new(
            "s3://bucket".into(),
            "wg".into(),
            "synty".into(),
            "raw_events".into(),
        )
        .unwrap();
        let mut backend = Backend {
            config,
            query: Box::new(FakeQuery {
                lines,
                calls: Arc::clone(&calls),
            }),
            streams: Some(vec!["edge-m-codex".into()]),
            days: Some(vec!["2026-07-22".into()]),
            cached: None,
        };

        let out = backend
            .search(
                "libxcb.so.1",
                None,
                None,
                None,
                Some("tool_result"),
                20,
                &ReadScope::default(),
            )
            .unwrap();

        assert!(out.contains("missing libxcb.so.1"));
        assert!(
            backend.cached.is_none(),
            "partial search stores must not satisfy a later trace_show"
        );
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 2, "match query plus metadata query");
        assert!(calls[0].contains("strpos(lower(line), 'libxcb.so.1') > 0"));
        assert!(calls[0].contains("$.kind"));
        assert!(calls[0].contains("'tool_result'"));
        assert!(calls[1].contains("'session_start', 'agent_meta'"));
        assert!(calls[1].contains("row_number() OVER"));
    }

    #[test]
    fn show_expands_the_resolved_session_in_one_follow_up_query() {
        let lines = vec![
            event(
                "start",
                "2026-07-22T10:00:00Z",
                "session_start",
                json!({"cwd":"/work/synty"}),
            ),
            event(
                "call-1",
                "2026-07-22T10:00:01Z",
                "tool_call",
                json!({"name":"exec_command","call_id":"c1","arguments":"{\"cmd\":\"cargo test\"}"}),
            ),
        ];
        let calls = Arc::new(Mutex::new(Vec::new()));
        let config = Config::new(
            "s3://bucket".into(),
            "wg".into(),
            "synty".into(),
            "raw_events".into(),
        )
        .unwrap();
        let mut backend = Backend {
            config,
            query: Box::new(FakeQuery {
                lines,
                calls: Arc::clone(&calls),
            }),
            streams: Some(vec!["edge-m-codex".into()]),
            days: Some(vec!["2026-07-22".into()]),
            cached: None,
        };

        let out = backend.show("call-1", 3, 5, &ReadScope::default()).unwrap();

        assert!(out.contains("cargo test"));
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 2, "id lookup plus one full-session query");
        assert!(calls[1].contains("$.session_id"));
        assert!(!calls[1].contains("synty_context_rank"));
    }

    #[test]
    fn show_uses_ulid_time_and_matching_stream_for_bounded_lookup() {
        let timestamp = parse_time("2026-07-22T10:00:01Z").unwrap();
        let id = crate::event::deterministic_ulid(timestamp.timestamp_millis() as u64, "call");
        let lines = vec![
            event(
                "start",
                "2026-07-22T10:00:00Z",
                "session_start",
                json!({"cwd":"/work/synty"}),
            ),
            event(
                &id,
                "2026-07-22T10:00:01Z",
                "tool_call",
                json!({"name":"exec_command","call_id":"c1","arguments":"{\"cmd\":\"cargo test\"}"}),
            ),
        ];
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut backend = Backend {
            config: Config::new(
                "s3://bucket".into(),
                "wg".into(),
                "synty".into(),
                "raw_events".into(),
            )
            .unwrap(),
            query: Box::new(FakeQuery {
                lines,
                calls: Arc::clone(&calls),
            }),
            streams: Some(vec![
                "edge-m-codex".into(),
                "edge-other-claudecode".into(),
            ]),
            days: Some(vec!["2026-07-22".into()]),
            cached: None,
        };

        let out = backend
            .show(&id, 3, 5, &ReadScope::default())
            .unwrap();

        assert!(out.contains("cargo test"));
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert!(calls[0].contains("2026-07-22T09:55:01+00:00"));
        assert!(calls[0].contains("2026-07-22T10:05:01+00:00"));
        assert!(calls[0].contains("'edge-other-claudecode'"));
        assert!(calls[1].contains("stream IN ('edge-m-codex')"));
        assert!(!calls[1].contains("'edge-other-claudecode'"));
    }

    #[test]
    fn compare_window_covers_both_ulid_timestamps() {
        let early = parse_time("2026-07-22T10:00:00Z").unwrap();
        let late = parse_time("2026-07-22T10:30:00Z").unwrap();
        let left = crate::event::deterministic_ulid(early.timestamp_millis() as u64, "left");
        let right = crate::event::deterministic_ulid(late.timestamp_millis() as u64, "right");

        let window = ids_lookup_window(&[&left, &format!("job:{right}")]).unwrap();

        assert_eq!(window.since, parse_time("2026-07-22T09:55:00Z").unwrap());
        assert_eq!(window.until, parse_time("2026-07-22T10:35:00Z").unwrap());
        assert!(ids_lookup_window(&[&left, "foreign-id"]).is_none());
    }

    #[test]
    fn listed_job_ids_query_the_native_span_id() {
        let lines = vec![
            event(
                "call-1",
                "2026-07-22T10:00:01Z",
                "tool_call",
                json!({"name":"exec_command","call_id":"c1","arguments":"{\"cmd\":\"cargo test\"}"}),
            ),
            event(
                "result-1",
                "2026-07-22T10:00:03Z",
                "tool_result",
                json!({"call_id":"c1","output":"Process running with session ID 42"}),
            ),
        ];
        let calls = Arc::new(Mutex::new(Vec::new()));
        let config = Config::new(
            "s3://bucket".into(),
            "wg".into(),
            "synty".into(),
            "raw_events".into(),
        )
        .unwrap();
        let mut backend = Backend {
            config,
            query: Box::new(FakeQuery {
                lines,
                calls: Arc::clone(&calls),
            }),
            streams: Some(vec!["edge-m-codex".into()]),
            days: Some(vec!["2026-07-22".into()]),
            cached: None,
        };

        let out = backend
            .show("job:call-1", 3, 5, &ReadScope::default())
            .unwrap();

        assert!(out.contains("cargo test"), "{out}");
        let calls = calls.lock().unwrap();
        assert!(calls[0].contains("'call-1'"));
        assert!(!calls[0].contains("'job:call-1'"));
    }

    #[test]
    fn stream_pruning_uses_exact_machine_and_canonical_source_suffixes() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let config = Config::new(
            "s3://bucket".into(),
            "wg".into(),
            "synty".into(),
            "raw_events".into(),
        )
        .unwrap();
        let mut backend = Backend {
            config,
            query: Box::new(FakeQuery {
                lines: Vec::new(),
                calls,
            }),
            streams: Some(vec![
                "edge-dev-box-codex".into(),
                "edge-dev-box-claudecode".into(),
                "edge-dev-box-2-codex".into(),
            ]),
            days: Some(vec!["2026-07-22".into()]),
            cached: None,
        };
        let scope = ReadScope {
            sources: vec!["codex_cli".into()],
            ..Default::default()
        };

        assert_eq!(
            backend
                .selected_streams(Some("dev-box"), Some("codex_cli"), &scope)
                .unwrap(),
            vec!["edge-dev-box-codex"]
        );
        assert!(
            backend
                .selected_streams(None, Some("unknown-agent"), &ReadScope::default())
                .is_err(),
            "an explicit unknown source must fail closed"
        );
        assert!(
            backend
                .selected_streams(
                    None,
                    None,
                    &ReadScope {
                        sources: vec!["unknown-agent".into()],
                        ..Default::default()
                    },
                )
                .is_err(),
            "an all-unknown source scope must fail closed"
        );
    }

    #[test]
    fn partition_ranges_select_delayed_and_unknown_physical_days() {
        use crate::bucket::Bucket;

        let root =
            std::env::temp_dir().join(format!("synty-athena-partitions-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let bucket = crate::bucket::LocalFs::new(&root);
        for day in ["2026-07-19", "2026-07-20", "2026-07-24"] {
            bucket
                .put(
                    &format!("events/edge-m-codex/chunks/track.{day}/one.jsonl"),
                    b"{}\n",
                )
                .unwrap();
        }
        bucket
            .put(
                "events/edge-m-codex/chunks/track.2026-07-20/old.jsonl",
                b"{}\n",
            )
            .unwrap();
        bucket
            .put(
                "event-partitions/edge-m-codex.json",
                serde_json::to_string(&json!({
                    "format": 1,
                    "stream": "edge-m-codex",
                    "legacy_days": ["2026-07-19", "2026-07-24"],
                    "partitions": {
                        "2026-07-20": {
                            "min_ts": "2026-07-22T10:00:00Z",
                            "max_ts": "2026-07-22T11:00:00Z"
                        }
                    },
                    "updated_at": "2026-07-24T12:00:00Z"
                }))
                .unwrap()
                .as_bytes(),
            )
            .unwrap();
        bucket
            .put(
                "event-partitions/edge-m-codex/track.2026-07-20.json",
                serde_json::to_string(&json!({
                    "format": 1,
                    "stream": "edge-m-codex",
                    "day": "2026-07-20",
                    "objects": {
                        "events/edge-m-codex/chunks/track.2026-07-20/one.jsonl": {
                            "min_ts": "2026-07-22T10:00:00Z",
                            "max_ts": "2026-07-22T11:00:00Z"
                        },
                        "events/edge-m-codex/chunks/track.2026-07-20/old.jsonl": {
                            "min_ts": "2026-06-21T10:00:00Z",
                            "max_ts": "2026-06-21T11:00:00Z"
                        }
                    }
                }))
                .unwrap()
                .as_bytes(),
            )
            .unwrap();
        let backend = Backend {
            config: Config {
                bucket: root.to_string_lossy().into_owned(),
                workgroup: "wg".into(),
                database: "synty".into(),
                table: "raw_events".into(),
            },
            query: Box::new(FakeQuery {
                lines: Vec::new(),
                calls: Arc::new(Mutex::new(Vec::new())),
            }),
            streams: Some(vec!["edge-m-codex".into()]),
            days: None,
            cached: None,
        };
        let selection = backend
            .selected_objects(
                &["edge-m-codex".into()],
                Window {
                    since: parse_time("2026-07-22T10:30:00Z").unwrap(),
                    until: parse_time("2026-07-22T10:45:00Z").unwrap(),
                },
            )
            .unwrap();

        assert_eq!(
            selection.days,
            ["2026-07-19", "2026-07-20", "2026-07-24"]
        );
        assert_eq!(selection.paths.len(), 3);
        assert!(!selection.paths.iter().any(|path| path.ends_with("/old.jsonl")));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn query_metrics_classify_timeouts_and_limits() {
        let timeout = Err::<(), _>(timeout_error("request timed out"))
            .context("Athena request")
            .unwrap_err();
        assert_eq!(
            query_outcome(&timeout),
            "timeout"
        );
        assert_eq!(
            query_outcome(&limit_error("selection exceeds 50000 events")),
            "limit"
        );
        assert_eq!(
            query_outcome(&anyhow::anyhow!("AWS rejected the request")),
            "error"
        );
    }

    #[test]
    fn legacy_streams_limit_object_paths_to_the_requested_physical_days() {
        use crate::bucket::Bucket;

        let root = std::env::temp_dir().join(format!(
            "synty-athena-legacy-window-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let bucket = crate::bucket::LocalFs::new(&root);
        for day in ["2026-07-20", "2026-07-22", "2026-07-23"] {
            bucket
                .put(
                    &format!("events/edge-m-codex/chunks/track.{day}/one.jsonl"),
                    b"{}\n",
                )
                .unwrap();
        }
        let backend = Backend {
            config: Config {
                bucket: root.to_string_lossy().into_owned(),
                workgroup: "wg".into(),
                database: "synty".into(),
                table: "raw_events".into(),
            },
            query: Box::new(FakeQuery {
                lines: Vec::new(),
                calls: Arc::new(Mutex::new(Vec::new())),
            }),
            streams: Some(vec!["edge-m-codex".into()]),
            days: None,
            cached: None,
        };

        let selection = backend
            .selected_objects(
                &["edge-m-codex".into()],
                Window {
                    since: parse_time("2026-07-22T10:00:00Z").unwrap(),
                    until: parse_time("2026-07-23T00:00:00Z").unwrap(),
                },
            )
            .unwrap();

        assert_eq!(selection.days, ["2026-07-22"]);
        assert_eq!(selection.paths.len(), 1);
        assert!(selection.paths[0].contains("track.2026-07-22"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn windows_fail_closed_before_scanning_more_than_seven_days() {
        let error = Window::parse(
            Some("2026-07-01T00:00:00Z"),
            Some("2026-07-09T00:00:00Z"),
            DEFAULT_LIST_HOURS,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("168 hours"));
    }

    #[test]
    fn malformed_rows_do_not_abort_session_discovery() {
        let sessions = event_sessions(&[
            "{not-json".into(),
            event("valid", "2026-07-22T10:00:00Z", "user_prompt", json!({})),
        ]);
        assert_eq!(sessions, BTreeSet::from(["session-1".into()]));
    }
}
