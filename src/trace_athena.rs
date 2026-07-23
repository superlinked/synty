// Read-only remote trace access over Athena. SQL prunes the immutable event
// lake by injected stream partitions and a bounded time window; the existing
// trace fold then reconstructs turns, spans, and jobs from only those rows.

use crate::{bucket, metrics, policy::ReadScope, trace};
use anyhow::{Context, Result, bail};
use aws_config::timeout::TimeoutConfig;
use aws_sdk_athena::types::{QueryExecutionContext, QueryExecutionState};
use chrono::{DateTime, Duration, NaiveDate, Utc};
use std::collections::BTreeSet;
use std::time::{Duration as StdDuration, Instant};

const DEFAULT_LIST_HOURS: i64 = 1;
const DEFAULT_LOOKUP_HOURS: i64 = 24 * 7;
const MAX_LOOKBACK_HOURS: i64 = 24 * 7;
const MAX_EVENTS: usize = 50_000;
const MAX_RESULT_BYTES: usize = 64 * 1024 * 1024;
const MAX_SESSIONS: usize = 200;
const QUERY_TIMEOUT: StdDuration = StdDuration::from_secs(50);
const AWS_CONNECT_TIMEOUT: StdDuration = StdDuration::from_secs(3);
const AWS_OPERATION_TIMEOUT: StdDuration = StdDuration::from_secs(10);

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

trait EventQuery: Send {
    fn run(&mut self, sql: &str) -> Result<QueryRows>;
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
}

impl EventQuery for AwsEventQuery {
    fn run(&mut self, sql: &str) -> Result<QueryRows> {
        anyhow::ensure!(
            sql.trim_start().to_ascii_uppercase().starts_with("SELECT "),
            "Athena trace only permits SELECT statements"
        );
        let started = Instant::now();
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

        let scanned_bytes = loop {
            if started.elapsed() >= QUERY_TIMEOUT {
                let _ = self.runtime.block_on(
                    self.client
                        .stop_query_execution()
                        .query_execution_id(&id)
                        .send(),
                );
                bail!(
                    "Athena trace query timed out after {} seconds",
                    QUERY_TIMEOUT.as_secs()
                );
            }
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
                Some(QueryExecutionState::Succeeded) => {
                    break query
                        .statistics()
                        .and_then(|statistics| statistics.data_scanned_in_bytes())
                        .unwrap_or(0);
                }
                Some(QueryExecutionState::Failed | QueryExecutionState::Cancelled) => {
                    bail!(
                        "Athena trace query {}: {}",
                        status
                            .state()
                            .map(|state| state.as_str())
                            .unwrap_or("failed"),
                        status.state_change_reason().unwrap_or("no reason returned")
                    );
                }
                _ => std::thread::sleep(StdDuration::from_millis(250)),
            }
        };

        let mut lines = Vec::new();
        let mut bytes = 0usize;
        let mut next_token = None;
        let mut first_page = true;
        loop {
            anyhow::ensure!(
                started.elapsed() < QUERY_TIMEOUT,
                "Athena trace result retrieval timed out after {} seconds",
                QUERY_TIMEOUT.as_secs()
            );
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
                    let Some(line) = row.data().first().and_then(|datum| datum.var_char_value())
                    else {
                        continue;
                    };
                    bytes = bytes.saturating_add(line.len());
                    anyhow::ensure!(
                        bytes <= MAX_RESULT_BYTES,
                        "Athena trace selection exceeds {} MiB; narrow the time, machine, source, or operation filter",
                        MAX_RESULT_BYTES / 1024 / 1024
                    );
                    lines.push(line.to_string());
                    anyhow::ensure!(
                        lines.len() <= MAX_EVENTS,
                        "Athena trace selection exceeds {MAX_EVENTS} events; narrow the time, machine, source, or operation filter"
                    );
                }
            }
            first_page = false;
            next_token = page.next_token().map(str::to_string);
            if next_token.is_none() {
                break;
            }
        }
        metrics::Run::new("athena_trace")
            .set("rows", lines.len())
            .set("result_bytes", bytes)
            .set("scanned_bytes", scanned_bytes)
            .set("elapsed_ms", started.elapsed().as_millis() as u64)
            .emit();
        Ok(QueryRows { lines })
    }
}

pub(crate) struct Backend {
    config: Config,
    query: Box<dyn EventQuery>,
    // Tests inject a fixed registry; production re-lists the bounded registry
    // for every tool call so newly enrolled streams appear without a restart.
    streams: Option<Vec<String>>,
    cached: Option<trace::TraceStore>,
}

impl Backend {
    pub(crate) fn new(config: Config) -> Result<Self> {
        let query = Box::new(AwsEventQuery::new(&config)?);
        Ok(Self {
            config,
            query,
            streams: None,
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
        let window = Window::parse(None, None, DEFAULT_LOOKUP_HOURS)?;
        let store = self.load_store(window, None, None, None, None, &[id], true, scope)?;
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
        let window = Window::parse(None, None, DEFAULT_LOOKUP_HOURS)?;
        let store = self.load_store(window, None, None, None, None, &[left, right], true, scope)?;
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
    ) -> Result<trace::TraceStore> {
        let streams = self.selected_streams(machine)?;
        let predicate = Predicate {
            source: source.map(str::to_string),
            scope_sources: scope.sources.clone(),
            needle: needle.map(str::to_string),
            kind_contains: kind.map(str::to_string),
            ids: ids.iter().map(|id| (*id).to_string()).collect(),
            ..Default::default()
        };
        let first = self.select(&streams, window, &predicate)?;
        let sessions = event_sessions(&first.lines)?;
        anyhow::ensure!(
            sessions.len() <= MAX_SESSIONS,
            "Athena trace selection spans more than {MAX_SESSIONS} sessions; narrow the time, machine, source, or operation filter"
        );
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
                &streams,
                context_window,
                &Predicate {
                    sessions: sessions.iter().cloned().collect(),
                    scope_sources: scope.sources.clone(),
                    ..Default::default()
                },
            )?
            .lines
        } else {
            first.lines
        };
        if !sessions.is_empty() && !expands_sessions {
            let mut contexts = self
                .select(
                    &streams,
                    context_window,
                    &Predicate {
                        sessions: sessions.into_iter().collect(),
                        kinds: vec!["session_start".into(), "agent_meta".into()],
                        scope_sources: scope.sources.clone(),
                        dedupe_session_kinds: true,
                        ..Default::default()
                    },
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
    ) -> Result<QueryRows> {
        let sql = select_sql(&self.config, streams, window, predicate, MAX_EVENTS + 1)?;
        self.query.run(&sql)
    }

    fn selected_streams(&mut self, machine: Option<&str>) -> Result<Vec<String>> {
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
        if let Some(machine) = machine {
            let machine = machine.to_ascii_lowercase();
            streams.retain(|stream| stream.to_ascii_lowercase().contains(&machine));
        }
        anyhow::ensure!(
            !streams.is_empty(),
            "no event streams match the machine filter"
        );
        Ok(streams)
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
        anyhow::ensure!(
            until - since <= Duration::hours(MAX_LOOKBACK_HOURS),
            "Athena trace windows are limited to {MAX_LOOKBACK_HOURS} hours"
        );
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
    window: Window,
    predicate: &Predicate,
    limit: usize,
) -> Result<String> {
    anyhow::ensure!(
        !streams.is_empty(),
        "Athena trace needs at least one stream"
    );
    let stream_values = streams
        .iter()
        .map(|stream| sql_string(stream))
        .collect::<Vec<_>>()
        .join(", ");
    let mut clauses = vec![
        format!("stream IN ({stream_values})"),
        format!(
            "day BETWEEN {} AND {}",
            sql_string(&window.since.format("%Y-%m-%d").to_string()),
            sql_string(&window.until.format("%Y-%m-%d").to_string())
        ),
        format!(
            "from_iso8601_timestamp(json_extract_scalar(line, '$.ts')) >= from_iso8601_timestamp({})",
            sql_string(&window.since.to_rfc3339())
        ),
        format!(
            "from_iso8601_timestamp(json_extract_scalar(line, '$.ts')) < from_iso8601_timestamp({})",
            sql_string(&window.until.to_rfc3339())
        ),
    ];
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

fn event_sessions(lines: &[String]) -> Result<BTreeSet<String>> {
    let mut sessions = BTreeSet::new();
    for line in lines {
        let event: crate::event::Event =
            serde_json::from_str(line).context("parse Athena event row")?;
        if !event.session_id.is_empty() {
            sessions.insert(event.session_id);
        }
    }
    Ok(sessions)
}

fn sql_string(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
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
        fn run(&mut self, sql: &str) -> Result<QueryRows> {
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
        assert!(sql.contains("day BETWEEN '2026-07-22' AND '2026-07-22'"));
        for mutating in ["INSERT", "UPDATE", "DELETE", "CREATE", "UNLOAD", "CTAS"] {
            assert!(!sql.to_ascii_uppercase().contains(mutating), "{mutating}");
        }
    }

    #[test]
    fn raw_athena_rows_reconstruct_the_existing_span_surface() {
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
            event(
                "result-1",
                "2026-07-22T10:00:03Z",
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
                Some("2026-07-22T10:00:00Z"),
                None,
                "recent",
                20,
                &ReadScope::default(),
            )
            .unwrap();
        assert!(out.contains("exec_command"));
        assert!(out.contains("call-1"));
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
}
