// `synty track` — the native tracker. Discovers each agent's session files,
// parses new bytes into canonical envelopes, and appends them to per-stream
// JSONL under `--out` (default corpus/local, where `ingest` reads). Per-file
// byte offsets persist in a cursor file, so each pass only parses what's new.
//
// One-shot by default; `--watch` polls forever, holding parser state in memory
// and synthesizing session_end after a session goes idle. `--install` writes a
// launchd/systemd unit that runs the watcher at login.

use crate::claudecode::ClaudeCode;
use crate::codex::Codex;
use crate::cowork::Cowork;
use crate::event::{kind, Event, Sequencer};
use crate::tail::{drive, ms_to_rfc3339, EmitCtx, Source};
use anyhow::{anyhow, bail, Result};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use walkdir::WalkDir;

const HEAD_BYTES: usize = 64 << 10;
const IDLE_MS: i64 = 10 * 60 * 1000;

pub struct Opts {
    pub which: String,
    pub out: String,
    pub max_age_days: u64,
    pub capture_since_ms: Option<i64>,
    pub machine: String,
    pub watch: bool,
    pub poll_secs: u64,
    pub upload_interval_secs: u64,
    pub campaign: Option<String>,
    pub role: Option<String>,
    pub install: Option<String>,
    pub cursors: String,
    /// If set, push drained event chunks to this bucket under events/ (the
    /// shared backplane for a fleet of trackers).
    pub bucket: Option<String>,
}

pub fn run(o: Opts) -> Result<()> {
    if let Some(kind) = &o.install {
        return install(kind, &o);
    }
    let mut t = Tracker::new(&o)?;
    if o.watch {
        t.watch(o.poll_secs.max(1))
    } else {
        let n = t.drain()?;
        t.flush_ends(unix_ms(SystemTime::now()), true)?;
        t.save_cursors()?;
        let pushed = t.push(true)?;
        let mut m = crate::metrics::Run::new("track");
        m.set("events", n)
            .set("sessions", t.session_count())
            .set("lines_skipped", t.skipped_count());
        m.emit();
        eprintln!(
            "track: {n} events ({} sessions) → {}",
            t.session_count(),
            o.out
        );
        if pushed > 0 {
            eprintln!(
                "track: pushed {pushed} event chunks → {}/events/",
                t.bucket.as_deref().unwrap_or("")
            );
        }
        Ok(())
    }
}

struct FileState {
    offset: i64,
    parser: Box<dyn crate::tail::FileParser>,
}

struct Stream {
    src: Box<dyn Source>,
    roots: Vec<String>,
    name: String,
    out: PathBuf,
    seq: Sequencer,
    started: HashSet<String>,
    files: HashMap<PathBuf, FileState>,
    /// last (ts_ms, ts) seen per still-open session, for idle session_end.
    open: HashMap<String, (i64, String)>,
    n_sessions: usize,
    /// Malformed lines the parser rejected — format drift made visible.
    n_skipped: usize,
    /// This machine's resolved actor, stamped into session_start payloads so a
    /// build on another machine attributes the session to its author, not to
    /// whoever runs the build.
    actor: String,
    campaign: String,
    campaign_role: String,
}

struct Tracker {
    streams: Vec<Stream>,
    cutoff_ms: i64,
    cursors_path: PathBuf,
    cursors: HashMap<String, i64>,
    out: String,
    bucket: Option<String>,
    upload_interval: Duration,
    last_push: Option<std::time::Instant>,
}

impl Tracker {
    fn new(o: &Opts) -> Result<Self> {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let cursors_path = PathBuf::from(&o.cursors);
        let cursors: HashMap<String, i64> = std::fs::read_to_string(&cursors_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        // Started session ids persist next to the cursors: a restart resumes
        // mid-file, and without this every resumed session would re-emit a
        // session_start (under a fresh deterministic id, so never deduped).
        let started_path = started_path(&cursors_path);
        let mut started_by_src: HashMap<String, HashSet<String>> = std::fs::read_to_string(&started_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        // "local" (the default) auto-derives a stable per-machine id; an explicit
        // --machine is taken as-is. Keeps fleet streams from colliding.
        let machine = crate::identity::resolve_machine(&o.machine);
        let actor = crate::identity::actor();
        let mut streams = Vec::new();
        for src in sources(&o.which)? {
            let name = format!("edge-{}-{}", machine, src.id());
            // The stream is a DIR of per-day files (track.<YYYY-MM-DD>.jsonl).
            // Sync turns only the newly appended complete lines into immutable
            // bucket chunks; the local daily file is never uploaded wholesale.
            let out = Path::new(&o.out).join(&name);
            std::fs::create_dir_all(&out)?;
            // Seed offsets from saved cursors so a restart resumes mid-file.
            let mut files = HashMap::new();
            let roots = default_roots(src.id(), &home);
            let started = started_by_src.remove(src.id()).unwrap_or_default();
            streams.push(Stream {
                roots,
                name,
                out,
                seq: Sequencer::new(),
                started,
                files: std::mem::take(&mut files),
                open: HashMap::new(),
                n_sessions: 0,
                n_skipped: 0,
                actor: actor.clone(),
                campaign: o.campaign.clone().unwrap_or_default(),
                campaign_role: o.role.clone().unwrap_or_default(),
                src,
            });
        }
        let cutoff_ms = o.capture_since_ms.unwrap_or_else(|| {
            if o.max_age_days == 0 {
                0
            } else {
                unix_ms(SystemTime::now()) - (o.max_age_days as i64) * 86_400_000
            }
        });
        Ok(Self {
            streams,
            cutoff_ms,
            cursors_path,
            cursors,
            out: o.out.clone(),
            bucket: o.bucket.clone(),
            upload_interval: Duration::from_secs(o.upload_interval_secs.max(1)),
            last_push: None,
        })
    }

    /// Push drained event chunks to the shared bucket, if configured.
    fn push(&mut self, force: bool) -> Result<usize> {
        if !force
            && self
                .last_push
                .is_some_and(|t| t.elapsed() < self.upload_interval)
        {
            return Ok(0);
        }
        self.last_push = Some(std::time::Instant::now());
        match &self.bucket {
            Some(b) => {
                let owned = self.streams.iter().map(|stream| stream.name.clone()).collect();
                crate::sync::push_events_for_streams(
                    b,
                    &self.out,
                    ".synty/uploads.json",
                    (self.cutoff_ms > 0).then_some(self.cutoff_ms),
                    &owned,
                )
            }
            None => Ok(0),
        }
    }

    /// One pass over every stream's files; returns the number of events emitted.
    fn drain(&mut self) -> Result<usize> {
        let cutoff = self.cutoff_ms;
        let mut total = 0;
        for st in &mut self.streams {
            total += st.drain(cutoff, &self.cursors)?;
        }
        Ok(total)
    }

    fn watch(&mut self, poll_secs: u64) -> Result<()> {
        eprintln!(
            "track --watch: {} stream(s), poll {poll_secs}s, idle session_end {}m. Ctrl-C to stop.",
            self.streams.len(),
            IDLE_MS / 60000
        );
        // GitHub runs on a slow sub-cadence of the session poll: the tracker is
        // model-free, so refreshing the org's PRs/issues here means freshness
        // no longer waits on someone opening a viewer. `refresh_github` itself
        // is throttled + token-gated, so untokened machines just pull.
        let gh_every = Duration::from_secs(crate::up::GITHUB_STALE_MIN.max(1) as u64 * 60);
        let mut gh_last: Option<std::time::Instant> = None;
        loop {
            let n = self.drain()?;
            let now = unix_ms(SystemTime::now());
            let ended = self.flush_ends(now, false)?;
            self.save_cursors()?;
            let pushed = self.push(false)?;
            if n > 0 || ended > 0 || pushed > 0 {
                let skipped = self.skipped_count();
                if skipped > 0 {
                    eprintln!(
                        "track: +{n} events, {ended} ended, {pushed} chunks pushed, {skipped} malformed lines skipped (total)"
                    );
                } else {
                    eprintln!("track: +{n} events, {ended} ended, {pushed} chunks pushed");
                }
            }
            if let Some(bucket) = self.bucket.clone() {
                if github_due(gh_last.map(|t| t.elapsed()), gh_every) {
                    crate::up::refresh_github(&bucket);
                    gh_last = Some(std::time::Instant::now());
                }
            }
            std::thread::sleep(Duration::from_secs(poll_secs));
        }
    }

    /// Emit session_end: `all` ends every open session (one-shot shutdown);
    /// otherwise only those idle past IDLE_MS.
    fn flush_ends(&mut self, now_ms: i64, all: bool) -> Result<usize> {
        let mut ended = 0;
        for st in &mut self.streams {
            ended += st.flush_ends(now_ms, all)?;
        }
        Ok(ended)
    }

    fn save_cursors(&mut self) -> Result<()> {
        for st in &self.streams {
            for (path, fs) in &st.files {
                self.cursors.insert(format!("{}\0{}", st.src.id(), path.display()), fs.offset);
            }
        }
        if let Some(p) = self.cursors_path.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::write(&self.cursors_path, serde_json::to_string(&self.cursors)?)?;
        let started: HashMap<&str, &HashSet<String>> =
            self.streams.iter().map(|st| (st.src.id(), &st.started)).collect();
        std::fs::write(started_path(&self.cursors_path), serde_json::to_string(&started)?)?;
        Ok(())
    }

    fn session_count(&self) -> usize {
        self.streams.iter().map(|s| s.n_sessions).sum()
    }

    fn skipped_count(&self) -> usize {
        self.streams.iter().map(|s| s.n_skipped).sum()
    }
}

impl Stream {
    fn drain(&mut self, cutoff_ms: i64, cursors: &HashMap<String, i64>) -> Result<usize> {
        let mut events = Vec::new();
        for path in discover(&self.roots, cutoff_ms) {
            let Ok(content) = std::fs::read(&path) else { continue };

            // First sight: detect version, build a parser, seed the offset from
            // a saved cursor so a restart resumes rather than re-emits.
            if !self.files.contains_key(&path) {
                let head = &content[..content.len().min(HEAD_BYTES)];
                let version = self.src.detect_version(head);
                let Some(parser) = self.src.new_parser(&version, head) else { continue };
                let offset = cursors.get(&format!("{}\0{}", self.src.id(), path.display())).copied().unwrap_or(0);
                self.files.insert(path.clone(), FileState { offset, parser });
            }
            let fs = self.files.get_mut(&path).unwrap();
            // Only complete lines (up to the last newline past the offset).
            if content.len() as i64 <= fs.offset {
                continue;
            }
            let slice = &content[fs.offset as usize..];
            let Some(last_nl) = slice.iter().rposition(|&b| b == b'\n') else {
                continue;
            };
            let complete = &slice[..=last_nl];
            let fallback_ms = file_mtime_ms(&path);

            let path_str = path.to_string_lossy();
            let mut ec = EmitCtx::new(
                self.name.clone(),
                &*self.src,
                &mut self.seq,
                &mut self.started,
            );
            let (mut evts, consumed, skipped) = drive(
                &mut *fs.parser,
                complete,
                &path_str,
                fs.offset,
                fallback_ms,
                &mut ec,
            );
            drop(ec);
            fs.offset += consumed;
            self.n_skipped += skipped;

            for e in &mut evts {
                if !self.campaign.is_empty() {
                    e.rollup_dim = self.campaign.clone();
                }
                if e.kind == kind::SESSION_START {
                    e.payload["actor"] = json!(self.actor);
                    // Which synty produced this stream — distinct from the
                    // agent's own `version` the parser may have captured.
                    // Lets the fleet roster report upgrade lag per machine.
                    e.payload["tracker_version"] = json!(env!("CARGO_PKG_VERSION"));
                    if !self.campaign.is_empty() {
                        e.payload["campaign_id"] = json!(self.campaign);
                    }
                    if !self.campaign_role.is_empty() {
                        e.payload["campaign_role"] = json!(self.campaign_role);
                    }
                    e.payload["backend"] = json!(self.src.envelope_source());
                }
            }
            if cutoff_ms > 0 {
                evts.retain(|e| {
                    event_time_ms(e).is_none_or(|t| t >= cutoff_ms)
                        // Stage session metadata locally so the upload ledger
                        // can attach it if this session later crosses the
                        // boundary. The upload gate never publishes it alone.
                        || e.kind == kind::SESSION_START
                });
            }
            self.n_sessions += evts
                .iter()
                .filter(|e| e.kind == kind::SESSION_START)
                .count();
            for e in &evts {
                if !e.session_id.is_empty() && e.kind != kind::SESSION_END {
                    self.open
                        .insert(e.session_id.clone(), (fallback_ms, e.ts.clone()));
                }
            }
            events.extend(evts);
        }
        let n = events.len();
        self.append(&events)?;
        Ok(n)
    }

    fn flush_ends(&mut self, now_ms: i64, all: bool) -> Result<usize> {
        let due: Vec<String> = self
            .open
            .iter()
            .filter(|(_, (ms, _))| all || now_ms - *ms > IDLE_MS)
            .map(|(sid, _)| sid.clone())
            .collect();
        if due.is_empty() {
            return Ok(0);
        }
        // session_end is synthesized (no source line), so its id is minted from
        // (source, session, last event ts) — deterministic, unlike the file
        // mtime/batch order it used before, so a re-emission by an overlapping
        // tracker dedups on read like every other event. Its ts is the last
        // activity, which is when the session effectively ended anyway.
        let mut events = Vec::new();
        for sid in &due {
            let (_, last_ts) = self.open.remove(sid).unwrap();
            let (ts_ms, ts) = crate::tail::resolve_ts(&last_ts, 0);
            let key = format!("{}\0session_end\0{sid}\0{ts}", self.src.id());
            events.push(Event {
                v: crate::event::ENVELOPE_V,
                event_id: crate::event::deterministic_ulid(ts_ms.max(0) as u64, &key),
                stream: self.name.clone(),
                seq: self.seq.next(&self.name),
                ts,
                source: self.src.envelope_source().to_string(),
                session_id: sid.clone(),
                kind: kind::SESSION_END.to_string(),
                payload: json!({"reason": if all {"shutdown"} else {"idle"}}),
                rollup_dim: self.campaign.clone(),
            });
        }
        self.append(&events)?;
        Ok(due.len())
    }

    fn append(&self, events: &[Event]) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        // Per-day local file in the stream dir. Sync reads only appended bytes
        // and publishes them as immutable chunks on its own cadence.
        std::fs::create_dir_all(&self.out)?;
        let path = self.out.join(format!(
            "track.{}.jsonl",
            day_stamp(unix_ms(SystemTime::now()))
        ));
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        let mut body = String::new();
        for e in events {
            body.push_str(&serde_json::to_string(e).map_err(|e| anyhow!("encode: {e}"))?);
            body.push('\n');
        }
        f.write_all(body.as_bytes())?;
        Ok(())
    }
}

fn event_time_ms(e: &Event) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(&e.ts)
        .ok()
        .map(|t| t.timestamp_millis())
}

/// The started-session-ids file, a sibling of the cursors file.
fn started_path(cursors: &Path) -> PathBuf {
    cursors.with_extension("started.json")
}

fn default_roots(id: &str, home: &str) -> Vec<String> {
    let claude_home = std::env::var("CLAUDE_CONFIG_DIR").ok();
    let codex_home = std::env::var("CODEX_HOME").ok();
    roots_with_overrides(id, home, claude_home.as_deref(), codex_home.as_deref())
}

fn roots_with_overrides(
    id: &str,
    home: &str,
    claude_home: Option<&str>,
    codex_home: Option<&str>,
) -> Vec<String> {
    match id {
        "claudecode" => {
            let base = claude_home.map(str::to_owned).unwrap_or_else(|| format!("{home}/.claude"));
            vec![format!("{base}/projects")]
        }
        "codex" => {
            let base = codex_home.map(str::to_owned).unwrap_or_else(|| format!("{home}/.codex"));
            vec![format!("{base}/sessions")]
        }
        "cowork" => vec![
            format!("{home}/Library/Application Support/Claude/local-agent-mode-sessions"),
            format!("{home}/Library/Application Support/Claude/claude-code-sessions"),
        ],
        _ => vec![],
    }
}

fn sources(which: &str) -> Result<Vec<Box<dyn Source>>> {
    let all: Vec<Box<dyn Source>> = vec![Box::new(ClaudeCode), Box::new(Codex), Box::new(Cowork)];
    if which == "all" {
        return Ok(all);
    }
    let picked: Vec<Box<dyn Source>> = all.into_iter().filter(|s| s.id() == which).collect();
    if picked.is_empty() {
        bail!("unknown source {which} (have: claudecode, codex, cowork)");
    }
    Ok(picked)
}

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

fn file_mtime_ms(p: &Path) -> i64 {
    std::fs::metadata(p)
        .and_then(|m| m.modified())
        .map(unix_ms)
        .unwrap_or(0)
}

fn unix_ms(t: SystemTime) -> i64 {
    t.duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

const SERVICE_LABEL: &str = "com.superlinked.synty";

/// Whether the login-time tracker is both installed and actually loaded. A
/// leftover plist/service file is not enough: status must not claim tracking
/// is active when the service manager has no such job.
pub fn autostart_enabled() -> bool {
    let Some((path, kind)) = autostart_unit() else {
        return false;
    };
    autostart_ready(Path::new(&path).exists(), service_loaded(kind))
}

fn autostart_ready(unit_exists: bool, service_loaded: bool) -> bool {
    unit_exists && service_loaded
}

/// The login-time autostart unit path + kind for this platform, if supported.
pub fn autostart_unit() -> Option<(String, &'static str)> {
    let home = std::env::var("HOME").ok()?;
    if cfg!(target_os = "macos") {
        Some((format!("{home}/Library/LaunchAgents/com.superlinked.synty.plist"), "launchd"))
    } else if cfg!(target_os = "linux") {
        Some((format!("{home}/.config/systemd/user/synty.service"), "systemd"))
    } else {
        None
    }
}

/// Turn login-time autostart on or off and verify the service manager accepted
/// it. A failed bootstrap is an error, not a green status badge.
pub fn autostart_set(on: bool) -> Result<()> {
    let (path, kind) =
        autostart_unit().ok_or_else(|| anyhow!("autostart unsupported on this platform"))?;
    if on {
        write_unit(kind, &path, "corpus/local", "local")?;
        loader(kind, &path, true)?;
    } else {
        loader(kind, &path, false)?;
        let _ = std::fs::remove_file(&path);
    }
    Ok(())
}

/// Restart the login-time tracker so a freshly installed binary takes over
/// (called by `synty upgrade`). Ok(false) when no unit is installed — nothing
/// to restart.
pub fn restart() -> Result<bool> {
    let Some((path, kind)) = autostart_unit() else {
        return Ok(false);
    };
    if !Path::new(&path).exists() {
        return Ok(false);
    }
    loader(kind, &path, false)?;
    loader(kind, &path, true)?;
    Ok(true)
}

/// The directory the autostart unit runs from. The current home when it holds
/// synty state (the dev-checkout case), else ~/.synty — created so a fresh
/// install's tracker has a stable, machine-wide home instead of whatever
/// directory `init` happened to run in.
fn unit_workdir() -> Result<String> {
    if Path::new(".synty").exists() {
        return Ok(std::env::current_dir()?.display().to_string());
    }
    let home = std::env::var("HOME").map_err(|_| anyhow!("no $HOME"))?;
    let d = Path::new(&home).join(".synty");
    std::fs::create_dir_all(&d)?;
    Ok(d.display().to_string())
}

fn launch_domain() -> String {
    #[cfg(unix)]
    unsafe {
        format!("gui/{}", libc::geteuid())
    }
    #[cfg(not(unix))]
    String::new()
}

fn service_loaded(kind: &str) -> bool {
    match kind {
        "launchd" => std::process::Command::new("launchctl")
            .args(["print", &format!("{}/{}", launch_domain(), SERVICE_LABEL)])
            .stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .is_ok_and(|o| launchd_report_running(o.status.success(), &o.stdout)),
        "systemd" => {
            std::process::Command::new("systemctl")
                .args(["--user", "is-enabled", "--quiet", "synty.service"])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
                && std::process::Command::new("systemctl")
                    .args(["--user", "is-active", "--quiet", "synty.service"])
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .is_ok_and(|s| s.success())
        }
        _ => false,
    }
}

/// Service managers may accept a job before reporting it active. Poll briefly
/// so initialization reflects the eventual startup result instead of racing it.
fn poll_service_ready(
    mut ready: impl FnMut() -> bool,
    attempts: usize,
    mut wait: impl FnMut(),
) -> bool {
    for attempt in 0..attempts {
        if ready() {
            return true;
        }
        if attempt + 1 < attempts {
            wait();
        }
    }
    false
}

// launchd can take several seconds to report a freshly bootstrapped background
// process as running while the machine is under load. Keep the wait bounded,
// but long enough that `init` and `upgrade` do not report a false failure.
const SERVICE_READY_ATTEMPTS: usize = 101;

fn service_registered(kind: &str) -> bool {
    kind == "launchd"
        && std::process::Command::new("launchctl")
            .args(["print", &format!("{}/{}", launch_domain(), SERVICE_LABEL)])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
}

fn launchd_report_running(success: bool, stdout: &[u8]) -> bool {
    success
        && String::from_utf8_lossy(stdout)
            .lines()
            .any(|line| line.trim() == "state = running")
}

fn run_service(cmd: &str, args: &[&str]) -> Result<()> {
    let out = std::process::Command::new(cmd)
        .args(args)
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| anyhow!("run {cmd}: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        let why = String::from_utf8_lossy(&out.stderr).trim().to_string();
        Err(anyhow!("{cmd} {} failed: {why}", args.join(" ")))
    }
}

/// (Un)load the unit via the service manager, then verify activation.
fn loader(kind: &str, path: &str, on: bool) -> Result<()> {
    loader_with(
        kind,
        path,
        on,
        run_service,
        service_registered,
        || service_loaded(kind),
        || std::thread::sleep(Duration::from_millis(100)),
    )
}

/// Loader flow with injectable service operations so lifecycle ordering and
/// readiness behavior can be exercised without touching the host manager.
fn loader_with(
    kind: &str,
    path: &str,
    on: bool,
    mut run: impl FnMut(&str, &[&str]) -> Result<()>,
    mut registered: impl FnMut(&str) -> bool,
    mut ready: impl FnMut() -> bool,
    mut wait: impl FnMut(),
) -> Result<()> {
    match (kind, on) {
        ("launchd", true) => {
            let domain = launch_domain();
            let target = format!("{domain}/{SERVICE_LABEL}");
            if registered(kind) {
                // The job may disappear between the registration probe and
                // bootout. The state transition below is authoritative.
                let _ = run("launchctl", &["bootout", &target]);
                if !poll_service_ready(
                    || !registered(kind),
                    SERVICE_READY_ATTEMPTS,
                    &mut wait,
                ) {
                    bail!("launchd did not finish unloading {SERVICE_LABEL}");
                }
            }
            run("launchctl", &["bootstrap", &domain, path])?;
            run("launchctl", &["kickstart", "-k", &target])?;
        }
        ("launchd", false) => {
            if registered(kind) {
                run(
                    "launchctl",
                    &["bootout", &format!("{}/{}", launch_domain(), SERVICE_LABEL)],
                )?;
            }
        }
        ("systemd", true) => {
            for args in SYSTEMD_LOAD_SEQUENCE {
                run("systemctl", args)?;
            }
        }
        ("systemd", false) => {
            // `is-active` can be false while an enabled failed service still
            // has a login symlink. Always disable before removing the unit.
            run(
                "systemctl",
                &["--user", "disable", "--now", "synty.service"],
            )?;
        }
        _ => bail!("unknown service manager: {kind}"),
    }
    if on && !poll_service_ready(&mut ready, SERVICE_READY_ATTEMPTS, &mut wait) {
        bail!("{kind} accepted the unit but {SERVICE_LABEL} is not loaded");
    }
    Ok(())
}

/// UTC day (YYYY-MM-DD) for the per-day output file. Pure (takes the timestamp)
/// so rotation is tested without a clock.
fn day_stamp(unix_ms: i64) -> String {
    chrono::DateTime::from_timestamp(unix_ms / 1000, 0)
        .map(|d| d.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| "0000-00-00".into())
}

/// Whether the GitHub sub-cadence is due: never-run (None) fires immediately,
/// otherwise once `every` has elapsed since the last run. Pure so the cadence is
/// tested without a clock or the network.
fn github_due(elapsed_since_last: Option<Duration>, every: Duration) -> bool {
    elapsed_since_last.map(|e| e >= every).unwrap_or(true)
}

/// Write the autostart unit file for `kind` (no output) — runs `synty track
/// --watch` at login, from a stable home, pushing to the configured fleet
/// bucket when one is set.
fn write_unit(kind: &str, path: &str, out: &str, machine: &str) -> Result<()> {
    let exe = std::env::current_exe()?.display().to_string();
    let cwd = unit_workdir()?;
    let mut args = vec![
        "track".to_string(),
        "--watch".to_string(),
        "--out".to_string(),
        out.to_string(),
        "--machine".to_string(),
        machine.to_string(),
    ];
    if let Some(b) = crate::config::resolve_bucket_opt(None) {
        args.push("--bucket".to_string());
        args.push(b);
    }
    let log_dir = Path::new(&cwd).join(".synty");
    std::fs::create_dir_all(&log_dir)?;
    let log = log_dir.join("track.log").display().to_string();
    let body = match kind {
        "launchd" => format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{SERVICE_LABEL}</string>
  <key>ProgramArguments</key><array>
    <string>{}</string>{}
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Background</string>
  <key>WorkingDirectory</key><string>{}</string>
  <key>StandardOutPath</key><string>{}</string>
  <key>StandardErrorPath</key><string>{}</string>
</dict></plist>
"#,
            xml(&exe),
            args.iter()
                .map(|a| format!("\n    <string>{}</string>", xml(a)))
                .collect::<String>(),
            xml(&cwd),
            xml(&log),
            xml(&log),
        ),
        "systemd" => format!(
            "[Unit]\nDescription=synty native tracker\n\n[Service]\nExecStart={} {}\nWorkingDirectory={}\nRestart=always\nRestartSec=5\nStandardOutput=append:{}\nStandardError=append:{}\n\n[Install]\nWantedBy=default.target\n",
            systemd_arg(&exe),
            args.iter()
                .map(|a| systemd_arg(a))
                .collect::<Vec<_>>()
                .join(" "),
            systemd_path(&cwd),
            systemd_path(&log),
            systemd_path(&log),
        ),
        _ => bail!("install kind must be launchd or systemd"),
    };
    if let Some(dir) = Path::new(path).parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, body)?;
    Ok(())
}

fn xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Quote one systemd unit argument. `%` must be doubled because systemd
/// expands specifiers even inside quotes.
fn systemd_arg(s: &str) -> String {
    format!(
        "\"{}\"",
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('%', "%%")
    )
}

/// Escape a standalone systemd directive path without wrapping it in quotes.
/// Directives such as WorkingDirectory= treat quote characters as path bytes;
/// hex escapes preserve spaces and other syntax-significant bytes instead.
fn systemd_path(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b == b'%' {
            out.push_str("%%");
        } else if b.is_ascii_alphanumeric() || b"/._:-".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("\\x{b:02x}"));
        }
    }
    out
}

/// `enable --now` leaves an already-active old process untouched. Reload,
/// enable, then restart so an idempotent init actually adopts the new binary,
/// bucket, and capture configuration.
const SYSTEMD_LOAD_SEQUENCE: &[&[&str]] = &[
    &["--user", "daemon-reload"],
    &["--user", "enable", "synty.service"],
    &["--user", "restart", "synty.service"],
];

/// CLI `track --install <kind>`: write the unit and print the manual load command.
fn install(kind: &str, o: &Opts) -> Result<()> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let path = match kind {
        "launchd" => format!("{home}/Library/LaunchAgents/com.superlinked.synty.plist"),
        "systemd" => format!("{home}/.config/systemd/user/synty.service"),
        _ => bail!("install kind must be launchd or systemd"),
    };
    write_unit(kind, &path, &o.out, &o.machine)?;
    match kind {
        "launchd" => println!(
            "wrote {path}\nload with:  launchctl bootstrap {} {path}",
            launch_domain()
        ),
        _ => println!("wrote {path}\nenable with:  systemctl --user enable --now synty.service"),
    }
    let _ = ms_to_rfc3339; // reserved for future status output
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_home_overrides_change_only_the_selected_source() {
        assert_eq!(
            roots_with_overrides("codex", "/home/a", None, Some("/var/codex")),
            vec!["/var/codex/sessions"]
        );
        assert_eq!(
            roots_with_overrides("claudecode", "/home/a", Some("/var/claude"), None),
            vec!["/var/claude/projects"]
        );
        assert_eq!(
            roots_with_overrides("codex", "/home/a", None, None),
            vec!["/home/a/.codex/sessions"]
        );
    }

    #[test]
    fn autostart_requires_both_unit_and_loaded_service() {
        assert!(autostart_ready(true, true));
        assert!(
            !autostart_ready(true, false),
            "orphan plist is not active tracking"
        );
        assert!(
            !autostart_ready(false, true),
            "loaded stale job needs the managed unit"
        );
    }

    #[test]
    fn launchd_job_must_be_running_not_merely_registered() {
        assert!(launchd_report_running(
            true,
            b"service = {\n\tstate = running\n\tpid = 42\n}"
        ));
        assert!(!launchd_report_running(
            true,
            b"service = {\n\tstate = waiting\n}"
        ));
        assert!(!launchd_report_running(false, b"state = running"));
    }

    // The tracker scrapes GitHub on a slow sub-cadence: immediately on first
    // pass, then only once the interval has elapsed (so the 30s session poll
    // doesn't hammer the API or re-pull the corpus every tick).
    #[test]
    fn github_cadence_fires_first_then_after_interval() {
        let every = Duration::from_secs(3600);
        assert!(github_due(None, every), "never-run → fire now");
        assert!(
            !github_due(Some(Duration::from_secs(30)), every),
            "30s in → not yet"
        );
        assert!(
            github_due(Some(Duration::from_secs(3601)), every),
            "past the interval → fire"
        );
    }

    // The per-day output filename is the UTC date of the event-write time;
    // bucket sync subsequently chunks only newly appended complete lines.
    #[test]
    fn day_stamp_is_the_utc_date() {
        assert_eq!(day_stamp(0), "1970-01-01");
        assert_eq!(day_stamp(1_749_859_200_000), "2025-06-14"); // a fixed 2025-06-14 UTC ms
    }

    #[test]
    fn service_unit_arguments_escape_manager_syntax() {
        assert_eq!(xml("a&<b>\""), "a&amp;&lt;b&gt;&quot;");
        assert_eq!(systemd_arg("/tmp/a b/%n\\x\""), "\"/tmp/a b/%%n\\\\x\\\"\"");
        assert_eq!(
            systemd_path("/tmp/a b/%n\\x\""),
            "/tmp/a\\x20b/%%n\\x5cx\\x22"
        );
    }

    #[test]
    fn autostart_waits_for_normal_manager_startup_latency() {
        let mut states = [false, false, true].into_iter();
        let mut waits = 0;
        assert!(poll_service_ready(
            || states.next().unwrap_or(false),
            3,
            || waits += 1,
        ));
        assert_eq!(waits, 2);
    }

    #[test]
    fn systemd_reinstall_restarts_and_tolerates_slow_startup() {
        let mut operations = Vec::new();
        let mut probes = 0;
        let mut waits = 0;
        loader_with(
            "systemd",
            "/tmp/synty.service",
            true,
            |command, args| {
                operations.push(format!("{command} {}", args.join(" ")));
                Ok(())
            },
            |_| false,
            || {
                probes += 1;
                probes == 51
            },
            || waits += 1,
        )
        .unwrap();
        assert_eq!(
            operations,
            &[
                "systemctl --user daemon-reload",
                "systemctl --user enable synty.service",
                "systemctl --user restart synty.service",
            ]
        );
        assert_eq!(probes, 51, "loader tolerates a five-second startup");
        assert_eq!(waits, 50, "loader waits between failed readiness probes");
    }

    #[test]
    fn launchd_reinstall_waits_for_bootout_before_bootstrap() {
        let mut operations = Vec::new();
        let mut registrations = [true, true, false].into_iter();
        let mut waits = 0;
        loader_with(
            "launchd",
            "/tmp/com.superlinked.synty.plist",
            true,
            |command, args| {
                operations.push(format!(
                    "{command} {}",
                    args.first().copied().unwrap_or_default()
                ));
                Ok(())
            },
            |_| registrations.next().unwrap_or(false),
            || true,
            || waits += 1,
        )
        .unwrap();
        assert_eq!(
            operations,
            &[
                "launchctl bootout",
                "launchctl bootstrap",
                "launchctl kickstart",
            ]
        );
        assert_eq!(waits, 1, "bootstrap waits until bootout is complete");
    }

    #[test]
    fn capture_boundary_stages_start_and_keeps_only_new_work() {
        let dir = std::env::temp_dir().join(format!("synty-track-since-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let root = dir.join("sessions");
        std::fs::create_dir_all(&root).unwrap();
        let old = r#"{"type":"user","sessionId":"S1","version":"2.1.30","timestamp":"2026-07-20T23:55:00Z","message":{"content":"old prompt"}}"#;
        let new = r#"{"type":"user","sessionId":"S1","version":"2.1.30","timestamp":"2026-07-21T00:05:00Z","message":{"content":"new prompt"}}"#;
        std::fs::write(root.join("s.jsonl"), format!("{old}\n{new}\n")).unwrap();
        let mut st = Stream {
            src: Box::new(ClaudeCode),
            roots: vec![root.to_string_lossy().into_owned()],
            name: "edge-t-claudecode".into(),
            out: dir.join("out"),
            seq: Sequencer::new(),
            started: HashSet::new(),
            files: HashMap::new(),
            open: HashMap::new(),
            n_sessions: 0,
            n_skipped: 0,
            actor: "tester".into(),
            campaign: String::new(),
            campaign_role: String::new(),
        };
        let cutoff = chrono::DateTime::parse_from_rfc3339("2026-07-21T00:00:00Z")
            .unwrap()
            .timestamp_millis();
        st.drain(cutoff, &HashMap::new()).unwrap();
        let body = read_stream(&dir.join("out"));
        assert!(
            body.contains("session_start") && body.contains("new prompt"),
            "{body}"
        );
        assert!(!body.contains("old prompt"), "{body}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Concatenate a stream dir's per-day files — the tracker now writes
    // track.<day>.jsonl, so a test reads the dir, not a fixed filename.
    fn read_stream(dir: &std::path::Path) -> String {
        let mut s = String::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.filter_map(|e| e.ok()) {
                if e.path().extension().and_then(|x| x.to_str()) == Some("jsonl") {
                    s.push_str(&std::fs::read_to_string(e.path()).unwrap_or_default());
                }
            }
        }
        s
    }

    // Two trackers seeing the same session must synthesize session_end under
    // the SAME deterministic id (minted from the last event ts, not file mtime
    // or batch order), so readers dedup the re-emission.
    #[test]
    fn session_end_id_is_deterministic_across_trackers() {
        let dir = std::env::temp_dir().join(format!("synty-end-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let root = dir.join("sessions");
        std::fs::create_dir_all(&root).unwrap();
        let l1 = r#"{"type":"user","sessionId":"S1","version":"2.1.30","timestamp":"2026-05-31T20:00:00Z","message":{"content":"only prompt"}}"#;
        std::fs::write(root.join("s.jsonl"), format!("{l1}\n")).unwrap();

        let end_id = |out: &str| {
            let mut st = Stream {
                src: Box::new(ClaudeCode),
                roots: vec![root.to_string_lossy().into_owned()],
                name: "edge-t-claudecode".into(),
                out: dir.join(out),
                seq: Sequencer::new(),
                started: HashSet::new(),
                files: HashMap::new(),
                open: HashMap::new(),
                n_sessions: 0,
                n_skipped: 0,
                actor: "tester".into(),
                campaign: String::new(),
                campaign_role: String::new(),
            };
            st.drain(0, &HashMap::new()).unwrap();
            st.flush_ends(i64::MAX - 1, true).unwrap();
            let body = read_stream(&dir.join(out));
            let line = body.lines().find(|l| l.contains("session_end")).expect("session_end emitted");
            serde_json::from_str::<Event>(line).unwrap().event_id
        };
        let (a, b) = (end_id("a"), end_id("b"));
        assert_eq!(a, b, "session_end must mint the same id from the same last-event ts");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // A tracker restart resumes mid-file with fresh in-memory state. The
    // persisted started set must keep a session that simply continued across
    // the restart from re-emitting session_start (a restart mints it under a
    // new line offset, so event-id dedup would never catch it).
    #[test]
    fn restart_does_not_duplicate_session_start() {
        let dir = std::env::temp_dir().join(format!("synty-track-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let root = dir.join("sessions");
        std::fs::create_dir_all(&root).unwrap();
        let log = root.join("s.jsonl");
        let l1 = r#"{"type":"user","sessionId":"S1","version":"2.1.30","timestamp":"2026-05-31T20:00:00Z","message":{"content":"first prompt"}}"#;
        std::fs::write(&log, format!("{l1}\n")).unwrap();

        let stream = |started: HashSet<String>| Stream {
            src: Box::new(ClaudeCode),
            roots: vec![root.to_string_lossy().into_owned()],
            name: "edge-t-claudecode".into(),
            out: dir.join("outdir"),
            seq: Sequencer::new(),
            started,
            files: HashMap::new(),
            open: HashMap::new(),
            n_sessions: 0,
            n_skipped: 0,
            actor: "tester".into(),
            campaign: String::new(),
            campaign_role: String::new(),
        };

        // First run: parse from the top, persist cursor + started set.
        let mut s1 = stream(HashSet::new());
        s1.drain(0, &HashMap::new()).unwrap();
        let offset = s1.files.values().next().unwrap().offset;
        let started = s1.started.clone();

        // Restart: the session's file grows; the new process seeds the cursor
        // and the persisted started set.
        let l2 = r#"{"type":"user","sessionId":"S1","version":"2.1.30","timestamp":"2026-05-31T20:05:00Z","message":{"content":"second prompt"}}"#;
        let mut f = std::fs::OpenOptions::new().append(true).open(&log).unwrap();
        f.write_all(format!("{l2}\n").as_bytes()).unwrap();
        let cursors: HashMap<String, i64> =
            [(format!("claudecode\0{}", log.display()), offset)].into_iter().collect();
        let mut s2 = stream(started);
        s2.drain(0, &cursors).unwrap();

        let out = read_stream(&dir.join("outdir"));
        let starts = out.lines().filter(|l| l.contains("\"session_start\"")).count();
        assert_eq!(starts, 1, "restart must not re-emit session_start:\n{out}");
        assert!(out.contains("second prompt"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // session_start carries who (actor) and which synty (tracker_version), so
    // the fleet roster can join machines to people and spot upgrade lag.
    #[test]
    fn session_start_carries_tracker_version() {
        let dir = std::env::temp_dir().join(format!("synty-ver-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let root = dir.join("sessions");
        std::fs::create_dir_all(&root).unwrap();
        let l1 = r#"{"type":"user","sessionId":"S1","version":"2.1.30","timestamp":"2026-05-31T20:00:00Z","message":{"content":"a prompt"}}"#;
        std::fs::write(root.join("s.jsonl"), format!("{l1}\n")).unwrap();

        let mut st = Stream {
            src: Box::new(ClaudeCode),
            roots: vec![root.to_string_lossy().into_owned()],
            name: "edge-t-claudecode".into(),
            out: dir.join("outdir"),
            seq: Sequencer::new(),
            started: HashSet::new(),
            files: HashMap::new(),
            open: HashMap::new(),
            n_sessions: 0,
            n_skipped: 0,
            actor: "tester".into(),
            campaign: "camp-1".into(),
            campaign_role: "investigator".into(),
        };
        st.drain(0, &HashMap::new()).unwrap();

        let out = read_stream(&dir.join("outdir"));
        let line = out.lines().find(|l| l.contains("\"session_start\"")).expect("session_start emitted");
        let e: Event = serde_json::from_str(line).unwrap();
        assert_eq!(e.payload["actor"], "tester", "actor stamp must survive alongside the version");
        assert_eq!(e.payload["tracker_version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(e.rollup_dim, "camp-1");
        assert_eq!(e.payload["campaign_id"], "camp-1");
        assert_eq!(e.payload["campaign_role"], "investigator");
        assert_eq!(e.payload["backend"], "claude_code");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Scenario: two teammates each run an agent on their own machine, tracking
    // to one shared bucket; whoever builds sees BOTH machines' sessions. This is
    // the core fleet promise (track everywhere, build once), exercised
    // end-to-end through the real tailer → push_events → ingest path.
    #[test]
    fn two_machines_converge_through_one_bucket() {
        let root = std::env::temp_dir().join(format!("synty-fleet-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let bucket = root.join("bucket");
        let bu = bucket.to_str().unwrap();

        // A synthetic Claude session per machine — distinct work.
        let session = |sessions: &std::path::Path, sid: &str, prompt: &str| {
            std::fs::create_dir_all(sessions).unwrap();
            let line = format!(
                r#"{{"type":"user","sessionId":"{sid}","version":"2.1.30","timestamp":"2026-05-31T20:00:00Z","message":{{"content":"{prompt}"}}}}"#
            );
            std::fs::write(sessions.join(format!("{sid}.jsonl")), format!("{line}\n")).unwrap();
        };
        session(&root.join("mA/sessions"), "SESSA", "machine A refactored the auth flow");
        session(&root.join("mB/sessions"), "SESSB", "machine B fixed the rate limiter");

        // Each machine tails its own sessions into its own corpus/local, then
        // pushes raw events to the shared bucket.
        let track_one = |machine: &str, sessions: &std::path::Path, local: &std::path::Path| {
            let mut st = Stream {
                src: Box::new(ClaudeCode),
                roots: vec![sessions.to_string_lossy().into_owned()],
                name: format!("edge-{machine}-claudecode"),
                out: local.join(format!("edge-{machine}-claudecode")),
                seq: Sequencer::new(),
                started: HashSet::new(),
                files: HashMap::new(),
                open: HashMap::new(),
                n_sessions: 0,
                n_skipped: 0,
                actor: machine.into(),
                campaign: String::new(),
                campaign_role: String::new(),
            };
            st.drain(0, &HashMap::new()).unwrap();
            st.flush_ends(i64::MAX - 1, true).unwrap(); // close the session so it becomes a unit
        };
        track_one("mA", &root.join("mA/sessions"), &root.join("mA/local"));
        track_one("mB", &root.join("mB/sessions"), &root.join("mB/local"));
        assert!(
            crate::sync::push_events_with(
                bu,
                root.join("mA/local").to_str().unwrap(),
                root.join("mA/uploads.json").to_str().unwrap(),
                None,
            )
            .unwrap()
                > 0
        );
        assert!(
            crate::sync::push_events_with(
                bu,
                root.join("mB/local").to_str().unwrap(),
                root.join("mB/uploads.json").to_str().unwrap(),
                None,
            )
            .unwrap()
                > 0
        );

        // A third machine builds: ingest pulls every device's events from the
        // bucket and turns them into docs — so the build sees both teammates.
        let docs = root.join("docs.jsonl");
        let builder = root.join("builder");
        crate::ingest::run(builder.to_str().unwrap(), docs.to_str().unwrap(), Some(bu)).unwrap();
        let text = std::fs::read_to_string(&docs).unwrap();
        assert!(text.contains("SESSA") && text.contains("SESSB"), "build sees both machines' sessions");
        assert!(text.contains("auth flow") && text.contains("rate limiter"), "both machines' work is present");

        // Re-running the build is idempotent — events skip by size, same docs.
        let before = std::fs::read_to_string(&docs).unwrap().len();
        crate::ingest::run(builder.to_str().unwrap(), docs.to_str().unwrap(), Some(bu)).unwrap();
        assert_eq!(std::fs::read_to_string(&docs).unwrap().len(), before, "re-build is stable");
        let _ = std::fs::remove_dir_all(&root);
    }
}
