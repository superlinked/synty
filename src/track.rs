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
    pub machine: String,
    pub watch: bool,
    pub poll_secs: u64,
    pub install: Option<String>,
    pub cursors: String,
    /// If set, push drained event files to this bucket under events/ (the
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
        let pushed = t.push()?;
        let mut m = crate::metrics::Run::new("track");
        m.set("events", n).set("sessions", t.session_count()).set("lines_skipped", t.skipped_count());
        m.emit();
        eprintln!("track: {n} events ({} sessions) → {}", t.session_count(), o.out);
        if pushed > 0 {
            eprintln!("track: pushed {pushed} event files → {}/events/", t.bucket.as_deref().unwrap_or(""));
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
}

struct Tracker {
    streams: Vec<Stream>,
    cutoff_ms: i64,
    cursors_path: PathBuf,
    cursors: HashMap<String, i64>,
    out: String,
    bucket: Option<String>,
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
            let out = Path::new(&o.out).join(&name).join("track.jsonl");
            if let Some(p) = out.parent() {
                std::fs::create_dir_all(p)?;
            }
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
                src,
            });
        }
        let cutoff_ms = if o.max_age_days == 0 {
            0
        } else {
            unix_ms(SystemTime::now()) - (o.max_age_days as i64) * 86_400_000
        };
        Ok(Self { streams, cutoff_ms, cursors_path, cursors, out: o.out.clone(), bucket: o.bucket.clone() })
    }

    /// Push drained event files to the shared bucket, if configured.
    fn push(&self) -> Result<usize> {
        match &self.bucket {
            Some(b) => crate::sync::push_events(b, &self.out),
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
        loop {
            let n = self.drain()?;
            let now = unix_ms(SystemTime::now());
            let ended = self.flush_ends(now, false)?;
            self.save_cursors()?;
            let pushed = self.push()?;
            if n > 0 || ended > 0 || pushed > 0 {
                let skipped = self.skipped_count();
                if skipped > 0 {
                    eprintln!("track: +{n} events, {ended} ended, {pushed} files pushed, {skipped} malformed lines skipped (total)");
                } else {
                    eprintln!("track: +{n} events, {ended} ended, {pushed} files pushed");
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
            let Some(last_nl) = slice.iter().rposition(|&b| b == b'\n') else { continue };
            let complete = &slice[..=last_nl];
            let fallback_ms = file_mtime_ms(&path);

            let path_str = path.to_string_lossy();
            let before = self.started.len();
            let mut ec = EmitCtx::new(self.name.clone(), &*self.src, &mut self.seq, &mut self.started);
            let (mut evts, consumed, skipped) = drive(&mut *fs.parser, complete, &path_str, fs.offset, fallback_ms, &mut ec);
            drop(ec);
            fs.offset += consumed;
            self.n_sessions += self.started.len() - before;
            self.n_skipped += skipped;

            for e in &mut evts {
                if e.kind == kind::SESSION_START {
                    e.payload["actor"] = json!(self.actor);
                }
            }
            for e in &evts {
                if !e.session_id.is_empty() && e.kind != kind::SESSION_END {
                    self.open.insert(e.session_id.clone(), (fallback_ms, e.ts.clone()));
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
        let mut events = Vec::new();
        let mut ec = EmitCtx::new(self.name.clone(), &*self.src, &mut self.seq, &mut self.started);
        for sid in &due {
            let (ms, ts) = self.open.remove(sid).unwrap();
            events.push(ec.event(ms, &ts, kind::SESSION_END, sid, json!({"reason": if all {"shutdown"} else {"idle"}})));
        }
        self.append(&events)?;
        Ok(due.len())
    }

    fn append(&self, events: &[Event]) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&self.out)?;
        let mut body = String::new();
        for e in events {
            body.push_str(&serde_json::to_string(e).map_err(|e| anyhow!("encode: {e}"))?);
            body.push('\n');
        }
        f.write_all(body.as_bytes())?;
        Ok(())
    }
}

/// The started-session-ids file, a sibling of the cursors file.
fn started_path(cursors: &Path) -> PathBuf {
    cursors.with_extension("started.json")
}

fn default_roots(id: &str, home: &str) -> Vec<String> {
    match id {
        "claudecode" => vec![format!("{home}/.claude/projects")],
        "codex" => vec![format!("{home}/.codex/sessions")],
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
    std::fs::metadata(p).and_then(|m| m.modified()).map(unix_ms).unwrap_or(0)
}

fn unix_ms(t: SystemTime) -> i64 {
    t.duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// Whether a login-time autostart unit has been installed (launchd or systemd).
pub fn autostart_enabled() -> bool {
    let home = std::env::var("HOME").unwrap_or_default();
    [
        format!("{home}/Library/LaunchAgents/com.superlinked.synty.plist"),
        format!("{home}/.config/systemd/user/synty.service"),
    ]
    .iter()
    .any(|p| Path::new(p).exists())
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

/// Turn login-time autostart on or off, then best-effort (un)load it so the
/// change takes effect now. Quiet (no stdout/stderr) — safe to call from the TUI.
pub fn autostart_set(on: bool) -> Result<()> {
    let (path, kind) = autostart_unit().ok_or_else(|| anyhow!("autostart unsupported on this platform"))?;
    if on {
        write_unit(kind, &path, "corpus/local", "local")?;
        loader(kind, &path, true);
    } else {
        loader(kind, &path, false);
        let _ = std::fs::remove_file(&path);
    }
    Ok(())
}

/// (un)load the unit via launchctl / systemctl, swallowing all output and errors.
fn loader(kind: &str, path: &str, on: bool) {
    let run = |cmd: &str, args: &[&str]| {
        let _ = std::process::Command::new(cmd)
            .args(args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    };
    match (kind, on) {
        ("launchd", true) => run("launchctl", &["load", "-w", path]),
        ("launchd", false) => run("launchctl", &["unload", "-w", path]),
        ("systemd", true) => {
            run("systemctl", &["--user", "daemon-reload"]);
            run("systemctl", &["--user", "enable", "--now", "synty.service"]);
        }
        ("systemd", false) => run("systemctl", &["--user", "disable", "--now", "synty.service"]),
        _ => {}
    }
}

/// Write the autostart unit file for `kind` (no output) — runs `synty track
/// --watch` at login.
fn write_unit(kind: &str, path: &str, out: &str, machine: &str) -> Result<()> {
    let exe = std::env::current_exe()?.display().to_string();
    let cwd = std::env::current_dir()?.display().to_string();
    let args = format!("track --watch --out {out} --machine {machine}");
    let body = match kind {
        "launchd" => format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.superlinked.synty</string>
  <key>ProgramArguments</key><array>
    <string>{exe}</string>{}
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>WorkingDirectory</key><string>{cwd}</string>
</dict></plist>
"#,
            args.split_whitespace().map(|a| format!("\n    <string>{a}</string>")).collect::<String>(),
        ),
        "systemd" => format!(
            "[Unit]\nDescription=synty native tracker\n\n[Service]\nExecStart={exe} {args}\nWorkingDirectory={cwd}\nRestart=always\n\n[Install]\nWantedBy=default.target\n",
        ),
        _ => bail!("install kind must be launchd or systemd"),
    };
    if let Some(dir) = Path::new(path).parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, body)?;
    Ok(())
}

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
        "launchd" => println!("wrote {path}\nload with:  launchctl load -w {path}"),
        _ => println!("wrote {path}\nenable with:  systemctl --user enable --now synty.service"),
    }
    let _ = ms_to_rfc3339; // reserved for future status output
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
            out: dir.join("out.jsonl"),
            seq: Sequencer::new(),
            started,
            files: HashMap::new(),
            open: HashMap::new(),
            n_sessions: 0,
            n_skipped: 0,
            actor: "tester".into(),
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

        let out = std::fs::read_to_string(dir.join("out.jsonl")).unwrap();
        let starts = out.lines().filter(|l| l.contains("\"session_start\"")).count();
        assert_eq!(starts, 1, "restart must not re-emit session_start:\n{out}");
        assert!(out.contains("second prompt"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
