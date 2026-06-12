// The human surface, built to design_tui.md: a header breadcrumb, a
// master/detail body, and a context footer; four views (Topics, Work, Search,
// Status) over units of work, with a comparable activity column and the brand
// palette. Session rows are two lines tall: the one-line summary on top, context
// below. The embedding model loads on a background thread (a search actor) so the
// first query is instant and the UI never blocks.

use crate::units::{self, Kind, Session, TopicUnits, Unit};
use crate::{first_line, load_docs, readmodel, short, Doc};
use anyhow::Result;
use chrono::{Datelike, NaiveDate};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Cell, Clear, Paragraph, Row, Table, TableState, Tabs, Wrap};
use ratatui::Frame;
use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Duration;

mod theme {
    use ratatui::style::Color;
    pub const BG: Color = Color::Rgb(0x1F, 0x1F, 0x2C);
    pub const FG: Color = Color::Rgb(0xF3, 0xF1, 0xF1);
    pub const ACCENT: Color = Color::Rgb(0xF3, 0x44, 0x1D);
    pub const BORDER: Color = Color::Rgb(0x5D, 0x6C, 0x80);
    pub const DIM: Color = Color::Rgb(0xA0, 0xB4, 0xC1);
    pub const GITHUB: Color = Color::Rgb(0x86, 0xA1, 0xBC); // sky
    pub const SESSION: Color = Color::Rgb(0xCE, 0xBC, 0xAA); // sand
    pub const SAGE: Color = Color::Rgb(0x86, 0x95, 0x82); // "on" / open
    pub const HILITE: Color = Color::Rgb(0x35, 0x3A, 0x4E); // selected-row background
    pub const MERGED: Color = Color::Rgb(0xA9, 0x8E, 0xDB); // PR merged (violet)
    pub const CLOSED: Color = Color::Rgb(0xC4, 0x6A, 0x6A); // PR/issue closed (muted red)
}

#[derive(Clone, Copy, PartialEq)]
enum View {
    Topics,
    Work,
    Search,
    Status,
}
const VIEWS: [View; 4] = [View::Topics, View::Work, View::Search, View::Status];
const VIEW_NAMES: [&str; 4] = ["Topics[1]", "Work[2]", "Search[3]", "Status[4]"];
const TL_DAYS: usize = 28; // topic activity strip: current + 3 prior weeks, by day

/// Commands into the search actor: a query, or "the pointer moved — reopen
/// the index" (the encoder stays loaded; only the mmap is swapped).
enum SearchCmd {
    Query(String),
    Reload,
}

enum SearchMsg {
    Ready,
    Searching,
    Results(Vec<i64>),
    /// Ack for SearchCmd::Reload — until it arrives, Results from the old
    /// index are dropped (its doc ids may not match the new docs).
    Reloaded,
    Err(String),
}

#[derive(PartialEq)]
enum Engine {
    Loading,
    Ready,
    Searching,
    Err(String),
}

/// Active topic-list filter: show only topics that touch this repo or person.
/// Cycled with `r` / `a`; wrapping past the last value clears it.
#[derive(Clone)]
enum Facet {
    Repo(String),
    Person(String),
}

impl Facet {
    fn matches(&self, t: &TopicUnits) -> bool {
        match self {
            Facet::Repo(n) => t.repos.iter().any(|r| r == n),
            Facet::Person(n) => t.authors.iter().any(|a| a == n),
        }
    }
    /// The same test for a single work unit (the Work view shares the filter).
    fn matches_unit(&self, u: &Unit) -> bool {
        match self {
            Facet::Repo(n) => &u.repo == n,
            Facet::Person(n) => &u.author == n,
        }
    }
    fn is_repo(&self) -> bool {
        matches!(self, Facet::Repo(_))
    }
    fn name(&self) -> &str {
        match self {
            Facet::Repo(n) | Facet::Person(n) => n,
        }
    }
    /// Short tag for the breadcrumb (`repo:sie-web` / `@alice`).
    fn tag(&self) -> String {
        match self {
            Facet::Repo(n) => format!("repo:{n}"),
            Facet::Person(n) => format!("@{n}"),
        }
    }
}

struct App {
    docs: Vec<Doc>,
    doc_by_id: HashMap<i64, usize>,
    sessions: Vec<Session>,
    sess_by_id: HashMap<String, usize>,
    work: Vec<Unit>,
    topics: Vec<TopicUnits>,
    status: crate::view::Status,
    view: View,
    sel: usize,
    drill_topic: Option<usize>, // Topics: viewing a topic's units (index into visible())
    drill_unit: bool,           // Topics, drilled: the selected unit's detail pane is open
    tool_drill: Option<units::ToolProfile>, // Status: the inspected tool's profile overlay
    filter: Option<Facet>,      // Topics/Work: show only items touching this repo/person
    query: String,
    results: Vec<i64>, // doc ids
    engine: Engine,
    autostart: bool,
    qtx: Option<Sender<SearchCmd>>,
    rrx: Option<Receiver<SearchMsg>>,
    quit: bool,
    cache: ViewCache,
    /// Fleet bucket for the background freshen + pulls.
    bucket: String,
    /// The running freshen child, if any; its latest phase shows in the footer.
    freshen: Option<Freshen>,
    freshen_note: Option<String>,
    last_freshen: Option<std::time::Instant>,
    /// Bundle reloads arrive on this channel (built off-thread, swapped between
    /// frames). `reload_pending` gates stale search Results until the actor
    /// acks the index reload.
    btx: Sender<Bundle>,
    brx: Receiver<Bundle>,
    reload_pending: bool,
}

/// Everything the views render, loaded as one unit so a hot reload swaps
/// atomically between frames.
struct Bundle {
    docs: Vec<Doc>,
    sessions: Vec<Session>,
    work: Vec<Unit>,
    topics: Vec<TopicUnits>,
    status: crate::view::Status,
    day_stats: HashMap<String, units::DayStat>,
}

impl Bundle {
    fn load() -> Self {
        let docs = load_docs(readmodel::docs_path()).unwrap_or_default();
        let sessions = units::sessions().unwrap_or_default();
        let work = units::units().unwrap_or_default();
        let topics = units::topic_units(12).unwrap_or_default();
        let status = crate::view::status().unwrap_or_else(|_| crate::view::Status {
            docs: docs.len(),
            github: 0,
            sessions: 0,
            by_kind: vec![],
            by_repo: vec![],
            by_user: vec![],
            by_tool: vec![],
            by_model: vec![],
            newest_ts: String::new(),
            last_indexed: None,
            last_tracked: None,
            autostart: false,
            stale: false,
        });
        Self { docs, sessions, work, topics, status, day_stats: units::day_stats() }
    }
}

/// The background freshen: a child `synty build --no-track` with stderr routed
/// to `.synty/build.log` — a file, never a pipe, so the child can't block on a
/// full buffer and an orphan finishes its build harmlessly. The TUI tails the
/// log for `@phase` markers (see progress.rs). Dropped (TUI quit) → the child
/// is killed; the bucket lease expires on its own.
struct Freshen {
    child: std::process::Child,
    log: std::fs::File,
}

const BUILD_LOG: &str = ".synty/build.log";

impl Freshen {
    fn spawn(bucket: &str) -> anyhow::Result<Self> {
        std::fs::create_dir_all(".synty")?;
        let log_w = std::fs::File::create(BUILD_LOG)?;
        // Low priority: a full re-index saturates every core by design
        // (k-means + quantization), and the freshen is a background courtesy —
        // it must never make the machine the user is typing on feel fried.
        let exe = std::env::current_exe()?;
        let child = std::process::Command::new("/usr/bin/nice")
            .arg("-n")
            .arg("15")
            .arg(exe)
            .args(["build", "--no-track", "--bucket", bucket])
            .env("SYNTY_PROGRESS", "1")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(log_w)
            .spawn()?;
        Ok(Self { child, log: std::fs::File::open(BUILD_LOG)? })
    }

    /// Read any new log lines (returning the latest phase note) and report
    /// Some(success) once the child has exited.
    fn poll(&mut self) -> (Option<String>, Option<bool>) {
        use std::io::Read;
        let mut buf = String::new();
        let _ = self.log.read_to_string(&mut buf);
        let note = buf
            .lines()
            .rev()
            .find_map(crate::progress::parse)
            .map(|(name, d, t)| crate::progress::describe(&name, d, t));
        let done = match self.child.try_wait() {
            Ok(Some(st)) => Some(st.success()),
            _ => None,
        };
        (note, done)
    }
}

impl Drop for Freshen {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Derived view data computed once at load: the TUI redraws several times a
/// second, and these depend only on `topics`/`work`, which never change after
/// load — recomputing facet tallies and parsing unit dates per frame is what
/// made large corpora scroll sluggishly.
#[derive(Default)]
struct ViewCache {
    repo_facets: Vec<String>,
    acct_facets: Vec<String>,
    repo_counts: HashMap<String, usize>,
    acct_counts: HashMap<String, usize>,
    /// Per topic, the day_num of each unit (parallel to `topics`).
    topic_days: Vec<Vec<i32>>,
    /// The Status stats panel: per metric, (label, per-day values over the
    /// 4-week window, window total) — plus the window anchors for the header.
    stats_rows: Vec<(&'static str, [u64; TL_DAYS], u64)>,
    stats_start: i32,
    stats_gmax: i32,
}

impl ViewCache {
    fn build(topics: &[TopicUnits], work: &[Unit], day_stats: &HashMap<String, units::DayStat>) -> Self {
        let facet = |repo: bool| {
            let mut ct: HashMap<&str, usize> = HashMap::new();
            for t in topics {
                for n in if repo { &t.repos } else { &t.authors } {
                    *ct.entry(n.as_str()).or_default() += 1;
                }
            }
            let mut v: Vec<(&str, usize)> = ct.into_iter().collect();
            v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
            v.into_iter().map(|(n, _)| n.to_string()).collect::<Vec<_>>()
        };
        let counts = |repo: bool| {
            let mut ct: HashMap<String, usize> = HashMap::new();
            for u in work {
                *ct.entry(if repo { u.repo.clone() } else { u.author.clone() }).or_default() += 1;
            }
            ct
        };
        // The stats time series, anchored to the most recent day with data so
        // the strips stay deterministic (no wall clock).
        let by_num: Vec<(i32, units::DayStat)> =
            day_stats.iter().filter_map(|(d, s)| Some((day_num(d)?, *s))).collect();
        let stats_gmax = by_num.iter().map(|(d, _)| *d).max().unwrap_or(0);
        let stats_start = week_start(stats_gmax);
        let metric: Vec<(&'static str, fn(&units::DayStat) -> u64)> = vec![
            ("tok out", |s| s.tok_out),
            ("tok in", |s| s.tok_in),
            ("cache r", |s| s.cache_read),
            ("cache w", |s| s.cache_create),
            ("tools", |s| s.tools),
            ("sessions", |s| s.sessions),
        ];
        let stats_rows = metric
            .into_iter()
            .map(|(label, get)| {
                let mut days = [0u64; TL_DAYS];
                for (d, s) in &by_num {
                    let off = d - stats_start;
                    if (0..TL_DAYS as i32).contains(&off) {
                        days[off as usize] += get(s);
                    }
                }
                let total: u64 = days.iter().sum();
                (label, days, total)
            })
            .collect();

        Self {
            repo_facets: facet(true),
            acct_facets: facet(false),
            repo_counts: counts(true),
            acct_counts: counts(false),
            topic_days: topics
                .iter()
                .map(|t| t.units.iter().filter_map(|u| day_num(&u.when)).collect())
                .collect(),
            stats_rows,
            stats_start,
            stats_gmax,
        }
    }
}

pub fn run(model_id: String, bucket: String) -> Result<()> {
    // Show the fleet's latest published read-model immediately; the background
    // freshen catches up on anything newer after the UI is on screen.
    if crate::sync::pull_if_stale(&bucket).unwrap_or(false) {
        eprintln!("pulled published read-model from {bucket}");
    }
    // Gag stderr first: the background model load (and candle/pylate-rs) write
    // device/diagnostic lines to stderr, which would scroll the alternate screen
    // and shove the header off the top. Restored when `_gag` drops.
    let _gag = StderrGag::new();
    let mut app = App::load(model_id, bucket);
    if app.status.stale {
        app.start_freshen();
    }
    let mut term = ratatui::init();
    let _ = term.clear();
    let res = app.run_loop(&mut term);
    ratatui::restore();
    res
}

/// Re-freshen this often while the TUI stays open (a no-op build is ~a second;
/// a real one runs in the child, never blocking the UI).
const FRESHEN_EVERY: std::time::Duration = std::time::Duration::from_secs(10 * 60);

/// Redirects fd 2 (stderr) to /dev/null while alive, restoring it on drop.
struct StderrGag {
    #[cfg(unix)]
    saved: i32,
}

impl StderrGag {
    fn new() -> Self {
        #[cfg(unix)]
        unsafe {
            let saved = libc::dup(2);
            let fd = libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY);
            if saved >= 0 && fd >= 0 {
                libc::dup2(fd, 2);
            }
            if fd >= 0 {
                libc::close(fd);
            }
            return Self { saved };
        }
        #[cfg(not(unix))]
        Self {}
    }
}

impl Drop for StderrGag {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            if self.saved >= 0 {
                libc::dup2(self.saved, 2);
                libc::close(self.saved);
            }
        }
    }
}

impl App {
    fn load(model_id: String, bucket: String) -> Self {
        let b = Bundle::load();
        let doc_by_id = b.docs.iter().enumerate().map(|(i, d)| (d.id, i)).collect();
        let sess_by_id = b.sessions.iter().enumerate().map(|(i, s)| (s.id.clone(), i)).collect();
        let (qtx, rrx) = spawn_search(model_id);
        let cache = ViewCache::build(&b.topics, &b.work, &b.day_stats);
        let (btx, brx) = channel::<Bundle>();
        Self {
            cache,
            docs: b.docs,
            doc_by_id,
            sessions: b.sessions,
            sess_by_id,
            work: b.work,
            topics: b.topics,
            status: b.status,
            view: View::Topics,
            sel: 0,
            drill_topic: None,
            drill_unit: false,
            tool_drill: None,
            filter: None,
            query: String::new(),
            results: vec![],
            engine: Engine::Loading,
            autostart: crate::track::autostart_enabled(),
            qtx: Some(qtx),
            rrx: Some(rrx),
            quit: false,
            bucket,
            freshen: None,
            freshen_note: None,
            last_freshen: None,
            btx,
            brx,
            reload_pending: false,
        }
    }

    fn run_loop(&mut self, term: &mut ratatui::DefaultTerminal) -> Result<()> {
        while !self.quit {
            term.draw(|f| self.draw(f))?;
            if event::poll(Duration::from_millis(150))? {
                if let Event::Key(k) = event::read()? {
                    if k.kind == KeyEventKind::Press {
                        self.on_key(k.code);
                    }
                }
            }
            self.drain_search();
            self.tick_freshen();
            if let Ok(bundle) = self.brx.try_recv() {
                self.apply(bundle);
            }
        }
        Ok(())
    }

    // ── background freshen ───────────────────────────────────────────────

    fn start_freshen(&mut self) {
        if self.freshen.is_some() {
            return;
        }
        self.last_freshen = Some(std::time::Instant::now());
        match Freshen::spawn(&self.bucket) {
            Ok(f) => {
                self.freshen = Some(f);
                self.freshen_note = Some("⟳ freshening".into());
            }
            Err(e) => self.freshen_note = Some(format!("freshen failed: {e}")),
        }
    }

    /// Drive the freshen child: surface its latest phase, and when it exits
    /// successfully, rebuild the data bundle off-thread (swapped in by
    /// run_loop when it lands).
    fn tick_freshen(&mut self) {
        if let Some(f) = &mut self.freshen {
            let (note, done) = f.poll();
            if let Some(n) = note {
                self.freshen_note = Some(n);
            }
            if let Some(ok) = done {
                self.freshen = None;
                if ok {
                    self.freshen_note = Some("⟳ reloading".into());
                    let tx = self.btx.clone();
                    std::thread::spawn(move || {
                        let _ = tx.send(Bundle::load());
                    });
                } else {
                    self.freshen_note = Some(format!("build failed — see {BUILD_LOG}"));
                }
            }
        } else if self.last_freshen.is_none_or(|t| t.elapsed() > FRESHEN_EVERY) {
            self.start_freshen();
        }
    }

    /// Swap in a freshly loaded bundle between frames, carrying the user's
    /// place across it: the drilled topic re-found by its stable key, the
    /// selection clamped, the search index reloaded (stale results gated
    /// until the actor acks) and the live query re-run.
    fn apply(&mut self, b: Bundle) {
        let drill_key = self
            .drill_topic
            .and_then(|vi| self.drilled(vi))
            .map(|t| t.cache_key.clone());

        self.cache = ViewCache::build(&b.topics, &b.work, &b.day_stats);
        self.doc_by_id = b.docs.iter().enumerate().map(|(i, d)| (d.id, i)).collect();
        self.sess_by_id = b.sessions.iter().enumerate().map(|(i, s)| (s.id.clone(), i)).collect();
        self.docs = b.docs;
        self.sessions = b.sessions;
        self.work = b.work;
        self.topics = b.topics;
        self.status = b.status;
        self.freshen_note = None;

        if let Some(key) = drill_key {
            let vis = self.visible();
            self.drill_topic = vis.iter().position(|&i| self.topics[i].cache_key == key);
        }
        if self.drill_topic.is_none() {
            self.drill_unit = false; // the drilled topic vanished; no unit to detail
        }
        self.sel = self.sel.min(self.list_len().saturating_sub(1));

        if let Some(tx) = &self.qtx {
            if tx.send(SearchCmd::Reload).is_ok() {
                self.reload_pending = true;
            }
        }
    }

    fn drain_search(&mut self) {
        let Some(rx) = &self.rrx else { return };
        while let Ok(msg) = rx.try_recv() {
            match msg {
                SearchMsg::Ready => self.engine = Engine::Ready,
                SearchMsg::Searching => self.engine = Engine::Searching,
                // Results racing an index reload may carry the OLD build's doc
                // ids — silently wrong against new docs, so drop them; the
                // query re-runs on the ack below.
                SearchMsg::Results(_) if self.reload_pending => {}
                SearchMsg::Results(ids) => {
                    self.results = ids.into_iter().filter(|id| self.doc_by_id.contains_key(id)).collect();
                    self.sel = 0;
                    self.engine = Engine::Ready;
                }
                SearchMsg::Reloaded => {
                    self.reload_pending = false;
                    if !self.query.trim().is_empty() {
                        if let Some(tx) = &self.qtx {
                            let _ = tx.send(SearchCmd::Query(self.query.clone()));
                        }
                    }
                }
                SearchMsg::Err(e) => self.engine = Engine::Err(e),
            }
        }
    }

    // ── current list / detail ────────────────────────────────────────────

    fn list_len(&self) -> usize {
        match self.view {
            View::Topics => match self.drill_topic {
                None => self.visible().len(),
                Some(t) => self.drilled(t).map(|t| t.units.len()).unwrap_or(0),
            },
            View::Work => self.visible_work().len(),
            View::Search => self.results.len(),
            View::Status => self.status.by_tool.len(),
        }
    }

    /// Topic indices passing the active filter, in their natural order (all when
    /// unfiltered). `sel`/`drill_topic` index into THIS list, not `topics`.
    fn visible(&self) -> Vec<usize> {
        (0..self.topics.len())
            .filter(|&i| self.filter.as_ref().is_none_or(|f| f.matches(&self.topics[i])))
            .collect()
    }

    /// The topic at visible position `vi` (resolves the filter indirection).
    fn drilled(&self, vi: usize) -> Option<&TopicUnits> {
        self.visible().get(vi).and_then(|&i| self.topics.get(i))
    }

    /// Distinct repos (`repo=true`) or people across topics, most-covered first —
    /// the order `r`/`a` cycle through (precomputed; see ViewCache).
    fn facet_names(&self, repo: bool) -> &[String] {
        if repo { &self.cache.repo_facets } else { &self.cache.acct_facets }
    }

    /// Step the repo (or person) filter to its next value, wrapping past the end
    /// to "no filter" so the one key both sets and clears.
    fn cycle_facet(&mut self, repo: bool) {
        let names = self.facet_names(repo);
        if names.is_empty() {
            return;
        }
        let cur = match &self.filter {
            Some(f) if f.is_repo() == repo => names.iter().position(|n| n == f.name()),
            _ => None, // unset, or filtering the other facet → start this one over
        };
        let next = match cur {
            None => Some(names[0].clone()),
            Some(i) if i + 1 < names.len() => Some(names[i + 1].clone()),
            Some(_) => None, // past the last value → clear the filter
        };
        self.filter = next.map(|n| if repo { Facet::Repo(n) } else { Facet::Person(n) });
        self.drill_topic = None;
        self.sel = 0;
    }

    /// Work-unit indices passing the active filter (the Work view's `visible`).
    fn visible_work(&self) -> Vec<usize> {
        (0..self.work.len())
            .filter(|&i| self.filter.as_ref().is_none_or(|f| f.matches_unit(&self.work[i])))
            .collect()
    }

    /// The always-on facet bar: two labeled rows — repos (sky) above accounts
    /// (sand) — with the active filter inverted in place. r cycles the repos
    /// row, p the accounts row.
    fn facet_bar(&self, width: u16) -> Text<'static> {
        Text::from(vec![self.facet_row("Repo[r]:", true, width), self.facet_row("Acct[a]:", false, width)])
    }

    /// One row of the facet bar: a dim label, then the repo (or account) chips,
    /// windowed around the active one so it stays visible (‹ › flag hidden chips).
    fn facet_row(&self, label: &str, repo: bool, width: u16) -> Line<'static> {
        let names = self.facet_names(repo);
        let mut spans = vec![Span::styled(format!("{label:<9} "), Style::new().fg(theme::DIM))];
        if names.is_empty() {
            spans.push(Span::styled("—", Style::new().fg(theme::DIM)));
            return Line::from(spans);
        }
        // Unit count per facet (how active each one is), from the Work units the
        // filter would select — rendered as name(n), like the Topics UNITS column.
        let counts = if repo { &self.cache.repo_counts } else { &self.cache.acct_counts };
        let chips: Vec<String> = names.iter().map(|n| format!("{n}({})", counts.get(n.as_str()).copied().unwrap_or(0))).collect();
        let active = self.filter.as_ref().filter(|f| f.is_repo() == repo).and_then(|f| names.iter().position(|n| n == f.name()));
        // Fit a window of chips that contains the active one, expanding outward.
        let avail = width.saturating_sub(12) as usize; // label + ‹ › slack
        let cw = |s: &str| s.chars().count() + 3; // chip + separator/padding slack
        let center = active.unwrap_or(0);
        let (mut start, mut end, mut used) = (center, center, cw(&chips[center]));
        loop {
            let mut grew = false;
            if end + 1 < chips.len() && used + cw(&chips[end + 1]) <= avail {
                end += 1;
                used += cw(&chips[end]);
                grew = true;
            }
            if start > 0 && used + cw(&chips[start - 1]) <= avail {
                start -= 1;
                used += cw(&chips[start]);
                grew = true;
            }
            if !grew {
                break;
            }
        }
        if start > 0 {
            spans.push(Span::styled("‹ ", Style::new().fg(theme::DIM)));
        }
        let color = if repo { theme::GITHUB } else { theme::SESSION };
        for (j, chip) in chips[start..=end].iter().enumerate() {
            if j > 0 {
                spans.push(Span::styled(" · ", Style::new().fg(theme::BORDER)));
            }
            let style = if active == Some(start + j) {
                Style::new().bg(theme::ACCENT).fg(theme::BG).add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(color)
            };
            spans.push(Span::styled(chip.clone(), style));
        }
        if end + 1 < chips.len() {
            spans.push(Span::styled(" ›", Style::new().fg(theme::DIM)));
        }
        Line::from(spans)
    }

    fn on_key(&mut self, code: KeyCode) {
        // Esc is the universal reset: it peels back state in every view.
        if code == KeyCode::Esc {
            return self.reset();
        }
        match code {
            KeyCode::Tab => return self.cycle(1),
            KeyCode::BackTab => return self.cycle(-1),
            KeyCode::Char(c @ '1'..='4') => return self.set_view(VIEWS[c as usize - '1' as usize]),
            _ => {}
        }
        if self.view == View::Search {
            return self.search_key(code);
        }
        match code {
            KeyCode::Char('q') => {
                if self.tool_drill.take().is_some() {
                } else if self.drill_unit {
                    self.drill_unit = false;
                } else if let Some(ti) = self.drill_topic.take() {
                    self.sel = ti; // back to the topic we drilled
                } else {
                    self.quit = true;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => self.move_sel(1),
            KeyCode::Up | KeyCode::Char('k') => self.move_sel(-1),
            KeyCode::Char('g') => self.sel = 0,
            KeyCode::Char('G') => self.sel = self.list_len().saturating_sub(1),
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                if self.view == View::Status {
                    if self.tool_drill.is_none() {
                        if let Some(t) = self.status.by_tool.get(self.sel) {
                            self.tool_drill = Some(units::tool_profile(&t.name));
                        }
                    }
                } else if self.view == View::Topics {
                    if self.drill_topic.is_none() {
                        if !self.visible().is_empty() {
                            self.drill_topic = Some(self.sel);
                            self.sel = 0;
                        }
                    } else if self.list_len() > 0 {
                        // second level: open the selected unit's detail pane
                        // (the same content Work shows for it).
                        self.drill_unit = true;
                    }
                }
            }
            KeyCode::Left | KeyCode::Char('h') => {
                if self.tool_drill.take().is_some() {
                } else if self.drill_unit {
                    self.drill_unit = false;
                } else if let Some(ti) = self.drill_topic.take() {
                    self.sel = ti;
                }
            }
            // Facet bar filter, shared by Topics and Work: r cycles the Repo row,
            // a the Acct row.
            KeyCode::Char('r') if matches!(self.view, View::Topics | View::Work) && self.drill_topic.is_none() => self.cycle_facet(true),
            KeyCode::Char('a') if matches!(self.view, View::Topics | View::Work) && self.drill_topic.is_none() => self.cycle_facet(false),
            // On the Status view, a toggles the login-time autostart tracker.
            KeyCode::Char('a') if self.view == View::Status => self.toggle_autostart(),
            // u kicks a freshen (build child) immediately; the footer shows it.
            KeyCode::Char('u') => self.start_freshen(),
            _ => {}
        }
    }

    fn search_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char(c) => self.query.push(c),
            KeyCode::Backspace => {
                self.query.pop();
            }
            KeyCode::Enter => {
                if !self.query.trim().is_empty() {
                    if let Some(tx) = &self.qtx {
                        let _ = tx.send(SearchCmd::Query(self.query.clone()));
                    }
                }
            }
            KeyCode::Down => self.move_sel(1),
            KeyCode::Up => self.move_sel(-1),
            _ => {}
        }
    }

    fn cycle(&mut self, d: i32) {
        let i = VIEWS.iter().position(|v| *v == self.view).unwrap_or(0) as i32;
        self.set_view(VIEWS[(i + d).rem_euclid(VIEWS.len() as i32) as usize]);
    }
    fn set_view(&mut self, v: View) {
        self.view = v;
        self.sel = 0;
        self.drill_topic = None;
        self.drill_unit = false;
        self.tool_drill = None;
    }
    fn move_sel(&mut self, d: i32) {
        let n = self.list_len();
        if n > 0 {
            self.sel = (self.sel as i32 + d).clamp(0, n as i32 - 1) as usize;
        }
    }

    /// Esc: peel back one layer of state — close a unit detail, exit a drill,
    /// then clear the filter, then clear a search query — and quit only when
    /// nothing is left to reset.
    fn reset(&mut self) {
        if self.tool_drill.take().is_some() {
        } else if self.drill_unit {
            self.drill_unit = false;
        } else if let Some(ti) = self.drill_topic.take() {
            self.sel = ti;
        } else if self.filter.take().is_some() {
            self.sel = 0;
        } else if self.view == View::Search && !self.query.is_empty() {
            self.query.clear();
            self.results.clear();
            self.sel = 0;
        } else {
            self.quit = true;
        }
    }

    /// Install or remove the login-time tracker, reflecting the new state.
    fn toggle_autostart(&mut self) {
        if crate::track::autostart_set(!self.autostart).is_ok() {
            self.autostart = crate::track::autostart_enabled();
            self.status.autostart = self.autostart;
        }
    }

    // ── render ───────────────────────────────────────────────────────────

    fn draw(&self, f: &mut Frame) {
        let [top, body, footer] =
            Layout::vertical([Constraint::Length(1), Constraint::Min(0), Constraint::Length(1)]).areas(f.area());
        self.draw_header(f, top);
        // An always-on facet bar sits above the Topics/Work list (hidden while
        // drilled): a Repos row over an Accounts row, the active filter inverted.
        let body = match self.view {
            View::Topics | View::Work if self.drill_topic.is_none() => {
                let [bar, rest] = Layout::vertical([Constraint::Length(2), Constraint::Min(0)]).areas(body);
                f.render_widget(Paragraph::new(self.facet_bar(bar.width)), bar);
                rest
            }
            _ => body,
        };
        match self.view {
            View::Status => self.draw_status(f, body),
            View::Topics => self.draw_topics(f, body),
            View::Work | View::Search => self.draw_master_detail(f, body),
        }
        // footer: contextual keys (left) · freshness (middle) · autostart (right)
        // glyph-first, matching the freshness cell ("✓ fresh").
        let auto = if self.autostart { " ✓ autostart " } else { " ✗ autostart " };
        let fresh = format!(" {} ", self.fresh_status());
        let [fkeys, ffresh, fauto] = Layout::horizontal([
            Constraint::Min(0),
            Constraint::Length(fresh.chars().count() as u16),
            Constraint::Length(auto.chars().count() as u16),
        ])
        .areas(footer);
        f.render_widget(Line::from(self.footer()).fg(theme::DIM), fkeys);
        let fresh_color = if self.freshen.is_some() || self.reload_pending {
            theme::ACCENT
        } else if self.status.stale {
            theme::CLOSED
        } else {
            theme::SAGE
        };
        f.render_widget(Line::from(fresh).fg(fresh_color).right_aligned(), ffresh);
        f.render_widget(
            Line::from(auto).fg(if self.autostart { theme::SAGE } else { theme::DIM }).right_aligned(),
            fauto,
        );
    }

    /// The freshness segment: the running build's phase, a stale warning, or a
    /// quiet ✓ — the data on screen always says where it stands.
    fn fresh_status(&self) -> String {
        if let Some(n) = &self.freshen_note {
            return n.clone();
        }
        if self.status.stale {
            "⚠ stale · u to refresh".into()
        } else {
            "✓ fresh".into()
        }
    }

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        // Paint the whole top row as a bar so it reads as navigation.
        let bar = Style::new().bg(theme::BG).fg(theme::FG);
        f.render_widget(Block::new().style(bar), area);

        let idx = VIEWS.iter().position(|v| *v == self.view).unwrap_or(0);
        let crumb = self.breadcrumb();
        // Tabs take the room they need; the breadcrumb gets a small slice on the
        // right (and nothing on a narrow terminal).
        let bc_w = (crumb.chars().count() as u16 + 2).min(area.width.saturating_sub(46));
        let [tabs, bc] = Layout::horizontal([Constraint::Min(0), Constraint::Length(bc_w)]).areas(area);
        f.render_widget(
            Tabs::new(VIEW_NAMES.to_vec())
                .select(idx)
                .style(bar)
                .highlight_style(Style::new().bg(theme::BG).fg(theme::ACCENT).bold()),
            tabs,
        );
        f.render_widget(Line::from(crumb).style(bar).fg(theme::DIM).right_aligned(), bc);
    }

    fn breadcrumb(&self) -> String {
        let filt = self.filter.as_ref().map(|f| format!(" · {}", f.tag())).unwrap_or_default();
        match (self.view, self.drill_topic) {
            (View::Topics, Some(t)) => {
                let base = format!("synty › Topics › {}", self.drilled(t).map(|x| x.title()).unwrap_or(""));
                match self.drilled(t).filter(|_| self.drill_unit).and_then(|x| x.units.get(self.sel)) {
                    Some(u) => format!("{base} › {}", crate::excerpt(&unit_lines(u).0, 40)),
                    None => base,
                }
            }
            (View::Topics, None) => format!("synty › Topics ({}){filt}", self.visible().len()),
            (View::Work, _) => format!("synty › Work ({})", self.work.len()),
            (View::Search, _) => format!("synty › Search ({})", self.results.len()),
            (View::Status, _) => "synty › Status".to_string(),
        }
    }

    fn draw_master_detail(&self, f: &mut Frame, area: Rect) {
        let split = if self.view == View::Search {
            let [q, s] = Layout::vertical([Constraint::Length(3), Constraint::Min(0)]).areas(area);
            let cursor = if matches!(self.engine, Engine::Ready | Engine::Searching) { "▏" } else { "" };
            let body = if self.query.is_empty() && cursor.is_empty() {
                Span::styled("type to search…", Style::new().fg(theme::DIM))
            } else {
                Span::styled(format!("{}{cursor}", self.query), Style::new().fg(theme::FG))
            };
            f.render_widget(
                Paragraph::new(Line::from(vec![Span::styled("⌕ ", Style::new().fg(theme::ACCENT)), body]))
                    .block(Block::bordered().border_style(Style::new().fg(theme::BORDER))),
                q,
            );
            s
        } else {
            area
        };
        let [left, right] = Layout::horizontal([Constraint::Percentage(46), Constraint::Percentage(54)]).areas(split);

        let (header, widths, rows) = self.build_table();
        let mut ts = TableState::default();
        if !rows.is_empty() {
            ts.select(Some(self.sel.min(rows.len() - 1)));
        }
        let table = Table::new(rows, widths)
            .header(Row::new(header).style(Style::new().fg(theme::DIM).add_modifier(Modifier::BOLD)))
            .row_highlight_style(Style::new().bg(theme::HILITE).add_modifier(Modifier::BOLD))
            .highlight_symbol("▌")
            .block(Block::bordered().border_style(Style::new().fg(theme::BORDER)));
        f.render_stateful_widget(table, left, &mut ts);

        f.render_widget(
            Paragraph::new(self.detail_lines())
                .wrap(Wrap { trim: false })
                .block(Block::bordered().border_style(Style::new().fg(theme::BORDER))),
            right,
        );
    }

    /// Topics: a full-width list; when drilled, a responsive overlay on top.
    fn draw_topics(&self, f: &mut Frame, area: Rect) {
        let (header, widths, rows) = self.build_table();
        let mut ts = TableState::default();
        let sel = self.drill_topic.unwrap_or(self.sel);
        if !rows.is_empty() {
            ts.select(Some(sel.min(rows.len() - 1)));
        }
        let table = Table::new(rows, widths)
            .header(Row::new(header).style(Style::new().fg(theme::DIM).add_modifier(Modifier::BOLD)))
            .row_highlight_style(Style::new().bg(theme::HILITE).add_modifier(Modifier::BOLD))
            .highlight_symbol("▌")
            .block(Block::bordered().border_style(Style::new().fg(theme::BORDER)));
        f.render_stateful_widget(table, area, &mut ts);

        if let Some(ti) = self.drill_topic {
            if let Some(t) = self.drilled(ti) {
                self.draw_topic_overlay(f, area, t);
            }
        }
    }

    /// The drill-down: full-screen on a narrow terminal, else an overlay over
    /// the right three-quarters of the list. A second Enter splits the unit
    /// list with the selected unit's detail — the same content Work's right
    /// pane shows — and the detail follows the selection.
    fn draw_topic_overlay(&self, f: &mut Frame, full: Rect, t: &TopicUnits) {
        let area = if full.width < 100 {
            full
        } else {
            let [_, r] = Layout::horizontal([Constraint::Percentage(24), Constraint::Percentage(76)]).areas(full);
            r
        };
        f.render_widget(Clear, area);
        let block = Block::bordered().border_style(Style::new().fg(theme::ACCENT)).title(format!(" {} ", t.title()));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let [facets, units] = Layout::vertical([Constraint::Length(9), Constraint::Min(0)]).areas(inner);
        f.render_widget(Paragraph::new(self.topic_facets(t)).wrap(Wrap { trim: false }), facets);

        let list = if self.drill_unit {
            // Master-detail inside the overlay; side-by-side when there's room.
            let (l, d) = if units.width >= 80 {
                let [l, d] = Layout::horizontal([Constraint::Percentage(46), Constraint::Percentage(54)]).areas(units);
                (l, d)
            } else {
                let [l, d] = Layout::vertical([Constraint::Percentage(45), Constraint::Percentage(55)]).areas(units);
                (l, d)
            };
            let detail = t.units.get(self.sel.min(t.units.len().saturating_sub(1))).map(|u| self.unit_detail(u)).unwrap_or_default();
            f.render_widget(
                Paragraph::new(detail)
                    .wrap(Wrap { trim: false })
                    .block(Block::bordered().border_style(Style::new().fg(theme::BORDER))),
                d,
            );
            l
        } else {
            units
        };

        let dim = Style::new().fg(theme::DIM);
        let rows: Vec<Row> = t
            .units
            .iter()
            .map(|u| {
                let (primary, secondary) = unit_lines(u);
                Row::new(vec![
                    Cell::from(u.when.clone()).style(dim),
                    type_cell(u.kind),
                    two_line(primary, secondary, kind_color(u.kind)),
                    state_cell(u),
                ])
                .height(2)
            })
            .collect();
        let mut ts = TableState::default();
        if !t.units.is_empty() {
            ts.select(Some(self.sel.min(t.units.len() - 1)));
        }
        let table = Table::new(rows, [Constraint::Length(11), Constraint::Length(8), Constraint::Min(20), Constraint::Length(8)])
            .row_highlight_style(Style::new().bg(theme::HILITE).add_modifier(Modifier::BOLD))
            .highlight_symbol("▌");
        f.render_stateful_widget(table, list, &mut ts);
    }

    /// Facets for a topic overlay: the reduced summary, then counts, repos,
    /// authors, activity, type mix — with the active filter's repo/person
    /// highlighted in the repos:/people: lines.
    fn topic_facets(&self, t: &TopicUnits) -> Text<'static> {
        let (sess, prs, issues) = t.mix;
        let a = last3(&t.activity);
        let repo_active = self.filter.as_ref().filter(|f| f.is_repo()).map(|f| f.name());
        let person_active = self.filter.as_ref().filter(|f| !f.is_repo()).map(|f| f.name());
        let dim = Style::new().fg(theme::DIM);
        let field = |label: &str, names: &[String], active: Option<&str>| {
            let mut spans = vec![Span::styled(format!("{label} "), dim)];
            spans.extend(styled_names(names, active, theme::FG, 6));
            Line::from(spans)
        };
        let mut lines: Vec<Line> = Vec::new();
        if let Some(s) = &t.summary {
            lines.push(Line::from(Span::styled(s.clone(), Style::new().fg(theme::FG))));
            lines.push(Line::from(""));
        }
        let when = t.span.as_ref().map(|(x, y)| format!("active {x} → {y}")).unwrap_or_else(|| format!("last active {}", t.last_active));
        lines.push(Line::from(Span::styled(format!("{} units · {when}", t.units.len()), dim)));
        lines.push(field("repos:", &t.repos, repo_active));
        lines.push(field("accounts:", &t.authors, person_active));
        lines.push(Line::from(Span::styled(format!("activity prior/last/this wk: {} / {} / {}", a[0], a[1], a[2]), dim)));
        lines.push(Line::from(Span::styled(format!("mix: {sess} sessions · {prs} PRs · {issues} issues"), dim)));
        Text::from(lines)
    }

    /// (header, column widths, rows) for the current view's table.
    fn build_table(&self) -> (Vec<String>, Vec<Constraint>, Vec<Row<'static>>) {
        let dim = Style::new().fg(theme::DIM);
        match self.view {
            // Topics always renders the topic list; its units live in the overlay.
            View::Topics => {
                // Per-day activity over the last 4 calendar weeks (Mon-aligned),
                // shaded on a shared scale; the header labels each week's Monday.
                let vis = self.visible();
                let gmax = vis.iter().flat_map(|&i| &self.cache.topic_days[i]).copied().max().unwrap_or(0);
                let start = week_start(gmax);
                let dailies: Vec<[u64; TL_DAYS]> = vis.iter().map(|&i| daily(&self.cache.topic_days[i], start)).collect();
                let cap = dailies.iter().flat_map(|d| d.iter().copied()).max().unwrap_or(0);
                let repo_active = self.filter.as_ref().filter(|f| f.is_repo()).map(|f| f.name());
                let person_active = self.filter.as_ref().filter(|f| !f.is_repo()).map(|f| f.name());
                let rows = vis
                    .iter()
                    .enumerate()
                    .map(|(row, &i)| {
                        let t = &self.topics[i];
                        // title() is the LLM name on top; show the summary below.
                        // When there's no name the title is already the summary, so
                        // put the keyphrases below instead of duplicating it.
                        let line = if t.name.is_some() {
                            t.summary.clone().or_else(|| t.units.iter().find_map(|u| u.summary.clone())).unwrap_or_default()
                        } else {
                            t.label.clone()
                        };
                        // repos on top, people below; the active facet inverted.
                        Row::new(vec![
                            two_line(t.title().to_string(), line, theme::FG),
                            Cell::from(Text::from(vec![
                                Line::from(styled_names(&t.repos, repo_active, theme::FG, 4)),
                                Line::from(styled_names(&t.authors, person_active, theme::FG, 4)),
                            ])),
                            day_strip(&dailies[row], cap),
                            Cell::from(t.units.len().to_string()).style(dim),
                        ])
                        .height(2)
                    })
                    .collect();
                (
                    vec!["".into(), "REPOS · ACCOUNTS".into(), week_header(start, gmax), "UNITS".into()],
                    vec![Constraint::Min(20), Constraint::Length(32), Constraint::Length(TL_DAYS as u16 + 3), Constraint::Length(5)],
                    rows,
                )
            }
            View::Work => {
                let repo_active = self.filter.as_ref().filter(|f| f.is_repo()).map(|f| f.name());
                let rows = self
                    .visible_work()
                    .into_iter()
                    .map(|i| {
                        let u = &self.work[i];
                        let (primary, secondary) = unit_lines(u);
                        let repo_style = if repo_active == Some(u.repo.as_str()) {
                            Style::new().fg(theme::ACCENT).add_modifier(Modifier::BOLD)
                        } else {
                            dim
                        };
                        Row::new(vec![
                            Cell::from(u.when.clone()).style(dim),
                            type_cell(u.kind),
                            Cell::from(u.repo.clone()).style(repo_style),
                            two_line(primary, secondary, theme::FG),
                            state_cell(u),
                        ])
                        .height(2)
                    })
                    .collect();
                (
                    ["WHEN", "TYPE", "REPO", "", "STATE"].map(String::from).to_vec(),
                    vec![Constraint::Length(11), Constraint::Length(8), Constraint::Length(12), Constraint::Min(20), Constraint::Length(8)],
                    rows,
                )
            }
            View::Search => {
                let rows = self
                    .results
                    .iter()
                    .filter_map(|id| self.doc_by_id.get(id))
                    .map(|&i| {
                        let d = &self.docs[i];
                        let (k, title) = doc_kind_title(d);
                        Row::new(vec![
                            type_cell(k),
                            Cell::from(if d.meta.repo.is_empty() { "—".into() } else { d.meta.repo.clone() }).style(dim),
                            Cell::from(title).style(Style::new().fg(kind_color(k))),
                        ])
                    })
                    .collect();
                (["TYPE", "REPO", ""].map(String::from).to_vec(), vec![Constraint::Length(8), Constraint::Length(12), Constraint::Min(20)], rows)
            }
            View::Status => (vec![], vec![], vec![]),
        }
    }

    fn detail_lines(&self) -> String {
        match self.view {
            View::Work => self.visible_work().get(self.sel).and_then(|&i| self.work.get(i)).map(|u| self.unit_detail(u)).unwrap_or_default(),
            View::Search => self.results.get(self.sel).and_then(|id| self.doc_by_id.get(id)).map(|&i| doc_detail(&self.docs[i])).unwrap_or_default(),
            _ => String::new(),
        }
    }

    fn unit_detail(&self, u: &Unit) -> String {
        match u.kind {
            Kind::Session => match u.session_id.as_ref().and_then(|s| self.sess_by_id.get(s)) {
                Some(&i) => self.session_detail(&self.sessions[i]),
                None => u.title.clone(),
            },
            _ => u.doc_id.and_then(|id| self.doc_by_id.get(&id)).map(|&i| doc_detail(&self.docs[i])).unwrap_or_default(),
        }
    }

    fn session_detail(&self, s: &Session) -> String {
        let mut o = format!(
            "session {} · {}\n{} → {}\n\neffort {}\n{} prompts · {} assistant · {} thinking · {} tool calls\n",
            short(&s.id),
            s.repo,
            day(&s.started),
            day(&s.ended),
            crate::view::meter(s.struggle),
            s.prompts,
            s.assistant,
            s.thinking,
            s.tools,
        );
        for line in [crate::view::usage_line(s), crate::view::tools_line(s)].into_iter().flatten() {
            o.push_str(&line);
            o.push('\n');
        }
        if let Some(sum) = &s.summary {
            o.push_str(&format!("summary: {sum}\n"));
        }
        if let Some(pr) = &s.linked_pr {
            o.push_str(&format!("linked PR: {pr}\n"));
        }
        if !s.files.is_empty() {
            o.push_str(&format!("files: {}\n", s.files.iter().take(10).cloned().collect::<Vec<_>>().join(", ")));
        }
        o.push_str(&format!("\nask:\n{}\n", s.ask));
        // a short representative arc: the session's user prompts
        let prompts: Vec<&Doc> = self
            .docs
            .iter()
            .filter(|d| d.meta.session_id == s.id && d.meta.kind == "user_prompt")
            .collect();
        if prompts.len() > 1 {
            o.push_str("\nturns:\n");
            for d in prompts.iter().take(8) {
                o.push_str(&format!("· {}\n", first_line(&d.text)));
            }
        }
        o
    }

    /// Status view: a totals/freshness/autostart header, the tokens & tools
    /// time-series panel, then two breakdown tables — docs · GitHub objects ·
    /// sessions per repo and per account.
    fn draw_status(&self, f: &mut Frame, area: Rect) {
        let [head, stats, cols] =
            Layout::vertical([Constraint::Length(6), Constraint::Length(9), Constraint::Min(0)]).areas(area);
        f.render_widget(
            Paragraph::new(self.status_head())
                .wrap(Wrap { trim: false })
                .block(Block::bordered().border_style(Style::new().fg(theme::BORDER)).title(" synty ")),
            head,
        );
        f.render_widget(
            Paragraph::new(self.stats_panel())
                .block(Block::bordered().border_style(Style::new().fg(theme::BORDER)).title(" tokens & tools · 4 weeks ")),
            stats,
        );
        let [upper, lower] = Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(cols);
        let [rcol, ucol] = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(upper);
        f.render_widget(facet_table("Repos", &self.status.by_repo), rcol);
        f.render_widget(facet_table("Accounts", &self.status.by_user), ucol);
        let [tcol, mcol] = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(lower);
        let mut ts = TableState::default();
        if !self.status.by_tool.is_empty() {
            ts.select(Some(self.sel.min(self.status.by_tool.len() - 1)));
        }
        f.render_stateful_widget(tools_table(&self.status.by_tool), tcol, &mut ts);
        f.render_widget(models_table(&self.status.by_model), mcol);
        if let Some(p) = &self.tool_drill {
            self.draw_tool_overlay(f, area, p);
        }
    }

    /// The tool drill-down: everything the envelopes hold about one tool —
    /// volume, latency, the day strip, argument-key frequencies with common
    /// values for the enum-ish keys, and the most recent invocations.
    fn draw_tool_overlay(&self, f: &mut Frame, full: Rect, p: &units::ToolProfile) {
        let area = if full.width < 100 {
            full
        } else {
            let [_, r] = Layout::horizontal([Constraint::Percentage(20), Constraint::Percentage(80)]).areas(full);
            r
        };
        f.render_widget(Clear, area);
        let block = Block::bordered()
            .border_style(Style::new().fg(theme::ACCENT))
            .title(format!(" {} · {} ", p.name, p.agent));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let dim = Style::new().fg(theme::DIM);
        let fg = Style::new().fg(theme::FG);
        let mut lines: Vec<Line> = Vec::new();
        let mut head = format!(
            "{} calls · {} errors",
            crate::view::fmt_tokens(p.calls),
            p.errs,
        );
        if p.timed > 0 {
            head.push_str(&format!(" · result p50 {}ms · p95 {}ms ({} timed)", p.p50_ms, p.p95_ms, p.timed));
        }
        if p.input_p95 > 0 {
            head.push_str(&format!(" · input p50 {} / p95 {} chars", p.input_p50, p.input_p95));
        }
        if p.chars > 0 {
            head.push_str(&format!(" · context ~{} tok", crate::view::fmt_tokens(p.chars / 4)));
        }
        lines.push(Line::from(Span::styled(head, fg)));
        lines.push(Line::from(""));

        // The 4-week day strip, same visual language as everywhere else.
        let by_num: Vec<(i32, u64)> = p.days.iter().filter_map(|(d, n)| Some((day_num(d)?, *n))).collect();
        if let Some(gmax) = by_num.iter().map(|(d, _)| *d).max() {
            let start = week_start(gmax);
            let mut days = [0u64; TL_DAYS];
            for (d, n) in &by_num {
                let off = d - start;
                if (0..TL_DAYS as i32).contains(&off) {
                    days[off as usize] += n;
                }
            }
            let cap = days.iter().copied().max().unwrap_or(0);
            lines.push(Line::from(Span::styled(week_header(start, gmax), dim)));
            lines.push(Line::from(strip_spans(&days, cap)));
            lines.push(Line::from(""));
        }

        if !p.arg_keys.is_empty() {
            lines.push(Line::from(Span::styled("args (share of calls):", dim)));
            for (key, n) in p.arg_keys.iter().take(10) {
                let pct = 100 * n / p.calls.max(1);
                let mut spans = vec![Span::styled(format!("  {key:<22} {n:>6} ({pct:>3}%)"), fg)];
                if let Some((_, tops)) = p.arg_tops.iter().find(|(k, _)| k == key) {
                    let vals: Vec<String> = tops.iter().map(|(v, c)| format!("{v}×{c}")).collect();
                    spans.push(Span::styled(format!("  {}", vals.join(" · ")), dim));
                }
                lines.push(Line::from(spans));
            }
            lines.push(Line::from(""));
        }
        if !p.samples.is_empty() {
            lines.push(Line::from(Span::styled("recent:", dim)));
            for s in &p.samples {
                lines.push(Line::from(Span::styled(format!("  {s}"), dim)));
            }
        }
        f.render_widget(Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }), inner);
    }

    /// The stats panel: one day-strip row per metric over the last four
    /// Mon-aligned weeks (the same window and shading as the Topics activity
    /// column), with the window total on the right. Each row shades against
    /// its own peak day — the scales differ by orders of magnitude.
    fn stats_panel(&self) -> Text<'static> {
        let c = &self.cache;
        if c.stats_gmax == 0 {
            return Text::from(Line::from(Span::styled("no usage data yet", Style::new().fg(theme::DIM))));
        }
        let dim = Style::new().fg(theme::DIM);
        let mut lines = vec![Line::from(Span::styled(
            format!("{:9}{}", "", week_header(c.stats_start, c.stats_gmax)),
            dim,
        ))];
        for (label, days, total) in &c.stats_rows {
            let mut spans = vec![Span::styled(format!("{label:<9}"), dim)];
            let cap = days.iter().copied().max().unwrap_or(0);
            spans.extend(strip_spans(days, cap));
            spans.push(Span::styled(
                format!("  {}", crate::view::fmt_tokens(*total)),
                Style::new().fg(theme::FG).add_modifier(Modifier::BOLD),
            ));
            lines.push(Line::from(spans));
        }
        Text::from(lines)
    }

    /// The Status header: totals, freshness, the autostart toggle state, and the
    /// per-kind doc composition.
    fn status_head(&self) -> Text<'static> {
        let s = &self.status;
        let dim = Style::new().fg(theme::DIM);
        let strong = |n: usize, c: Color| Span::styled(n.to_string(), Style::new().fg(c).add_modifier(Modifier::BOLD));
        let kinds = {
            let mut sp = vec![Span::styled("kinds  ", dim)];
            for (i, (k, n)) in s.by_kind.iter().take(8).enumerate() {
                if i > 0 {
                    sp.push(Span::styled(" · ", Style::new().fg(theme::BORDER)));
                }
                sp.push(Span::styled(format!("{k}({n})"), Style::new().fg(theme::FG)));
            }
            Line::from(sp)
        };
        Text::from(vec![
            Line::from(vec![
                strong(s.docs, theme::FG),
                Span::styled(" docs · ", dim),
                strong(s.github, theme::GITHUB),
                Span::styled(" github · ", dim),
                strong(s.sessions, theme::SESSION),
                Span::styled(" sessions", dim),
            ]),
            Line::from(vec![
                Span::styled(
                    format!(
                        "newest {} · indexed {} · tracked {}",
                        if s.newest_ts.is_empty() { "—" } else { &s.newest_ts },
                        s.last_indexed.as_deref().unwrap_or("never"),
                        s.last_tracked.as_deref().unwrap_or("never"),
                    ),
                    dim,
                ),
                if s.stale {
                    Span::styled("  ⚠ stale — events newer than index", Style::new().fg(theme::CLOSED).add_modifier(Modifier::BOLD))
                } else {
                    Span::raw("")
                },
            ]),
            Line::from(vec![
                Span::styled("autostart[a]  ", dim),
                if self.autostart {
                    Span::styled("ON", Style::new().fg(theme::SAGE).add_modifier(Modifier::BOLD))
                } else {
                    Span::styled("OFF", Style::new().fg(theme::CLOSED).add_modifier(Modifier::BOLD))
                },
            ]),
            kinds,
        ])
    }

    fn footer(&self) -> String {
        let keys = match self.view {
            View::Search => match &self.engine {
                Engine::Loading => "loading model…",
                Engine::Searching => "searching…",
                Engine::Err(e) => return format!("  {e}"),
                Engine::Ready => "type · Enter search · ↑↓ results · Esc clear",
            },
            _ if self.drill_unit => "↑↓ units · Esc/h close · q quit",
            _ if self.drill_topic.is_some() => "↑↓ units · Enter detail · Esc/h back · q quit",
            View::Topics => "↑↓ move · Enter drill · Esc reset · Tab cycle · q quit",
            _ if self.tool_drill.is_some() => "Esc/h close · q quit",
            View::Status => "↑↓ tools · Enter inspect · a autostart · Tab cycle · q quit",
            View::Work => "↑↓ move · Esc reset · Tab cycle · q quit",
        };
        format!("  {keys}")
    }
}

// ── row + detail formatting (free fns; 'static lines) ────────────────────────

fn kind_tag(k: Kind) -> &'static str {
    match k {
        Kind::Session => "session",
        Kind::Pr => "pr",
        Kind::Issue => "issue",
    }
}
fn kind_color(k: Kind) -> Color {
    match k {
        Kind::Session => theme::SESSION,
        Kind::Pr | Kind::Issue => theme::GITHUB,
    }
}

/// A two-line table cell: `primary` in `color` on top, `secondary` dimmed below.
fn two_line(primary: String, secondary: String, color: Color) -> Cell<'static> {
    Cell::from(Text::from(vec![
        Line::from(Span::styled(primary, Style::new().fg(color))),
        Line::from(Span::styled(secondary, Style::new().fg(theme::DIM))),
    ]))
}

/// `names` as styled spans, comma-joined: the active facet inverted (ACCENT
/// bold), the rest in `base`; a dim "—" when empty. Caps at `max` names — the
/// shared highlighter for the topic column and the drill overlay's facet lines.
fn styled_names(names: &[String], active: Option<&str>, base: Color, max: usize) -> Vec<Span<'static>> {
    if names.is_empty() {
        return vec![Span::styled("—", Style::new().fg(theme::DIM))];
    }
    let mut spans = Vec::new();
    for (i, n) in names.iter().take(max).enumerate() {
        if i > 0 {
            spans.push(Span::styled(", ", Style::new().fg(theme::DIM)));
        }
        let style = if active == Some(n.as_str()) {
            Style::new().fg(theme::ACCENT).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(base)
        };
        spans.push(Span::styled(n.clone(), style));
    }
    spans
}

/// A Status breakdown table: name + sessions/github/docs columns, a dim "·" for
/// zeros so the meaningful counts stand out. Rows arrive pre-sorted (most docs).
// Status-table convention: the first numeric column is the sort key — Repos
// and Accounts sort by DOCS, Tools by ~TOK, Models by OUT — and it alone
// wears the standout color; every other numeric column is dim. Names stay
// foreground, zeros are a dim "·" everywhere.
fn facet_table(title: &str, rows: &[crate::view::Tally]) -> Table<'static> {
    let dim = Style::new().fg(theme::DIM);
    let num = |n: u64, c: Color| {
        if n == 0 {
            Cell::from("·").style(dim)
        } else {
            Cell::from(crate::view::fmt_tokens(n)).style(Style::new().fg(c))
        }
    };
    let body: Vec<Row> = rows
        .iter()
        .map(|t| {
            Row::new(vec![
                Cell::from(t.name.clone()).style(Style::new().fg(theme::FG)),
                num(t.docs as u64, theme::SESSION), // the sort key stands out
                num(t.sessions as u64, theme::DIM),
                num(t.github as u64, theme::DIM),
                num(t.tok_out, theme::DIM),
                num(t.tools, theme::DIM),
            ])
        })
        .collect();
    Table::new(body, [Constraint::Min(8), Constraint::Length(6), Constraint::Length(5), Constraint::Length(5), Constraint::Length(7), Constraint::Length(6)])
        .header(Row::new(["", "DOCS", "SESS", "GH", "TOK", "TOOLS"].map(Cell::from)).style(dim.add_modifier(Modifier::BOLD)))
        .block(Block::bordered().border_style(Style::new().fg(theme::BORDER)).title(format!(" {title} ({}) ", rows.len())))
}

/// The fleet-wide tool mix: which tools the sessions actually call, how
/// often, and how many calls errored (error attribution is per-name where the
/// source reports it — Claude's tool_result.is_error; codex reports none).
fn tools_table(rows: &[crate::view::ToolTally]) -> Table<'static> {
    let dim = Style::new().fg(theme::DIM);
    let body: Vec<Row> = rows
        .iter()
        .map(|t| {
            Row::new(vec![
                Cell::from(t.name.clone()).style(Style::new().fg(theme::FG)),
                if t.chars == 0 {
                    Cell::from("·").style(dim)
                } else {
                    Cell::from(format!("~{}", crate::view::fmt_tokens(t.est_tokens()))).style(Style::new().fg(theme::SESSION))
                },
                Cell::from(crate::view::fmt_tokens(t.calls)).style(dim),
                if t.errs == 0 {
                    Cell::from("·").style(dim)
                } else {
                    Cell::from(t.errs.to_string()).style(dim)
                },
                Cell::from(t.agent.clone()).style(dim),
            ])
        })
        .collect();
    Table::new(body, [Constraint::Min(8), Constraint::Length(7), Constraint::Length(6), Constraint::Length(5), Constraint::Length(14)])
        .header(Row::new(["", "~TOK", "CALLS", "ERR", "AGENT"].map(Cell::from)).style(dim.add_modifier(Modifier::BOLD)))
        .row_highlight_style(Style::new().bg(theme::HILITE).add_modifier(Modifier::BOLD))
        .highlight_symbol("▌")
        .block(Block::bordered().border_style(Style::new().fg(theme::BORDER)).title(format!(" Tools ({}) · Enter inspects ", rows.len())))
}

/// The fleet's token spend per model — the four classes plus deduped turns.
/// Codex sessions report no model, so their share rides under "codex".
fn models_table(rows: &[units::ModelUsage]) -> Table<'static> {
    let dim = Style::new().fg(theme::DIM);
    let tok = |n: u64, c: Color| {
        if n == 0 {
            Cell::from("·").style(dim)
        } else {
            Cell::from(crate::view::fmt_tokens(n)).style(Style::new().fg(c))
        }
    };
    let body: Vec<Row> = rows
        .iter()
        .map(|m| {
            Row::new(vec![
                Cell::from(m.model.clone()).style(Style::new().fg(theme::FG)),
                tok(m.tok_out, theme::SESSION), // the sort key stands out
                tok(m.tok_in, theme::DIM),
                tok(m.cache_read, theme::DIM),
                tok(m.cache_create, theme::DIM),
                if m.turns == 0 { Cell::from("·").style(dim) } else { Cell::from(m.turns.to_string()).style(dim) },
            ])
        })
        .collect();
    Table::new(
        body,
        [Constraint::Min(12), Constraint::Length(7), Constraint::Length(7), Constraint::Length(7), Constraint::Length(7), Constraint::Length(6)],
    )
    .header(Row::new(["", "OUT", "IN", "CACHE-R", "CACHE-W", "TURNS"].map(Cell::from)).style(dim.add_modifier(Modifier::BOLD)))
    .block(Block::bordered().border_style(Style::new().fg(theme::BORDER)).title(format!(" Models ({}) ", rows.len())))
}

/// (primary, secondary) text for a unit row: the one-line summary on top when
/// present (for sessions, PRs, and issues alike), with the title/ask as context
/// below. PR/issue status moves to its own column, so it's not a line here.
fn unit_lines(u: &Unit) -> (String, String) {
    match &u.summary {
        Some(s) => (s.clone(), u.title.clone()),
        None => (u.title.clone(), u.outcome.clone()),
    }
}

/// The three most recent weekly buckets as [prior, last, this].
fn last3(activity: &[u64]) -> [u64; 3] {
    let n = activity.len();
    let g = |i: usize| n.checked_sub(i).and_then(|x| activity.get(x)).copied().unwrap_or(0);
    [g(3), g(2), g(1)]
}

/// Grayscale activity ramp, darkest (idle) → brightest (busiest week).
const SHADES: [Color; 5] = [
    Color::Rgb(0x33, 0x35, 0x3e),
    Color::Rgb(0x60, 0x62, 0x6e),
    Color::Rgb(0x95, 0x97, 0xa4),
    Color::Rgb(0xc6, 0xc8, 0xd2),
    Color::Rgb(0xf0, 0xf1, 0xf5),
];

/// Shade level 0..4 for a week's count against the global max.
fn shade(count: u64, gmax: u64) -> Color {
    let lvl = if gmax == 0 || count == 0 { 0 } else { (((count * 4) + gmax - 1) / gmax).min(4) as usize };
    SHADES[lvl]
}

/// Days from the CE epoch for a YYYY-MM-DD day.
fn day_num(day: &str) -> Option<i32> {
    NaiveDate::parse_from_str(day, "%Y-%m-%d").ok().map(|d| d.num_days_from_ce())
}

/// The oldest visible Monday (num_days_from_ce): the Monday of `gmax`'s calendar
/// week, back three more weeks — so the strip shows four Mon-Sun weeks ending
/// with the week of the most recent activity.
fn week_start(gmax: i32) -> i32 {
    let dow = NaiveDate::from_num_days_from_ce_opt(gmax).map(|d| d.weekday().num_days_from_monday() as i32).unwrap_or(0);
    gmax - dow - (TL_DAYS as i32 - 7)
}

/// Header for the activity column: each week's Monday date (M/D), left-aligned in
/// an 8-char slot so the label sits over the first block of its week.
fn week_header(start: i32, gmax: i32) -> String {
    if gmax == 0 {
        return String::new();
    }
    let label = |w: i32| match NaiveDate::from_num_days_from_ce_opt(start + w * 7) {
        Some(d) => format!("{}/{}", d.month(), d.day()),
        None => String::new(),
    };
    format!("{:<8}{:<8}{:<8}{:<7}", label(0), label(1), label(2), label(3))
}

/// A topic's per-day unit counts (from its precomputed day numbers) over the
/// four calendar weeks beginning at `start` (num_days_from_ce of the oldest
/// Monday), oldest day first.
fn daily(days: &[i32], start: i32) -> [u64; TL_DAYS] {
    let mut d = [0u64; TL_DAYS];
    for &day in days {
        let off = day - start;
        if (0..TL_DAYS as i32).contains(&off) {
            d[off as usize] += 1;
        }
    }
    d
}

/// One block per day, shaded by that day's activity, grouped into weeks with a
/// thin `│` divider every 7 days. Newest day on the right. Rendered two rows
/// tall (the same strip on both lines) so each day is a taller, easier-to-compare
/// column that lines up with the two-line title/repos cells.
fn day_strip(days: &[u64; TL_DAYS], cap: u64) -> Cell<'static> {
    let spans = strip_spans(days, cap);
    Cell::from(Text::from(vec![Line::from(spans.clone()), Line::from(spans)]))
}

/// The day-strip spans themselves — shared between the two-line table cell
/// above and the Status stats panel's single-line rows.
fn strip_spans(days: &[u64; TL_DAYS], cap: u64) -> Vec<Span<'static>> {
    let mut spans: Vec<Span> = Vec::with_capacity(TL_DAYS + TL_DAYS / 7);
    for (d, &n) in days.iter().enumerate() {
        if d > 0 && d % 7 == 0 {
            spans.push(Span::styled("│", Style::new().fg(theme::BORDER)));
        }
        spans.push(if n == 0 {
            Span::styled("·", Style::new().fg(theme::HILITE))
        } else {
            Span::styled("█", Style::new().fg(shade(n, cap)))
        });
    }
    spans
}

fn type_cell(k: Kind) -> Cell<'static> {
    Cell::from(kind_tag(k)).style(Style::new().fg(kind_color(k)))
}

/// The per-unit state column: a session's 5-dot effort meter, or a PR/issue's
/// status (OPEN/MERGED/CLOSED) colored. One column carries both.
fn state_cell(u: &Unit) -> Cell<'static> {
    match u.kind {
        Kind::Session => Cell::from(crate::view::meter(u.struggle)).style(Style::new().fg(theme::ACCENT)),
        _ => {
            let st = u.outcome.to_uppercase();
            let color = match st.as_str() {
                "OPEN" => theme::SAGE,
                "MERGED" => theme::MERGED,
                "CLOSED" => theme::CLOSED,
                _ => theme::DIM,
            };
            Cell::from(st).style(Style::new().fg(color))
        }
    }
}

fn doc_kind_title(d: &Doc) -> (Kind, String) {
    match d.meta.kind.as_str() {
        "pull_request" => (Kind::Pr, format!("{}#{} {}", d.meta.repo, d.meta.number.unwrap_or(0), first_line(&d.text))),
        "issue" => (Kind::Issue, format!("{}#{} {}", d.meta.repo, d.meta.number.unwrap_or(0), first_line(&d.text))),
        _ => (Kind::Session, format!("{} {}", short(&d.meta.session_id), first_line(&d.text))),
    }
}

fn doc_detail(d: &Doc) -> String {
    let mut o = match d.meta.kind.as_str() {
        "pull_request" | "issue" => {
            let url = d.meta.url.clone().unwrap_or_default();
            format!("{} {}#{} [{}]\n{}\n\n", d.meta.kind, d.meta.repo, d.meta.number.unwrap_or(0), d.meta.state.clone().unwrap_or_default(), url)
        }
        _ => format!("{} · {} · {}\n\n", d.meta.kind, if d.meta.repo.is_empty() { "—" } else { &d.meta.repo }, d.meta.ts),
    };
    o.push_str(&d.text);
    o
}

fn day(ts: &str) -> String {
    ts.split('T').next().unwrap_or("").to_string()
}

fn spawn_search(model_id: String) -> (Sender<SearchCmd>, Receiver<SearchMsg>) {
    let (qtx, qrx) = channel::<SearchCmd>();
    let (rtx, rrx) = channel::<SearchMsg>();
    std::thread::spawn(move || {
        use crate::encode::Encoder;
        use next_plaid::{MmapIndex, SearchParameters};
        let (mut enc, mut idx) = match (Encoder::load(&model_id), MmapIndex::load(&readmodel::index_dir().to_string_lossy())) {
            (Ok(e), Ok(i)) => (e, i),
            _ => {
                let _ = rtx.send(SearchMsg::Err("index/model unavailable (run `index`)".into()));
                return;
            }
        };
        let _ = rtx.send(SearchMsg::Ready);
        while let Ok(cmd) = qrx.recv() {
            let q = match cmd {
                SearchCmd::Reload => {
                    match MmapIndex::load(&readmodel::index_dir().to_string_lossy()) {
                        Ok(new) => {
                            idx = new;
                            let _ = rtx.send(SearchMsg::Reloaded);
                        }
                        Err(e) => {
                            let _ = rtx.send(SearchMsg::Err(format!("reload index: {e}")));
                        }
                    }
                    continue;
                }
                SearchCmd::Query(q) => q,
            };
            let _ = rtx.send(SearchMsg::Searching);
            match enc.encode_query(&q) {
                Ok(qe) => {
                    let params = SearchParameters { top_k: 30, ..Default::default() };
                    match idx.search(&qe, &params, None) {
                        Ok(r) => {
                            let _ = rtx.send(SearchMsg::Results(r.passage_ids));
                        }
                        Err(e) => {
                            let _ = rtx.send(SearchMsg::Err(e.to_string()));
                        }
                    }
                }
                Err(e) => {
                    let _ = rtx.send(SearchMsg::Err(e.to_string()));
                }
            }
        }
    });
    (qtx, rrx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::Unit;
    use crate::Meta;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn doc(id: i64, kind: &str, repo: &str, sid: &str, text: &str) -> Doc {
        Doc {
            id,
            text: text.into(),
            meta: Meta {
                source: if kind == "pull_request" { "github".into() } else { "agent".into() },
                kind: kind.into(),
                repo: repo.into(),
                author: String::new(),
                session_id: sid.into(),
                ts: "2026-05-31T10:00:00Z".into(),
                number: Some(7),
                url: None,
                state: Some("OPEN".into()),
                labels: vec![],
            },
        }
    }

    fn app() -> App {
        let docs = vec![doc(0, "pull_request", "sie-web", "", "fix docs search"), doc(1, "user_prompt", "sie", "S1", "add OCR adapter")];
        let doc_by_id = docs.iter().enumerate().map(|(i, d)| (d.id, i)).collect();
        let session = Session {
            id: "S1".into(),
            repo: "sie".into(),
            started: "2026-05-31T09:00:00Z".into(),
            ended: "2026-05-31T10:00:00Z".into(),
            ask: "add OCR adapter".into(),
            prompts: 3,
            assistant: 5,
            thinking: 2,
            tools: 9,
            files: vec!["ocr.rs".into()],
            linked_pr: None,
            topic: Some(0),
            struggle: 0.6,
            tok_in: 4_200,
            tok_out: 18_900,
            cache_read: 310_000,
            cache_create: 12_000,
            usage_turns: 7,
            tools_by_name: vec![("Bash".into(), 5, 1, 2_000), ("Edit".into(), 3, 0, 800)],
            tool_err: 1,
            by_model: vec![units::ModelUsage { model: "claude-fable-5".into(), tok_in: 4_200, tok_out: 18_900, cache_read: 310_000, cache_create: 12_000, turns: 7 }],
            source: "claude_code".into(),
            summary: Some("Added an OCR adapter to the sie pipeline.".into()),
            author: String::new(),
        };
        let work = vec![
            Unit { kind: Kind::Session, when: "2026-05-31".into(), repo: "sie".into(), title: "add OCR adapter".into(), outcome: "1 files".into(), summary: Some("Added an OCR adapter to the sie pipeline.".into()), topic: Some(0), rank: i64::MAX, dup: false, struggle: 0.6, author: String::new(), doc_id: None, session_id: Some("S1".into()) },
            Unit { kind: Kind::Pr, when: "2026-05-31".into(), repo: "sie-web".into(), title: "sie-web#7 fix docs search".into(), outcome: "OPEN".into(), summary: None, topic: Some(0), rank: i64::MAX, dup: false, struggle: 0.0, author: "alice".into(), doc_id: Some(0), session_id: None },
        ];
        let topics = vec![TopicUnits { id: 0, cache_key: "t0".into(), label: "ocr, docs".into(), units: work.iter().map(clone_unit).collect(), last_active: "2026-05-31".into(), activity: vec![1, 0, 2, 3], mix: (1, 5, 3), repos: vec!["sie".into(), "sie-web".into()], authors: vec!["alice".into()], summary: Some("OCR adapter work across the sie pipeline and docs search.".into()), name: Some("OCR & Document Extraction".into()), span: Some(("2026-05-29".into(), "2026-05-31".into())) }];
        App {
            cache: ViewCache::build(&topics, &work, &test_day_stats()),
            docs,
            doc_by_id,
            sess_by_id: [("S1".to_string(), 0)].into_iter().collect(),
            sessions: vec![session],
            work,
            topics,
            status: crate::view::Status { docs: 2, github: 1, sessions: 1, by_kind: vec![("user_prompt".into(), 1), ("pull_request".into(), 1)], by_repo: vec![crate::view::Tally { name: "sie".into(), docs: 1, github: 0, sessions: 1, tok_out: 18_900, tools: 8 }], by_user: vec![crate::view::Tally { name: "alice".into(), docs: 1, github: 1, sessions: 0, tok_out: 0, tools: 0 }], by_tool: vec![crate::view::ToolTally { name: "Bash".into(), agent: "claude".into(), calls: 5, errs: 1, chars: 81_200 }, crate::view::ToolTally { name: "Edit".into(), agent: "claude".into(), calls: 3, errs: 0, chars: 0 }], by_model: vec![units::ModelUsage { model: "claude-fable-5".into(), tok_in: 4_200, tok_out: 18_900, cache_read: 310_000, cache_create: 12_000, turns: 7 }], newest_ts: "2026-05-31".into(), last_indexed: None, last_tracked: None, autostart: false, stale: false },
            view: View::Topics,
            sel: 0,
            drill_topic: None,
            drill_unit: false,
            tool_drill: None,
            filter: None,
            query: String::new(),
            results: vec![],
            engine: Engine::Loading,
            autostart: false,
            qtx: None,
            rrx: None,
            quit: false,
            bucket: ".synty".into(),
            freshen: None,
            freshen_note: None,
            last_freshen: Some(std::time::Instant::now()),
            btx: channel::<Bundle>().0,
            brx: channel::<Bundle>().1,
            reload_pending: false,
        }
    }

    fn test_day_stats() -> HashMap<String, units::DayStat> {
        let mut m = HashMap::new();
        m.insert("2026-05-31".to_string(), units::DayStat { tok_in: 4_200, tok_out: 18_900, cache_read: 310_000, cache_create: 12_000, tools: 8, sessions: 1 });
        m.insert("2026-05-30".to_string(), units::DayStat { tok_out: 2_000, tools: 3, sessions: 1, ..Default::default() });
        m
    }

    fn clone_unit(u: &Unit) -> Unit {
        Unit { kind: u.kind, when: u.when.clone(), repo: u.repo.clone(), title: u.title.clone(), outcome: u.outcome.clone(), summary: u.summary.clone(), topic: u.topic, rank: u.rank, dup: u.dup, struggle: u.struggle, author: u.author.clone(), doc_id: u.doc_id, session_id: u.session_id.clone() }
    }

    // app() plus a second, unrelated topic and its work unit, so filters have
    // something to narrow against (infra/bob vs the fixture's sie/alice).
    fn app2() -> App {
        let mut a = app();
        let infra = Unit { kind: Kind::Pr, when: "2026-05-30".into(), repo: "infra".into(), title: "infra#3 terraform runner".into(), outcome: "MERGED".into(), summary: Some("Terraform runner IAM work.".into()), topic: Some(1), rank: i64::MAX, dup: false, struggle: 0.0, author: "bob".into(), doc_id: None, session_id: None };
        a.work.push(clone_unit(&infra));
        a.topics.push(TopicUnits {
            id: 1,
            cache_key: "t1".into(),
            label: "infra".into(),
            units: vec![infra],
            last_active: "2026-05-30".into(),
            activity: vec![0, 1, 0, 0],
            mix: (0, 1, 0),
            repos: vec!["infra".into()],
            authors: vec!["bob".into()],
            summary: Some("Terraform runner IAM work.".into()),
            name: Some("Infra & Terraform".into()),
            span: Some(("2026-05-28".into(), "2026-05-30".into())),
        });
        a.cache = ViewCache::build(&a.topics, &a.work, &test_day_stats());
        a
    }

    #[test]
    fn renders_every_view() {
        let mut term = Terminal::new(TestBackend::new(110, 32)).unwrap();
        for v in VIEWS {
            let mut a = app();
            a.view = v;
            term.draw(|f| a.draw(f)).unwrap();
        }
    }

    // The footer's freshness segment narrates the background build's phase,
    // warns when stale, and goes quiet when current.
    #[test]
    fn footer_shows_freshness_state() {
        let mut term = Terminal::new(TestBackend::new(110, 32)).unwrap();
        let text = |a: &App, term: &mut Terminal<TestBackend>| -> String {
            term.draw(|f| a.draw(f)).unwrap();
            term.backend().buffer().content().iter().map(|c| c.symbol()).collect()
        };
        let mut a = app();
        a.freshen_note = Some("⟳ encoding 120/470".into());
        assert!(text(&a, &mut term).contains("encoding 120/470"), "running phase missing");
        a.freshen_note = None;
        a.status.stale = true;
        assert!(text(&a, &mut term).contains("stale"), "stale warning missing");
        a.status.stale = false;
        assert!(text(&a, &mut term).contains("✓ fresh"), "fresh state missing");
    }

    // A hot reload re-finds the drilled topic by its stable key even when the
    // new clustering reordered the topic list.
    #[test]
    fn reload_keeps_the_drilled_topic_by_key() {
        let mut a = app2();
        a.drill_topic = Some(0); // drilled into "t0" (OCR), first in the list
        let mut donor = app2();
        donor.topics.reverse(); // new build orders t1 first
        let b = Bundle {
            docs: donor.docs,
            sessions: donor.sessions,
            work: donor.work,
            topics: donor.topics,
            status: donor.status,
            day_stats: test_day_stats(),
        };
        a.apply(b);
        let vi = a.drill_topic.expect("drill survives the reload");
        let drilled = a.drilled(vi).expect("drilled topic resolves");
        assert_eq!(drilled.cache_key, "t0", "remap follows the stable key, not the position");
    }

    // The nav bar shows every tab's full label, not a clipped "5".
    #[test]
    fn navbar_shows_all_labels() {
        let mut term = Terminal::new(TestBackend::new(110, 32)).unwrap();
        let a = app();
        term.draw(|f| a.draw(f)).unwrap();
        let text: String = term.backend().buffer().content().iter().map(|c| c.symbol()).collect();
        for label in ["Topics", "Work", "Search", "Status"] {
            assert!(text.contains(label), "nav missing {label}");
        }
        // glyph-first, like the freshness cell ("✓ fresh" / "✗ autostart").
        assert!(text.contains("✗ autostart"), "footer missing glyph-first autostart status");
        // breadcrumb-only chrome: no redundant TOPIC header, but functional ones remain
        assert!(text.contains("REPOS · ACCOUNTS"), "topics table missing column headers");
        assert!(!text.contains("TOPIC"), "redundant TOPIC header should be gone");
        assert!(text.contains("synty › Topics"), "breadcrumb missing");
    }

    #[test]
    fn topic_drill_opens_overlay_with_units_and_facets() {
        let mut term = Terminal::new(TestBackend::new(160, 40)).unwrap();
        let mut a = app();
        a.view = View::Topics;
        assert_eq!(a.list_len(), 1); // one topic
        a.on_key(KeyCode::Enter); // drill
        assert_eq!(a.drill_topic, Some(0));
        assert_eq!(a.list_len(), 2); // navigation now moves over its two units
        term.draw(|f| a.draw(f)).unwrap();
        let text: String = term.backend().buffer().content().iter().map(|c| c.symbol()).collect();
        // overlay shows facets (repos/authors) and a member unit's summary
        assert!(text.contains("repos:") && text.contains("accounts:"), "overlay missing facets: {text}");
        assert!(text.contains("OCR adapter to the sie"), "overlay missing unit summary");
        // h restores selection to the drilled topic
        a.on_key(KeyCode::Char('h'));
        assert!(a.drill_topic.is_none());
        assert_eq!(a.sel, 0);
    }

    // Inside a drilled topic, Enter opens the selected unit's detail — the
    // same content Work's right pane shows — the detail follows ↑↓, and
    // Esc peels one layer at a time: detail → overlay → list.
    #[test]
    fn unit_drill_shows_detail_inside_topic_overlay() {
        let mut term = Terminal::new(TestBackend::new(160, 40)).unwrap();
        let mut a = app();
        a.on_key(KeyCode::Enter); // drill the topic
        a.on_key(KeyCode::Enter); // open unit 0 (the session) detail
        assert!(a.drill_unit);
        term.draw(|f| a.draw(f)).unwrap();
        let text: String = term.backend().buffer().content().iter().map(|c| c.symbol()).collect();
        assert!(text.contains("3 prompts · 5 assistant"), "session detail missing: {text}");
        assert!(text.contains("ask:"), "session detail missing the ask");
        // the detail follows the selection: move to the PR unit
        a.on_key(KeyCode::Down);
        term.draw(|f| a.draw(f)).unwrap();
        let text: String = term.backend().buffer().content().iter().map(|c| c.symbol()).collect();
        assert!(!text.contains("3 prompts"), "detail should follow the selection off the session");
        // Esc closes only the detail, then only the overlay
        a.on_key(KeyCode::Esc);
        assert!(!a.drill_unit && a.drill_topic.is_some());
        term.draw(|f| a.draw(f)).unwrap();
        let text: String = term.backend().buffer().content().iter().map(|c| c.symbol()).collect();
        assert!(!text.contains("ask:"), "detail pane should be gone");
        a.on_key(KeyCode::Esc);
        assert!(a.drill_topic.is_none());
    }

    #[test]
    fn view_switch_by_number() {
        let mut a = app();
        a.on_key(KeyCode::Char('2'));
        assert!(matches!(a.view, View::Work));
        a.on_key(KeyCode::Char('4'));
        assert!(matches!(a.view, View::Status));
    }

    // The topics list shows a per-day activity strip with week dividers.
    #[test]
    fn topics_show_day_activity_strip() {
        let mut term = Terminal::new(TestBackend::new(120, 20)).unwrap();
        let mut a = app();
        a.view = View::Topics;
        term.draw(|f| a.draw(f)).unwrap();
        let text: String = term.backend().buffer().content().iter().map(|c| c.symbol()).collect();
        assert!(text.contains("5/25"), "week Monday date header missing (fixture week of 2026-05-25)");
        assert!(text.contains('│'), "week divider missing from activity strip");
        assert!(text.contains("REPOS · ACCOUNTS"), "repos/people column header missing");
        assert!(text.contains("sie, sie-web"), "repos line missing from topics row");
    }

    // Work rows surface the session's one-line summary, not just the ask.
    #[test]
    fn work_rows_show_summary() {
        let mut term = Terminal::new(TestBackend::new(160, 32)).unwrap();
        let mut a = app();
        a.view = View::Work;
        term.draw(|f| a.draw(f)).unwrap();
        let text: String = term.backend().buffer().content().iter().map(|c| c.symbol()).collect();
        assert!(text.contains("synty › Work"), "work breadcrumb missing");
        assert!(text.contains("OCR adapter to the sie"), "work row missing summary text: {text}");
    }

    // Pressing r/p filters the topic list to one repo / person; cycling past the
    // last value clears the filter, and the breadcrumb names the active facet.
    #[test]
    fn repo_person_filter_narrows_topic_list() {
        let mut a = app2();
        // Two distinct topics: sie/sie-web/alice vs infra/bob.
        assert_eq!(a.visible().len(), 2);

        // r selects the first repo facet → the list narrows to the topic it touches.
        a.on_key(KeyCode::Char('r'));
        assert!(a.filter.is_some(), "r should set a repo filter");
        assert_eq!(a.visible().len(), 1, "repo filter should narrow to one topic");
        assert!(a.breadcrumb().contains("repo:"), "breadcrumb should name the repo facet");

        // Cycling through every repo (infra, sie, sie-web) then once more clears it.
        for _ in 0..3 {
            a.on_key(KeyCode::Char('r'));
        }
        assert!(a.filter.is_none(), "cycling past the last repo clears the filter");
        assert_eq!(a.visible().len(), 2);

        // a filters by account and is mutually exclusive with the repo facet.
        a.on_key(KeyCode::Char('a'));
        assert!(a.breadcrumb().contains('@'), "breadcrumb should name the account facet");
        assert_eq!(a.visible().len(), 1, "person filter should narrow to one topic");

        // Drilling while filtered resolves through the visible list, not topics[sel].
        a.sel = 0;
        a.on_key(KeyCode::Enter);
        assert_eq!(a.drill_topic, Some(0), "drill indexes the visible list");
        let drilled = a.drilled(0).expect("a visible topic");
        let want = a.filter.as_ref().unwrap().name();
        assert!(drilled.authors.iter().any(|p| p.as_str() == want), "drilled topic matches the active person filter");
    }

    // r/a drive the filter from the Work view too, not just Topics.
    #[test]
    fn filter_keys_work_in_work_view() {
        let mut a = app2();
        a.view = View::Work;
        a.on_key(KeyCode::Char('r'));
        assert!(a.filter.is_some(), "r sets a filter in the Work view");
        assert!(a.visible_work().len() < a.work.len(), "Work list narrows under the filter");
    }

    // Esc peels back state — drill, then filter — and only quits when clean.
    #[test]
    fn esc_resets_then_quits() {
        let mut a = app2();
        a.on_key(KeyCode::Char('a')); // set an account filter
        assert!(a.filter.is_some());
        a.on_key(KeyCode::Esc);
        assert!(a.filter.is_none() && !a.quit, "Esc clears the filter without quitting");
        a.on_key(KeyCode::Enter); // drill a topic
        assert!(a.drill_topic.is_some());
        a.on_key(KeyCode::Esc);
        assert!(a.drill_topic.is_none() && !a.quit, "Esc exits the drill without quitting");
        a.on_key(KeyCode::Esc);
        assert!(a.quit, "Esc on a clean screen quits");
    }

    // The Status view shows totals, per-repo/account breakdowns, and the
    // autostart state. (The toggle itself has side effects, so it isn't pressed.)
    #[test]
    fn status_view_shows_breakdowns_and_autostart() {
        let mut term = Terminal::new(TestBackend::new(120, 34)).unwrap();
        let mut a = app();
        a.view = View::Status;
        term.draw(|f| a.draw(f)).unwrap();
        let text: String = term.backend().buffer().content().iter().map(|c| c.symbol()).collect();
        assert!(text.contains("docs") && text.contains("github") && text.contains("sessions"), "totals row missing: {text}");
        assert!(text.contains("Repos (1)") && text.contains("Accounts (1)"), "breakdown table titles missing: {text}");
        assert!(text.contains("SESS") && text.contains("GH") && text.contains("DOCS"), "table headers missing");
        assert!(text.contains("sie") && text.contains("alice"), "facet rows missing");
        assert!(text.contains("autostart") && text.contains("OFF"), "autostart state missing");
        // the tokens & tools time-series panel sits above the breakdowns, one
        // strip row per metric with the 4-week total (18.9k + 2k out).
        assert!(text.contains("tokens & tools"), "stats panel missing: {text}");
        assert!(text.contains("tok out") && text.contains("cache r") && text.contains("sessions"), "metric rows missing");
        assert!(text.contains("20.9k"), "window total missing: {text}");
        // segmentation: repo rows carry token/tool spend, the fleet-wide tool
        // mix names tools with their agent, and models get their own table.
        assert!(text.contains("TOK") && text.contains("TOOLS"), "facet spend columns missing: {text}");
        assert!(text.contains("18.9k"), "repo token spend missing");
        assert!(text.contains("Tools (2)") && text.contains("CALLS") && text.contains("Bash"), "tools table missing: {text}");
        assert!(text.contains("AGENT") && text.contains("claude"), "tools agent column missing: {text}");
        assert!(text.contains("~TOK") && text.contains("~20.3k"), "estimated context column missing: {text}");
        assert!(text.contains("Models (1)") && text.contains("claude-fable-5") && text.contains("CACHE-R"), "models table missing: {text}");
        // the toggle uses the keycap convention, not an explanation sentence.
        assert!(text.contains("autostart[a]"), "autostart keycap hint missing: {text}");
        assert!(!text.contains("press a to toggle"), "explanation sentence should be gone");
    }

    // Enter on a Status tool opens its profile overlay — argument-key shares,
    // common values, latency — and Esc peels it like every other drill.
    #[test]
    fn tool_drill_overlay_renders_profile() {
        let mut term = Terminal::new(TestBackend::new(140, 40)).unwrap();
        let mut a = app();
        a.view = View::Status;
        a.tool_drill = Some(units::ToolProfile {
            name: "Bash".into(),
            agent: "claude".into(),
            calls: 3665,
            errs: 106,
            chars: 5_600_000,
            days: [("2026-05-31".to_string(), 12u64)].into_iter().collect(),
            arg_keys: vec![("command".into(), 3665), ("run_in_background".into(), 3665), ("timeout".into(), 420)],
            arg_tops: vec![("run_in_background".into(), vec![("false".into(), 3600), ("true".into(), 65)])],
            p50_ms: 740,
            p95_ms: 12_400,
            timed: 3600,
            input_p50: 180,
            input_p95: 900,
            samples: vec![r#"{"command":"cargo test"}"#.into()],
        });
        term.draw(|f| a.draw(f)).unwrap();
        let text: String = term.backend().buffer().content().iter().map(|c| c.symbol()).collect();
        assert!(text.contains("Bash · claude"), "overlay title missing: {text}");
        assert!(text.contains("3.7k calls") && text.contains("106 errors"), "volume line missing: {text}");
        assert!(text.contains("p50 740ms"), "latency missing");
        assert!(text.contains("context ~1.4M tok"), "context estimate missing: {text}");
        assert!(text.contains("args (share of calls):") && text.contains("command"), "args section missing");
        assert!(text.contains("false×3600"), "common values missing");
        assert!(text.contains("cargo test"), "recent sample missing");
        a.on_key(KeyCode::Esc);
        assert!(a.tool_drill.is_none(), "Esc closes the overlay");
    }

    // The facet bar has a Repos row over an Accounts row, and inverts the active.
    #[test]
    fn facet_bar_lists_and_highlights() {
        let mut a = app2();
        let bar = a.facet_bar(120);
        let repos: String = bar.lines[0].spans.iter().map(|s| s.content.to_string()).collect();
        let accounts: String = bar.lines[1].spans.iter().map(|s| s.content.to_string()).collect();
        assert!(repos.starts_with("Repo[r]:") && repos.contains("sie") && repos.contains("infra"), "repos row: {repos}");
        assert!(accounts.starts_with("Acct[a]:") && accounts.contains("alice") && accounts.contains("bob"), "accounts row: {accounts}");
        // each chip carries a unit count, like the Topics UNITS column.
        assert!(repos.contains("infra("), "repo chip should show a unit count: {repos}");
        // the active facet's chip is inverted (accent background).
        a.filter = Some(Facet::Repo("infra".into()));
        let bar = a.facet_bar(120);
        let chip = bar.lines.iter().flat_map(|l| &l.spans).find(|s| s.content.starts_with("infra")).expect("infra chip");
        assert_eq!(chip.style.bg, Some(theme::ACCENT), "active chip should be inverted");
    }

    // styled_names accents the active facet name and leaves the others in base.
    #[test]
    fn active_facet_name_is_highlighted() {
        let names = vec!["sie".to_string(), "infra".to_string()];
        let spans = styled_names(&names, Some("infra"), theme::FG, 4);
        let infra = spans.iter().find(|s| s.content == "infra").expect("infra span");
        assert_eq!(infra.style.fg, Some(theme::ACCENT), "active name should be accented");
        let sie = spans.iter().find(|s| s.content == "sie").expect("sie span");
        assert_eq!(sie.style.fg, Some(theme::FG), "inactive name keeps the base color");
    }
}
