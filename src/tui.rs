// The human surface, built to design_tui.md: a header breadcrumb, a
// master/detail body, and a context footer; four views (Topics, Work, Search,
// Status) over units of work, with a comparable activity column and the brand
// palette. Session rows are two lines tall: the one-line summary on top, context
// below. The embedding model loads on a background thread (a search actor) so the
// first query is instant and the UI never blocks.

use crate::units::{self, Kind, Session, TopicUnits, Unit};
use crate::{first_line, load_docs, short, Doc, DOCS_PATH, INDEX_PATH};
use anyhow::Result;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Cell, Paragraph, Row, Table, TableState, Tabs, Wrap};
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
}

#[derive(Clone, Copy, PartialEq)]
enum View {
    Topics,
    Work,
    Search,
    Status,
}
const VIEWS: [View; 4] = [View::Topics, View::Work, View::Search, View::Status];
const VIEW_NAMES: [&str; 4] = ["1 Topics", "2 Work", "3 Search", "4 Status"];

enum SearchMsg {
    Ready,
    Searching,
    Results(Vec<i64>),
    Err(String),
}

#[derive(PartialEq)]
enum Engine {
    Loading,
    Ready,
    Searching,
    Err(String),
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
    drill_topic: Option<usize>, // Topics: viewing a topic's units
    query: String,
    results: Vec<i64>, // doc ids
    engine: Engine,
    autostart: bool,
    qtx: Option<Sender<String>>,
    rrx: Option<Receiver<SearchMsg>>,
    quit: bool,
}

pub fn run(model_id: String) -> Result<()> {
    // Gag stderr first: the background model load (and candle/pylate-rs) write
    // device/diagnostic lines to stderr, which would scroll the alternate screen
    // and shove the header off the top. Restored when `_gag` drops.
    let _gag = StderrGag::new();
    let mut app = App::load(model_id);
    let mut term = ratatui::init();
    let _ = term.clear();
    let res = app.run_loop(&mut term);
    ratatui::restore();
    res
}

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
    fn load(model_id: String) -> Self {
        let docs = load_docs(DOCS_PATH).unwrap_or_default();
        let doc_by_id = docs.iter().enumerate().map(|(i, d)| (d.id, i)).collect();
        let sessions = units::sessions().unwrap_or_default();
        let sess_by_id = sessions.iter().enumerate().map(|(i, s)| (s.id.clone(), i)).collect();
        let work = units::units().unwrap_or_default();
        let topics = units::topic_units(12).unwrap_or_default();
        let status = crate::view::status().unwrap_or_else(|_| crate::view::Status {
            docs: docs.len(),
            github: 0,
            sessions: 0,
            by_kind: vec![],
            by_repo: vec![],
            newest_ts: String::new(),
            last_indexed: None,
            last_tracked: None,
        });
        let (qtx, rrx) = spawn_search(model_id);
        Self {
            docs,
            doc_by_id,
            sessions,
            sess_by_id,
            work,
            topics,
            status,
            view: View::Topics,
            sel: 0,
            drill_topic: None,
            query: String::new(),
            results: vec![],
            engine: Engine::Loading,
            autostart: crate::track::autostart_enabled(),
            qtx: Some(qtx),
            rrx: Some(rrx),
            quit: false,
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
        }
        Ok(())
    }

    fn drain_search(&mut self) {
        let Some(rx) = &self.rrx else { return };
        while let Ok(msg) = rx.try_recv() {
            match msg {
                SearchMsg::Ready => self.engine = Engine::Ready,
                SearchMsg::Searching => self.engine = Engine::Searching,
                SearchMsg::Results(ids) => {
                    self.results = ids.into_iter().filter(|id| self.doc_by_id.contains_key(id)).collect();
                    self.sel = 0;
                    self.engine = Engine::Ready;
                }
                SearchMsg::Err(e) => self.engine = Engine::Err(e),
            }
        }
    }

    // ── current list / detail ────────────────────────────────────────────

    fn list_len(&self) -> usize {
        match self.view {
            View::Topics => match self.drill_topic {
                None => self.topics.len(),
                Some(t) => self.topics.get(t).map(|t| t.units.len()).unwrap_or(0),
            },
            View::Work => self.work.len(),
            View::Search => self.results.len(),
            View::Status => 0,
        }
    }

    fn on_key(&mut self, code: KeyCode) {
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
            KeyCode::Char('q') | KeyCode::Esc => {
                if self.drill_topic.is_some() {
                    self.drill_topic = None;
                    self.sel = 0;
                } else {
                    self.quit = true;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => self.move_sel(1),
            KeyCode::Up | KeyCode::Char('k') => self.move_sel(-1),
            KeyCode::Char('g') => self.sel = 0,
            KeyCode::Char('G') => self.sel = self.list_len().saturating_sub(1),
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                if self.view == View::Topics && self.drill_topic.is_none() && !self.topics.is_empty() {
                    self.drill_topic = Some(self.sel);
                    self.sel = 0;
                }
            }
            KeyCode::Left | KeyCode::Char('h') => {
                if self.drill_topic.is_some() {
                    self.drill_topic = None;
                    self.sel = 0;
                }
            }
            _ => {}
        }
    }

    fn search_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc => self.quit = true,
            KeyCode::Char(c) => self.query.push(c),
            KeyCode::Backspace => {
                self.query.pop();
            }
            KeyCode::Enter => {
                if !self.query.trim().is_empty() {
                    if let Some(tx) = &self.qtx {
                        let _ = tx.send(self.query.clone());
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
    }
    fn move_sel(&mut self, d: i32) {
        let n = self.list_len();
        if n > 0 {
            self.sel = (self.sel as i32 + d).clamp(0, n as i32 - 1) as usize;
        }
    }

    // ── render ───────────────────────────────────────────────────────────

    fn draw(&self, f: &mut Frame) {
        let [top, body, footer] =
            Layout::vertical([Constraint::Length(1), Constraint::Min(0), Constraint::Length(1)]).areas(f.area());
        self.draw_header(f, top);
        match self.view {
            View::Status => f.render_widget(self.status_para(), body),
            View::Topics | View::Work | View::Search => self.draw_master_detail(f, body),
        }
        // footer: contextual keys (left) · autostart status (right)
        let auto = if self.autostart { " autostart ✓ " } else { " autostart ✗ " };
        let [fkeys, fauto] =
            Layout::horizontal([Constraint::Min(0), Constraint::Length(auto.chars().count() as u16)]).areas(footer);
        f.render_widget(Line::from(self.footer()).fg(theme::DIM), fkeys);
        f.render_widget(
            Line::from(auto).fg(if self.autostart { theme::SAGE } else { theme::DIM }).right_aligned(),
            fauto,
        );
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
        match (self.view, self.drill_topic) {
            (View::Topics, Some(t)) => format!("synty › Topics › {}", self.topics.get(t).map(|x| x.label.as_str()).unwrap_or("")),
            _ => "synty".to_string(),
        }
    }

    fn draw_master_detail(&self, f: &mut Frame, area: Rect) {
        let split = if self.view == View::Search {
            let [q, s] = Layout::vertical([Constraint::Length(3), Constraint::Min(0)]).areas(area);
            let cursor = if matches!(self.engine, Engine::Ready | Engine::Searching) { "▏" } else { "" };
            f.render_widget(
                Paragraph::new(format!("{}{cursor}", self.query))
                    .block(Block::bordered().border_style(Style::new().fg(theme::BORDER)).title("search")),
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
            .row_highlight_style(Style::new().fg(theme::ACCENT).bg(theme::HILITE).add_modifier(Modifier::BOLD))
            .highlight_symbol("▌")
            .block(Block::bordered().border_style(Style::new().fg(theme::BORDER)).title(self.list_title()));
        f.render_stateful_widget(table, left, &mut ts);

        f.render_widget(
            Paragraph::new(self.detail_lines())
                .wrap(Wrap { trim: false })
                .block(Block::bordered().border_style(Style::new().fg(theme::BORDER)).title("detail")),
            right,
        );
    }

    /// (header, column widths, rows) for the current view's table.
    fn build_table(&self) -> (Vec<&'static str>, Vec<Constraint>, Vec<Row<'static>>) {
        let dim = Style::new().fg(theme::DIM);
        match self.view {
            View::Topics => match self.drill_topic {
                None => {
                    // Shade the 3-week activity on a global scale so topics compare.
                    let gmax = self.topics.iter().map(|t| last3(&t.activity).into_iter().max().unwrap_or(0)).max().unwrap_or(0);
                    let rows = self
                        .topics
                        .iter()
                        .map(|t| {
                            // keyphrase label on top, the latest session summary in the topic below
                            let latest = t.units.iter().find_map(|u| u.summary.clone()).unwrap_or_default();
                            Row::new(vec![
                                two_line(t.label.clone(), latest, theme::FG),
                                activity_cell(last3(&t.activity), gmax),
                                Cell::from(t.units.len().to_string()).style(dim),
                                Cell::from(t.last_active.clone()).style(dim),
                            ])
                            .height(2)
                        })
                        .collect();
                    (
                        vec!["TOPIC", "ACTIVITY", "UNITS", "LAST"],
                        vec![Constraint::Min(20), Constraint::Length(8), Constraint::Length(5), Constraint::Length(11)],
                        rows,
                    )
                }
                Some(ti) => {
                    let rows = self
                        .topics
                        .get(ti)
                        .map(|t| {
                            t.units
                                .iter()
                                .map(|u| {
                                    let (primary, secondary) = unit_lines(u);
                                    Row::new(vec![
                                        Cell::from(u.when.clone()).style(dim),
                                        type_cell(u.kind),
                                        two_line(primary, secondary, kind_color(u.kind)),
                                    ])
                                    .height(2)
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    (vec!["WHEN", "TYPE", "SUMMARY"], vec![Constraint::Length(11), Constraint::Length(8), Constraint::Min(20)], rows)
                }
            },
            View::Work => {
                let rows = self
                    .work
                    .iter()
                    .map(|u| {
                        let (primary, secondary) = unit_lines(u);
                        Row::new(vec![
                            Cell::from(u.when.clone()).style(dim),
                            type_cell(u.kind),
                            Cell::from(u.repo.clone()).style(dim),
                            two_line(primary, secondary, theme::FG),
                            effort_cell(u),
                        ])
                        .height(2)
                    })
                    .collect();
                (
                    vec!["WHEN", "TYPE", "REPO", "SUMMARY", "EFFORT"],
                    vec![Constraint::Length(11), Constraint::Length(8), Constraint::Length(12), Constraint::Min(20), Constraint::Length(6)],
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
                            Cell::from(if d.meta.repo.is_empty() { "local".into() } else { d.meta.repo.clone() }).style(dim),
                            Cell::from(title).style(Style::new().fg(kind_color(k))),
                        ])
                    })
                    .collect();
                (vec!["TYPE", "REPO", "TITLE"], vec![Constraint::Length(8), Constraint::Length(12), Constraint::Min(20)], rows)
            }
            View::Status => (vec![], vec![], vec![]),
        }
    }

    fn list_title(&self) -> String {
        match self.view {
            View::Topics if self.drill_topic.is_none() => format!("topics · recent first ({})", self.topics.len()),
            View::Topics => "units".into(),
            View::Work => format!("work ({})", self.work.len()),
            View::Search => format!("results ({})", self.results.len()),
            _ => String::new(),
        }
    }

    fn detail_lines(&self) -> String {
        match self.view {
            View::Topics => match self.drill_topic {
                None => self.topics.get(self.sel).map(|t| self.topic_detail(t)).unwrap_or_default(),
                Some(ti) => self
                    .topics
                    .get(ti)
                    .and_then(|t| t.units.get(self.sel))
                    .map(|u| self.unit_detail(u))
                    .unwrap_or_default(),
            },
            View::Work => self.work.get(self.sel).map(|u| self.unit_detail(u)).unwrap_or_default(),
            View::Search => self.results.get(self.sel).and_then(|id| self.doc_by_id.get(id)).map(|&i| doc_detail(&self.docs[i])).unwrap_or_default(),
            _ => String::new(),
        }
    }

    fn topic_detail(&self, t: &TopicUnits) -> String {
        let (gh, asst, prompt) = t.mix;
        let n = t.activity.len();
        let wk = |i: usize| n.checked_sub(i).and_then(|x| t.activity.get(x)).copied().unwrap_or(0);
        let (this, last, prior) = (wk(1), wk(2), wk(3));
        let mx = this.max(last).max(prior).max(1);
        let bar = |c: u64| "█".repeat(((c * 16) / mx) as usize);
        let mut o = format!(
            "{}\n\n{} units · last active {}\n\nactivity (docs/week)\n  this   {:<16} {}\n  last   {:<16} {}\n  prior  {:<16} {}\n\nmix: github {gh} · assistant {asst} · prompt {prompt}\n\nunits:\n",
            t.label, t.units.len(), t.last_active,
            bar(this), this, bar(last), last, bar(prior), prior,
        );
        for u in t.units.iter().take(30) {
            o.push_str(&format!("· {} {:<7} {}{}\n", u.when, kind_tag(u.kind), u.title, self.unit_keyphrases(u)));
        }
        o
    }

    /// Compact session keyphrases for a unit row (empty for PRs/issues).
    fn unit_keyphrases(&self, u: &Unit) -> String {
        if u.kind != Kind::Session {
            return String::new();
        }
        match u.session_id.as_ref().and_then(|s| self.sess_by_id.get(s)).map(|&i| &self.sessions[i].keyphrases) {
            Some(k) if !k.is_empty() => format!("  [{}]", k.join(", ")),
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
            "session {} · {}\n{} → {}\n\nstruggle {}\n{} prompts · {} assistant · {} thinking · {} tool calls\n",
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
        if !s.keyphrases.is_empty() {
            o.push_str(&format!("about: {}\n", s.keyphrases.join(", ")));
        }
        match &s.summary {
            Some(sum) => o.push_str(&format!("summary: {sum}\n")),
            None if !s.gist.is_empty() => o.push_str(&format!("gist: {}\n", s.gist)),
            None => {}
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

    fn status_para(&self) -> Paragraph<'static> {
        Paragraph::new(crate::view::status_md(&self.status))
            .wrap(Wrap { trim: false })
            .block(Block::bordered().border_style(Style::new().fg(theme::BORDER)).title("status"))
    }

    fn footer(&self) -> String {
        let keys = match self.view {
            View::Search => match &self.engine {
                Engine::Loading => "loading model…",
                Engine::Searching => "searching…",
                Engine::Err(e) => return format!("  {e}"),
                Engine::Ready => "type · Enter search · ↑↓ results · 1-4 views · Esc quit",
            },
            View::Topics if self.drill_topic.is_some() => "↑↓ move · h back · 1-4 views · q quit",
            View::Topics => "↑↓ move · Enter drill · 1-4 views · q quit",
            View::Status => "1-4 views · Tab cycle · q quit",
            View::Work => "↑↓ move · 1-4 views · Tab cycle · q quit",
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

/// (primary, secondary) text for a unit row: the one-line summary on top when
/// present, with the original ask (or PR/issue state) as context below.
fn unit_lines(u: &Unit) -> (String, String) {
    match (u.kind, &u.summary) {
        (Kind::Session, Some(s)) => (s.clone(), u.title.clone()),
        _ => (u.title.clone(), u.outcome.clone()),
    }
}

/// The three most recent weekly buckets as [prior, last, this].
fn last3(activity: &[u64]) -> [u64; 3] {
    let n = activity.len();
    let g = |i: usize| n.checked_sub(i).and_then(|x| activity.get(x)).copied().unwrap_or(0);
    [g(3), g(2), g(1)]
}

/// 3 boxes shaded on a grayscale ramp by each week's activity (vs the global
/// max), so topics are comparable at a glance.
fn activity_cell(weeks: [u64; 3], gmax: u64) -> Cell<'static> {
    const SHADES: [Color; 5] = [
        Color::Rgb(0x33, 0x35, 0x3e),
        Color::Rgb(0x60, 0x62, 0x6e),
        Color::Rgb(0x95, 0x97, 0xa4),
        Color::Rgb(0xc6, 0xc8, 0xd2),
        Color::Rgb(0xf0, 0xf1, 0xf5),
    ];
    let spans: Vec<Span> = weeks
        .iter()
        .map(|&c| {
            let lvl = if gmax == 0 || c == 0 { 0 } else { (((c * 4) + gmax - 1) / gmax).min(4) as usize };
            Span::styled("██", Style::new().fg(SHADES[lvl]))
        })
        .collect();
    Cell::from(Line::from(spans))
}

fn type_cell(k: Kind) -> Cell<'static> {
    Cell::from(kind_tag(k)).style(Style::new().fg(kind_color(k)))
}

/// A 3-cell effort bar from a session's struggle score; empty for PR/issue.
fn effort_cell(u: &Unit) -> Cell<'static> {
    if u.kind != Kind::Session {
        return Cell::from("");
    }
    let filled = (u.struggle * 3.0).round().clamp(0.0, 3.0) as usize;
    Cell::from(format!("{}{}", "▰".repeat(filled), "▱".repeat(3 - filled))).style(Style::new().fg(theme::ACCENT))
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
        _ => format!("{} · {} · {}\n\n", d.meta.kind, if d.meta.repo.is_empty() { "local" } else { &d.meta.repo }, d.meta.ts),
    };
    o.push_str(&d.text);
    o
}

fn day(ts: &str) -> String {
    ts.split('T').next().unwrap_or("").to_string()
}

fn spawn_search(model_id: String) -> (Sender<String>, Receiver<SearchMsg>) {
    let (qtx, qrx) = channel::<String>();
    let (rtx, rrx) = channel::<SearchMsg>();
    std::thread::spawn(move || {
        use crate::encode::Encoder;
        use next_plaid::{MmapIndex, SearchParameters};
        let (mut enc, idx) = match (Encoder::load(&model_id), MmapIndex::load(INDEX_PATH)) {
            (Ok(e), Ok(i)) => (e, i),
            _ => {
                let _ = rtx.send(SearchMsg::Err("index/model unavailable (run `index`)".into()));
                return;
            }
        };
        let _ = rtx.send(SearchMsg::Ready);
        while let Ok(q) = qrx.recv() {
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
            keyphrases: vec!["ocr".into(), "adapter".into()],
            gist: "add OCR adapter".into(),
            summary: Some("Added an OCR adapter to the sie pipeline.".into()),
        };
        let work = vec![
            Unit { kind: Kind::Session, when: "2026-05-31".into(), repo: "sie".into(), title: "add OCR adapter".into(), outcome: "1 files".into(), summary: Some("Added an OCR adapter to the sie pipeline.".into()), topic: Some(0), struggle: 0.6, doc_id: None, session_id: Some("S1".into()) },
            Unit { kind: Kind::Pr, when: "2026-05-31".into(), repo: "sie-web".into(), title: "sie-web#7 fix docs search".into(), outcome: "OPEN".into(), summary: None, topic: Some(0), struggle: 0.0, doc_id: Some(0), session_id: None },
        ];
        let topics = vec![TopicUnits { id: 0, label: "ocr, docs".into(), units: work.iter().map(clone_unit).collect(), last_active: "2026-05-31".into(), activity: vec![1, 0, 2, 3], mix: (1, 5, 3) }];
        App {
            docs,
            doc_by_id,
            sess_by_id: [("S1".to_string(), 0)].into_iter().collect(),
            sessions: vec![session],
            work,
            topics,
            status: crate::view::Status { docs: 2, github: 1, sessions: 1, by_kind: vec![], by_repo: vec![], newest_ts: "2026-05-31".into(), last_indexed: None, last_tracked: None },
            view: View::Topics,
            sel: 0,
            drill_topic: None,
            query: String::new(),
            results: vec![],
            engine: Engine::Loading,
            autostart: false,
            qtx: None,
            rrx: None,
            quit: false,
        }
    }

    fn clone_unit(u: &Unit) -> Unit {
        Unit { kind: u.kind, when: u.when.clone(), repo: u.repo.clone(), title: u.title.clone(), outcome: u.outcome.clone(), summary: u.summary.clone(), topic: u.topic, struggle: u.struggle, doc_id: u.doc_id, session_id: u.session_id.clone() }
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

    // The nav bar shows every tab's full label, not a clipped "5".
    #[test]
    fn navbar_shows_all_labels() {
        let mut term = Terminal::new(TestBackend::new(110, 32)).unwrap();
        let mut a = app();
        term.draw(|f| a.draw(f)).unwrap();
        let text: String = term.backend().buffer().content().iter().map(|c| c.symbol()).collect();
        for label in ["Topics", "Work", "Search", "Status"] {
            assert!(text.contains(label), "nav missing {label}");
        }
        assert!(text.contains("autostart"), "footer missing autostart status");
        assert!(text.contains("TOPIC") && text.contains("ACTIVITY"), "topics table missing headers");
    }

    #[test]
    fn topic_drill_shows_units_then_session_detail() {
        let mut a = app();
        a.view = View::Topics;
        assert_eq!(a.list_len(), 1); // one topic
        a.on_key(KeyCode::Enter); // drill
        assert_eq!(a.drill_topic, Some(0));
        assert_eq!(a.list_len(), 2); // its two units
        // session unit detail mentions struggle + counts
        let d = a.detail_lines();
        assert!(d.contains("struggle"), "detail: {d}");
        a.on_key(KeyCode::Char('h'));
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

    // Work rows surface the session's one-line summary, not just the ask.
    #[test]
    fn work_rows_show_summary() {
        let mut term = Terminal::new(TestBackend::new(160, 32)).unwrap();
        let mut a = app();
        a.view = View::Work;
        term.draw(|f| a.draw(f)).unwrap();
        let text: String = term.backend().buffer().content().iter().map(|c| c.symbol()).collect();
        assert!(text.contains("SUMMARY"), "work header missing SUMMARY");
        assert!(text.contains("OCR adapter to the sie"), "work row missing summary text: {text}");
    }
}
