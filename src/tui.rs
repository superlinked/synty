// Interactive terminal UI: a tracker-status pane plus browse/drill over topics,
// recent activity, and search. It reuses the same view-models as the CLI, so the
// two surfaces stay at parity — the TUI just shows more at once and lets you
// drill in. Tabs: Status / Topics / Recent / Search.

use crate::encode::Encoder;
use crate::view::{self, title_of};
use crate::{first_line, Doc, DOCS_PATH, INDEX_PATH};
use anyhow::Result;
use next_plaid::{MmapIndex, SearchParameters};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style, Stylize};
use ratatui::text::Line;
use ratatui::widgets::{Block, List, ListState, Paragraph, Tabs, Wrap};
use ratatui::Frame;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Status,
    Topics,
    Recent,
    Search,
}
const TABS: [Tab; 4] = [Tab::Status, Tab::Topics, Tab::Recent, Tab::Search];

struct App {
    model_id: String,
    docs: Vec<Doc>,
    id_to_idx: HashMap<i64, usize>,
    tab: Tab,
    status: view::Status,
    topics: Vec<(String, Vec<i64>)>, // (label, member doc ids)
    topic_drill: Option<usize>,      // Some(topic) → listing its members
    recent: Vec<usize>,              // doc indices, newest first
    query: String,
    results: Vec<usize>,
    engine: Option<(Encoder, MmapIndex)>,
    sel: usize,
    note: String,
    quit: bool,
}

pub fn run(model_id: String) -> Result<()> {
    let mut app = App::load(model_id)?;
    let mut term = ratatui::init();
    let res = app.run_loop(&mut term);
    ratatui::restore();
    res
}

impl App {
    fn load(model_id: String) -> Result<Self> {
        let docs = crate::load_docs(DOCS_PATH).unwrap_or_default();
        let id_to_idx = docs.iter().enumerate().map(|(i, d)| (d.id, i)).collect();
        let status = view::status().unwrap_or(view::Status {
            docs: docs.len(),
            github: 0,
            sessions: 0,
            by_kind: vec![],
            by_repo: vec![],
            newest_ts: String::new(),
            last_indexed: None,
            last_tracked: None,
        });
        let topics = view::topic_groups().unwrap_or_default();
        let mut recent: Vec<usize> = (0..docs.len())
            .filter(|&i| matches!(docs[i].meta.kind.as_str(), "pull_request" | "issue" | "user_prompt"))
            .collect();
        recent.sort_by(|&a, &b| docs[b].meta.ts.cmp(&docs[a].meta.ts));
        Ok(Self {
            model_id,
            docs,
            id_to_idx,
            tab: Tab::Status,
            status,
            topics,
            topic_drill: None,
            recent,
            query: String::new(),
            results: vec![],
            engine: None,
            sel: 0,
            note: String::new(),
            quit: false,
        })
    }

    fn run_loop(&mut self, term: &mut ratatui::DefaultTerminal) -> Result<()> {
        while !self.quit {
            term.draw(|f| self.draw(f))?;
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press {
                    self.on_key(k.code);
                }
            }
        }
        Ok(())
    }

    // ── data for the active view ──────────────────────────────────────────

    /// Lines for the left-hand list in the current tab/drill.
    fn list_lines(&self) -> Vec<String> {
        match self.tab {
            Tab::Status => vec![],
            Tab::Topics => match self.topic_drill {
                None => self.topics.iter().map(|(l, ids)| format!("{l}  ({} docs)", ids.len())).collect(),
                Some(t) => self.member_docs(t).iter().map(|&i| title_of(&self.docs[i])).collect(),
            },
            Tab::Recent => self.recent.iter().map(|&i| recent_line(&self.docs[i])).collect(),
            Tab::Search => self.results.iter().map(|&i| title_of(&self.docs[i])).collect(),
        }
    }

    /// Detail pane for the current selection.
    fn detail(&self) -> String {
        match self.tab {
            Tab::Status => view::status_md(&self.status),
            Tab::Topics => match self.topic_drill {
                None => match self.topics.get(self.sel) {
                    Some((label, ids)) => self.topic_detail(label, ids),
                    None => String::new(),
                },
                Some(t) => self.member_docs(t).get(self.sel).map(|&i| doc_detail(&self.docs[i])).unwrap_or_default(),
            },
            Tab::Recent => self.recent.get(self.sel).map(|&i| doc_detail(&self.docs[i])).unwrap_or_default(),
            Tab::Search => self.results.get(self.sel).map(|&i| doc_detail(&self.docs[i])).unwrap_or_default(),
        }
    }

    fn member_docs(&self, topic: usize) -> Vec<usize> {
        self.topics
            .get(topic)
            .map(|(_, ids)| ids.iter().filter_map(|id| self.id_to_idx.get(id).copied()).collect())
            .unwrap_or_default()
    }

    fn topic_detail(&self, label: &str, ids: &[i64]) -> String {
        let docs: Vec<&Doc> = ids.iter().filter_map(|id| self.id_to_idx.get(id).map(|&i| &self.docs[i])).collect();
        let mut o = format!("{label}\n{} docs\n\n", docs.len());
        for d in docs.iter().take(40) {
            o.push_str(&format!("- {}\n", title_of(d)));
        }
        o
    }

    // ── input ─────────────────────────────────────────────────────────────

    fn on_key(&mut self, code: KeyCode) {
        // Tab switching is global.
        match code {
            KeyCode::Tab => {
                let i = TABS.iter().position(|t| *t == self.tab).unwrap_or(0);
                self.set_tab(TABS[(i + 1) % TABS.len()]);
                return;
            }
            KeyCode::BackTab => {
                let i = TABS.iter().position(|t| *t == self.tab).unwrap_or(0);
                self.set_tab(TABS[(i + TABS.len() - 1) % TABS.len()]);
                return;
            }
            _ => {}
        }

        if self.tab == Tab::Search {
            self.search_key(code);
            return;
        }

        match code {
            KeyCode::Char('q') | KeyCode::Esc => {
                if self.topic_drill.is_some() {
                    self.topic_drill = None;
                    self.sel = 0;
                } else {
                    self.quit = true;
                }
            }
            KeyCode::Char(c @ '1'..='4') => {
                self.set_tab(TABS[c as usize - '1' as usize]);
            }
            KeyCode::Down | KeyCode::Char('j') => self.move_sel(1),
            KeyCode::Up | KeyCode::Char('k') => self.move_sel(-1),
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                if self.tab == Tab::Topics && self.topic_drill.is_none() && !self.topics.is_empty() {
                    self.topic_drill = Some(self.sel);
                    self.sel = 0;
                }
            }
            KeyCode::Left | KeyCode::Char('h') => {
                if self.topic_drill.is_some() {
                    self.topic_drill = None;
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
            KeyCode::Enter => self.run_search(),
            KeyCode::Down => self.move_sel(1),
            KeyCode::Up => self.move_sel(-1),
            _ => {}
        }
    }

    fn set_tab(&mut self, t: Tab) {
        self.tab = t;
        self.sel = 0;
        self.topic_drill = None;
    }

    fn move_sel(&mut self, delta: i32) {
        let n = self.list_lines().len();
        if n == 0 {
            return;
        }
        let cur = self.sel as i32 + delta;
        self.sel = cur.clamp(0, n as i32 - 1) as usize;
    }

    fn run_search(&mut self) {
        if self.query.trim().is_empty() {
            return;
        }
        if self.engine.is_none() {
            self.note = "loading model…".into();
            match (Encoder::load(&self.model_id), MmapIndex::load(INDEX_PATH)) {
                (Ok(e), Ok(i)) => self.engine = Some((e, i)),
                _ => {
                    self.note = "index/model unavailable (run `index`)".into();
                    return;
                }
            }
        }
        let (enc, idx) = self.engine.as_mut().unwrap();
        let q = match enc.encode_query(&self.query) {
            Ok(q) => q,
            Err(e) => {
                self.note = format!("encode error: {e}");
                return;
            }
        };
        let params = SearchParameters { top_k: 20, ..Default::default() };
        match idx.search(&q, &params, None) {
            Ok(r) => {
                self.results = r.passage_ids.iter().filter_map(|id| self.id_to_idx.get(id).copied()).collect();
                self.sel = 0;
                self.note = format!("{} results", self.results.len());
            }
            Err(e) => self.note = format!("search error: {e}"),
        }
    }

    // ── render ──────────────────────────────────────────────────────────────

    fn draw(&self, f: &mut Frame) {
        let [top, body, hint] =
            Layout::vertical([Constraint::Length(1), Constraint::Min(0), Constraint::Length(1)]).areas(f.area());

        let idx = TABS.iter().position(|t| *t == self.tab).unwrap_or(0);
        f.render_widget(
            Tabs::new(vec!["1 Status", "2 Topics", "3 Recent", "4 Search"]).select(idx).highlight_style(Style::new().bold().reversed()),
            top,
        );

        match self.tab {
            Tab::Status => f.render_widget(Paragraph::new(self.detail()).wrap(Wrap { trim: false }).block(Block::bordered().title("status")), body),
            _ => self.draw_list_detail(f, body),
        }

        f.render_widget(Line::from(self.hints()).dim(), hint);
    }

    fn draw_list_detail(&self, f: &mut Frame, area: Rect) {
        // Search reserves a query line above the split.
        let (query_area, split) = if self.tab == Tab::Search {
            let [q, s] = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(area);
            (Some(q), s)
        } else {
            (None, area)
        };
        if let Some(q) = query_area {
            f.render_widget(Paragraph::new(format!("/ {}", self.query)).block(Block::bordered()), q);
        }

        let [left, right] = Layout::horizontal([Constraint::Percentage(42), Constraint::Percentage(58)]).areas(split);

        let lines = self.list_lines();
        let title = self.list_title(lines.len());
        let items: Vec<String> = lines.iter().map(|l| truncate(l, left.width.saturating_sub(2) as usize)).collect();
        let mut st = ListState::default();
        if !items.is_empty() {
            st.select(Some(self.sel.min(items.len() - 1)));
        }
        f.render_stateful_widget(
            List::new(items).block(Block::bordered().title(title)).highlight_style(Style::new().add_modifier(Modifier::REVERSED)),
            left,
            &mut st,
        );

        f.render_widget(Paragraph::new(self.detail()).wrap(Wrap { trim: false }).block(Block::bordered().title("detail")), right);
    }

    fn list_title(&self, n: usize) -> String {
        match self.tab {
            Tab::Topics => match self.topic_drill {
                None => format!("topics ({n})"),
                Some(t) => format!("{} ▸ members", self.topics.get(t).map(|(l, _)| l.as_str()).unwrap_or("")),
            },
            Tab::Recent => format!("recent ({n})"),
            Tab::Search => format!("results ({n})"),
            Tab::Status => String::new(),
        }
    }

    fn hints(&self) -> String {
        let nav = match self.tab {
            Tab::Search => "type to query · Enter search · ↑↓ results · Esc quit",
            Tab::Topics if self.topic_drill.is_some() => "↑↓ move · ←/h back · Tab next · q quit",
            Tab::Topics => "↑↓ move · Enter drill · 1-4 tabs · q quit",
            _ => "↑↓ move · 1-4 tabs · Tab next · q quit",
        };
        if self.note.is_empty() {
            format!("  {nav}")
        } else {
            format!("  {nav}   [{}]", self.note)
        }
    }
}

fn recent_line(d: &Doc) -> String {
    let day = d.meta.ts.split('T').next().unwrap_or("");
    let repo = if d.meta.repo.is_empty() { "local" } else { &d.meta.repo };
    format!("{day}  {:<13} {repo}: {}", d.meta.kind, first_line(&d.text))
}

fn doc_detail(d: &Doc) -> String {
    let mut o = String::new();
    match d.meta.kind.as_str() {
        "pull_request" | "issue" => {
            o.push_str(&format!("{} {}#{}  [{}]\n", d.meta.kind, d.meta.repo, d.meta.number.unwrap_or(0), d.meta.state.clone().unwrap_or_default()));
            if let Some(u) = &d.meta.url {
                o.push_str(&format!("{u}\n"));
            }
        }
        _ => {
            o.push_str(&format!("{} · {} · {}\n", d.meta.kind, if d.meta.repo.is_empty() { "local" } else { &d.meta.repo }, d.meta.ts));
        }
    }
    o.push('\n');
    o.push_str(&d.text);
    o
}

fn truncate(s: &str, w: usize) -> String {
    if w == 0 || s.chars().count() <= w {
        s.to_string()
    } else {
        s.chars().take(w.saturating_sub(1)).collect::<String>() + "…"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Meta;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn doc(id: i64, kind: &str, repo: &str, text: &str) -> Doc {
        Doc {
            id,
            text: text.into(),
            meta: Meta {
                source: if kind == "pull_request" { "github".into() } else { "agent".into() },
                kind: kind.into(),
                repo: repo.into(),
                author: String::new(),
                session_id: "S1".into(),
                ts: "2026-05-31T10:00:00Z".into(),
                number: Some(7),
                url: None,
                state: None,
                labels: vec![],
            },
        }
    }

    fn app() -> App {
        let docs = vec![doc(0, "pull_request", "sie-web", "fix docs search"), doc(1, "user_prompt", "sie", "add OCR adapter")];
        let id_to_idx = docs.iter().enumerate().map(|(i, d)| (d.id, i)).collect();
        App {
            model_id: "m".into(),
            docs,
            id_to_idx,
            tab: Tab::Topics,
            status: view::Status { docs: 2, github: 1, sessions: 1, by_kind: vec![("issue".into(), 1)], by_repo: vec![("sie".into(), 1)], newest_ts: "2026-05-31".into(), last_indexed: None, last_tracked: None },
            topics: vec![("docs search".into(), vec![0]), ("ocr adapter".into(), vec![1])],
            topic_drill: None,
            recent: vec![0, 1],
            query: String::new(),
            results: vec![],
            engine: None,
            sel: 0,
            note: String::new(),
            quit: false,
        }
    }

    // Renders every tab to an off-screen buffer without panicking, and the
    // visible text reflects the active view.
    #[test]
    fn renders_all_tabs() {
        let mut term = Terminal::new(TestBackend::new(100, 30)).unwrap();
        for tab in [Tab::Status, Tab::Topics, Tab::Recent, Tab::Search] {
            let mut a = app();
            a.tab = tab;
            term.draw(|f| a.draw(f)).unwrap();
        }
        // Topics tab shows a topic label in the buffer.
        let mut a = app();
        term.draw(|f| a.draw(f)).unwrap();
        let text = buffer_text(term.backend());
        assert!(text.contains("topics"));
        assert!(text.contains("docs search") || text.contains("ocr"));
    }

    // Drilling into a topic switches the list to its members and back.
    #[test]
    fn topic_drill_and_back() {
        let mut a = app();
        assert_eq!(a.list_lines().len(), 2); // two topics
        a.on_key(KeyCode::Enter); // drill into topic 0
        assert_eq!(a.topic_drill, Some(0));
        assert_eq!(a.list_lines().len(), 1); // its one member
        a.on_key(KeyCode::Char('h')); // back
        assert!(a.topic_drill.is_none());
    }

    // Number keys switch tabs; selection navigation clamps.
    #[test]
    fn tab_switch_and_nav_clamp() {
        let mut a = app();
        a.on_key(KeyCode::Char('3'));
        assert!(matches!(a.tab, Tab::Recent));
        a.on_key(KeyCode::Up); // already at top
        assert_eq!(a.sel, 0);
        a.on_key(KeyCode::Down);
        assert_eq!(a.sel, 1);
        a.on_key(KeyCode::Down); // clamp at last (2 recent items)
        assert_eq!(a.sel, 1);
    }

    fn buffer_text(b: &TestBackend) -> String {
        b.buffer().content().iter().map(|c| c.symbol()).collect()
    }
}
