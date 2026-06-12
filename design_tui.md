# synty — TUI design

**Status:** v0.1 · **Owner:** Daniel Svonava

The human surface. `design.md` owns the system; this owns how the terminal UI
looks, behaves, and what data it needs. The CLI renders the same view-models, so
the two stay at parity (the TUI is just more expressive on screen).

## Jobs to be done

Work backwards from what the user is trying to do; each maps to one view.

- **A — task-start context:** "what's the prior work on X?" → Search.
- **B — situational awareness:** "what's been happening / what's active?" → Work.
- **C — catch up on a thread:** "what happened with topic X, and how is it going?" → Topics.
- **D — orient / prioritize:** "where should I look first?" → Overview.
- **E — is it healthy:** "is tracking fresh, what's indexed?" → Status.

## Principles

1. **Keyboard-first and conventional.** Borrow muscle memory; invent nothing.
   `↑↓`/`jk` move, `Enter` drill, `Esc` back (the universal "go back"), `/`
   live-filter, `:` jump-to-view, `?` help, `Tab` cycle, `q` quit, `1`–`5` jump.
2. **One spine: master → detail → drill, with a breadcrumb.** Every screen is a
   list plus a live preview; `Enter` goes deeper (topic → units → unit →
   messages / commits / comments), `Esc` comes back; the header shows where you
   are.
3. **Context footer always shows the actionable keys** for the current view/mode.
4. **Units of work are the default altitude; messages are the basement.** Lists
   are sessions / PRs / issues. Raw turns appear only at the deepest drill, and
   even then representative, not exhaustive.
5. **Time is a first-class visual.** Sort-by-recency everywhere; sparklines on
   topics and units; an activity overview.
6. **Color carries meaning, consistently.** Hierarchy: most text in the default
   foreground; bold for headers; dim for metadata; semantic colors for state;
   accent only for the current selection / interactive element.
7. **One job per view; min size 80×24** (show a resize hint below it);
   constraint-based layout, never absolute positions.

## Palette (Superlinked brand)

Truecolor; degrades to the nearest 16-color on terminals without it.

| Role | Color | Hex |
|---|---|---|
| Background (bars, fills) | dark blue | `#1F1F2C` (or black `#000000`) |
| Foreground (body text) | off-white | `#F3F1F1` |
| Accent (selection, active tab, emphasis) | coral-red | `#F3441D` |
| Accent soft (secondary highlight) | coral | `#FC6445` |
| Borders | slate | `#5D6C80` |
| Dim / metadata | mist | `#A0B4C1` |
| Source: github | sky | `#86A1BC` |
| Source: session/assistant | sand | `#CEBCAA` |
| Source: prompt | peach | `#FECCBE` |
| State: open | sage | `#869582` |
| State: merged | accent | `#F3441D` |
| State: closed | slate | `#5D6C80` |
| Struggle heat (low→high) | blush → peach → coral → accent | `#FDE4DD` `#FECCBE` `#FC6445` `#F3441D` |

## Layout

```
┌ synty · Topics › "gateway isolation" › session 6e6cf0d1 ───────────────┐  header: app · breadcrumb
│ ┌ list (master) ──────────┐ ┌ detail / preview (live) ───────────────┐ │
│ │ > unit                   │ │  selected unit, rendered rich          │ │
│ │   unit                   │ │  (summary, sparkline, metadata)        │ │
│ └──────────────────────────┘ └────────────────────────────────────────┘ │
│ /filter…                                                                  │  (when filtering)
│ ↑↓ move · enter drill · / filter · s sort · ? help · q quit       [note]  │  context footer
└────────────────────────────────────────────────────────────────────────┘
```

Master/detail split ~40/60. The detail pane scrolls. A top tab strip names the
four views (Topics, Work, Search, Status) and highlights the active one in
accent. The Topics list carries a per-day activity strip (last four weeks, week
dividers) and drilling a topic opens an overlay (the right three-quarters of
the screen; full-screen when narrow) with its description, repos, people, and
member units. Enter on a member unit splits the overlay with that unit's
detail — the same content Work's right pane shows — following the selection;
Esc/h peel one layer at a time (detail → overlay → list).

## Keymap

**Global:** `1`–`5` jump to view · `Tab`/`Shift-Tab` cycle · `:` jump-to-view by
name · `?` help overlay · `q` / `Ctrl-C` quit.

**Lists:** `j`/`↓` `k`/`↑` move · `g`/`G` top/bottom · `Enter`/`l`/`→` drill ·
`Esc`/`h`/`←` back · `/` live-filter (Esc clears) · `s` cycle sort
(recency · size/activity · struggle).

**Search view:** the query line is always focused — type to edit, `Enter` runs,
`↑↓` move through results, `Esc` quits. Model is loaded at startup so the first
search is instant.

## Views

- **Overview** (job D) — landing dashboard: top *active* topics with sparklines
  (activity over the last N weeks), the most recent work units, and a one-line
  health glance. Answers "where do I look?".
- **Topics** (job C) — list sorted by recency/activity, each row: label
  (keyphrases) · size · last-active · sparkline. Drill → topic page: a summary
  header, an activity sparkline, a type-mix bar (github / session / prompt), and
  a time-ordered list of **work units** (sessions as ask→outcome, PRs as
  title→state). Drill a unit → unit detail.
- **Work** (job B) — one unified, filter/sortable list of units (sessions, PRs,
  issues): `when · type · repo · title/ask · outcome · struggle`. The "recent"
  feed, reframed to units. Drill → unit detail.
- **Search** (job A) — query → results as units → drill to detail → messages.
- **Status** (job E) — tracker/index health: docs by source/kind/repo, newest
  item, last indexed, last tracked, per-source freshness; a tokens & tools
  panel tracks the four token classes, tool calls, and active sessions per day
  over the last four Mon-aligned weeks (same strips as the Topics activity
  column, each row shaded against its own peak), above the repo/account
  breakdowns — which carry TOK/TOOLS spend columns — and a Tools table naming
  every tool with calls and per-name error counts.

### Unit detail (the drill target)

- **Session:** header (ask, repo, time span, duration, prompt/tool/thinking
  counts, struggle score, linked PR), files touched, and a *representative arc*
  of the session — key turns, not every message. Drill → messages.
- **PR / issue:** title, body, state, labels, linked session; drill → comments.

## Data-layer contract

The CLI and TUI consume these view-models (computed from docs.jsonl + the raw
envelopes under corpus/local + clusters.json). Sketch, not exact types:

- `Session { id, repo, started, ended, duration, prompts, assistant_msgs,
  thinking_blocks, tool_calls, retries, files: [String], ask, linked_pr,
  topic, struggle }`
- `Unit { kind: Session|Pr|Issue, when, repo, title, outcome, topic, struggle }`
  — the unified list item.
- `Topic { id, label, units: [UnitRef], span, activity: [(bucket, count)],
  type_mix, summary }`
- `Rollup` — activity counts per time bucket (day/week), per topic/unit/repo.

`thinking_blocks`, `tool_calls`, etc. come from the raw envelopes (which carry
`ts`, `session_id`, `attachment_ref` for files, `pr_link` for the PR linkage).

### Struggle score

Thinking *text* is unavailable (Claude Code stores only signatures; Codex
encrypts all but a summary on ~⅓ of items), so struggle is **derived from
structure**, which is what actually correlates with difficulty:

```
struggle = z(thinking_blocks) + z(tool_calls) + z(turns) + z(duration) + z(retries)
```

normalized across sessions and bucketed low→high to the heat ramp. Codex
reasoning *summaries* (the `summary[]` field) are captured when present and shown
in the session arc; they are a bonus signal, not the score's basis.

## Notes

- The shipped binary stays single and self-contained; the TUI uses
  `ratatui`/`crossterm` only (no web).
- CLI parity: `topic`, `recent`/`work`, `status`, `search` render the same
  view-models as Markdown.
- An MCP server and a richer web frontend are future work (see `roadmap.md`).
