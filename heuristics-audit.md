# heuristics-audit.md — ad-hoc fixes and how to retire them

A sweep of the hand-tuned heuristics, magic constants, and hardcoded lists
across the binary, and a prioritized plan to replace them with simpler,
authoritative, project-generalizable mechanisms. The goal is **fewer moving
parts**: most fixes below *delete* a heuristic by deferring to a ground-truth
signal or the data's own distribution, not by adding another knob.

This is a robustness/generalization track. `quality-roadmap.md` owns clustering
*quality* interventions; this owns "stop baking this corpus/org/model into the
logic."

## The patterns (the systematic view)

Every finding is an instance of one of six recurring shapes. Fix the pattern,
not the 50 symptoms.

1. **Authoritative signal over heuristic.** When a ground truth exists (GitHub
   `__typename` User/Bot, org membership, git identity), use it and demote the
   heuristic to a *fallback only when the signal is absent*. Today heuristics
   run even when the authoritative signal is present — and can override it.
2. **Run-relative over absolute thresholds.** Derive cutoffs from the data's own
   distribution so they survive an encoder/model/corpus change. The codebase
   already does this in **three** places — `topics.rs` neighbor `FLOOR` (ratio
   to best neighbor), `qwen.rs` `EMBED_FLOOR` (0.6×median), `topics.rs`
   `GRABBAG_FRAC` (0.8×median). The holdouts that still use an absolute,
   encoder-calibrated number are the bugs.
3. **Data out of code.** Org-, stack-, time-, and English-specific *lists* are
   data, not logic. They belong in config or a data file, or be derived — not in
   a `const` array that needs a recompile and bakes in "the agents that exist in
   2026."
4. **One adapter per platform.** Platform-specific parsing (envelope markers,
   token-usage semantics, repo-naming convention) belongs in the per-source
   tailer, not scattered through `ingest`/`units`. New platform → new adapter,
   no edits to the core.
5. **Policy in config, mechanism in code.** A genuinely team-tunable value
   (activity window, retention cap) is policy → config with a documented default
   in *one* place. A structural constant is mechanism → code.
6. **Restraint.** A display cap (`take(12)`, `fmt_tokens` breakpoints, the
   5-dot meter) is *fine* as a constant. Parameterizing it per-corpus is just
   sprawl in the other direction. Centralize and document; do not configure.

## Worked example of the discipline

The sweep flagged `topics.rs:22 FLOOR=0.6` as "absolute, breaks on encoder swap,
CRITICAL." Reading `topics.rs:522` — `let d = (s / best)` — it's a **ratio to
the per-unit best neighbor**, i.e. already run-relative (pattern 2 done right).
The real holdouts are `DUP` and `RES_SCALE`. Verify the mechanism before
generalizing from a label; that habit is the actual fix for "unsystematic."

## Prioritized remediation

### P0 — generalizes AND removes code (do first)

| # | Problem (current heuristics) | Robust, simple replacement | Net | Status |
|---|---|---|---|---|
| 0.1 | **Bot/agent identity.** `is_bot` substring `"bot"` + hardcoded CI set; `BOT_AUTHORS`/`AGENT_MARKS` tables; `is_bot` runs even when org membership (authoritative) is known and can override it (`fleet.rs`). | Org membership authoritative when known (`is_bot` doesn't run on that path → no override); `is_bot` is the *members-unknown fallback only*. Service-account members (`slrelease`) → a user-owned `config.fleet_ignore` list, **not** a name in code. `__typename` (User vs Bot) noted as the no-members-path hardening. | −override bug | **done** (membership-authoritative + `fleet_ignore`; `__typename` deferred) |
| 0.2 | **Topic-name gate stack.** Four stacked gates (`clean_name` shape, `name_grounded` unigram overlap, `repo_ban`, embedding gate) with no single authority; the unigram gate rejects good paraphrases; the proposed hyphen-count "mush" tweak is arbitrary. | The slug problem (`SIE-Server`) → one *structural* rule: a single mashed token with ≥2 word-parts is an identifier → reject to the summary fallback (not a hyphen count). Collapsing the gate stack onto the embedding gate (drop `name_grounded`) needs a name-quality eval first. | slug→summary | **partial** (structural slug rule done, `name_dupes` 2→0; `name_grounded` removal deferred — needs a name-quality eval, and it regressed once already) |
| 0.3 | **`DUP=0.95`** near-duplicate cutoff is absolute, calibrated to "non-dup pairs ≈0.85" on the *current* encoder (`topics.rs:36`). Swap the encoder → silently wrong. | Derive from the run's own same-repo pair-score distribution (a high percentile), or make it relative like `FLOOR`. Same discipline already used three places. | generalizes core | **deferred** — touches the cluster core; gate behind the anchor/cluster eval |

`RES_SCALE=2.5` (Louvain, `topics.rs:32`) is the scariest single constant —
calibrated to *this* corpus's anchor eval; a different corpus size/shape gets
grab-bags or over-fragmentation. **Do not touch casually**: it needs the anchor
eval (quality-roadmap) to re-derive, ideally replaced by picking resolution from
a modularity/coherence curve rather than a fixed multiplier. Tracked, not P0.

### P1 — robustness + correct layering

| # | Problem | Replacement |
|---|---|---|
| 1.1 | Policy knobs as scattered consts: `GH_WINDOW_DAYS`, `CAP` vs `config.max_docs` (two sources of truth), `DEFAULT_REPOS`, `FLOOR_SLACK_MIN`, hysteresis 10%. | The team-tunable ones → `config` with one defaults table + documented rationale. Collapse the `CAP`/`max_docs` split. Leave the rest as documented mechanism. |
| 1.2 | Platform parsing scattered: `is_noise` markers (Claude-Code envelope shapes) in `ingest`; codex cumulative-token semantics + the "2–4× streamed repeat" msg_id assumption in `units`. | Move into the per-source tailer modules (`src/<tool>/`), which already exist for the trackers. Core stays platform-agnostic; the next tracker (Cursor, deferred) adds an adapter, edits nothing. |
| 1.3 | `repo folding` assumes a `-` delimiter (`sie-web-backbutton`→`sie-web`); breaks on `_`/`.`/`/` or monorepos (`units.rs`). | Lean on the known-repo set + git-remote basename (both already present); make the worktree-suffix separator explicit/derivable rather than hardcoded `-`. |
| 1.4 | `fmt=2` manifest version is a manual bump — forget it and a derivation change silently serves stale docs (`ingest.rs`). | Hash the derivation version (the relevant code/schema) into the manifest so it can't be forgotten. |
| 1.5 | `machine_id` / `actor` brittle on cloud/ephemeral/CI (hostname, `$USER`, no stale-check on pinned login) (`identity.rs`). | Detect CI/ephemeral and require/auto-pass `--machine`; document the precedence; revisit pinned-login staleness. Medium — only bites multi-machine/CI fleets. |

### P2 — centralize + document, do NOT parameterize (restraint)

- The `take(N)` swarm (6/8/10/12/16 across `view.rs`/`topics.rs`), `fmt_tokens`
  breakpoints, the 5-dot meter, excerpt lengths (200/320/500/600/1500): collect
  into one documented `display` constants block. They are appropriately simple
  constants; configuring them per-corpus would be sprawl.
- English/Latin-script assumptions (`MIN_TEXT` char counts, `STOPWORDS`,
  `title_case` rules): note and defer — the corpus is English engineering work;
  revisit only if a non-English corpus appears.

## Sequencing

- 0.1 and 0.2 are independent, self-contained, and each delete code — best first,
  and 0.2 closes out the topic-naming churn that prompted this.
- 0.3 touches the cluster core → gate behind the anchor/quality eval so a
  threshold change is measured, not eyeballed (AGENTS.md metrics rule).
- 1.2 pays off when the next tracker lands; do it alongside that, not speculatively.
- P2 is a half-day cleanup; do it last, as one commit, to avoid touching it
  repeatedly.

## Principle to hold onto

Prefer the fix that *removes* a decision over the one that *adds a knob*. An
authoritative signal (membership, `__typename`) and a run-relative threshold
(median) both mean "no number to hand-tune per project." A config knob is still
a knob — use it only for genuine team policy, not to paper over a missing
signal.
