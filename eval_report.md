# synty experiment — evaluation

**Date:** 2026-05-31 · **Owner:** Daniel Svonava
**Question:** does late-interaction retrieval + lightweight clustering + extractive summaries, on the production stack, produce something useful — to a human and a coding agent — **with no generative model**?

## What ran

One self-contained Rust binary, the same stack we would ship:

- **Encode:** `pylate-rs` (ColBERT on Candle, CPU), model `mxbai-edge-colbert-v0-32m` (ModernBERT, 32 M params, 127 MB), loaded from a **local dir** — no network, no Python, no Docker, no server.
- **Index/search:** `next-plaid` crate (PLAID late-interaction, SQLite metadata filtering).
- **Corpus:** real data — Claude Code / Codex / Cowork sessions on this machine (60 d) + GitHub PRs/issues from 6 `superlinked` repos (90 d), pulled with the existing v1 agent + `gh`.

Measured: **3,288 docs → 665,070 token embeddings**, encoded in **477 s** (~7 docs/s, plain CPU), PLAID index built in **88 s**. Query encode + filtered search is sub-second. Index on disk is a few hundred MB.

Subcommands: `ingest`, `index`, `search <q> [--filter col=value]`, `cluster`, `summarize`, `eval`. Search prints Markdown to stdout — the agent surface, no server.

## Tests

14 scenario-derived unit tests (`cargo test`, all green), written from user expectations rather than the implementation: GitHub PR → one doc with correct metadata; empty item dropped; session messages chunked with repo from cwd; system-injected pseudo-prompts (`<task-notification>`, `<bash-*>`) dropped; recency cap renumbers ids; ColBERT `#` references parsed; mutual-kNN groups only reciprocated neighbors; one-way neighbors are not edges; a GitHub link bridges groups; pad rows dropped in the Tensor→Array2 conversion; helper formatting.

## Scorecard

### 1. Retrieval — **PASS (12/12; bar was ≥7/10)**

Every probe query returned a relevant result in the top 3, across both GitHub and session content, with strong semantic matching even when wording differed:

- "OCR document parsing adapter MinerU docling" → `sie-internal#1148/#1149` (MinerU), `#687/#762` (Docling).
- "close security vulnerabilities dependabot CodeQL" → `#1133` ("close ~190 Dependabot alerts"), `#1141`, `#1130`.
- "gateway generation isolation guardrails" → `#1136` (exact), plus the queue-isolation PRs.
- Filters work: `repo=sie-web` returns only sie-web; `kind=pull_request` filters correctly; `source=agent` surfaces my own sessions.

### 2. Agent usefulness — **PASS (3/3)**

Task-start questions answered with materially better context than nothing:

- "has anyone worked on OCR recently" → OCR showcase issue `#607`, `#496`, `#579` + a session debating the OCR/extract design.
- "status of generation isolation guardrails" → `#1136` + the related isolation fixes.
- "what was decided about the synty rewrite language and embeddings" → retrieved this conversation's own decision messages (Rust + next-plaid).

This is the core value: an agent (or human) pointing at synty before starting gets the relevant prior work, across tools and GitHub, locally, with no LLM.

### 3. Summaries — **session: PASS · topic digests: WEAK (tied to clustering)**

Extractive session summaries are specific and accurate without any model: each shows the opening ask verbatim, prompt count, and files touched. Examples: the current pivot session (8 prompts; `.gitignore`, `design.md`); a model-self-hostability research session (slide-deck files); a metrics session (10 deck components). Useful to read and to hand to an agent.

Topic digests are only as good as the clusters they summarize (see below): coherent session clusters digest well; the over-merged GitHub blob does not, and session-only clusters get thin labels.

### 4. Clustering — **PARTIAL → see re-run**

First run (no LLM): mutual-kNN over late-interaction similarity ∪ GitHub `#`-references → connected components. The **session clusters were genuinely recognizable** (a maintainer-dashboard build, the synty v1 exploration, an AWS/Docker setup, two benchmark slide-deck sessions). Two problems: a **1,143-doc giant component** (the `#`-reference unioning transitively chained most of sie-internal, and dense semantic neighborhoods bridged), and a **junk cluster of `<task-notification>` system messages**.

Fix applied and **verified on a re-run** (ingest noise filter + relative-score floor ≥60% of best neighbor + K=6):

- Noise cluster **eliminated** — 131 system-message docs dropped (3,288 → 3,157); the `<task-notification>` "topic" is gone.
- Giant component **shrank 1,143 → 710 docs**, and sessions **unbridged** from GitHub (assistant-messages in the big cluster fell 291 → 28).
- **Session clusters are now clean and recognizable**: the synty-rewrite thread, the GTM/PDL deep-pull + enrichment-dashboard work, the benchmark slide decks.
- **Residual:** the 710-doc cluster is ~95% GitHub (622 sie-internal PRs/issues). Connected-components over the `#`-reference graph transitively merges a homogeneous repo into one blob; a kNN score floor cannot fix graph transitivity.

**Clustering: partial.** Recognizable for sessions, over-merged for GitHub. The fix is scoped and known: community detection (Louvain/Leiden) with a resolution knob, and treat GitHub links as a weighted signal, not a hard transitive union. Retrieval, search, and summaries do not depend on it.

## Verdict: **GO**, with clustering as a scoped refinement

The core hypothesis holds. On real data, with **no generative model and nothing leaving the machine**, late-interaction retrieval + filtering is excellent, the agent-facing search surface is immediately useful, and extractive session summaries are good. The whole thing is one self-contained Rust binary — the exact stack to ship.

The one area needing work is **clustering into topics**: connected-components over a kNN graph is too coarse for a homogeneous corpus. The fix path is clear and scoped (community detection with a resolution knob, e.g. Louvain/Leiden; keep GitHub links as a signal, not a transitive merge). Retrieval, search, and summaries — the primary value — do not depend on it.

## Notes for step 2

- **Model bundling:** mxbai-edge (127 MB) is small enough to `include_bytes!` into the binary for a true single artifact; or ship alongside. ModernBERT-only is fine (pylate-rs constraint).
- **Encode speed:** ~7 docs/s on plain CPU. Enable Candle `accelerate` (macOS) / `openblas` (Linux) and make hydration incremental for live use; batch backfill is fine as-is.
- **hf_hub hang:** the blocking downloader stalled on a large model with no timeout. Ship with a local/bundled model or a `curl`-style resumable fetch; do not rely on hf_hub at runtime.
- **Filters:** next-plaid WHERE wants bound parameters, so the surface is `col=value` (e.g. `repo=sie-web`), not inline SQL literals.
- **Clustering:** add Louvain/Leiden community detection; cache per-doc neighbors so re-clustering does not re-encode.
- Remaining product milestones (unchanged): Rust tailers, bucket-as-backplane sync, TUI, optional team HTTP frontend, privacy guardrails.

## M1 update — clustering refined, encode accelerated (2026-05-31)

The scoped clustering refinement and the encode/iteration items from "Notes for
step 2" are done. Re-validated on a fresh full pull (3,938 docs / 770 K
embeddings, 90 d GitHub + 90 d sessions).

**Clustering → PASS.** Connected-components is replaced by **Louvain** modularity
optimization over a weighted graph: kNN similarity (normalized per-doc, floored,
summed both directions) plus GitHub `#`-references as a **fixed-weight edge
signal**, not a transitive union. A `--resolution` knob trades granularity
(8 → 38 clusters on a 7 d corpus; 22 clusters at res 1.0 on the full corpus),
and modularity is reported (**0.75**).

- The **710-doc GitHub blob is gone.** Largest cluster is now 462 docs (the
  slide-deck *sessions*, not a GitHub blob); the old blob is distributed across
  recognizable topics — OCR (`ocr, sie server, olmocr, docling`), gateway
  (`worker, gateway, config`), MTEB (`mteb, target, quality`), infra
  (`superlinked sie, terraform, aws`), docs (`docs, sdk, mdx`).
- **Labels are extractive c-TF-IDF keyphrases** (no LLM): the "local" blobs
  became nameable themes (`next plaid, rust, synty`; `pdl, people, company`;
  `rlm, daytona, sandbox, e2b`).

**Retrieval → still PASS (12/12).** The same probe set returns the same relevant
top hits (OCR `#1149`, gateway isolation `#1136`, SOC2 `infrastructure#225`,
docs-search `sie-web#91`, VLM cache `#1143`, dense loader `#1144`); the
clustering change does not touch retrieval.

**Encode + iteration.** `--features metal` runs encode on the Apple GPU, ~5.7×
faster than the hardcoded CPU path (37 vs 6.5 docs/s; 3,938 docs in 97 s).
`index` persists per-doc embeddings and `cluster` caches the weighted graph, so
a resolution sweep re-runs only Louvain (~0.7 s) and never re-encodes or
re-searches. The shipped default stays plain CPU.

**Known cost.** A *fresh* full-corpus cluster build is search-bound (~335 s at
3,938 docs, even with approximate search params); cached re-runs are
sub-second. A candidate for later optimization (lower probe, or pooled-vector
kNN for the graph), not on the M1 path.

Tests: 22 scenario unit tests green (Louvain split/resolution/modularity,
c-TF-IDF labels, same-repo link edges, plus the originals).
