# synty — System Design

**Status:** v0.9 (fleet) · **Date:** 2026-06-11 · **Owner:** Daniel Svonava

synty is a passively-collected memory of how work actually happens: it ingests
coding-agent sessions (Claude Code, Codex, Cowork) and GitHub activity, and makes
the result searchable and readable by both humans and agents. The pivot from v1
(`superlinked/synty-legacy`) is to a **single self-contained Rust binary** built
on **late-interaction retrieval, with generation only for local summaries** —
nothing leaves the machine, no API keys, runs offline.

This document owns the architecture and the target end-state, and marks what is
already built. `roadmap.md` owns the sequence to get there. `eval_report.md`
records the validation of the kernel on real data.

---

## Principles

- **Nothing leaves the machine.** The core — retrieval (ColBERT late-interaction),
  clustering, and keyphrases — is deterministic and embedding-only, never an LLM.
  Session summaries are the one exception: a small local model (Qwen3-0.6B on
  candle, the `llm` feature) generates them offline. No data leaves the machine
  either way; that is the actual feature, and a local model preserves it.
- **Local-first, self-contained.** One static binary. The model downloads once
  and runs offline thereafter. No server, no Python, no Docker required to get
  value.
- **The bucket is the backplane.** Events are the durable source of truth; the
  search index and SQLite metadata are derived projections, rebuildable from
  events. Solo = a local directory; team = a shared S3/GCS bucket.
- **Agents are first-class readers.** The primary agent surface is a CLI that
  prints Markdown to stdout; humans get a TUI; a team optionally runs a web
  frontend over the same data.

## Engine

- **Encode:** `pylate-rs` (ColBERT on Candle, ModernBERT backend). Default
  model `mixedbread-ai/mxbai-edge-colbert-v0-32m` (32 M params, 127 MB, PyLate
  format). `model.rs` resolves a model to a local dir, downloading on first use
  under `~/.cache/synty/models` with connect/read timeouts and retry (avoids the
  hf_hub no-timeout hang); a directory spec is used verbatim. The shipped build
  is plain CPU and portable; opt-in cargo features pick a faster backend —
  `metal` (Apple GPU, ~5.7× encode on Apple Silicon, with CPU fallback if the
  GPU can't init), `accelerate` (macOS CPU BLAS), `mkl` (Linux CPU BLAS).
- **Index / search:** `next-plaid` (PLAID multi-vector index + MaxSim scoring,
  SQLite metadata store). Filtered search resolves a `column=value` predicate to
  a doc-id subset via the metadata DB, then runs MaxSim over it.
- **Why late interaction:** one ~128-dim vector per token (not per document)
  retrieves far better than single-vector embeddings on short, code-adjacent
  text, and the SQLite side gives exact metadata filtering for free.

## Data model

A **document** is the indexed unit: `{id, text, meta}` where `meta` carries
`source` (`github | agent`), `kind` (`pull_request | issue | user_prompt |
assistant_message | thinking`), `repo`, `author`, `session_id`, `ts`, and
GitHub-only `number / url / state / labels`. Sessions are chunked one document
per user/assistant/thinking message; system-injected pseudo-prompts (hook
echoes, tool output, reminders) are filtered out. GitHub items are title+body.

Events (the raw envelopes behind documents) are the source of truth and live in
the bucket; `docs.jsonl` + the index are derived.

## Ingestion

| Source | Mechanism | Status |
|---|---|---|
| Claude Code / Codex / Cowork | `synty track` tails local session files → canonical envelopes | **Built** (native Rust tailers, one per tool; `--watch` + per-file cursors + launchd/systemd autostart) |
| GitHub (PRs, issues) | `synty github` GraphQL backfill (token) | **Built** — incremental (UPDATED_AT-floored at the last scrape, so steady-state runs fetch only changes incl. state flips); the corpus + manifest share through the bucket, and `build` refreshes when stale and a token is present, so one tokened machine scrapes for the fleet. App install-token + webhooks planned |

`synty track` is the source of truth for sessions: a `Source` per tool detects
the file format version and builds a parser; a shared driver mints canonical
envelopes with deterministic event_ids (so a re-parse never duplicates). The
GitHub path pulls PRs/issues straight from the GraphQL API with a token, so it
runs on CI or a server without a developer machine.

## Derivations (all without an LLM)

- **Search** — filtered late-interaction retrieval. *Built.*
- **Clusters (topics)** — emergent, no taxonomy. Clustering is over **units of
  work** (sessions, PRs, issues), not the raw message firehose: each unit's
  one-line summary is embedded (multi-vector ColBERT; sessions lead with their
  repo and touched files, summary appended) and **Louvain** runs over a
  MaxSim kNN graph of those embeddings (normalized per-unit, floored,
  summed over both directions), with near-duplicate units — the same work item
  re-run — collapsed onto one representative before any graph work. A topic is
  therefore a coherent *set of units*,
  so its members, facets (repos/authors), label, and summary are consistent by
  construction — no doc-vs-unit reconciliation. A `--resolution` knob trades
  topic count vs size; modularity is reported. Each cluster's provisional label
  is its most concise member summary, replaced once `summarize` titles the topic
  — the title is gated on the cluster's distinctive (c-TF-IDF) terms, banned
  from being a bare repo slug, and embedding-checked against the members
  themselves, with an extractive keyword title as the fallback;
  summary embeddings are content-addressed in the shared store (encode-once).
  Clustering the denoised summaries also beats clustering the chat firehose.
  *Built (unit-summary Louvain + resolution + gated topic names).*
- **Summaries.** Per session: opening ask, c-TF-IDF keyphrases, files touched,
  effort, linked PR, and a one-line abstractive summary from a local Qwen3-0.6B
  (greedy decode, cached by input hash in `.synty/` — and shared fleet-wide as
  write-once bucket objects — so the reader never runs
  the model at view time; falls back to an extractive representative line without
  the `llm` feature). Per topic: a one-line summary reduced from the member
  summaries plus the short gated name, same model and cache; counts and repos
  stay extractive. *Built (extractive + local LLM session summaries).*
- **Session stats (tokens & tool calls).** Per-session token totals and tool
  mix, derived at view time from the raw envelopes — the agent's own usage
  records, never tokenizer estimates. Capture per source: Claude Code emits a
  usage envelope per raw assistant line (streamed turns repeat the identical
  object across the lines of one message id — measured 2–4× — so aggregation
  dedups by msg_id; embedding usage on the message payload would silently drop
  tool-use-only turns, hence its own envelope); Codex token_count snapshots
  are cumulative, so the last one is the session total, normalized at read
  time (fresh in = input − cached; the corpus keeps codex's raw semantics, so
  the policy is retroactively revisable — historical codex sessions lit up
  with no re-tracking); Cowork records no usage. The four classes stay
  separate end-to-end (in · out · cache-read · cache-write — a cache read is
  not a fresh input), tool calls tally by name with errors from
  `tool_result.is_error`, and a session without usage shows no number at all
  (never a fake 0). Subagent (`agent-`) sessions count separately — parent
  totals exclude children until a rollup over the existing `subagent_parent`
  edges; per-model tables, tool durations, and an optional pricing map are
  deferred with it. `[metrics stats]` reports `usage_coverage_pct` — the
  share of sessions whose source recorded usage. *Built (P1).*
- **Fleet coverage (install rate).** Whether synty runs everywhere agents run,
  from data already in the bucket: the roster folds every `edge-<machine>-…`
  stream (liveness from envelope timestamps — file mtimes lie after a pull;
  actor stamps; `tracker_version` for upgrade lag) and joins it against the
  GitHub authors active in a trailing window. Untracked authors whose PRs name
  an agent (Co-authored-by trailer, Generated-with footer, bot author — flagged
  at ingest as `agent_attr`, precision-first) are the actionable "runs agents,
  unwatched" list; a machine that streamed and went quiet is tracker rot,
  distinct from never-installed. `ingest` emits a `[metrics coverage]` block
  (machines, actors, install rate). *Built (M8).*

## Surfaces

- **CLI → stdout (agents):** `search`, `topic`, `recent`, `status` print Markdown
  an agent reads over the shell — no server, no auth, no network. *Built.*
- **TUI (humans):** `synty tui` — tabs for topics, work, search, stats (usage
  charts + spend tables), and status (self-health + fleet roster), with
  browse/drill (topic → members → full document), reusing the CLI's
  view-models. *Built.*
- **MCP server:** `synty mcp` serves `synty_search / synty_topics /
  synty_recent / synty_status` as agent tools over stdio (hand-rolled JSON-RPC,
  no new deps), so a coding agent consults past work mid-session. *Built.*
- **JSON output:** `--json` on `search / topic / recent / status` for scripts.
  *Built.*
- **Team web frontend (optional):** the same view models served over HTTP for a
  shared bucket. *Planned.*

## Tiers and the trust boundary

- **Solo / local:** binary writes events to a local dir, builds the index
  locally, answers from the CLI/TUI. No mediation needed — you are the only
  reader. No server, no creds.
- **Team / company:** a shared bucket plus **one frontend** that mediates reads.
  The frontend is where the privacy guardrails live, because read rules cannot
  be enforced on a client with raw bucket access.

## Privacy (team tier)

Mediated at the frontend: a publication delay before a session is visible to
non-originators; secret/credential redaction before shared reads; per-session
erasure. Solo mode needs none of this. Rollups are by work (topic/repo/period),
never by person.

## Storage layout (bucket)

```
events/<stream>/…                     append-only envelopes (source of truth);
                                        stream = edge-<machine>-<source>, so many
                                        trackers' files coexist without collision
embeddings/<hash[..2]>/<hash>.emb      content-addressed f16 vectors (write-once)
summaries/<kh[..2]>/<kh>-<ihash>.json  per-(unit, input-hash) LLM summaries
                                        (write-once: first viewer generates for
                                        the fleet; empty = tried + gate-rejected)
blobs/<fnv>                            content-addressed build files (index
                                        chunks, docs snapshot, clusters) — shared
                                        across builds, so appends upload deltas
builds/<build>.<rev>.json              manifest: filename → blob, per (build, rev)
current.json                           the pointer, PUT last — readers never see
                                        a torn build; rev versions the clusters
lease/build                            soft TTL lease electing one index builder
```

A `Bucket` trait abstracts this store (local dir always; S3/GCS behind
`--features s3/gcs`, with conditional PUT for write-once and the lease). The
fleet model is **no designated builder**: every tracker pushes events; whoever
opens a viewer pulls the published read-model, then contributes a build.
Write-once stores are the collaboration primitive — a viewer encodes and
summarizes only what no other machine has (pending lists shuffle per machine,
so concurrent viewers split the work). The lease only prevents duplicate index
builds; losing it wastes compute, never corrupts, because publishes are
immutable-prefix + pointer-swap. The local layout mirrors this:
`index/builds/<build>/` + an atomically repointed `index/current.json`;
incremental appends clone the previous build (CoW), so a reader's live mmap is
never mutated under it. `synty up` loops locally; `synty build` is the
one-shot fleet-aware build.

## Data compatibility

- **Envelopes are add-only, forever.** Fields are never renamed or repurposed;
  readers skip unknown kinds and fields and default absent ones. Each envelope
  carries `v` (currently 1), bumped only for a breaking change we intend never
  to make. This is the contract that keeps raw tracked data useful forever —
  everything else is a regenerable projection of it.
- **Derived artifacts are versioned by what produced them.** Embeddings are
  namespaced by encoder model (the default model keeps the original layout);
  summary hashes are salted by prompt version and any non-default summarizer
  model. Different models never share artifacts; changing a version
  regenerates exactly the affected entries, fleet-wide, once.
- **The read-model pointer carries `format` + the writer's version.** A reader
  meeting a newer format refuses to pull it and says to upgrade; an unreadable
  derived blob is a cache miss, never an error.

## What's built (kernel)

A working binary: `up` (solo loop), `build` (one-shot fleet-aware pipeline),
`track` (native tailers → envelope streams, `--bucket` to push), `github`
(GraphQL backfill), `ingest` (envelopes + GitHub → `corpus/docs.jsonl`,
`--bucket` to pull), `index` (encode + content-addressed store + versioned
build + publish), `search [--filter col=value] [--json]`, `topic`, `recent`,
`status`, `tui`, `mcp`, `cluster [--resolution]`, `summarize`, `eval`, plus the
scenario test suite (`cargo test`, pure). The bucket backplane (local always,
S3/GCS opt-in) gives fleet-wide encode-once and collaborative builds.
Validated at M0/M1 on real data (3,938 docs / 770 K embeddings): retrieval 12/12 relevant
top-3, agent task-start dogfood 3/3, session summaries specific and accurate
(extractive in the core; one-line abstractive from a local Qwen3-0.6B under the
`llm` feature, with retrieval and clustering staying LLM-free). Clustering (M1) is Louvain over the
weighted graph: the prior GitHub over-merge (a 710-doc blob) is gone — 22
recognizable keyphrase-labeled topics, largest 462 docs, modularity 0.75. Full
results in `eval_report.md`.

## Stack

Rust (edition 2024). `pylate-rs`, `next-plaid`, `candle-core`, `ndarray`,
`serde`, `clap`, `ureq`. Cross-compiles to darwin-arm64/amd64 + linux-amd64;
CPU by default, opt-in `metal` (Apple GPU) / `accelerate` (macOS) / `mkl`
(Linux) features for faster encode.
