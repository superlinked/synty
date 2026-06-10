# synty — System Design

**Status:** v0.8 (post-pivot) · **Date:** 2026-05-31 · **Owner:** Daniel Svonava

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
| GitHub (PRs, issues, comments) | `synty github` GraphQL backfill (token) | **Built** (machine-independent, no `gh`); App install-token + webhooks planned |

`synty track` is the source of truth for sessions: a `Source` per tool detects
the file format version and builds a parser; a shared driver mints canonical
envelopes with deterministic event_ids (so a re-parse never duplicates). The
GitHub path pulls PRs/issues straight from the GraphQL API with a token, so it
runs on CI or a server without a developer machine.

## Derivations (all without an LLM)

- **Search** — filtered late-interaction retrieval. *Built.*
- **Clusters (topics)** — emergent, no taxonomy. Clustering is over **units of
  work** (sessions, PRs, issues), not the raw message firehose: each unit's
  one-line summary is embedded (multi-vector ColBERT) and **Louvain** runs over a
  MaxSim kNN graph of those summary embeddings (normalized per-unit, floored,
  summed over both directions). A topic is therefore a coherent *set of units*,
  so its members, facets (repos/authors), label, and summary are consistent by
  construction — no doc-vs-unit reconciliation. A `--resolution` knob trades
  topic count vs size; modularity is reported. Labels are extractive c-TF-IDF
  keyphrases over the member summaries; summary embeddings are content-addressed
  in the shared store (encode-once). Clustering the denoised summaries also beats
  clustering the chat firehose. *Built (unit-summary Louvain + resolution +
  keyphrase labels).*
- **Summaries.** Per session: opening ask, c-TF-IDF keyphrases, files touched,
  effort, linked PR, and a one-line abstractive summary from a local Qwen3-0.6B
  (greedy decode, cached by input hash next to the index so the reader never runs
  the model at view time; falls back to an extractive representative line without
  the `llm` feature). Per topic: counts, repos, notable titles — extractive.
  *Built (extractive + local LLM session summaries).*

## Surfaces

- **CLI → stdout (agents):** `search`, `topic`, `recent`, `status` print Markdown
  an agent reads over the shell — no server, no auth, no network. *Built.*
- **TUI (humans):** `synty tui` — tabs for status, topics, recent, and search,
  with browse/drill (topic → members → full document), reusing the CLI's
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
opens a viewer pulls the published read-model instantly, then contributes a
build — encoding/summarizing only what no other machine has done (the
write-once stores are the collaboration primitive; concurrent viewers shuffle
the pending list with per-machine seeds and split it). The lease only prevents
duplicate index builds; losing it can't corrupt anything because publishes are
immutable-prefix + pointer-swap. Locally the same shape: builds live under
`index/builds/<build>/`, `index/current.json` repoints atomically, incremental
appends clone the previous build (CoW), and readers — including a TUI holding
a live mmap — are never mutated under. `synty up` runs the loop locally for
solo use; `synty build` is the one-shot fleet-aware build.

## What's built (kernel)

A working binary: `up` (one-command solo loop), `track` (native tailers →
envelope streams, `--bucket` to push), `github` (GraphQL backfill), `ingest`
(envelopes + GitHub → `corpus/docs.jsonl`, `--bucket` to pull), `index` (encode
+ build + content-addressed store + publish), `search [--filter col=value]`
(pulls the published read-model), `topic`, `recent`, `status`, `tui`,
`cluster [--resolution]`, `summarize`, `eval`, plus 55 scenario tests. The
bucket backplane (local always, S3/GCS opt-in) gives fleet-wide encode-once and
build-once-read-many.
Validated on real data (3,938 docs / 770 K embeddings): retrieval 12/12 relevant
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
