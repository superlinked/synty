# synty — System Design

**Status:** v0.8 (post-pivot) · **Date:** 2026-05-31 · **Owner:** Daniel Svonava

synty is a passively-collected memory of how work actually happens: it ingests
coding-agent sessions (Claude Code, Codex, Cowork) and GitHub activity, and makes
the result searchable and readable by both humans and agents. The pivot from v1
(`superlinked/synty-legacy`) is to a **single self-contained Rust binary** built
on **late-interaction retrieval with no generative model** — nothing leaves the
machine, no API keys, runs offline.

This document owns the architecture and the target end-state, and marks what is
already built. `roadmap.md` owns the sequence to get there. `eval_report.md`
records the validation of the kernel on real data.

---

## Principles

- **No generative model.** Embeddings (ColBERT late-interaction) + deterministic
  logic + extractive summarization. For a tool that ingests dev transcripts,
  "your data never leaves the machine" is the feature.
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

- **Encode:** `pylate-rs` (ColBERT on Candle, CPU, ModernBERT backend). Default
  model `mixedbread-ai/mxbai-edge-colbert-v0-32m` (32 M params, 127 MB, PyLate
  format). `model.rs` resolves a model to a local dir, downloading on first use
  under `~/.cache/synty/models` with connect/read timeouts and retry (avoids the
  hf_hub no-timeout hang); a directory spec is used verbatim.
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
| Claude Code / Codex / Cowork | tail local session files → canonical envelopes | **reuses v1 Go agent dumps today**; native Rust tailers planned |
| GitHub (PRs, issues, comments) | `gh` / GraphQL backfill + webhooks | **`gh` JSON today**; App-based ingestion planned |

The target is native Rust tailers (one per tool) that write envelopes to the
bucket, plus a GitHub path that does not depend on a developer machine.

## Derivations (all without an LLM)

- **Search** — filtered late-interaction retrieval. *Built.*
- **Clusters (topics)** — emergent, no taxonomy. Today: mutual-kNN over
  similarity ∪ GitHub `#`-references → connected components. This over-merges a
  homogeneous repo (one giant component); the target is **community detection
  (Louvain/Leiden) with a resolution knob**, treating GitHub links as a weighted
  signal rather than a transitive union. *Built (interim); refinement planned.*
- **Summaries** — extractive. Per session: opening ask, files touched, prompt
  count, linked PR. Per topic: counts, repos, notable titles. *Built.*

## Surfaces

- **CLI → stdout (agents):** `search`, `topic`, `recent` print Markdown an agent
  reads over the shell — no server, no auth, no network. *Built: `search`.*
- **TUI (humans):** tracker status (is it running, what it sees, throughput,
  autostart) + browse/drill. *Planned.*
- **MCP server (optional):** expose `synty_search / synty_topic / synty_session`
  as agent tools. *Planned.*
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
events/<source>/<yyyy-mm-dd>/…       append-only envelopes (source of truth)
index/                               next-plaid PLAID index (derived projection)
meta.sqlite                          metadata for WHERE-filtering (derived)
```

A web/worker node rebuilds `index/` + `meta.sqlite` from `events/`, the same way
the binary does locally.

## What's built (kernel)

A working binary: `ingest` (v1 dumps + `gh` → `corpus/docs.jsonl`), `index`
(encode + build), `search [--filter col=value]`, `cluster`, `summarize`, `eval`,
plus 16 scenario tests. Validated on real data (3,157 docs / 652 K embeddings):
retrieval 12/12 relevant top-3, agent task-start dogfood 3/3, extractive session
summaries specific and accurate, all with no generative model. Clustering works
for sessions; GitHub over-merges (the named refinement above). Full results in
`eval_report.md`.

## Stack

Rust (edition 2024). `pylate-rs`, `next-plaid`, `candle-core`, `ndarray`,
`serde`, `clap`, `ureq`. Cross-compiles to darwin-arm64/amd64 + linux-amd64;
CPU by default, optional Candle `accelerate`/`openblas` for faster encode.
