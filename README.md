# synty

**A passive memory of how your work actually happens.** synty quietly watches
your coding-agent sessions (Claude Code, Codex, Cowork) and your GitHub activity,
and makes all of it searchable — by you, and by the agents you work with. Before
starting a task, ask *"has anyone touched the auth flow?"* or *"what did we
decide about rate limiting?"* and get the relevant prior sessions and PRs
back in seconds.

It runs as a single local binary. **No API keys, nothing leaves your machine** —
not even the summaries, which a small model generates locally. For a tool that
ingests your dev transcripts, that is the whole point.

## Why synty

- **Private by design.** Retrieval is late-interaction embeddings (ColBERT) plus
  deterministic logic — no LLM in the loop. Session summaries are written by a
  small local model (Qwen3-0.6B) that runs offline on your CPU, so your sessions
  are never sent anywhere.
- **Local-first, one binary.** No server, no Python, no Docker to get value. The
  embedding model (~127 MB) downloads once and runs offline after; the optional
  summarizer (~1.2 GB) downloads the first time anything summarizes
  (`build`, `tui`, or `summarize`).
- **Agents are first-class readers.** The main surface is a CLI that prints
  Markdown to stdout — exactly what a coding agent reads before starting, so it
  builds on prior work instead of starting cold.

## Quickstart

```sh
cargo build --release

# One command from nothing to tracking + a first build. Omit the bucket for a
# local-only trial; pass your team bucket to join the fleet (re-run with a
# bucket later to switch a local trial onto the team).
cargo run --release -- join            # local trial
cargo run --release -- join gs://my-team   # or join the fleet

# Or run the loop yourself: track, ingest, and index every minute — your
# machine is both the tracker (lightweight, model-free) and the builder
# (downloads the embedding model on first run; `build`/`tui` also fetch the
# summarizer). In fleet mode machines run only the tracker.
cargo run --release -- up
```

`join` pins your GitHub identity (so sessions merge with your PRs), enables the
login-time tracker, and runs the first build — non-interactive and idempotent.
`up` tails your local Claude Code / Codex / Cowork session files, pulls the
org's recent PRs/issues, and rebuilds the search index every minute. Once you
see `indexed N docs`, query it from another terminal:

```sh
cargo run --release -- search "rate limiting middleware"
cargo run --release -- search "fix flaky login test" --filter repo=api
```

Results are ranked Markdown cards — PRs, issues, and session moments — that you
or an agent can read directly. No accounts, no network calls.

Prefer to browse? `cargo run --release -- tui` opens an interactive terminal UI:
tabs for **Topics**, **Work**, **Search**, **Stats**, and **Status**, with
drill-down from a topic into its sessions and the full document text. Filter
any list to one repo or account with `r`/`a`, refresh on demand with `u` (the
footer shows build progress and staleness). The Stats tab charts what the
agents consume against what the work produces and breaks the spend down per
repo, account, tool, and model; the Status tab shows tracker health, the fleet
roster (who runs synty where, who runs agents untracked), whether you're running
**local (trial)** or **activated** on a team bucket, and toggles login-time
tracking. The CLI has the same surface for agents and scripts:

```sh
cargo run --release -- related          # prior work related to what you're doing now (from this repo's git)
cargo run --release -- topic            # emergent topics (or `topic auth` to filter)
cargo run --release -- recent           # latest PRs, issues, and prompts
cargo run --release -- status           # health: what's indexed, freshness, activation, the fleet roster
cargo run --release -- stats            # usage: tokens/tools/sessions vs LOC/PRs/issues per week
cargo run --release -- tool Bash        # one tool's profile: volume, latency, argument mix
cargo run --release -- show a1b2c3d4    # drill into a session, PR/issue (repo#123), or topic key
```

`related` is the zero-effort entry point: with no query, it reads this repo's
recent commits and changed files, builds a query from them, and surfaces prior
sessions and PRs across every repo synty has seen — run it before starting a
task to build on past work.

Output is built to be drilled: `search`, `recent`, and `topic` print stable
ids inline (`[a1b2c3d4]` for sessions, `repo#123` for PRs/issues, `[72a778f8]`
for topics) that feed `synty show <id>`. Every read command takes `--json`
for scripts; the output is one versioned envelope,
`{"v": 1, "kind": "…", "data": …}` — check `v` once, dispatch on `kind`. For
coding agents there's an MCP server — add it to the agent's MCP config and it
gets the same read surface as tools (`synty_search`, `synty_related`,
`synty_topics`, `synty_recent`, `synty_status`, `synty_stats`, `synty_tool`,
`synty_show`; `synty_related` takes the agent's repo path and needs no query;
ids in any tool's output feed `synty_show`):

```sh
cargo run --release -- mcp              # MCP over stdio
```

### Or run the pipeline yourself

`build` runs the whole pipeline once (track → ingest → index → summarize →
cluster → topic summaries), so `topic` and the TUI are fully populated:

```sh
cargo run --release -- build
```

The individual steps, for scripting (and what `up` loops over):

```sh
cargo run --release -- track     # tail agent sessions → canonical event envelopes
cargo run --release -- github    # pull the org's active PRs/issues (org from `join`; $GITHUB_TOKEN or gh)
cargo run --release -- ingest    # turn events + GitHub into searchable documents
cargo run --release -- index     # encode with ColBERT and build the index
cargo run --release -- search "<query>" [--filter col=value]
cargo run --release -- summarize                  # local-LLM one-line summary per unit
cargo run --release -- cluster --resolution 2.0   # group units by summary embedding → topics
cargo run --release -- summarize                  # again: reduce each topic from its members
```

`cluster` groups *units of work* (sessions, PRs, issues) by their summary, so run
`summarize` first; the topic summaries are then a second `summarize` pass.

## Fleet mode (optional)

Solo, the "bucket" is a local directory. For a team it's one shared S3/GCS
bucket — and the bucket is the *only* shared infrastructure: no build server,
no cron, no coordination service. That works because everything in it is
append-only (events), write-once (embeddings, summaries), or swapped
atomically behind a pointer (the index) — machines cooperate without ever
talking to each other.

Three roles, one binary:

- **Every machine writes.** The tracker runs at login: model-free, near-zero
  footprint, it tails local agent session files and pushes raw events.
- **One machine scrapes GitHub.** Whichever machine has a token refreshes the
  org's PRs/issues during its builds — incrementally, fetching only what
  changed since the last scrape — and shares the result. Nobody else needs a
  token.
- **Viewers build.** Opening `synty tui` (or running `synty build`) pulls the
  latest published read-model, then freshens in the background: encode and
  summarize only what no machine has done yet (the first to need something
  generates it for the fleet; concurrent viewers split the pending list), one
  soft lease per index build, publish, pointer swap. Idle for a week? Events
  accumulate; the next viewer pays an incremental catch-up.

```sh
# One paste per machine (or baked into VM images): installs the binary, runs
# `synty join <bucket>` (identity + login-time tracker + first build), and opens
# the viewer. Omit the bucket for a local trial; re-run with one to activate.
curl -fsSL <internal-url>/install.sh | sh -s -- s3://my-team
```

The binary is internally hosted (`$SYNTY_BINARY_URL`), not a public package or
Homebrew tap — distribution is team-first while the rollout proceeds. The
configured bucket is the default everywhere; `--bucket` overrides. Cloud buckets
need `--features s3` / `gcs`. The TUI footer shows where things stand —
activation (`● local` → `✓ activated`) and freshness (`⟳ encoding 120/470` ·
`⚠ stale` · `✓ fresh`).

Caveat: every fleet member has raw bucket access and can read everyone's
sessions. Fine for high-trust teams; the mediated-frontend tier (publication
delay, redaction) is the planned answer where it isn't.

## How it works

- **Encode** — `pylate-rs` runs a small ColBERT model (ModernBERT, 32 M params)
  that represents each document as one vector *per token*. This late-interaction
  approach retrieves far better than single-vector embeddings on short,
  code-adjacent text.
- **Index & search** — `next-plaid` (PLAID) stores those vectors and scores
  queries with MaxSim, backed by SQLite for exact metadata filters
  (`--filter repo=...`, `kind=pull_request`, …).
- **Topics** — units of work (sessions, PRs, issues) clustered by the embedding
  of their one-line summary plus project context (Louvain over a MaxSim kNN
  graph, near-duplicate reruns collapsed onto one representative), each named
  by the local model with a short title that must share one of the cluster's
  *distinctive* terms (c-TF-IDF against the other clusters), may not be a
  bare repo name, and must embed close to its members (MaxSim against the
  cluster's own embeddings) — otherwise an extractive keyword title takes
  its place; a
  `--resolution` knob trades more, smaller topics for fewer, larger ones. A
  topic is a coherent set of units, not a bag of messages.
- **Source of truth** — sessions and GitHub items become append-only event
  envelopes; the index and its metadata are derived projections, rebuildable
  from the events at any time (and shareable through a bucket).
- **Session stats** — per-session token totals (in · out · cache-read ·
  cache-write, from the agent's own usage records, never estimated) and the
  tool-call mix with error counts; sessions whose source recorded no usage
  show no number rather than a fake zero.

Architecture and rationale live in `design.md`; the milestone plan in
`roadmap.md`; validation on real data in `eval_report.md`.

## Speed & offline

- On Apple Silicon, build with `--features metal` for GPU encode (~5.7× faster).
  `accelerate` (macOS) and `mkl` (Linux) are CPU-BLAS alternatives. The default
  build is plain CPU and fully portable.
- The model `mixedbread-ai/mxbai-edge-colbert-v0-32m` downloads on first use and
  is cached. For a guaranteed-offline setup, fetch it once and point
  `SYNTY_MODEL` at the local directory:

  ```sh
  m=models/mxbai; mkdir -p $m/1_Dense
  base=https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m/resolve/main
  for f in tokenizer.json config.json config_sentence_transformers.json \
           special_tokens_map.json 1_Dense/config.json 1_Dense/model.safetensors model.safetensors; do
    curl -sL --retry 8 --continue-at - "$base/$f" -o "$m/$f"
  done
  export SYNTY_MODEL="$PWD/models/mxbai"
  ```
