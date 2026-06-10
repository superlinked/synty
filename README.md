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
  summarizer (~1.2 GB) downloads on first `summarize`.
- **Agents are first-class readers.** The main surface is a CLI that prints
  Markdown to stdout — exactly what a coding agent reads before starting, so it
  builds on prior work instead of starting cold.

## Quickstart

```sh
cargo build --release

# One-time: connect GitHub, pick the org to track, enable login-time tracking.
cargo run --release -- setup

# Track your agent sessions + GitHub and rebuild a fresh index, on a loop.
# (The embedding model downloads automatically on first run.)
cargo run --release -- up
```

`setup` verifies your GitHub token, lists the orgs (and your own account) it can
see, and lets you pick one — its most-recently-active repos are what gets
back-filled — then offers to start the tracker at login. `up` tails your local Claude Code / Codex / Cowork
session files, pulls the org's recent PRs/issues, and rebuilds the search index
every minute. (No GitHub yet? `up` still tracks local sessions; run `setup`
anytime to add it.) Once you see `indexed N docs`, query it from another
terminal:

```sh
cargo run --release -- search "rate limiting middleware"
cargo run --release -- search "fix flaky login test" --filter repo=api
```

Results are ranked Markdown cards — PRs, issues, and session moments — that you
or an agent can read directly. No accounts, no network calls.

Prefer to browse? `cargo run --release -- tui` opens an interactive terminal UI:
tabs for **Topics**, **Work**, **Search**, and **Status**, with drill-down from
a topic into its sessions and the full document text. Filter any list to one
repo or person with `r`/`p`, and the Status tab breaks activity down per repo
and account and toggles login-time tracking. The CLI has the same surface for
agents and scripts:

```sh
cargo run --release -- topic            # emergent topics (or `topic auth` to filter)
cargo run --release -- recent           # latest PRs, issues, and prompts
cargo run --release -- status           # what's indexed and how fresh it is
```

### Or run the steps yourself

`up` is just a loop over these (handy for one-off use or scripting):

```sh
cargo run --release -- track     # tail agent sessions → canonical event envelopes
cargo run --release -- github    # pull the org's active PRs/issues (org from `setup`; $GITHUB_TOKEN or gh)
cargo run --release -- ingest    # turn events + GitHub into searchable documents
cargo run --release -- index     # encode with ColBERT and build the index
cargo run --release -- search "<query>" [--filter col=value]
cargo run --release -- summarize                  # local-LLM one-line summary per unit
cargo run --release -- cluster --resolution 2.0   # group units by summary embedding → topics
cargo run --release -- summarize                  # again: reduce each topic from its members
```

`cluster` groups *units of work* (sessions, PRs, issues) by their summary, so run
`summarize` first; the topic summaries are then a second `summarize` pass.

## Team mode (optional)

Solo, the "bucket" is a local directory. For a team, point every command at a
shared S3/GCS bucket and many machines, VMs, and sandboxes converge there:

```sh
cargo run --release --features s3 -- track  --bucket s3://my-team   # each device pushes
cargo run --release --features s3 -- ingest --bucket s3://my-team   # a builder pulls all
cargo run --release --features s3 -- index  --bucket s3://my-team   # build once
cargo run --release --features s3 -- search "..." --bucket s3://my-team  # others just query
```

Every device's events coexist in the bucket, each message is encoded only once
across the whole fleet (content-addressed), and one machine builds the index
while the rest download and query it. Use `gs://` with `--features gcs`. Register
the tracker to start at login with `track --install launchd|systemd`.

## How it works

- **Encode** — `pylate-rs` runs a small ColBERT model (ModernBERT, 32 M params)
  that represents each document as one vector *per token*. This late-interaction
  approach retrieves far better than single-vector embeddings on short,
  code-adjacent text.
- **Index & search** — `next-plaid` (PLAID) stores those vectors and scores
  queries with MaxSim, backed by SQLite for exact metadata filters
  (`--filter repo=...`, `kind=pull_request`, …).
- **Topics** — units of work (sessions, PRs, issues) clustered by the embedding
  of their one-line summary (Louvain over a MaxSim kNN graph), each named by the
  local model with a short title that's checked for faithfulness to its members
  (grounded fallback if it drifts); a `--resolution` knob trades more, smaller
  topics for fewer, larger ones. A topic is a coherent set of units, not a bag
  of messages.
- **Source of truth** — sessions and GitHub items become append-only event
  envelopes; the index and its metadata are derived projections, rebuildable
  from the events at any time (and shareable through a bucket).

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
