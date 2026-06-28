# synty

**Your coding agents start every session from zero. So do you.**

synty quietly records every coding-agent session (Claude Code, Codex, Cowork)
and your GitHub activity, building one local, searchable memory of how the work
actually happened. The tracker installs once and runs everywhere, and the data
it captures is yours to keep and read directly.

**Two ways to use it:**

- **Find past work in seconds.** Ask *"has anyone touched the auth flow?"*, or
  just run `synty related`, and it surfaces the sessions and PRs that matter. You
  browse in the TUI; your agents read the same answers over the CLI and MCP.
- **Build on the raw data.** Everything synty stores is open: raw events as plain
  JSONL on disk (and in the shared bucket), a SQLite metadata database, `--json`
  on every command. See where agents get stuck, which tools burn the most tokens,
  what to fine-tune on, and which tools your agents actually need. Nothing is
  trapped in a built-in viewer.

**Local-first:** one binary, no API keys, nothing leaves your machine, and even
the one-line summaries are written by a small model on your CPU. synty's topic
clustering and cross-source links (PRs, issues, and sessions, joined) mean the
data comes already structured, so you're not analyzing a pile of raw logs.

## See it in action

Browse your work memory in the terminal: topics, drill-down, search, stats, the
fleet roster.

![synty's TUI: topics, drill-down, search, stats, status](docs/tui.gif)

One paste from nothing to tracking plus a viewer. Start local, point at a bucket
later (`◐ local` → `✓ on the team`).

![synty init onboarding: local trial, then join the team](docs/install.gif)

The agent surface: `related` / `search` / `status` printing Markdown to stdout.

![synty CLI: related, search, status](docs/cli.gif)

<sub>Rendered from `docs/*.tape` with [vhs](https://github.com/charmbracelet/vhs); see [CONTRIBUTING](CONTRIBUTING.md#rendering-the-demo-gifs) to re-render.</sub>

## Install & update

One paste takes you from nothing to tracking plus a viewer. The bucket is
optional: omit it to trial synty against your own sessions, add it later to
share memory across your machines or your team.

```sh
curl -fsSL https://raw.githubusercontent.com/superlinked/synty/main/install.sh | sh                      # local trial
curl -fsSL https://raw.githubusercontent.com/superlinked/synty/main/install.sh | sh -s -- gs://my-team   # share a bucket
```

The installer puts the binary on PATH, runs `synty init [bucket]` (GitHub
identity, login-time tracker, first build), and opens the viewer. The binary is
the prebuilt asset from the latest [GitHub
Release](https://github.com/superlinked/synty/releases).

**Update** with `synty upgrade`: it pulls the latest release, checks the sha256,
swaps the binary in place, and restarts the tracker. It's a no-op when you're
current, and `synty status` (plus the TUI footer) flags a newer build.

*Building from source? See [Build & offline](#build--offline) and replace `synty`
with `cargo run --release --`.*

## Start local, then connect your machines or team

You don't need a bucket to get value. Start on one machine; add a bucket when you
want synty to span more than one.

**1. Local-first (recommended).**

- `synty init` pins your identity, turns on the login-time tracker, and builds an
  index from your local sessions (plus your org's PRs if you have a
  `gh`/`GITHUB_TOKEN`). Status reads **`◐ local`**.
- Use it: the tracker runs at login, you browse in `synty tui`, and you run
  `synty related` before tasks. Everything stays on your machine; the fleet
  can't see you.
- Add a bucket when you're ready: `synty init gs://my-team`. The bucket is the
  shared memory, whether that's your own laptop and desktop or your whole team.
  Status flips to **`✓ my-team`**.

**2. Straight to a bucket.** If you already trust it, install with the bucket
(`… | sh -s -- gs://my-team`) and land at `✓ my-team`.

The only thing that activates you is setting a bucket. The login-time tracker is
on by default the whole time, with its own indicator; it's never a second gate.

## Use it: you browse, your agents read

**You, in the TUI.** `synty tui` opens tabs for **Topics**, **Work**,
**Search**, **Stats**, and **Status**, with drill-down from a topic into its
sessions and the full text. Filter to one repo or account (`r`/`a`), refresh on
demand (`u`). The footer shows where you stand: activation (`◐ local` →
`✓ <bucket>`) and freshness (`◐ encoding 120/470` · `⚠ stale` · `✓ fresh`).

**Your agents, over the CLI and MCP.** Every read command prints Markdown to
stdout, no server or auth. `synty related` is the zero-effort entry point: no
query, it reads the current repo's recent commits and changed files and surfaces
prior work across everything synty has seen.

```sh
synty related                  # prior work for what you're doing now (from this repo's git)
synty search "rate limiting"   # semantic search (--filter repo=api, kind=pull_request, …)
synty recent                   # latest PRs, issues, and prompts
synty topic                    # emergent topics (or `topic auth` to filter)
synty status                   # health: indexed, freshness, activation, the fleet roster
synty stats                    # usage: tokens/tools/sessions vs LOC/PRs/issues per week
synty show a1b2c3d4            # drill into a session, PR/issue (repo#123), or topic
```

Each result is a ranked Markdown card, PRs/issues and session moments together
(illustrative):

```text
## rate limiting middleware

1. [24.3] **pull_request api#1487** — Add a token-bucket limiter to the gateway
   merged · https://github.com/acme/api/pull/1487
2. [21.8] _user_prompt · api · a3f1c2d9_
   how do we share the per-tenant limiter's state across pods? settled on Redis…
3. [19.0] **issue api#1502** — 429s under burst load on the search endpoint
   open · https://github.com/acme/api/issues/1502
```

Ids ride inline (`[a1b2c3d4]` sessions, `repo#123` PRs/issues, `[72a778f8]`
topics) and feed `synty show <id>`. Add `--json` to any read command for a
versioned envelope (`{"v": 1, "kind": …, "data": …}`). `synty mcp` serves the
whole surface to a coding agent over stdio.

## Own your data

synty is a capture layer, not a walled garden. Everything it records sits in
plain files under your synty home (`~/.synty` by default), so you can analyze it
however you like without going back through synty.

- **Raw events** (the source of truth): append-only JSONL under `corpus/local/`,
  and `events/<stream>/` in a shared bucket. The envelope schema is in
  [`docs/design.md`](docs/design.md).
- **Documents** (sessions + GitHub, ingested): `corpus/docs.jsonl`, one object
  per line with `meta` (`source`, `kind`, `repo`, `author`, `session_id`, `ts`).
- **Metadata**: a SQLite database under `index/`; the `meta` fields are the
  columns, so any SQLite tool can query it.

The shaped views are already a pipe away:

```sh
# weekly token, tool, and session totals
synty stats --json | jq '.data.weeks[] | {week: .start, tok_in, tok_out, tools}'

# one tool's profile: calls, errors, latency (where agents get stuck)
synty tool Bash --json | jq '{calls: .data.calls, errors: .data.errors, p95_ms: .data.p95_ms}'

# count documents by kind, straight from the raw file
jq -r '.meta.kind' ~/.synty/corpus/docs.jsonl | sort | uniq -c
```

Build a dashboard on it, fine-tune a model on your own sessions, decide which
tools your agents actually need. And because synty already clusters the work and
links across sources (PRs, issues, sessions), your analysis starts from
structured data instead of a heap of logs.

## Fleet mode

Solo, the "bucket" is a local directory. For a team, or just your own laptop and
desktop, it's one shared S3/GCS bucket, and that bucket is the *only* shared
infrastructure: no build server and no coordination service. It works because
everything in it is append-only (events), write-once (embeddings, summaries), or
swapped atomically behind a pointer (the index).

Three roles, one binary:

- **Every machine writes.** The tracker runs at login: model-free, near-zero
  footprint, tailing local session files and pushing raw events.
- **One machine scrapes GitHub.** Whoever has a token refreshes the org's
  PRs/issues during its builds and shares the result. Nobody else needs a token.
- **Viewers build.** Opening `synty tui` (or `synty build`) pulls the latest
  read-model, then freshens in the background: encode and summarize only what no
  machine has done yet, one soft lease per build, then a pointer swap.

Cloud buckets need `--features s3` / `gcs`. To exercise the whole sync path with
no cloud account, `scripts/fleet-smoke.sh` runs publish, a cold pull, then a
delta pull against a temp local bucket and checks the `[metrics sync]` numbers.

> **Heads up:** every member with the bucket can read every session in it. synty
> is built for high-trust groups; there's no redaction or per-reader mediation.
> If that doesn't fit, stay local, or share a bucket only among people who can
> already see each other's work.

## Run the pipeline yourself

`synty build` runs the whole thing (track → ingest → index → summarize →
cluster); `synty up` loops it every minute. The individual steps, for scripting:

```sh
synty track     # tail agent sessions → event envelopes
synty github    # pull the org's active PRs/issues (token or gh)
synty ingest    # events + GitHub → searchable documents
synty index     # encode with ColBERT and build the index
synty summarize # local one-line summary per unit (run again for topic summaries)
synty cluster --resolution 2.0   # group units by summary embedding → topics
```

## How it works

- **Late-interaction retrieval.** `pylate-rs` runs a small ColBERT model
  (ModernBERT, 32 M params) that encodes each document as one vector *per
  token*; `next-plaid` (PLAID) scores queries with MaxSim, backed by SQLite for
  exact metadata filters. Late interaction beats single-vector embeddings on
  short, code-adjacent text.
- **Summaries and names from a small local model.** The one-line session and
  topic summaries, and the gated topic names, come from Qwen3-0.6B on your CPU,
  never a remote API. A name has to share a distinctive cluster term and embed
  close to its members, or an extractive title wins.
- **Events are the source of truth.** Sessions and GitHub items become
  append-only envelopes; the index and metadata are derived projections,
  rebuildable any time and shareable through a bucket.
- **Activation is a bucket.** A machine is a fleet member exactly when a bucket
  is set. That single fact is what the status surface tracks.

Architecture and rationale live in [`docs/design.md`](docs/design.md); the
on-real-data validation lives in [`evals/`](evals/).

## Build & offline

```sh
cargo build --release        # plain CPU, portable, dependency-light (the shipped build)
```

On Apple Silicon, add `--features metal` for GPU encode (~5.7× faster);
`accelerate` (macOS) and `mkl` (Linux) are CPU-BLAS alternatives. None is the
default.

The embedding model (`mixedbread-ai/mxbai-edge-colbert-v0-32m`, ~127 MB)
downloads on first use; the summarizer (~1.2 GB) the first time anything
summarizes. For a guaranteed-offline setup, fetch the embedding model once and
point `SYNTY_MODEL` at the directory:

```sh
m=models/mxbai; mkdir -p $m/1_Dense
base=https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m/resolve/main
for f in tokenizer.json config.json config_sentence_transformers.json \
         special_tokens_map.json 1_Dense/config.json 1_Dense/model.safetensors model.safetensors; do
  curl -sL --retry 8 --continue-at - "$base/$f" -o "$m/$f"
done
export SYNTY_MODEL="$PWD/models/mxbai"
```

Cutting a release (version tags, the CI build matrix) is a maintainer task: see
[CONTRIBUTING](CONTRIBUTING.md#releasing).
