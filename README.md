# synty

**Your coding agents start every session from zero. So do you.**

synty passively records every coding-agent session (Claude Code, Codex, Cowork)
and your GitHub activity into one local, searchable memory, for you and the
agents you work with.

![synty's TUI: browse topics, sessions, search, and stats in the terminal](docs/tui.gif)

<sub>Demos rendered from [`docs/*.tape`](CONTRIBUTING.md#rendering-the-demo-gifs) with [vhs](https://github.com/charmbracelet/vhs).</sub>

- **Recall, not re-discovery.** Ask *"has anyone touched the auth flow?"*, or just
  run `synty related`, and get the sessions and PRs that matter. You browse in
  the TUI; your agents read the same over the CLI and MCP.
- **Your data stays yours.** Everything is open on disk: raw events as JSONL, a
  SQLite index, `--json` on every command. Chart agent friction, fine-tune a
  model, build your own dashboards. Nothing is trapped in a viewer.
- **Local-first.** One binary, no API keys, nothing leaves your machine. Even the
  one-line summaries run on a small local model.

## Quick start

```sh
curl -fsSL https://raw.githubusercontent.com/superlinked/synty/main/install.sh | sh
```

One paste installs the binary, turns on the login-time tracker, and opens the
viewer. After that, two commands cover the daily loop:

```sh
synty tui        # browse your work memory: topics, sessions, search, stats
synty related    # surface prior work for whatever you're doing now (no query)
```

The tracker runs at login, so the memory keeps building on its own. Update any
time with `synty upgrade`. Building from source instead? See [Build](#build).

## Commands

Every read command prints Markdown to stdout, or `--json` for a versioned envelope.

| Command | What it does |
|---|---|
| `synty related` | prior work for your current task, from this repo's git (no query) |
| `synty search "<query>"` | semantic search; add `--filter repo=…,kind=pull_request` |
| `synty recent` | latest PRs, issues, and prompts |
| `synty topic [name]` | emergent topics, or one topic's members |
| `synty show <id>` | open a session, PR/issue (`repo#123`), or topic |
| `synty status` | what's indexed, freshness, activation, the fleet roster |
| `synty stats` | tokens / tools / sessions vs LOC / PRs / issues per week |
| `synty tui` | interactive browser: tabs, drill-down, filter by repo or account |
| `synty mcp` | serve the read surface to agents over stdio |
| `synty build` / `synty up` | rebuild the index once / keep it fresh on a loop |

A result is a ranked Markdown card with ids you can drill (`[a1b2c3d4]` sessions,
`repo#123` PRs/issues, `[72a778f8]` topics) that feed `synty show`:

```text
## rate limiting middleware

1. [24.3] **pull_request api#1487** — Add a token-bucket limiter to the gateway
   merged · https://github.com/acme/api/pull/1487
2. [21.8] _user_prompt · api · a3f1c2d9_
   how do we share the per-tenant limiter's state across pods? settled on Redis…
3. [19.0] **issue api#1502** — 429s under burst load on the search endpoint
   open · https://github.com/acme/api/issues/1502
```

![synty's agent surface: related, search, and status as Markdown on stdout](docs/cli.gif)

## Own your data

synty is a capture layer, not a walled garden. Everything it records sits in
plain files under `~/.synty`, so you can build on it directly:

- **Raw events** (the source of truth): append-only JSONL under `corpus/local/`
  (and `events/<stream>/` in a shared bucket).
- **Documents**: `corpus/docs.jsonl`, one object per line with `meta`
  (`source`, `kind`, `repo`, `author`, `session_id`, `ts`).
- **Metadata**: a SQLite database under `index/`, queryable with any SQLite tool.

```sh
synty stats --json | jq '.data.weeks[] | {week: .start, tok_in, tok_out}'   # weekly token trend
synty tool Bash --json | jq '{calls: .data.calls, errors: .data.errors}'    # where agents get stuck
jq -r '.meta.kind' ~/.synty/corpus/docs.jsonl | sort | uniq -c              # straight from the raw file
```

Because synty already clusters the work and links across sources (PRs, issues,
sessions), your analysis starts from structured data, not a heap of logs.

## Teams

Solo, the "bucket" is a local directory. For a team, or just your own laptop and
desktop, point synty at one shared S3/GCS bucket and you build one memory:

```sh
synty init gs://my-team
```

That bucket is the only shared infrastructure: no build server, no coordination
service. Every machine's tracker pushes events; whoever opens a viewer builds the
index and publishes it for the rest; one tokened machine scrapes GitHub for
everyone. Cloud buckets need `--features s3` / `gcs`.

> **Heads up:** every member with the bucket can read every session in it. synty
> is built for high-trust groups; there's no per-reader redaction. If that
> doesn't fit, stay local, or scope the bucket to people who already see each
> other's work.

## How it works

- **Retrieval is late interaction.** `pylate-rs` runs a small ColBERT model
  (ModernBERT, 32 M params) that encodes each document as one vector per token;
  `next-plaid` scores queries with MaxSim over a SQLite metadata filter. It beats
  single-vector embeddings on short, code-adjacent text.
- **Summaries and topic names** come from a small local model (Qwen3-0.6B) on
  your CPU, never a remote API, with an extractive fallback.
- **Events are the source of truth.** The index and metadata are derived
  projections, rebuildable any time and shareable through a bucket.

Architecture and rationale live in [`docs/design.md`](docs/design.md); the
on-real-data validation lives in [`evals/`](evals/).

## Build

```sh
cargo build --release        # plain CPU, portable, dependency-light (the shipped build)
```

On Apple Silicon, add `--features metal` for GPU encode (~5.7× faster);
`accelerate` (macOS) and `mkl` (Linux) are CPU-BLAS alternatives. The embedding
model (~127 MB) downloads on first use; the summarizer (~1.2 GB) the first time
anything summarizes. For an air-gapped setup and the per-stage pipeline, see
[`docs/design.md`](docs/design.md) and [CONTRIBUTING](CONTRIBUTING.md).

Cutting a release is a maintainer task: see [CONTRIBUTING](CONTRIBUTING.md#releasing).
