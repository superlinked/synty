# synty

**A passive memory of how your work actually happens.** synty quietly watches
your coding-agent sessions (Claude Code, Codex, Cowork) and your GitHub activity,
and makes all of it searchable — by you, and by the agents you work with. Before
starting a task, ask *"has anyone touched the auth flow?"*, or just run `synty
related` to pull the relevant prior sessions and PRs **without typing a query**.

It runs as a single local binary. **No API keys, nothing leaves your machine** —
not even the summaries, which a small model writes locally. Retrieval is
late-interaction embeddings (ColBERT) plus deterministic logic — no LLM in the
loop. For a tool that ingests your dev transcripts, that privacy *is* the point.

## See it in action

Browse your work memory in the terminal UI — topics, drill-down, search, stats,
the fleet roster:

![synty's TUI: topics, drill-down, search, stats, status](docs/tui.gif)

One command from nothing to tracking + a viewer — a local trial, then join the
team (`◐ local` → `✓ on the team`):

![synty init onboarding: local trial, then join the team](docs/install.gif)

The agent surface — `related` / `search` / `status` printing Markdown to stdout:

![synty CLI: related, search, status](docs/cli.gif)

<sub>Rendered from [`docs/*.tape`](docs/demos.md) with [vhs](https://github.com/charmbracelet/vhs); re-render when the UI changes.</sub>

## Install

One paste from nothing to "tracking + a viewer". The bucket is optional — omit
it to trial synty against your own sessions first, add it later to join your
team:

```sh
curl -fsSL https://raw.githubusercontent.com/superlinked/synty/main/install.sh | sh                      # local trial
curl -fsSL https://raw.githubusercontent.com/superlinked/synty/main/install.sh | sh -s -- gs://my-team   # join the team
```

The installer puts the binary on PATH, runs `synty init [bucket]` (pins your
GitHub identity, enables the login-time tracker, runs the first build), then
opens the viewer. It pulls the prebuilt binary from the latest [GitHub
Release](https://github.com/superlinked/synty/releases); `synty upgrade`
self-updates from the same place. *Building from source? See [Build &
offline](#build--offline); replace `synty` with `cargo run --release --` in any
command below.*

## Start local, then join your team

**1 · Local-first (recommended) — see the value, then join.**

1. **Install with no bucket.** `synty init` pins your identity, turns on the
   login-time tracker, and builds an index from your local sessions (+ your
   org's PRs if you have a `gh`/`GITHUB_TOKEN`). Status reads **`◐ local`**.
2. **Use it.** The tracker runs at login; you browse in `synty tui` and run
   `synty related` / `synty search` before tasks. Everything stays on your
   machine — you push nothing, the fleet can't see you.
3. **Join the team** when you're ready: `synty init gs://my-team`. That sets the
   bucket; the next build pulls the fleet's shared memory and your tracker starts
   contributing. Status flips to **`✓ my-team`** — you're an activated member.

**2 · Straight to the team.** Trust it already? Install with the bucket (journey
1, but the one-liner carries `gs://my-team`). You land directly at `✓ my-team`.

**3 · Everyday — you browse, your agents read.** Same index, two surfaces:

- **You:** `synty tui` — an interactive terminal UI for exploring.
- **Your agents:** `synty related` / `synty search` (Markdown to stdout), or the
  MCP server, so a coding agent consults past work mid-session.

The only thing that moves you from `◐ local` to `✓ <bucket>` is setting a bucket.
Autostart (login-time tracking) is on by default the whole time and has its own
indicator — it's never a second gate.

## Browse it yourself, or let your agents read it

### The TUI — browse, filter, drill in

`synty tui` opens tabs for **Topics**, **Work**, **Search**, **Stats**, and
**Status**, with drill-down from a topic into its sessions and the full document
text. Filter any list to one repo or account with `r`/`a`; refresh on demand
with `u`. The footer always says where you stand — activation (`◐ local` →
`✓ <bucket>`) and freshness (`◐ encoding 120/470` · `⚠ stale` · `✓ fresh`). The
Stats tab charts what the agents consume against what the work produces; the
Status tab shows tracker health and the fleet roster (who runs synty where, who
runs agents untracked) and toggles login-time tracking. On startup it pulls the
fleet's published index and freshens in the background.

### The CLI & MCP — Markdown your agents read

Every read command prints Markdown to stdout — exactly what a coding agent reads
over the shell, no server or auth. `synty related` is the zero-effort entry
point: **non-interactive, no query** — it reads this repo's recent commits and
changed files, builds a query from them, and surfaces prior work across every
repo synty has seen.

```sh
synty related                  # prior work related to what you're doing now (from this repo's git)
synty search "rate limiting"   # semantic search (--filter repo=api, kind=pull_request, …)
synty recent                   # latest PRs, issues, and prompts
synty topic                    # emergent topics (or `topic auth` to filter)
synty status                   # health: what's indexed, freshness, activation, the fleet roster
synty stats                    # usage: tokens/tools/sessions vs LOC/PRs/issues per week
synty show a1b2c3d4            # drill into a session, PR/issue (repo#123), or topic key
```

Each result is a ranked Markdown card — PRs/issues and session moments together
(illustrative output):

```text
## rate limiting middleware

1. [24.3] **pull_request api#1487** — Add a token-bucket limiter to the gateway
   merged · https://github.com/acme/api/pull/1487
2. [21.8] _user_prompt · api · a3f1c2d9_
   how do we share the per-tenant limiter's state across pods? settled on Redis…
3. [19.0] **issue api#1502** — 429s under burst load on the search endpoint
   open · https://github.com/acme/api/issues/1502
```

Output is built to be drilled: `search`, `related`, `recent`, and `topic` print
stable ids inline (`[a1b2c3d4]` sessions, `repo#123` PRs/issues, `[72a778f8]`
topics) that feed `synty show <id>`. Every read command takes `--json` — one
versioned envelope, `{"v": 1, "kind": "…", "data": …}`, so a script checks `v`
once and dispatches on `kind`.

For coding agents, `synty mcp` serves the same surface over stdio (`synty_search`,
`synty_related`, `synty_topics`, `synty_recent`, `synty_status`, `synty_stats`,
`synty_tool`, `synty_show`). `synty_related` takes the agent's repo path and
needs no query; ids in any tool's output feed `synty_show`.

## Fleet mode (teams)

Solo, the "bucket" is a local directory. For a team it's one shared S3/GCS
bucket — and the bucket is the *only* shared infrastructure: no build server, no
cron, no coordination service. That works because everything in it is
append-only (events), write-once (embeddings, summaries), or swapped atomically
behind a pointer (the index) — machines cooperate without ever talking to each
other. Three roles, one binary:

- **Every machine writes.** The tracker runs at login: model-free, near-zero
  footprint, it tails local agent session files and pushes raw events.
- **One machine scrapes GitHub.** Whichever machine has a token refreshes the
  org's PRs/issues during its builds (incrementally) and shares the result.
  Nobody else needs a token.
- **Viewers build.** Opening `synty tui` (or `synty build`) pulls the latest
  published read-model, then freshens in the background — encoding and
  summarizing only what no machine has done yet, one soft lease per build, then a
  pointer swap.

The configured bucket is the default everywhere; `--bucket` overrides. Cloud
buckets need `--features s3` / `gcs`. To exercise the whole sync path locally
(no cloud account): `scripts/fleet-smoke.sh` runs publish → cold pull → delta
pull against a temp local-dir bucket and checks the `[metrics sync]` numbers.

> **Caveat:** every fleet member has raw bucket access and can read everyone's
> sessions. Fine for high-trust teams; the mediated-frontend tier (publication
> delay, redaction) is the planned answer where it isn't.

## Run the pipeline yourself

`synty build` runs the whole pipeline once (track → ingest → index → summarize →
cluster → topic summaries); `synty up` loops it every minute (your machine is
both the model-free tracker and the builder). The individual steps, for
scripting:

```sh
synty track     # tail agent sessions → canonical event envelopes
synty github    # pull the org's active PRs/issues (org from `init`; $GITHUB_TOKEN or gh)
synty ingest    # turn events + GitHub into searchable documents
synty index     # encode with ColBERT and build the index
synty summarize                  # local-LLM one-line summary per unit
synty cluster --resolution 2.0   # group units by summary embedding → topics
synty summarize                  # again: reduce each topic from its members
```

`cluster` groups *units of work* (sessions, PRs, issues) by their summary, so run
`summarize` first; the topic summaries are a second `summarize` pass.

## How it works — and why nothing leaves your machine

- **Retrieval has no LLM in it.** `pylate-rs` runs a small ColBERT model
  (ModernBERT, 32 M params) that represents each document as one vector *per
  token*; `next-plaid` (PLAID) scores queries with MaxSim, backed by SQLite for
  exact metadata filters. Late interaction retrieves far better than
  single-vector embeddings on short, code-adjacent text.
- **Summaries are local.** The one-line unit/topic summaries come from a small
  model (Qwen3-0.6B) running offline on your CPU — never a remote API, so your
  sessions are never sent anywhere. Topic *names* must share a distinctive
  cluster term and embed close to their members, else an extractive title wins.
- **Events are the source of truth.** Sessions and GitHub items become
  append-only event envelopes; the index and its metadata are derived
  projections, rebuildable from the events at any time and shareable through a
  bucket.
- **Activation = a bucket.** A machine is a fleet member exactly when a bucket is
  set; that single fact is what the status surface tracks.

Architecture and rationale live in `design.md`; the milestone plan in
`roadmap.md`; validation on real data in `eval_report.md`.

## Build & offline

```sh
cargo build --release        # plain CPU, portable, dependency-light — the shipped build
```

On Apple Silicon, build with `--features metal` for GPU encode (~5.7× faster);
`accelerate` (macOS) and `mkl` (Linux) are CPU-BLAS alternatives. None is the
default.

The embedding model (`mixedbread-ai/mxbai-edge-colbert-v0-32m`, ~127 MB)
downloads on first use; the summarizer (~1.2 GB) downloads the first time
anything summarizes. For a guaranteed-offline setup, fetch the embedding model
once and point `SYNTY_MODEL` at the local directory:

```sh
m=models/mxbai; mkdir -p $m/1_Dense
base=https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m/resolve/main
for f in tokenizer.json config.json config_sentence_transformers.json \
         special_tokens_map.json 1_Dense/config.json 1_Dense/model.safetensors model.safetensors; do
  curl -sL --retry 8 --continue-at - "$base/$f" -o "$m/$f"
done
export SYNTY_MODEL="$PWD/models/mxbai"
```

## Releasing & upgrades

Binaries ship as **GitHub Release** assets, built by CI. Each platform gets the
fastest build for it — **metal on Apple Silicon, plain CPU on Linux**, each with
a runtime fallback — so nobody picks a build.

**Cut a release** — bump the version and push a tag; `.github/workflows/release.yml`
runs the test suite, builds each platform, and attaches `synty-<os>-<arch>` (plus
a `.sha256`) to the release:

```sh
# edit Cargo.toml version → e.g. 0.2.0, commit
git tag v0.2.0 && git push origin v0.2.0
```

The matrix builds `macos-14` (`--features metal,s3,gcs`) and `ubuntu-latest`
(`--features s3,gcs`); the `s3`/`gcs` features are for reading the team's *data*
bucket and are independent of where the binary ships. Add matrix rows for more
platforms (Intel Mac `macos-13`, `ubuntu-24.04-arm`) as needed.

**Upgrade** — once a newer release exists, `synty status`, the TUI footer
(`⬆ <v>`), and `synty up` show a nag; the swap stays explicit:

```sh
synty upgrade        # download this platform's asset, verify sha256, replace in place, restart the tracker
```

`upgrade` fetches the release directly — no extra credential (a private repo
reuses the GitHub token synty already has for PRs/issues; override the source
repo with `$SYNTY_RELEASE_REPO`). It's a no-op when you're current and refuses
on a checksum mismatch. `synty --version` reports the running build. The install
one-liner pulls the same assets — public download, or `gh` when the repo is
private.
