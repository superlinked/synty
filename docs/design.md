# synty: System Design

**Status:** v0.2.4 (durable fleet rollout) · **Owner:** Daniel Svonava

synty is a passively-collected memory of how work actually happens: it ingests
coding-agent sessions (Claude Code, Codex, Cowork) and GitHub activity, and makes
the result searchable and readable by both humans and agents. The pivot from v1
(`superlinked/synty-legacy`) is to a **single self-contained Rust binary** built
on **late-interaction retrieval, with generation only for local summaries and
topic names**.
Retrieval and generation run locally and require no remote model API. Solo mode
runs offline; team mode explicitly moves data through the configured bucket.

This document owns the architecture and the target end-state, and marks what is
already built. `evals/` holds the eval suite that validates the kernel on real
data.

---

## Principles

- **Local computation.** The core (retrieval via ColBERT
  late-interaction, clustering, and keyphrases) is deterministic and
  embedding-only, never an LLM. Session summaries and topic names are the
  exception: a small local model (Qwen3-0.6B on candle, the `llm` feature)
  generates them offline.
  No prompt or summary is sent to a model service. A configured team bucket is
  the one deliberate data boundary: members with bucket access can read its raw
  sessions.
- **Local-first, self-contained.** One static binary. The model downloads once
  and runs offline thereafter. No server, no Python, no Docker required to get
  value.
- **The bucket is the backplane.** Events are the durable source of truth; the
  search index and SQLite metadata are derived projections, rebuildable from
  events. Solo = a local directory; team = a shared S3/GCS bucket.
- **Agents are first-class readers.** The primary agent surface is a CLI that
  prints Markdown to stdout; humans get a TUI over the same data.

## Engine

- **Encode:** `pylate-rs` (ColBERT on Candle, ModernBERT backend). Default
  model `mixedbread-ai/mxbai-edge-colbert-v0-32m` (32 M params, 127 MB, PyLate
  format). `model.rs` resolves a model to a local dir, downloading on first use
  under `~/.cache/synty/models` with connect/read timeouts and retry (this
  avoids the hf_hub no-timeout hang); a directory spec is used verbatim. The
  shipped build is plain CPU and portable; opt-in cargo features pick a faster
  backend: `metal` (Apple GPU, ~5.7× encode on Apple Silicon, with CPU fallback
  if the GPU can't init), `accelerate` (macOS CPU BLAS), `mkl` (Linux CPU BLAS).
- **Index / search:** `next-plaid` (PLAID multi-vector index + MaxSim scoring,
  SQLite metadata store). Filtered search resolves a `column=value` predicate to
  a doc-id subset via the metadata DB, then runs MaxSim over it. Fleet builds
  inventory shared vectors once, then fetch, encode, and append bounded document
  batches so a long retention window never holds every token matrix in RAM.
- **Why late interaction:** one ~128-dim vector per token (not per document)
  means a specific term still carries its own signal instead of being averaged
  into a single pooled vector; that advantage over single-vector embeddings
  holds generally and widens on longer documents, where pooling discards more.
  The SQLite side gives exact metadata filtering for free.

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
| GitHub (PRs, issues) | `synty github` GraphQL backfill (token) | **Built**, incremental (UPDATED_AT-floored at the last scrape, so steady-state runs fetch only changes incl. state flips); the corpus + manifest share through the bucket, and `build` refreshes when stale and a token is present, so one tokened machine scrapes for the fleet. App install-token + webhooks planned |

`synty track` is the source of truth for sessions: a `Source` per tool detects
the file format version and builds a parser; a shared driver mints canonical
envelopes with deterministic event_ids (so a re-parse never duplicates). It
polls source logs every 30 seconds and batches network publication separately
(60 seconds by default): only newly appended complete lines become immutable,
line-safe objects under `events/<stream>/chunks/`. An absolute `capture_since`
gate applies both at tailing and again at upload/derivation, retaining only the
session-start metadata needed for a session that crosses the boundary. The
GitHub path pulls PRs/issues straight from the GraphQL API with a token, so it
runs on CI or a server without a developer machine.

## Derivations

- **Search.** Filtered late-interaction retrieval. *Built.*
- **Clusters (topics).** Emergent, no taxonomy. Clustering is over **units of
  work** (sessions, PRs, issues), not the raw message firehose: each unit's
  one-line summary is embedded (multi-vector ColBERT; sessions lead with their
  repo and touched files, summary appended) and **Louvain** runs over a
  MaxSim kNN graph of those embeddings (normalized per-unit, floored, summed
  over both directions), with near-duplicate units (the same work item re-run)
  collapsed onto one representative before any graph work. A topic is therefore
  a coherent *set of units*, so its members, facets (repos/authors), label, and
  summary are consistent by construction, with no doc-vs-unit reconciliation. A
  `--resolution` knob trades topic count vs size; modularity is reported. Each
  cluster's provisional label is its most concise member summary, replaced once
  `summarize` titles the topic. The title is gated on the cluster's distinctive
  (c-TF-IDF) terms, banned from being a bare repo slug, and embedding-checked
  against the members themselves, with an extractive keyword title as the
  fallback; summary embeddings are content-addressed in the shared store
  (encode-once). Clustering the denoised summaries also beats clustering the
  chat firehose. *Built (unit-summary Louvain + resolution + gated topic
  names).*
- **Summaries.** Per session: opening ask, c-TF-IDF keyphrases, files touched,
  effort, linked PR, and a one-line abstractive summary from a local Qwen3-0.6B
  (greedy decode, cached by input hash in `.synty/`, and shared fleet-wide as
  write-once bucket objects, so the reader never runs the model at view time;
  falls back to an extractive representative line without the `llm` feature).
  Per topic: a one-line summary reduced from the member summaries plus the short
  gated name, same model and cache; counts and repos stay extractive. *Built
  (extractive + local LLM session summaries).*
- **Session stats (tokens & tool calls).** Per-session token totals and tool
  mix, derived at view time from the raw envelopes, the agent's own usage
  records, never tokenizer estimates. Capture per source: Claude Code emits a
  usage envelope per raw assistant line (streamed turns repeat the identical
  object across the lines of one message id, measured 2–4×, so aggregation
  dedups by msg_id; embedding usage on the message payload would silently drop
  tool-use-only turns, hence its own envelope); Codex token_count snapshots
  are cumulative, so the last one is the session total, normalized at read
  time (fresh in = input − cached; the corpus keeps codex's raw semantics, so
  the policy is retroactively revisable; historical codex sessions lit up
  with no re-tracking); Cowork records no usage. The four classes stay
  separate end-to-end (in · out · cache-read · cache-write, since a cache read
  is not a fresh input), tool calls tally by name with errors from
  `tool_result.is_error`, and a session without usage shows no number at all
  (never a fake 0). Subagent (`agent-`) sessions count separately; parent
  totals exclude children until a rollup over the existing `subagent_parent`
  edges; per-model tables, tool durations, and an optional pricing map are
  deferred with it. `[metrics stats]` reports `usage_coverage_pct`, the
  share of sessions whose source recorded usage. *Built (P1).*
- **Fleet coverage (install rate).** Whether synty runs everywhere agents run,
  from data already in the bucket: the roster folds every `edge-<machine>-…`
  stream (liveness from envelope timestamps, since file mtimes lie after a pull;
  actor stamps; `tracker_version` for upgrade lag) and joins it against the
  org's **members** active on GitHub in a trailing window, scoped to the team
  (`synty github` caches `membersWithRole`), not every external contributor who
  opened one PR. Membership is authoritative: when the member list is known it
  is the sole filter, and `config.fleet_ignore` drops a service-account member
  by data, not a name in code; the `is_bot` login heuristic is only the
  no-member-list fallback. Everyone active and uncovered is listed (the install
  gap a lead acts on), with agent-attribution (Co-authored-by trailer,
  Generated-with footer, bot author, flagged at ingest as `agent_attr`,
  precision-first) shown as a per-author marker rather than the membership test,
  since most agent users leave no artifact. A machine that streamed and went
  quiet is tracker rot, distinct from never-installed. `ingest` emits a
  `[metrics coverage]` block (machines, actors, install rate). *Built (M8).*

## Surfaces

- **CLI → stdout (agents):** `search`, `related`, `topic`, `recent`, `status`,
  `stats`, `tool`, `show`, and `trace` print Markdown an agent reads over the shell: no
  server or application auth. In team mode a read first syncs the fleet's raw
  chunks and latest published read-model; offline it keeps using the last
  complete local snapshot. `related` takes no query: it derives one from the
  repo's recent commits + changed files and searches cross-repo, so an agent can
  pull prior work on its current task for free. Stable ids ride inline (sessions
  `[id8]`, PRs `repo#N`, topics `[key8]`), and `show <id>` drills into any of
  them, so list views and detail views close the loop without JSON. *Built.*
- **TUI (humans):** `synty tui` opens tabs for topics, work, search, stats
  (usage charts + spend tables), and status (self-health + fleet roster), with
  browse/drill (topic → members → full document), reusing the CLI's view-models.
  *Built.*
- **MCP server:** `synty mcp` serves the CLI's read surface as agent tools over
  stdio by default (hand-rolled JSON-RPC): search, related, topics, recent,
  status, stats, tool, show, plus forensic `synty_trace_*` tools. Role/tool
  policy and optional read scope bind mediated clients; responses apply a
  redaction profile. A restricted scope is applied before topics and detail
  aggregates are built; global health surfaces are unavailable. `sources`
  names native producers, matching both indexed and trace data. `--http`
  (feature `mcp-http`) exposes the same dispatcher as authenticated Streamable
  HTTP at `/mcp` for remote agents. It requires a 32-byte bearer token; any
  non-loopback `--bind` also requires `--listen-public`, `--tls-cert`, and
  `--tls-key`. `--scope`, `--redaction`, and repeatable `--allowed-origin`
  configure the mediated boundary. The transport validates exact browser
  origins and protocol versions, bounds requests, and
  keeps health responsive while tool calls run. It pulls the published query
  model before serving and continues potentially large raw-event transfers in
  the background. Raw-history analysis is serialized separately so it cannot
  block semantic search. Each dispatcher has a bounded queue; HTTP clients have
  a 120-second response deadline and a per-client 120-request/minute window,
  while stale queued work is cancelled before execution. Remote related-work
  queries accept context text rather than server filesystem paths. *Built.*
- **Harness import:** `synty import` normalizes campaign/Devin NDJSON into
  owned envelope streams with deterministic ids, capture boundaries, and
  import-time redaction. Identity derives from native ids or canonical event
  content, independent of input path and line order; concurrent writers lock
  the owned stream. `--format`, `--machine`, `--campaign`, `--role`, `--repo`,
  `--actor`, `--since`, `--redaction`, `--quarantine`, and `--bucket` describe
  the foreign source and publication policy; `--dry-run` writes neither streams
  nor quarantine files. *Built.*
- **JSON output:** `--json` on every read command, one versioned envelope
  (`{"v": 1, "kind": …, "data": …}`) so scripts check the format once. *Built.*

### Execution-trace query surface

`synty trace` is the narrow forensic layer over canonical raw events. It gives
the investigating agent compact, composable facts without embedding an analyst
or a remediation workflow in synty itself:

- `trace list --type turns|spans|jobs` filters by repo, machine, source, status,
  operation, time, error presence, and duration, then sorts by recency or
  duration. Jobs may instead sort by source-reported wait time.
- `trace show <id>` expands a source-native turn, paired tool call/result, raw
  event, associated job, or session into a bounded evidence timeline. Turn
  timelines collapse same-turn job polls into one navigable job row.
- `trace search <literal>` searches the raw envelopes, including prompts,
  commands, outputs, and metadata; `trace compare <left> <right>` returns only
  factual field differences between two turns, two spans, or two jobs.

Turn boundaries use source-native task/turn markers where they exist and
prompt boundaries otherwise. Tool spans pair call/result ids within a session.
For Codex asynchronous commands, a job associates the initiating `exec_command`
with later `write_stdin` calls by the exact process-session id present in both
records. It reports lifecycle elapsed time separately from the sum of
source-reported tool waits: a long-lived background server is not automatically
a long wait. A continuation without a captured initiating call remains visible
and is labeled `continuation_only` rather than guessed onto another command.
Every duration says whether the source reported it or it is merely an event
gap; the latter may include human approval or idle time and is deliberately not
presented as tool runtime. The projection is rebuilt on demand from JSONL for
now, keeping raw envelopes authoritative and avoiding a new index contract
until corpus-scale use demonstrates one is needed. *Built.*

## Tiers and the trust boundary

- **Solo / local:** the binary writes events to a local dir, builds the index
  locally, and answers from the CLI/TUI. No mediation needed, since you are the
  only reader. No server, no creds.
- **Team / company:** a shared bucket that every member reads and writes
  directly. synty assumes a high-trust team for raw bucket credentials: anyone
  with the bucket can read every stored object, because read rules cannot be
  enforced on a client with that access. A separate mediated path exists for
  agents: `synty mcp` (stdio or authenticated HTTP) applies role/tool policy,
  optional read scope, and response redaction; upload sync can also redact
  before writing team event chunks. Upload redaction is off by default so raw
  events remain a rebuildable source of truth. Opting in records the profile in
  the upload ledger and refuses a later profile change after offsets advance.
  `init --capture-repo` is enforced at upload and import boundaries; a session
  without an allowed start record is not uploaded. Repository policy is also
  fixed once offsets advance because immutable chunks cannot be retracted and
  newly allowed history cannot be recovered from an advanced cursor.

Usage and topic rollups are by work (topic, repo, period); the one per-person
view is the fleet-coverage roster, which names who runs agents untracked so a
lead can close the install gap.

**Onboarding & the local→bucket ramp.** One command does it: `synty init
[bucket]` validates bucket access, pins the GitHub identity, enables and verifies
the login-time tracker, and runs the
first build (one onboarding path, not the old interactive `setup` plus more).
Omit the bucket to trial synty against local sessions (invisible to the fleet,
since it pushes no events); re-run with a bucket and that same `init` is the
switch onto the team (config gains the bucket; the next build's event sync does
the rest, no migration). A machine is **activated**, a real fleet member,
exactly when a bucket is set; the bucket is the only thing that moves the badge.
Autostart (the login-time tracker) is turned on by `init`/install and is on by
default thereafter, reported in its own indicator, not a second activation gate.
Containers and process supervisors pass `--no-autostart` and run
`synty track --watch` themselves. An orphan plist/unit is not reported as
active, and initialization fails visibly if the service manager does not load
the watcher. `--capture-since` persists an absolute event boundary and
`--upload-interval` sets the network batching cadence. Optional `--campaign` /
`--role` persist campaign stamps used by later track/import runs.
`--capture-repo`, `--upload-redaction`, and `--mcp-redaction` persist the
corresponding privacy policy. A systemd
user service starts at boot on a headless developer VM when the administrator
enables lingering for that user (`loginctl enable-linger`).
The state shows on `status` and the TUI footer (`◐ local`, accent → `✓
<bucket>`, sage), so the ramp is legible. The install one-liner carries the
bucket and drops into the viewer, so a paste goes from nothing to tracking.

## Storage layout (bucket)

```text
events/<stream>/chunks/<track-day>/<range-hash>.jsonl
                                      immutable append deltas (source of truth);
                                        stream = edge-<machine>-<source>, so many
                                        trackers' files coexist without collision
event-streams/<stream>                immutable bounded discovery registry;
                                        readers continue each stream by key cursor
members/<machine>/activation.json     immutable init access marker (no session data)
embeddings/<hash[..2]>/<hash>.emb      content-addressed f16 vectors (write-once)
summaries/<kh[..2]>/<kh>-<ihash>.json  per-(unit, input-hash) LLM summaries
                                        (write-once: first viewer generates for
                                        the fleet; empty = tried + gate-rejected)
blobs/<fnv>                            content-addressed build files (index
                                        chunks, docs snapshot, clusters), shared
                                        across builds, so appends upload deltas
builds/<build>.<rev>.json              manifest: filename → blob, per (build, rev)
current.json                           the pointer, PUT last; readers never see
                                        a torn build; rev versions the clusters
lease/build                            soft TTL lease electing one index builder
```

A `Bucket` trait abstracts this store (local dir always; S3/GCS behind
`--features s3/gcs`, with conditional PUT for write-once and the lease). The
fleet model is **no designated builder**: every tracker pushes events; whoever
opens a viewer pulls all raw streams and the published read-model, then
contributes a build. All CLI/MCP readers perform the same pull; commands can
therefore inspect every machine, while semantic results cover the latest
published build and warn when newer raw chunks are pending.
Write-once stores are the collaboration primitive: a viewer encodes and
summarizes only what no other machine has (pending lists shuffle per machine,
so concurrent viewers split the work). The lease only prevents duplicate index
builds; losing it wastes compute, never corrupts, because publishes are
immutable-prefix + pointer-swap. The local layout mirrors this:
`index/builds/<build>/` + an atomically repointed `index/current.json`;
incremental appends clone the previous build (CoW), so a reader's live mmap is
never mutated under it. `synty up` loops locally; `synty build` is the
one-shot fleet-aware build.

**S3 credentials.** synty never persists access keys. With no named profile,
the S3 backend uses environment, web-identity, container, or instance metadata
credentials, which is the intended VM/container path. `init --aws-profile
<name>` instead loads that AWS shared-config profile through the SDK's rotating
provider chain; unattended workstations should make it a direct
`credential_process` profile (for example IAM Roles Anywhere), not an AWS SSO
session that requires periodic interactive login. The generated launchd/systemd
unit reads the same config and therefore needs no temporary CLI environment.

The **binary distributes through GitHub Releases**, kept off the data bucket on
purpose: CI builds a per-platform artifact on each version tag and attaches
`synty-<os>-<arch>` (+ a `.sha256`) to the release. `synty upgrade` finds the
latest release and downloads its platform's asset (a public release asset over
plain HTTPS; a private repo reuses the GitHub token synty already holds for
PRs/issues), so there is no extra credential, no presigning, and the data bucket
stays purely data. It verifies the sha256, replaces the running binary in place,
and restarts the tracker; each machine gets the fastest build for its platform
(metal on Apple Silicon, CPU on Linux, each with runtime fallback) without
choosing. A passive, cached nag (`status`, footer, `up`) flags when a machine is
behind; the install one-liner bootstraps from the same public release asset (or
`gh` for a private repo).

The container image is a separate release artifact. Tag builds publish a
`linux/amd64` + `linux/arm64` manifest to
`851725219920.dkr.ecr.eu-central-1.amazonaws.com/synty:<version>` as an immutable tag
through a GitHub OIDC role. Each architecture builds on a native hosted runner;
the release verifies both platforms after composing the index. The Helm chart
follows `Chart.appVersion`, runs each container as UID 10001 with a read-only
root filesystem, and exposes MCP only as a cluster Service. Remote MCP is
disabled by default; enabling it requires an application TLS Secret and a
NetworkPolicy with explicit source selectors.

## Data compatibility

- **Envelopes are add-only, forever.** Fields are never renamed or repurposed;
  readers skip unknown kinds and fields and default absent ones. Each envelope
  carries `v` (currently 1), bumped only for a breaking change we intend never
  to make. This is the contract that keeps raw tracked data useful forever;
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
`status`, `trace list/show/search/compare` (turns, spans, async jobs), `tui`,
`mcp` (stdio + optional HTTP), `import`, `cluster [--resolution]`, `summarize`,
`eval`, plus the scenario test suite (`cargo test`, pure). The bucket backplane
(local always, S3/GCS opt-in) gives fleet-wide encode-once and collaborative
builds. MCP-HTTP is an optional mediated transport for remote clients.
Validated at M0/M1 on real data (3,938 docs / 770 K embeddings): retrieval 12/12
relevant top-3, agent task-start dogfood 3/3, session summaries specific and
accurate (extractive in the core; one-line abstractive from a local Qwen3-0.6B
under the `llm` feature, with retrieval and clustering staying LLM-free).
Clustering (M1) is Louvain over the weighted graph: the prior GitHub over-merge
(a 710-doc blob) is gone, replaced by 22 recognizable keyphrase-labeled topics,
largest 462 docs, modularity 0.75. Full results via `synty eval`; the eval suite
lives in `evals/`.

## Stack

Rust (edition 2024). `pylate-rs`, `next-plaid`, `candle-core`, `ndarray`,
`serde`, `clap`, `ureq`. Cross-compiles to darwin-arm64/amd64 + linux-amd64;
CPU by default, opt-in `metal` (Apple GPU) / `accelerate` (macOS) / `mkl`
(Linux) features for faster encode.
