# synty — Roadmap

Path from the current kernel to the system in `design.md`. Milestones are
ordered; each lists its projects as one-liners. Design detail lives in
`design.md`, not here.

## M0 — Kernel · done
- pylate-rs + next-plaid engine in one Rust binary; `ingest / index / search / cluster / summarize / eval`; 16 scenario tests.
- Model download-on-demand (mxbai-edge) with local cache.
- Validated on real data — retrieval, agent dogfood, extractive summaries (see `eval_report.md`).

## M1 — Retrieval & topics solid · done
- Louvain community detection with a `--resolution` knob; GitHub links as a weighted edge signal, not a transitive union.
- Topic labels from extractive c-TF-IDF keyphrases.
- Opt-in Candle `metal`/`accelerate`/`mkl`; incremental (re-)indexing (content-hash embedding cache + unchanged fast-path).

## M2 — Native tracker · done
- Rust tailers for Claude Code / Codex / Cowork (`synty track`), validated against the v1 agent as oracle; codex version gap fixed.
- GitHub ingestion independent of a developer machine (`synty github` GraphQL backfill, token-based); App install-token + webhooks still planned.
- Autostart registration (launchd / systemd) + `--watch` with per-file cursors.

## M3 — Local mode & bucket backplane · done
- `synty up`: one command to track + ingest + index on a loop, zero config (solo).
- Bucket trait (local dir always; S3/GCS first-class behind `--features s3/gcs` via object_store). Events sync to the bucket (push/pull) so many trackers converge; index + docs publish as a rebuildable read-model clients pull.
- Content-addressed f16 embedding store: each message encoded once across the fleet; a second device rebuilds with no re-encode.

## M4 — Surfaces · done
- CLI surface complete: `search / topic / recent / status` print Markdown to stdout.
- TUI (humans): tracker status + autostart toggle + browse/drill topics and sessions, at rough feature parity with the CLI (more expressive where a terminal UI allows).

## M5 — Team frontend & privacy
- Optional HTTP frontend over a shared bucket (same view models as the CLI/TUI).
- Guardrails mediated at the frontend: publication delay, secret/credential redaction, per-session erasure.

## M6 — Distribution & OSS
- Single-binary packaging (optional model bundling via `include_bytes!`); cross-platform release artifacts.
- Install one-liner; license; public docs; OSS release.

## M7 — Fleet: collaborative build over a shared bucket · done
No designated builder and no processing infra: lightweight trackers everywhere
push raw events; whoever opens a viewer contributes compute, incrementally
building the shared derived artifacts in the bucket.
- Multi-writer safety: envelopes dedup by event_id on read (session_end ids made deterministic), so overlapping trackers can never double-count.
- Versioned read-model: immutable builds (local `index/builds/<build>/`, bucket content-addressed blobs + per-(build,rev) manifests) behind an atomically swapped pointer — no torn publishes, no mutating a reader's live mmap; incremental appends clone the previous build (CoW).
- Bucket-shared derived artifacts: per-(unit, input-hash) write-once summary objects (first viewer generates for the fleet); clusters travel with the build as additive revs, prev-cluster key lineage read from the published build.
- Collaborative summarize: machine-seeded shuffle of the pending list + a store re-check per job — lock-free unit-level parallelism; a soft TTL lease (conditional PUT; S3 needs ETag conditional-put enabled, wired in) serializes only the index build + publish.
- Viewer loop: `tui` pulls the read-model and renders instantly, freshens via a background `build --no-track` child (stderr → .synty/build.log, `@phase` markers), hot-reloads with selection remapped by stable topic key and the search index reloaded behind an ack; footer shows ⟳ phase / ⚠ stale / ✓ fresh.
- Tracker rollout: `install.sh` one-liner (binary + ~/.synty home + bucket config + login-time tracker) for laptops and dev-VM/Modal images; `config.bucket` is the default everywhere (flag still wins); `$SYNTY_HOME`/~/.synty resolution makes every command work from any directory.
- Deferred: Cursor tailer (needs a machine with Cursor session data); hosted agents (Claude Code web, Devin) need per-platform log-export exploration.

## Future work (after the milestones)
- ~~MCP server exposing agent tools over stdio~~ — done (`synty mcp`: synty_search / synty_topics / synty_recent / synty_status).
