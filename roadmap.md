# synty — Roadmap

Path from the current kernel to the system in `design.md`. Milestones are
ordered; each lists its projects as one-liners. Design detail lives in
`design.md`, not here.

## M0 — Kernel · done
- pylate-rs + next-plaid engine in one Rust binary; `ingest / index / search / cluster / summarize / eval`; 16 scenario tests.
- Model download-on-demand (mxbai-edge) with local cache.
- Validated on real data — retrieval, agent dogfood, extractive summaries (see `eval_report.md`).

## M1 — Retrieval & topics solid
- Community-detection clustering (Louvain/Leiden with a resolution knob); GitHub links as a weighted signal, not a transitive union.
- Topic labels from extractive keyphrases.
- Optional Candle `accelerate`/`openblas`; incremental (re-)indexing.

## M2 — Native tracker
- Rust tailers for Claude Code / Codex / Cowork (replace the v1 Go agent as the source).
- GitHub ingestion independent of a developer machine (App + backfill/webhooks).
- Autostart registration (launchd / systemd).

## M3 — Local mode & bucket backplane
- `synty up`: one command to track + index + serve locally, zero config.
- Events to a bucket (local dir / S3 / GCS); index + metadata as rebuildable projections; incremental sync.

## M4 — Surfaces
- Agent surface complete: `search / topic / recent` Markdown to stdout; MCP server exposing the same as tools.
- TUI: tracker status + autostart toggle + browse/drill.

## M5 — Team frontend & privacy
- Optional HTTP frontend over a shared bucket (same view models as the CLI/TUI).
- Guardrails mediated at the frontend: publication delay, secret/credential redaction, per-session erasure.

## M6 — Distribution & OSS
- Single-binary packaging (optional model bundling via `include_bytes!`); cross-platform release artifacts.
- Install one-liner; license; public docs; OSS release.
