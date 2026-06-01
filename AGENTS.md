# AGENTS.md — synty

Working rules for changing code in this repo. Read `design.md` for the
architecture and target end-state, `roadmap.md` for the milestone sequence, and
`eval_report.md` for the kernel validation before making changes. This file
does not restate them.

Parsing of coding-agent session logs and GitHub ingestion are ported from the
v1 implementation in `../synty-legacy` (Go tailers under
`internal/source/<tool>/`, TypeScript GitHub backfill under
`server/src/lib/github/`). The new implementation is a single self-contained
Rust binary.

## Commit messages

- Never mention Claude, Claude Code, Anthropic, or any AI assistant anywhere in
  a commit message, and never add a `Co-Authored-By` trailer naming an
  assistant. Commits are authored by the human committer, full stop.
- Subject is imperative and scoped (e.g. `M1: Louvain topics ...`). The body
  explains the why and the measured effect, preferring numbers (docs/s, cluster
  counts, repo and PR numbers, dates) over adjectives.

## Build / test / run

- `cargo test` runs the scenario suite. It is pure: no model, corpus, or
  network needed.
- `cargo build --release` is the shipped build: plain CPU, portable,
  dependency-light. Keep it that way.
- On Apple Silicon, develop with `cargo build --release --features metal` (GPU
  encode, ~5.7x faster). `accelerate` (macOS CPU BLAS) and `mkl` (Linux) are
  the other opt-in backends. None may become default.
- The encoder loads the model from `$SYNTY_MODEL` (point it at a local dir for
  offline use; it otherwise downloads the default model on first run).
- Pipeline: `ingest` → `index` (encode + build + cache per-doc embeddings) →
  `search` / `cluster [--resolution]` / `summarize` / `eval`. `index` wipes and
  rebuilds the `index/` dir, which also invalidates the embedding and kNN-graph
  caches that `cluster` reuses.

## Code

- Match the surrounding style: comment density, naming, idiom. Each module opens
  with a short comment on what it owns and why.
- Every behavioral change gets a scenario-style unit test written from user
  expectations, not from the implementation (see the existing `#[cfg(test)]`
  blocks).
- Derivations stay LLM-free: embeddings, deterministic logic, and extractive
  text only. "Nothing leaves the machine" is the feature, not a constraint to
  work around.
- New per-tool session tailers (M2) each get their own module, mirroring
  `synty-legacy/internal/source/<tool>/`; keep platform-specific code inside the
  tailer that needs it.

## Temporary files

Scratch scripts, debug output, and corpus dumps go in `tmp/` or `corpus/` at the
repo root (both gitignored). Don't drop scratch files elsewhere.
