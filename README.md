# synty (experiment)

Late-interaction work memory on the production stack, **no generative model**: a
single self-contained Rust binary that encodes agent sessions + GitHub activity
with ColBERT (`pylate-rs`) and indexes them in `next-plaid` (PLAID multi-vector
search + SQLite metadata filtering). Search prints Markdown to stdout — the
surface a coding agent calls.

This is the kernel for the synty rewrite (see `design.md`). It validates the
approach on real data before building the tracker / TUI / team frontend. Findings
in `eval_report.md`.

## Run

The model is loaded from a **local dir** (avoids hf_hub's no-timeout hang). Fetch
a PyLate-format ModernBERT ColBERT once:

```sh
m=models/mxbai; mkdir -p $m/1_Dense
base=https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m/resolve/main
for f in tokenizer.json config.json config_sentence_transformers.json \
         special_tokens_map.json 1_Dense/config.json 1_Dense/model.safetensors model.safetensors; do
  curl -sL --retry 8 --continue-at - --speed-limit 3000 --speed-time 20 "$base/$f" -o "$m/$f"
done
export SYNTY_MODEL="$PWD/models/mxbai"
```

Track local agent sessions and pull GitHub, then index and query:

```sh
cargo run --release -- up                     # solo: track + ingest + index loop
# …or the steps individually:
cargo run --release -- track                  # tail Claude Code/Codex/Cowork → envelopes
cargo run --release -- github                 # PRs/issues via GraphQL ($GITHUB_TOKEN, or gh)
cargo run --release -- ingest                 # → corpus/docs.jsonl
cargo run --release -- index                  # encode + build + content-addressed store
cargo run --release -- search "OCR adapter"   # filtered semantic search
cargo run --release -- search "docs search fix" --filter repo=sie-web
cargo run --release -- cluster --resolution 2.0   # Louvain topics → clusters.json
cargo run --release -- summarize              # extractive sessions + topic digests
cargo run --release -- eval                   # retrieval probe set → eval_runs.md
cargo test                                    # scenario tests
```

For a team, point commands at a shared bucket: `track --bucket s3://b`,
`ingest --bucket s3://b`, `index --bucket s3://b`, `search --bucket s3://b`
(build with `--features s3` or `gcs`). Events from every device converge there,
each message is encoded once across the fleet, and clients pull the published
index. `track --install launchd|systemd` registers the watcher at login.

On Apple Silicon, add `--features metal` to `build`/`run` for GPU encode (~5.7×
faster); `accelerate` (macOS) and `mkl` (Linux) are CPU-BLAS alternatives. The
default build is plain CPU and portable.

`SYNTY_MODEL` defaults to `mixedbread-ai/mxbai-edge-colbert-v0-32m` (downloaded
on first use); set it to a local dir for offline use.
