#!/bin/sh
# Manual fleet smoke — model-heavy, so it is NOT part of `cargo test` (which
# stays pure). It exercises the real bucket path on a temp local-dir bucket and
# two temp SYNTY_HOMEs:
#
#   machine A  →  publish the read-model to the bucket
#   machine B  →  COLD pull (full: blobs_reused=0)
#   machine B  →  pull an incremental build that shares A's blobs (DELTA: reused>0, ~0 bytes)
#
# and asserts the `[metrics sync]` lines at each step. Run from the repo root.
# It needs a prior local build (the existing `.synty/embeddings` + `index/`), so
# it can publish without re-encoding the whole corpus.
#
# Gotchas baked in: $SYNTY_HOME dirs are created up-front (a missing one would
# make the binary fall back to the real home), and the real $HOME is kept so the
# embedding-model cache is reused rather than re-downloaded.
set -eu

BIN="${SYNTY_BIN:-./target/release/synty}"
[ -x "$BIN" ] || { echo "build the binary first:  cargo build --release [--features metal]"; exit 1; }
[ -d .synty/embeddings ] && [ -f index/current.json ] || {
  echo "need a prior local build first:  $BIN build"; exit 1
}

W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
B="$W/bucket"; mkdir -p "$B" "$W/hB"
# Skip re-encode: point the bucket's content stores at the existing local ones.
ln -s "$PWD/.synty/embeddings" "$B/embeddings"
ln -s "$PWD/.synty/summaries"  "$B/summaries"

say()  { printf '\n=== %s ===\n' "$1"; }
fail() { echo "FAIL: $1"; exit 1; }

say "machine A: publish the current read-model → bucket"
A="$("$BIN" index --bucket "$B" 2>&1)"; echo "$A" | grep -E 'metrics sync' || true
echo "$A" | grep -q 'phase=publish_up' || fail "no publish_up metric"

say "machine B (fresh home): COLD pull — expect blobs_reused=0"
B1="$(SYNTY_HOME="$W/hB" "$BIN" search "rate limiting" --bucket "$B" 2>&1)"; echo "$B1" | grep -E 'metrics sync' || true
echo "$B1" | grep -qE 'phase=pull_down .*blobs_reused=0' || fail "cold pull was not reused=0"

say "simulate an incremental build that reuses A's blobs (new build name, same files)"
python3 - "$B" <<'PY'
import json, sys, shutil
B = sys.argv[1]
cur = json.load(open(f"{B}/current.json")); x, rev = cur["build"], cur["rev"]
shutil.copy(f"{B}/builds/{x}.{rev}.json", f"{B}/builds/feedfacefeedface.0.json")
json.dump({"build": "feedfacefeedface", "rev": 0, "format": 1, "writer": "smoke"},
          open(f"{B}/current.json", "w"))
PY

say "machine B: pull the new build — expect blobs_reused>0 and ~0 bytes (all blobs already local)"
B2="$(SYNTY_HOME="$W/hB" "$BIN" search "rate limiting" --bucket "$B" 2>&1)"; echo "$B2" | grep -E 'metrics sync' || true
echo "$B2" | grep -qE 'phase=pull_down .*blobs_reused=[1-9]' || fail "delta pull did not reuse on-disk blobs"
echo "$B2" | grep -qE 'phase=pull_down .*blobs_fetched=0'    || echo "WARN: expected blobs_fetched=0 on a fully-shared build"

say "PASS — publish, cold pull, and delta pull all behaved"
