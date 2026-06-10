#!/bin/sh
# synty fleet installer: put the binary on PATH, create the machine's home
# (~/.synty), record the shared bucket, and register the login-time tracker.
# Idempotent — safe to bake into dev-VM / sandbox images:
#
#   SYNTY_BUCKET=s3://my-team ./install.sh [path/to/synty-binary]
#
# The tracker is model-free and near-zero footprint: it tails local agent
# session files and pushes raw events to the bucket. Heavy work (encoding,
# summarizing, indexing) happens on whichever machine opens a viewer.
set -eu

BIN="${1:-target/release/synty}"
[ -x "$BIN" ] || { echo "no synty binary at $BIN (build with: cargo build --release)"; exit 1; }

DEST="${SYNTY_PREFIX:-$HOME/.local/bin}"
mkdir -p "$DEST"
install -m 755 "$BIN" "$DEST/synty"
echo "installed $DEST/synty"

SYNTY_HOME="$HOME/.synty"
mkdir -p "$SYNTY_HOME/.synty"

if [ -n "${SYNTY_BUCKET:-}" ]; then
  CFG="$SYNTY_HOME/.synty/config.json"
  if [ -f "$CFG" ]; then
    # Merge the bucket into the existing config without disturbing it.
    python3 - "$CFG" "$SYNTY_BUCKET" << 'EOF'
import json, sys
p, bucket = sys.argv[1], sys.argv[2]
cfg = json.load(open(p))
cfg["bucket"] = bucket
json.dump(cfg, open(p, "w"), indent=2)
EOF
  else
    printf '{\n  "bucket": "%s"\n}\n' "$SYNTY_BUCKET" > "$CFG"
  fi
  echo "configured bucket $SYNTY_BUCKET"
fi

# Register the tracker at login (launchd on macOS, systemd --user on Linux).
case "$(uname -s)" in
  Darwin) KIND=launchd ;;
  Linux)  KIND=systemd ;;
  *) echo "no autostart support for $(uname -s); run 'synty track --watch' yourself"; exit 0 ;;
esac
cd "$SYNTY_HOME" && "$DEST/synty" track --install "$KIND"
echo "done — the tracker starts at login; open a viewer anywhere with: synty tui"
