#!/bin/sh
# synty installer — one paste from nothing to "tracking + a viewer":
#
#   curl -fsSL <internal-url>/install.sh | sh                      # local trial
#   curl -fsSL <internal-url>/install.sh | sh -s -- gs://my-team   # join the team
#
# It puts the binary on PATH, runs `synty join [bucket]` (pins your GitHub
# identity, enables the login-time tracker, runs the first build), then opens
# the viewer. Omit the bucket to try synty against your local sessions first;
# re-paste with a bucket later and that same `join` switches you onto the team.
# Idempotent — safe to bake into dev-VM / sandbox images.
#
# Binary source, in order: $SYNTY_BINARY_URL (downloaded), $SYNTY_BINARY (a
# local path), else ./target/release/synty (a local build). Distribution is
# internal for now — no public package or Homebrew tap while the team rolls out.
set -eu

BUCKET="${1:-}"

# 1. Resolve the binary.
if [ -n "${SYNTY_BINARY_URL:-}" ]; then
  TMP="$(mktemp)"
  trap 'rm -f "$TMP"' EXIT
  echo "downloading synty from $SYNTY_BINARY_URL"
  curl -fsSL "$SYNTY_BINARY_URL" -o "$TMP"
  chmod +x "$TMP"
  BIN="$TMP"
else
  BIN="${SYNTY_BINARY:-target/release/synty}"
fi
[ -x "$BIN" ] || {
  echo "no synty binary at $BIN (set SYNTY_BINARY_URL, set SYNTY_BINARY, or build with: cargo build --release)"
  exit 1
}

# 2. Put it on PATH.
DEST="${SYNTY_PREFIX:-$HOME/.local/bin}"
mkdir -p "$DEST"
install -m 755 "$BIN" "$DEST/synty"
echo "installed $DEST/synty"

# 3. Pin a stable machine-wide home so config, tracker, and build all agree no
#    matter which directory `curl | sh` ran in.
SYNTY_HOME="${SYNTY_HOME:-$HOME/.synty}"
export SYNTY_HOME
mkdir -p "$SYNTY_HOME"

# 4. One step: config + GitHub identity + login-time tracker + first build.
#    With a bucket it's the local→bucket switch; without, a local trial.
if [ -n "$BUCKET" ]; then
  "$DEST/synty" join "$BUCKET"
else
  "$DEST/synty" join
fi

# 5. Drop into the viewer when there's a terminal to drive it. Under `curl | sh`
#    stdin is the script, so reattach the controlling TTY; with no terminal
#    (image bake / CI), just say how to open it.
if [ -t 1 ] && [ -e /dev/tty ]; then
  exec "$DEST/synty" tui </dev/tty
else
  echo "done — the tracker starts at login; open the viewer anywhere with: synty tui"
fi
