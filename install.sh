#!/bin/sh
# synty installer — one paste from nothing to "tracking + a viewer":
#
#   curl -fsSL <internal-url>/install.sh | sh                      # local trial
#   curl -fsSL <internal-url>/install.sh | sh -s -- gs://my-team   # join the team
#
# It puts the binary on PATH, runs `synty init [bucket]` (pins your GitHub
# identity, enables the login-time tracker, runs the first build), then opens
# the viewer. Omit the bucket to try synty against your local sessions first;
# re-paste with a bucket later and that same `init` switches you onto the team.
# Idempotent — safe to bake into dev-VM / sandbox images.
#
# Binary source, in order: $SYNTY_BINARY_URL (a direct artifact URL — a public
# mirror or presigned link), else the latest GitHub Release of
# $SYNTY_RELEASE_REPO (default superlinked/synty) via `gh` (which authenticates
# to the private repo for you), else $SYNTY_BINARY / ./target/release/synty (a
# local build). Distribution is internal for now — no public package or Homebrew
# tap. After install, `synty upgrade` self-updates from the same releases.
set -eu

BUCKET="${1:-}"
RELEASE_REPO="${SYNTY_RELEASE_REPO:-superlinked/synty}"

# 1. Resolve the binary. This machine's platform key matches `release::platform_key`.
os=$(uname -s); arch=$(uname -m)
case "$os" in Darwin) os=darwin ;; Linux) os=linux ;; *) echo "unsupported OS: $os"; exit 1 ;; esac
case "$arch" in arm64 | aarch64) arch=arm64 ;; x86_64 | amd64) arch=x64 ;; *) echo "unsupported arch: $arch"; exit 1 ;; esac
PLAT="synty-$os-$arch"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [ -n "${SYNTY_BINARY_URL:-}" ]; then
  echo "downloading $PLAT from $SYNTY_BINARY_URL"
  curl -fsSL "$SYNTY_BINARY_URL" -o "$WORK/$PLAT"
  BIN="$WORK/$PLAT"
elif command -v gh >/dev/null 2>&1; then
  echo "downloading $PLAT from the latest $RELEASE_REPO release"
  gh release download --repo "$RELEASE_REPO" --pattern "$PLAT" --dir "$WORK" --clobber \
    || { echo "no $PLAT in the latest $RELEASE_REPO release (is one published for your platform?)"; exit 1; }
  BIN="$WORK/$PLAT"
else
  BIN="${SYNTY_BINARY:-target/release/synty}"
fi
[ -x "$BIN" ] || chmod +x "$BIN" 2>/dev/null || true
[ -f "$BIN" ] || {
  echo "no synty binary: install \`gh\` (and \`gh auth login\`) to pull from $RELEASE_REPO, or set SYNTY_BINARY_URL / SYNTY_BINARY, or build with: cargo build --release"
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
  "$DEST/synty" init "$BUCKET"
else
  "$DEST/synty" init
fi

# 5. Drop into the viewer when there's a terminal to drive it. Under `curl | sh`
#    stdin is the script, so reattach the controlling TTY; with no terminal
#    (image bake / CI), just say how to open it.
if [ -t 1 ] && [ -e /dev/tty ]; then
  exec "$DEST/synty" tui </dev/tty
else
  echo "done — the tracker starts at login; open the viewer anywhere with: synty tui"
fi
