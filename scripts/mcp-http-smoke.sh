#!/usr/bin/env bash
# Exercise the real HTTP listener and CLI argument path without a bucket,
# model, corpus, or network dependency beyond loopback.
set -euo pipefail

bin="${1:-target/debug/synty}"
work="$(mktemp -d "${TMPDIR:-/tmp}/synty-mcp-smoke.XXXXXX")"
port="${SYNTY_MCP_SMOKE_PORT:-18765}"
token="0123456789abcdef0123456789abcdef0123456789abcdef"
server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf "$work"
}
trap cleanup EXIT INT TERM

mkdir -p "$work/home"
SYNTY_HOME="$work/home" "$bin" mcp --http --bind "127.0.0.1:$port" --token "$token" \
  >"$work/server.out" 2>"$work/server.err" &
server_pid="$!"

for _ in {1..80}; do
  if curl -fsS "http://127.0.0.1:$port/health" >"$work/health.json" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    sed -n '1,120p' "$work/server.err" >&2
    exit 1
  fi
  sleep 0.25
done
grep -q '"status":"ok"' "$work/health.json"

common=(
  -H "Authorization: Bearer $token"
  -H 'Accept: application/json, text/event-stream'
  -H 'Content-Type: application/json'
)
status="$(curl -sS -o "$work/unauthorized.txt" -w '%{http_code}' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  --data '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
  "http://127.0.0.1:$port/mcp")"
[[ "$status" == 401 ]]

status="$(curl -sS -o "$work/initialize.json" -w '%{http_code}' "${common[@]}" \
  --data '{"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"smoke","version":"1"}}}' \
  "http://127.0.0.1:$port/mcp")"
[[ "$status" == 200 ]]
grep -q '"protocolVersion":"2025-03-26"' "$work/initialize.json"

status="$(curl -sS -o "$work/tools.json" -w '%{http_code}' "${common[@]}" \
  -H 'MCP-Protocol-Version: 2025-03-26' \
  --data '{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{}}' \
  "http://127.0.0.1:$port/mcp")"
[[ "$status" == 200 ]]
grep -q '"name":"synty_search"' "$work/tools.json"

status="$(curl -sS -o "$work/protocol.txt" -w '%{http_code}' "${common[@]}" \
  -H 'MCP-Protocol-Version: 2099-01-01' \
  --data '{"jsonrpc":"2.0","id":4,"method":"ping","params":{}}' \
  "http://127.0.0.1:$port/mcp")"
[[ "$status" == 400 ]]

status="$(curl -sS -o "$work/origin.txt" -w '%{http_code}' "${common[@]}" \
  -H 'Origin: https://untrusted.example' \
  -H 'MCP-Protocol-Version: 2025-03-26' \
  --data '{"jsonrpc":"2.0","id":5,"method":"ping","params":{}}' \
  "http://127.0.0.1:$port/mcp")"
[[ "$status" == 403 ]]

for id in {1..116}; do
  status="$(curl -sS -o /dev/null -w '%{http_code}' "${common[@]}" \
    -H 'MCP-Protocol-Version: 2025-03-26' \
    --data "{\"jsonrpc\":\"2.0\",\"id\":$((100 + id)),\"method\":\"ping\",\"params\":{}}" \
    "http://127.0.0.1:$port/mcp")"
  [[ "$status" == 200 ]]
done
status="$(curl -sS -o "$work/rate.txt" -w '%{http_code}' "${common[@]}" \
  -H 'MCP-Protocol-Version: 2025-03-26' \
  --data '{"jsonrpc":"2.0","id":999,"method":"ping","params":{}}' \
  "http://127.0.0.1:$port/mcp")"
[[ "$status" == 429 ]]

printf 'MCP HTTP smoke passed: health, auth, initialize, tools, protocol, origin, rate limit\n'
