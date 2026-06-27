// Who and where a tracked session came from. The session author used to be a
// hardcoded "daniel"; this derives a real identity instead, and a stable machine
// id for the `edge-<machine>-<source>` stream names so a fleet doesn't all
// collide on the "local" default. Resolution is offline and best-effort —
// `synty setup` (later) pins the exact GitHub login into .synty/identity, which
// takes precedence here the moment it exists.

use std::path::Path;

const ID_DIR: &str = ".synty";

/// The actor to stamp on this machine's sessions. Precedence: a pinned GitHub
/// login (.synty/identity, written by setup) → the local part of git's
/// user.email → $USER. So a person's sessions merge with their GitHub PRs
/// instead of every session reading as one hardcoded name.
pub fn actor() -> String {
    pick_actor(crate::config::load().github_login, git_email_local(), env_user())
}

fn pick_actor(pinned: Option<String>, git_email: Option<String>, user: Option<String>) -> String {
    pinned.or(git_email).or(user).unwrap_or_else(|| "unknown".into())
}

/// A stable machine id for stream names. Returns the value pinned in
/// .synty/machine if present, else derives `<hostname>-<short>` and persists it
/// — so two boxes never both call themselves "local". Ephemeral/cloud hosts
/// (random or colliding hostnames, throwaway disks) should pass `--machine`.
pub fn machine_id() -> String {
    if let Some(m) = read_id("machine") {
        return m;
    }
    let host = hostname();
    let id = machine_name(&host, short_nonce(&host));
    let _ = write_id("machine", &id);
    id
}

/// Resolve the `--machine` argument: the "local" default (or empty) auto-derives
/// a stable id; an explicit value is taken as-is (sanitized).
pub fn resolve_machine(arg: &str) -> String {
    if arg.is_empty() || arg == "local" {
        machine_id()
    } else {
        sanitize(arg)
    }
}

fn machine_name(host: &str, nonce: u64) -> String {
    format!("{}-{:04x}", sanitize(host), nonce & 0xffff)
}

/// Pin the authenticated GitHub login so this machine's sessions attribute to
/// the same identity as its PRs. Called by the GitHub backfill (which already
/// holds a token); `actor()` then reads it offline. Idempotent.
pub fn cache_github_login(login: &str) {
    let login = sanitize(login);
    if login.is_empty() {
        return;
    }
    let mut cfg = crate::config::load();
    if cfg.github_login.as_deref() == Some(login.as_str()) {
        return;
    }
    cfg.github_login = Some(login);
    let _ = crate::config::save(&cfg);
}

// ── sources (best-effort, offline) ──────────────────────────────────────────

fn env_user() -> Option<String> {
    std::env::var("USER").ok().map(|u| sanitize(&u)).filter(|u| !u.is_empty())
}

fn git_email_local() -> Option<String> {
    let o = std::process::Command::new("git").args(["config", "user.email"]).output().ok()?;
    if !o.status.success() {
        return None;
    }
    let email = String::from_utf8_lossy(&o.stdout);
    email.trim().split('@').next().map(sanitize).filter(|s| !s.is_empty())
}

/// Short, dotless hostname (`Daniels-MacBook-Pro-2.local` → `Daniels-MacBook-Pro-2`).
fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().split('.').next().unwrap_or("").to_string())
        .map(|h| sanitize(&h))
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "host".into())
}

/// FNV-1a over hostname + wall-clock nanos + pid → a stable-once suffix seed.
/// Only used when minting a new machine id, which is then persisted.
fn short_nonce(host: &str) -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    let pid = std::process::id() as u64;
    let mut h = 0xcbf29ce484222325u64;
    for b in host.bytes().chain(nanos.to_le_bytes()).chain(pid.to_le_bytes()) {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn sanitize(s: &str) -> String {
    s.trim().chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' }).collect()
}

// ── persisted ids ───────────────────────────────────────────────────────────

fn read_id(name: &str) -> Option<String> {
    let s = std::fs::read_to_string(Path::new(ID_DIR).join(name)).ok()?;
    let s = s.trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn write_id(name: &str, val: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(ID_DIR)?;
    std::fs::write(Path::new(ID_DIR).join(name), val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actor_prefers_pinned_then_git_then_user() {
        assert_eq!(pick_actor(Some("svonava".into()), Some("dan".into()), Some("d".into())), "svonava");
        assert_eq!(pick_actor(None, Some("svonava".into()), Some("ubuntu".into())), "svonava");
        assert_eq!(pick_actor(None, None, Some("ubuntu".into())), "ubuntu");
        assert_eq!(pick_actor(None, None, None), "unknown");
    }

    #[test]
    fn machine_name_appends_short_suffix_and_sanitizes() {
        assert_eq!(machine_name("Daniels-MacBook-Pro-2", 0xabcd), "Daniels-MacBook-Pro-2-abcd");
        // dots/spaces in a host become dashes; suffix is masked to 4 hex.
        assert_eq!(machine_name("ip 10.0.1.23", 0x1_2345), "ip-10-0-1-23-2345");
    }

    #[test]
    fn resolve_machine_passes_explicit_through() {
        assert_eq!(resolve_machine("sie-ci"), "sie-ci");
        assert_eq!(resolve_machine("runner/7"), "runner-7");
    }

    #[test]
    fn sanitize_keeps_word_chars_and_dashes() {
        assert_eq!(sanitize("svonava@gmail.com"), "svonava-gmail-com");
        assert_eq!(sanitize("  Daniel's Box "), "Daniel-s-Box");
    }
}
