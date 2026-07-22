// First-run configuration: which GitHub account/org to back-fill, the resolved
// identity, and how many repos to pull. Persisted as .synty/config.json so the
// tracker, backfill, and TUI all read the same choices. `synty init` writes it;
// everything else reads it and falls back to sane defaults until it exists.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

const PATH: &str = ".synty/config.json";
pub const DEFAULT_REPOS: usize = 20;

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct Config {
    /// GitHub org (or user) whose PRs & issues we back-fill.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub org: Option<String>,
    /// The authenticated GitHub login — also the actor for this machine's sessions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub github_login: Option<String>,
    /// How many of the org's most-recently-active repos to track.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backfill_repos: Option<usize>,
    /// Repo names resolved by the last back-fill — the known set used to fold a
    /// session's worktree dir (`sie-web-backbutton`) to its repo (`sie-web`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub repos: Vec<String>,
    /// Recency cap on the indexed corpus (docs). Oldest docs beyond it are
    /// dropped from the index — a memory tool should be loud about forgetting,
    /// so `ingest` warns whenever the cap actually bites.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_docs: Option<usize>,
    /// The fleet's shared bucket (s3://…, gs://…, or a path). Every command
    /// defaults to it; an explicit --bucket flag still wins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bucket: Option<String>,
    /// Named AWS shared-config profile used by the S3 credential chain. For an
    /// unattended workstation this should use credential_process (including
    /// IAM Roles Anywhere); ordinary shared credentials also work. Empty means
    /// object_store's environment/instance/task/workload-role chain on VMs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aws_profile: Option<String>,
    /// Absolute lower bound for newly collected events. RFC3339 in UTC; `init`
    /// also accepts YYYY-MM-DD and `now`, then persists the resolved timestamp.
    /// Upload sync applies the same bound, so an existing local corpus cannot
    /// leak older history merely because a bucket was added later.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_since: Option<String>,
    /// Seconds between event uploads from the watch process (default 60). The
    /// local tail still polls every 30s; this is the network batching cadence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upload_interval_secs: Option<u64>,
    /// Days of stream silence before a machine's tracker counts as gone quiet
    /// on the fleet roster (default 7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fleet_quiet_days: Option<u64>,
    /// GitHub logins to drop from the fleet roster even when they're org
    /// members — service/release accounts (e.g. a release bot that's a real
    /// member) that no one should be "told to install synty". Data, not a
    /// name baked into code: edit it per org.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fleet_ignore: Vec<String>,
}

/// Bucket precedence: explicit flag > config > the local default.
pub fn resolve_bucket(flag: Option<String>) -> String {
    flag.or_else(|| load().bucket).unwrap_or_else(|| ".synty".into())
}

/// Same for commands where "no bucket" is meaningful (track's event push).
pub fn resolve_bucket_opt(flag: Option<String>) -> Option<String> {
    flag.or_else(|| load().bucket)
}

pub const DEFAULT_UPLOAD_INTERVAL_SECS: u64 = 60;

pub fn upload_interval_secs() -> u64 {
    load()
        .upload_interval_secs
        .unwrap_or(DEFAULT_UPLOAD_INTERVAL_SECS)
        .max(1)
}

/// Resolve the persisted capture boundary to milliseconds since the epoch.
/// Invalid hand-edited values are ignored here; `init` validates before save.
pub fn capture_since_ms() -> Option<i64> {
    load()
        .capture_since
        .as_deref()
        .and_then(|s| parse_capture_since(s).ok())
}

/// `now`, YYYY-MM-DD (UTC midnight), or RFC3339 → canonical RFC3339 UTC.
pub fn normalize_capture_since(raw: &str) -> Result<String> {
    let now = chrono::Utc::now();
    let dt = if raw.eq_ignore_ascii_case("now") {
        now
    } else if let Ok(d) = chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        d.and_hms_opt(0, 0, 0)
            .map(|x| x.and_utc())
            .ok_or_else(|| anyhow::anyhow!("invalid capture date: {raw}"))?
    } else {
        chrono::DateTime::parse_from_rfc3339(raw)
            .map_err(|_| {
                anyhow::anyhow!("capture boundary must be `now`, YYYY-MM-DD, or RFC3339: {raw}")
            })?
            .with_timezone(&chrono::Utc)
    };
    Ok(dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
}

fn parse_capture_since(raw: &str) -> Result<i64> {
    let canonical = normalize_capture_since(raw)?;
    Ok(chrono::DateTime::parse_from_rfc3339(&canonical)?.timestamp_millis())
}

/// Unknown/missing timestamps stay visible for forward compatibility; only a
/// record proven older than the configured boundary is excluded.
pub fn captured_at(ts: &str, cutoff_ms: Option<i64>) -> bool {
    let Some(cutoff) = cutoff_ms else { return true };
    chrono::DateTime::parse_from_rfc3339(ts)
        .map(|t| t.timestamp_millis() >= cutoff)
        .unwrap_or(true)
}

pub fn load() -> Config {
    std::fs::read_to_string(PATH).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

pub fn save(c: &Config) -> Result<()> {
    if let Some(dir) = Path::new(PATH).parent() {
        std::fs::create_dir_all(dir)?;
    }
    crate::write_atomic(PATH, serde_json::to_string_pretty(c)?.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_boundary_accepts_day_and_timestamp() {
        assert_eq!(
            normalize_capture_since("2026-07-21").unwrap(),
            "2026-07-21T00:00:00Z"
        );
        assert_eq!(
            normalize_capture_since("2026-07-21T08:15:30-07:00").unwrap(),
            "2026-07-21T15:15:30Z"
        );
        assert!(normalize_capture_since("last Tuesday").is_err());
        let cutoff = parse_capture_since("2026-07-21").unwrap();
        assert!(!captured_at("2026-07-20T23:59:59Z", Some(cutoff)));
        assert!(captured_at("2026-07-21T00:00:00Z", Some(cutoff)));
        assert!(captured_at("future-envelope-time", Some(cutoff)));
    }
}
