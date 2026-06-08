// First-run configuration: which GitHub account/org to back-fill, the resolved
// identity, and how many repos to pull. Persisted as .synty/config.json so the
// tracker, backfill, and TUI all read the same choices. `synty setup` writes it;
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
}

pub fn load() -> Config {
    std::fs::read_to_string(PATH).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

pub fn save(c: &Config) -> Result<()> {
    if let Some(dir) = Path::new(PATH).parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(PATH, serde_json::to_string_pretty(c)?)?;
    Ok(())
}
