// `synty upgrade` — self-update from GitHub Releases. CI builds a per-platform
// artifact on each tag and publishes it as a release asset
// (`synty-<os>-<arch>` + a `.sha256` sidecar); a machine pulls the one built
// for its platform — metal on Apple Silicon, plain CPU on Linux, each with
// runtime fallback — so nobody picks a build. The binary fetches releases with
// the same GitHub token synty already uses for PRs/issues, so there's no extra
// credential and no second distribution channel.

use crate::track;
use anyhow::{anyhow, bail, Result};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const RELEASE_REPO_DEFAULT: &str = "superlinked/synty";

/// Where releases are published. Overridable for forks / private mirrors.
fn release_repo() -> String {
    std::env::var("SYNTY_RELEASE_REPO").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| RELEASE_REPO_DEFAULT.to_string())
}

/// macos/aarch64 → "darwin-arm64", etc. None for a platform we don't ship, so
/// the caller turns that into a clear error instead of guessing a key.
pub fn platform_key_for(os: &str, arch: &str) -> Option<String> {
    let os = match os {
        "macos" => "darwin",
        "linux" => "linux",
        _ => return None,
    };
    let arch = match arch {
        "aarch64" => "arm64",
        "x86_64" => "x64",
        _ => return None,
    };
    Some(format!("{os}-{arch}"))
}

pub fn platform_key() -> Option<String> {
    platform_key_for(std::env::consts::OS, std::env::consts::ARCH)
}

/// True when `a` is a strictly newer x.y.z than `b`. Malformed → false, so a
/// garbage tag never triggers a spurious "upgrade available".
pub fn version_gt(a: &str, b: &str) -> bool {
    match (parse_semver(a), parse_semver(b)) {
        (Some(a), Some(b)) => a > b,
        _ => false,
    }
}

fn parse_semver(v: &str) -> Option<(u64, u64, u64)> {
    let mut it = v.trim().trim_start_matches('v').split('.');
    let major = it.next()?.parse().ok()?;
    let minor = it.next()?.parse().ok()?;
    // Tolerate a pre-release/build suffix on the patch ("1-rc.2" → 1).
    let patch = it.next().unwrap_or("0").split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()?;
    Some((major, minor, patch))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Self-update from the latest release: if it's a newer version with a build for
/// this platform, download it, verify its sha256 against the `.sha256` sidecar,
/// replace the running binary in place, and restart the tracker. No-op when
/// already current; refuses on a checksum mismatch.
pub fn upgrade() -> Result<()> {
    let repo = release_repo();
    let rel = crate::github::latest_release(&repo)?.ok_or_else(|| anyhow!("no releases published for {repo} yet"))?;
    let latest = rel.tag.trim_start_matches('v');
    if !version_gt(latest, VERSION) {
        eprintln!("upgrade: already up to date (v{VERSION})");
        return Ok(());
    }
    let plat = platform_key()
        .ok_or_else(|| anyhow!("{}/{} has no published build", std::env::consts::OS, std::env::consts::ARCH))?;
    let name = format!("synty-{plat}");
    let asset =
        rel.asset(&name).ok_or_else(|| anyhow!("release {} has no {name} (assets: {})", rel.tag, rel.asset_names()))?;
    let bytes = crate::github::download_asset(&asset.api_url)?;

    // Integrity: verify against the .sha256 sidecar when the release carries one.
    match rel.asset(&format!("{name}.sha256")) {
        Some(sc) => {
            let raw = crate::github::download_asset(&sc.api_url)?;
            let want = String::from_utf8_lossy(&raw).split_whitespace().next().unwrap_or("").to_lowercase();
            let got = sha256_hex(&bytes);
            if want != got {
                bail!("sha256 mismatch for {name} (expected {want}, got {got}) — refusing to install");
            }
        }
        None => eprintln!("upgrade: no {name}.sha256 in the release — skipping checksum verification"),
    }

    replace_self(&bytes)?;
    eprintln!("upgrade: v{VERSION} → v{latest} installed");
    match track::restart() {
        Ok(true) => eprintln!("upgrade: tracker restarted on the new binary"),
        Ok(false) => eprintln!("upgrade: open a new shell (or `synty up`) to use the new binary"),
        Err(e) => eprintln!("upgrade: installed, but tracker restart failed ({e}) — restart it yourself"),
    }
    let _ = std::fs::remove_file(CHECK_PATH); // status should reflect the new version now
    Ok(())
}

/// Replace the running binary in place: temp sibling → mode 0755 → atomic
/// rename over `current_exe()`. The live process keeps the old inode, so this
/// call returns normally and the new file takes effect on the next launch.
fn replace_self(bytes: &[u8]) -> Result<()> {
    let exe = std::env::current_exe().map_err(|e| anyhow!("locate the running binary: {e}"))?;
    let dir = exe.parent().ok_or_else(|| anyhow!("binary has no parent directory"))?;
    let tmp = dir.join(format!(".synty-upgrade.{}", std::process::id()));
    std::fs::write(&tmp, bytes).map_err(|e| anyhow!("write {}: {e}", tmp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))?;
    }
    std::fs::rename(&tmp, &exe).map_err(|e| anyhow!("replace {}: {e}", exe.display()))?;
    Ok(())
}

const CHECK_TTL_SECS: u64 = 6 * 3600;
const CHECK_PATH: &str = ".synty/upgrade_check.json";

#[derive(serde::Serialize, serde::Deserialize)]
struct Check {
    checked: u64,
    latest: String,
}

fn now_unix() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Best-effort: the newer version published to the release repo, or None (no
/// token, no releases, or we're current). Cached for a few hours in the local
/// store so `status`/the TUI don't query GitHub on every refresh. Never errors
/// — it feeds the nag, not a gate.
pub fn available() -> Option<String> {
    if let Ok(raw) = std::fs::read_to_string(CHECK_PATH) {
        if let Ok(c) = serde_json::from_str::<Check>(&raw) {
            if now_unix().saturating_sub(c.checked) < CHECK_TTL_SECS {
                return version_gt(&c.latest, VERSION).then_some(c.latest);
            }
        }
    }
    let rel = crate::github::latest_release(&release_repo()).ok()??;
    let latest = rel.tag.trim_start_matches('v').to_string();
    let _ = std::fs::create_dir_all(".synty");
    if let Ok(raw) = serde_json::to_string(&Check { checked: now_unix(), latest: latest.clone() }) {
        let _ = std::fs::write(CHECK_PATH, raw);
    }
    version_gt(&latest, VERSION).then_some(latest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_key_maps_known_targets_only() {
        assert_eq!(platform_key_for("macos", "aarch64").as_deref(), Some("darwin-arm64"));
        assert_eq!(platform_key_for("linux", "x86_64").as_deref(), Some("linux-x64"));
        assert_eq!(platform_key_for("windows", "x86_64"), None, "we don't ship windows");
        assert_eq!(platform_key_for("linux", "riscv64"), None, "unknown arch → no guess");
    }

    #[test]
    fn version_gt_compares_semver_handles_v_prefix_and_garbage() {
        assert!(version_gt("0.2.0", "0.1.0"));
        assert!(version_gt("0.1.10", "0.1.9"), "numeric, not lexical");
        assert!(version_gt("v0.2.0", "0.1.0"), "tolerates a leading v from a tag");
        assert!(!version_gt("0.1.0", "0.1.0"), "equal is not newer");
        assert!(!version_gt("0.1.0", "0.2.0"));
        assert!(!version_gt("garbage", "0.1.0"), "malformed never nags");
        assert!(!version_gt("0.1.0", ""));
    }

    #[test]
    fn sha256_hex_is_64_lowercase_hex() {
        let h = sha256_hex(b"synty");
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }
}
