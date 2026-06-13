// `synty publish` / `synty upgrade` — binary distribution through the same
// bucket that already carries events and the index. The bucket is synty's only
// shared infrastructure, so the binary lives there too: immutable, per-version,
// per-platform artifacts under `bin/<version>/synty-<os>-<arch>` behind a
// `bin/latest.json` pointer — the same immutable-object + atomic-pointer shape
// the index uses. A machine pulls the one artifact built for its platform
// (metal on Apple Silicon, plain CPU on Linux), so nobody picks a build.

use crate::{bucket, track};
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const MANIFEST_KEY: &str = "bin/latest.json";

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Artifact {
    pub key: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Manifest {
    pub version: String,
    pub artifacts: BTreeMap<String, Artifact>,
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
/// garbage manifest never triggers a spurious "upgrade available".
pub fn version_gt(a: &str, b: &str) -> bool {
    match (parse_semver(a), parse_semver(b)) {
        (Some(a), Some(b)) => a > b,
        _ => false,
    }
}

fn parse_semver(v: &str) -> Option<(u64, u64, u64)> {
    let mut it = v.trim().split('.');
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

/// Add/replace `plat`'s artifact. A new version starts a fresh set (old
/// binaries stay in the bucket under their own dir, just not pointed-to); the
/// same version merges, so each platform publishes independently.
pub fn merge_artifact(mut m: Manifest, version: &str, plat: &str, art: Artifact) -> Manifest {
    if m.version != version {
        m = Manifest { version: version.to_string(), artifacts: BTreeMap::new() };
    }
    m.artifacts.insert(plat.to_string(), art);
    m
}

fn read_manifest(b: &dyn bucket::Bucket) -> Result<Option<Manifest>> {
    match b.get(MANIFEST_KEY)? {
        Some(raw) => Ok(Some(serde_json::from_slice(&raw).context("parse bin/latest.json")?)),
        None => Ok(None),
    }
}

/// Upload this binary as the current platform's artifact for its version, then
/// merge it into the `latest.json` pointer. Run the freshly built release binary
/// so its version always matches the bytes it uploads.
pub fn publish(bucket_uri: &str, binary: Option<String>) -> Result<()> {
    let plat = platform_key().ok_or_else(|| {
        anyhow!(
            "{}/{} isn't a published target — build and host it yourself",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    })?;
    let path = match binary {
        Some(p) => std::path::PathBuf::from(p),
        None => std::env::current_exe().context("locate the running binary")?,
    };
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let key = format!("bin/{VERSION}/synty-{plat}");

    let b = bucket::open(bucket_uri)?;
    b.put(&key, &bytes)?;
    let art = Artifact { key: key.clone(), sha256: sha256_hex(&bytes), bytes: bytes.len() as u64 };
    let updated = merge_artifact(read_manifest(&*b)?.unwrap_or_default(), VERSION, &plat, art);
    b.put(MANIFEST_KEY, &serde_json::to_vec_pretty(&updated)?)?;

    let plats: Vec<&str> = updated.artifacts.keys().map(String::as_str).collect();
    eprintln!(
        "publish: {key} ({} MiB) → {bucket_uri}\npublish: latest.json now v{VERSION} [{}]",
        bytes.len() / 1024 / 1024,
        plats.join(", ")
    );
    Ok(())
}

/// Self-update from the bucket: if `latest.json` names a newer version with a
/// build for this platform, download it, verify its sha256, replace the running
/// binary in place, and restart the tracker. No-op when already current.
pub fn upgrade(bucket_uri: &str) -> Result<()> {
    let b = bucket::open(bucket_uri)?;
    let manifest = read_manifest(&*b)?
        .ok_or_else(|| anyhow!("no published binaries in {bucket_uri} (nothing at {MANIFEST_KEY})"))?;
    if !version_gt(&manifest.version, VERSION) {
        eprintln!("upgrade: already up to date (v{VERSION})");
        return Ok(());
    }
    let plat = platform_key()
        .ok_or_else(|| anyhow!("{}/{} has no published build", std::env::consts::OS, std::env::consts::ARCH))?;
    let art = manifest.artifacts.get(&plat).ok_or_else(|| {
        anyhow!(
            "no {plat} build published for v{} (have: {})",
            manifest.version,
            manifest.artifacts.keys().cloned().collect::<Vec<_>>().join(", ")
        )
    })?;
    let bytes = b.get(&art.key)?.ok_or_else(|| anyhow!("manifest points at {}, which is missing", art.key))?;
    let got = sha256_hex(&bytes);
    if got != art.sha256 {
        bail!("sha256 mismatch for {} (expected {}, got {got}) — refusing to install", art.key, art.sha256);
    }
    replace_self(&bytes)?;
    eprintln!("upgrade: v{VERSION} → v{} installed", manifest.version);
    match track::restart() {
        Ok(true) => eprintln!("upgrade: tracker restarted on the new binary"),
        Ok(false) => eprintln!("upgrade: open a new shell (or `synty up`) to use the new binary"),
        Err(e) => eprintln!("upgrade: installed, but tracker restart failed ({e}) — restart it yourself"),
    }
    Ok(())
}

/// Replace the running binary in place: temp sibling → mode 0755 → atomic
/// rename over `current_exe()`. The live process keeps the old inode, so this
/// call returns normally and the new file takes effect on the next launch.
fn replace_self(bytes: &[u8]) -> Result<()> {
    let exe = std::env::current_exe().context("locate the running binary")?;
    let dir = exe.parent().ok_or_else(|| anyhow!("binary has no parent directory"))?;
    let tmp = dir.join(format!(".synty-upgrade.{}", std::process::id()));
    std::fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))?;
    }
    std::fs::rename(&tmp, &exe).with_context(|| format!("replace {}", exe.display()))?;
    Ok(())
}

/// Best-effort: the newer version published to `bucket`, or None (no access, no
/// manifest, parse error, or we're current). Never errors — it feeds the status
/// nag, not a gate.
pub fn available(bucket_uri: &str) -> Option<String> {
    let b = bucket::open(bucket_uri).ok()?;
    let raw = b.get(MANIFEST_KEY).ok()??;
    let m: Manifest = serde_json::from_slice(&raw).ok()?;
    version_gt(&m.version, VERSION).then_some(m.version)
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
    fn version_gt_compares_semver_and_ignores_garbage() {
        assert!(version_gt("0.2.0", "0.1.0"));
        assert!(version_gt("0.1.10", "0.1.9"), "numeric, not lexical");
        assert!(version_gt("1.0.0", "0.9.9"));
        assert!(!version_gt("0.1.0", "0.1.0"), "equal is not newer");
        assert!(!version_gt("0.1.0", "0.2.0"));
        assert!(!version_gt("garbage", "0.1.0"), "malformed never nags");
        assert!(!version_gt("0.1.0", ""));
    }

    #[test]
    fn merge_keeps_platforms_within_a_version_and_resets_across() {
        let art = |k: &str| Artifact { key: k.into(), sha256: "x".into(), bytes: 1 };
        let m = merge_artifact(Manifest::default(), "0.1.0", "darwin-arm64", art("a"));
        let m = merge_artifact(m, "0.1.0", "linux-x64", art("b"));
        assert_eq!(m.version, "0.1.0");
        assert_eq!(m.artifacts.len(), 2, "same version accumulates platforms");
        // A new version starts fresh — stale platforms don't linger in the pointer.
        let m = merge_artifact(m, "0.2.0", "darwin-arm64", art("c"));
        assert_eq!(m.version, "0.2.0");
        assert_eq!(m.artifacts.len(), 1);
        assert_eq!(m.artifacts["darwin-arm64"].key, "c");
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let m = merge_artifact(
            Manifest::default(),
            "0.1.0",
            "darwin-arm64",
            Artifact { key: "bin/0.1.0/synty-darwin-arm64".into(), sha256: "abc".into(), bytes: 42 },
        );
        let raw = serde_json::to_vec(&m).unwrap();
        assert_eq!(serde_json::from_slice::<Manifest>(&raw).unwrap(), m);
    }

    #[test]
    fn sha256_hex_is_64_lowercase_hex() {
        let h = sha256_hex(b"synty");
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }
}
