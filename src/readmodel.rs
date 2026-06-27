// The versioned read-model. Builds are immutable directories under
// `index/builds/<build>/`; a tiny pointer file (`index/current.json`,
// written atomically, LAST) names the build readers should open. Nothing ever
// mutates a directory a reader may have mmapped — an incremental update
// (append or patch) CLONES the previous build (CoW where the filesystem
// allows), mutates only the clone, and repoints; and a
// torn build can never become current because the pointer only moves after the
// new directory is complete. `rev` versions the derived artifacts (clusters)
// added to a build after indexing: `unit_clusters.<rev>.json` files are
// additive, so bumping rev never rewrites anything either.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const ROOT: &str = "index";
const POINTER: &str = "index/current.json";

/// The read-model layout version. Bumped only for breaking layout changes;
/// a reader that meets a NEWER format refuses to pull it (and says to upgrade)
/// instead of misparsing it.
pub const FORMAT: u32 = 1;

/// The build a reader should open, resolved through the pointer.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Current {
    pub build: String,
    pub rev: u64,
    #[serde(default = "format_default")]
    pub format: u32,
    /// The synty version that wrote this pointer — for debugging mixed fleets.
    #[serde(default)]
    pub writer: String,
}

fn format_default() -> u32 {
    1 // pointers from before the field are format 1
}

/// Identity is (build, rev): the same read-model regardless of which binary
/// version wrote the pointer — mixed-version fleets must not churn on it.
impl PartialEq for Current {
    fn eq(&self, o: &Self) -> bool {
        self.build == o.build && self.rev == o.rev
    }
}

impl Current {
    pub fn dir(&self) -> PathBuf {
        build_dir(&self.build)
    }
    pub fn docs(&self) -> PathBuf {
        self.dir().join("docs.jsonl")
    }
    pub fn clusters(&self) -> PathBuf {
        self.dir().join(format!("unit_clusters.{}.json", self.rev))
    }
}

pub fn build_dir(build: &str) -> PathBuf {
    Path::new(ROOT).join("builds").join(build)
}

/// Resolve the current build, or None when nothing is built yet.
pub fn current() -> Option<Current> {
    std::fs::read_to_string(POINTER).ok().and_then(|s| serde_json::from_str(&s).ok())
}

/// Atomically repoint readers at `build`/`rev`. The build directory must be
/// complete before this is called — the pointer move IS the publish.
pub fn repoint(build: &str, rev: u64) -> Result<()> {
    std::fs::create_dir_all(ROOT)?;
    let cur = Current {
        build: build.into(),
        rev,
        format: FORMAT,
        writer: env!("CARGO_PKG_VERSION").into(),
    };
    crate::write_atomic(POINTER, serde_json::to_string(&cur)?.as_bytes())
}

/// Reader conveniences: resolve through the pointer, with the working-corpus
/// paths as the no-pointer fallback (a fresh checkout reads what `ingest`
/// wrote before any build exists).
pub fn docs_path() -> PathBuf {
    current().map(|c| c.docs()).unwrap_or_else(|| PathBuf::from(crate::DOCS_PATH))
}

pub fn clusters_path() -> PathBuf {
    current().map(|c| c.clusters()).unwrap_or_else(|| PathBuf::from("unit_clusters.json"))
}

pub fn index_dir() -> PathBuf {
    current().map(|c| c.dir()).unwrap_or_else(|| PathBuf::from(crate::INDEX_PATH))
}

/// The mtime of the pointer — "when was the read-model last refreshed", for
/// staleness checks.
pub fn last_updated() -> Option<std::time::SystemTime> {
    std::fs::metadata(POINTER).ok()?.modified().ok()
}

/// Clone a build directory for an incremental append: CoW where the filesystem
/// supports it (APFS `cp -c`, reflink on XFS/btrfs), full copy otherwise. The
/// clone diverges on write, so mutating the copy never touches the original.
pub fn clone_build(from: &Path, to: &Path) -> Result<()> {
    let _ = std::fs::remove_dir_all(to);
    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let args: &[&str] = if cfg!(target_os = "macos") { &["-Rc"] } else { &["-a", "--reflink=auto"] };
    let ok = std::process::Command::new("cp")
        .args(args)
        .arg(from)
        .arg(to)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok {
        return Ok(());
    }
    copy_tree(from, to).with_context(|| format!("clone {} -> {}", from.display(), to.display()))
}

fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        let dest = to.join(e.file_name());
        if e.file_type()?.is_dir() {
            copy_tree(&e.path(), &dest)?;
        } else {
            std::fs::copy(e.path(), &dest)?;
        }
    }
    Ok(())
}

/// Drop build directories no longer referenced, keeping the current build and
/// `also_keep` (the previous one — a reader may still hold its mmap; on unix
/// even a deleted mmap stays readable, this is just tidiness with a grace).
pub fn gc(also_keep: &[String]) {
    let Some(cur) = current() else { return };
    let builds = Path::new(ROOT).join("builds");
    let Ok(entries) = std::fs::read_dir(&builds) else { return };
    for e in entries.filter_map(|e| e.ok()) {
        let name = e.file_name().to_string_lossy().into_owned();
        if name != cur.build && !also_keep.contains(&name) {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

/// The build id for a doc-hash manifest: identity of the corpus snapshot.
pub fn build_id(doc_hashes: &[u64]) -> String {
    let mut bytes = Vec::with_capacity(doc_hashes.len() * 8);
    for h in doc_hashes {
        bytes.extend_from_slice(&h.to_le_bytes());
    }
    format!("{:016x}", crate::index::fnv1a(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The pointer only moves after a build is complete, and readers resolve
    // docs/clusters paths through it (rev selects the clusters file).
    #[test]
    fn pointer_resolves_versioned_paths() {
        let c = Current { build: "abc123".into(), rev: 2, format: FORMAT, writer: String::new() };
        assert_eq!(c.dir(), Path::new("index/builds/abc123"));
        assert!(c.docs().ends_with("index/builds/abc123/docs.jsonl"));
        assert!(c.clusters().ends_with("index/builds/abc123/unit_clusters.2.json"));
    }

    // Pointers from before the format field read as format 1; identity ignores
    // the writer (mixed binary versions must not churn the pointer); a newer
    // format is detectable for the upgrade gate.
    #[test]
    fn pointer_versioning_contract() {
        let old: Current = serde_json::from_str(r#"{"build":"abc","rev":1}"#).unwrap();
        assert_eq!(old.format, 1);
        let newer: Current =
            serde_json::from_str(r#"{"build":"abc","rev":1,"format":9,"writer":"9.9.9"}"#).unwrap();
        assert!(newer.format > FORMAT, "upgrade gate can see the newer format");
        assert_eq!(old, newer, "identity is (build, rev) only");
    }

    #[test]
    fn build_id_is_stable_and_distinguishing() {
        assert_eq!(build_id(&[1, 2, 3]), build_id(&[1, 2, 3]));
        assert_ne!(build_id(&[1, 2, 3]), build_id(&[1, 2, 4]));
    }

    #[test]
    fn clone_build_copies_a_tree() {
        let dir = std::env::temp_dir().join(format!("synty-clone-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (a, b) = (dir.join("a"), dir.join("b"));
        std::fs::create_dir_all(a.join("sub")).unwrap();
        std::fs::write(a.join("x.npy"), b"big").unwrap();
        std::fs::write(a.join("sub/meta.json"), b"{}").unwrap();
        clone_build(&a, &b).unwrap();
        assert_eq!(std::fs::read(b.join("x.npy")).unwrap(), b"big");
        assert_eq!(std::fs::read(b.join("sub/meta.json")).unwrap(), b"{}");
        // Mutating the clone must not touch the original (CoW divergence).
        std::fs::write(b.join("x.npy"), b"changed").unwrap();
        assert_eq!(std::fs::read(a.join("x.npy")).unwrap(), b"big");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
