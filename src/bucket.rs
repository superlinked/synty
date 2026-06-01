// The backplane. Events are the durable source of truth; the index and its
// metadata are derived projections rebuilt from events. A `Bucket` is the flat
// key/value store underneath: a local directory for solo use, a shared
// S3/GCS bucket for a fleet of trackers. Everything above this trait is
// storage-agnostic.
//
// Keys are '/'-separated paths. The factory picks a backend from the URI:
//   ./dir or file://dir   local filesystem (always available)
//   s3://bucket/prefix    S3   (build --features s3)
//   gs://bucket/prefix    GCS  (build --features gcs)

use anyhow::Result;
use std::path::{Path, PathBuf};

#[allow(dead_code)] // `list` is used by the incremental event reader (M3.4)
pub trait Bucket: Send + Sync {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()>;
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    fn exists(&self, key: &str) -> Result<bool>;
    /// Byte size of an object, or None if absent — a cheap change check for
    /// append-only event files (no full download).
    fn size(&self, key: &str) -> Result<Option<u64>>;
    /// All keys under `prefix` (recursive), relative to the bucket root.
    fn list(&self, prefix: &str) -> Result<Vec<String>>;
}

/// Open the bucket named by `uri`. A bare path or `file://` is local.
pub fn open(uri: &str) -> Result<Box<dyn Bucket>> {
    if let Some(rest) = uri.strip_prefix("s3://") {
        return open_cloud("s3", rest);
    }
    if let Some(rest) = uri.strip_prefix("gs://") {
        return open_cloud("gcs", rest);
    }
    let path = uri.strip_prefix("file://").unwrap_or(uri);
    Ok(Box::new(LocalFs::new(path)))
}

/// Local-filesystem bucket rooted at a directory. Keys map to paths under it.
pub struct LocalFs {
    root: PathBuf,
}

impl LocalFs {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self { root: root.as_ref().to_path_buf() }
    }
    fn path(&self, key: &str) -> PathBuf {
        self.root.join(key)
    }
}

impl Bucket for LocalFs {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        let p = self.path(key);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Write to a temp sibling then rename, so a reader never sees a partial
        // object (atomic on the same filesystem).
        let tmp = p.with_extension("part");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, &p)?;
        Ok(())
    }
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        match std::fs::read(self.path(key)) {
            Ok(b) => Ok(Some(b)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.path(key).exists())
    }
    fn size(&self, key: &str) -> Result<Option<u64>> {
        match std::fs::metadata(self.path(key)) {
            Ok(m) => Ok(Some(m.len())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let base = self.path(prefix);
        let start = if base.is_dir() { base } else { self.root.join(prefix) };
        let mut out = Vec::new();
        for e in walkdir::WalkDir::new(&start).into_iter().filter_map(|e| e.ok()) {
            if !e.file_type().is_file() {
                continue;
            }
            if let Ok(rel) = e.path().strip_prefix(&self.root) {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
        out.sort();
        Ok(out)
    }
}

#[cfg(not(any(feature = "s3", feature = "gcs")))]
fn open_cloud(kind: &str, _rest: &str) -> Result<Box<dyn Bucket>> {
    anyhow::bail!("{kind} bucket needs a cloud backend; rebuild with --features {kind}")
}

#[cfg(any(feature = "s3", feature = "gcs"))]
fn open_cloud(kind: &str, rest: &str) -> Result<Box<dyn Bucket>> {
    cloud::open(kind, rest)
}

#[cfg(any(feature = "s3", feature = "gcs"))]
mod cloud;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_roundtrip_and_list() {
        let dir = std::env::temp_dir().join(format!("synty-bucket-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let b = LocalFs::new(&dir);
        assert!(!b.exists("events/a/x.jsonl").unwrap());
        b.put("events/a/x.jsonl", b"hi").unwrap();
        b.put("events/a/y.jsonl", b"yo").unwrap();
        b.put("index/manifest.json", b"{}").unwrap();
        assert!(b.exists("events/a/x.jsonl").unwrap());
        assert_eq!(b.get("events/a/x.jsonl").unwrap().as_deref(), Some(&b"hi"[..]));
        assert_eq!(b.get("missing").unwrap(), None);
        let keys = b.list("events").unwrap();
        assert_eq!(keys, vec!["events/a/x.jsonl", "events/a/y.jsonl"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn uri_scheme_picks_backend() {
        assert!(open("/tmp/x").is_ok());
        assert!(open("file:///tmp/x").is_ok());
        // Cloud schemes only resolve when their feature is built in.
        let s3 = open("s3://bucket/prefix");
        #[cfg(not(feature = "s3"))]
        assert!(s3.is_err());
        let _ = s3;
    }
}
