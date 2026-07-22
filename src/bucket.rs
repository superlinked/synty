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
    /// Fetch a batch while preserving input order. Cloud backends override
    /// this to overlap request latency; the local/test default stays simple.
    fn get_many(&self, keys: &[String]) -> Result<Vec<Option<Vec<u8>>>> {
        keys.iter().map(|key| self.get(key)).collect()
    }
    fn exists(&self, key: &str) -> Result<bool>;
    /// Byte size of an object, or None if absent — used for legacy mutable
    /// event compatibility and other metadata-only checks.
    fn size(&self, key: &str) -> Result<Option<u64>>;
    /// All keys under `prefix` (recursive), relative to the bucket root.
    fn list(&self, prefix: &str) -> Result<Vec<String>>;
    /// Keys under `prefix` whose full key sorts after `offset`. Cloud stores
    /// push the cursor into their paginated LIST; local/test stores filter the
    /// already-sorted result.
    fn list_after(&self, prefix: &str, offset: &str) -> Result<Vec<String>> {
        let mut keys = self.list(prefix)?;
        keys.retain(|key| key.as_str() > offset);
        keys.sort();
        Ok(keys)
    }
    /// Create the object only if absent; Ok(true) = we created it, Ok(false) =
    /// someone else holds it. The atomic primitive under leases and write-once
    /// stores (local: hard-link-if-absent; cloud: conditional PUT).
    fn put_if_absent(&self, key: &str, bytes: &[u8]) -> Result<bool>;
    /// Write a batch with the same write-once contract. Returns how many this
    /// caller created; racing writers of identical content are harmless.
    fn put_many_if_absent(&self, objects: &[(String, Vec<u8>)]) -> Result<usize> {
        let mut created = 0;
        for (key, bytes) in objects {
            created += usize::from(self.put_if_absent(key, bytes)?);
        }
        Ok(created)
    }
    /// Remove an object; absent is not an error.
    fn delete(&self, key: &str) -> Result<()>;
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
    /// A process-unique temp sibling for `p` — a fixed name would let two
    /// concurrent writers of the same key rename each other's half-written tmp
    /// into place.
    fn tmp_for(p: &Path) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let name = p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        p.with_file_name(format!("{name}.part.{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)))
    }
}

/// Tmp-file marker — `list` skips these so a killed writer's leftovers never
/// read as real objects.
fn is_part(name: &str) -> bool {
    name.ends_with(".part") || name.contains(".part.")
}

impl Bucket for LocalFs {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        let p = self.path(key);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Write to a temp sibling then rename, so a reader never sees a partial
        // object (atomic on the same filesystem).
        let tmp = Self::tmp_for(&p);
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
            if !e.file_type().is_file() || is_part(&e.file_name().to_string_lossy()) {
                continue;
            }
            if let Ok(rel) = e.path().strip_prefix(&self.root) {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
        out.sort();
        Ok(out)
    }

    fn put_if_absent(&self, key: &str, bytes: &[u8]) -> Result<bool> {
        let p = self.path(key);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // hard_link is atomic fail-if-exists on APFS/ext4 — the local stand-in
        // for a conditional PUT.
        let tmp = Self::tmp_for(&p);
        std::fs::write(&tmp, bytes)?;
        let created = match std::fs::hard_link(&tmp, &p) {
            Ok(()) => true,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => false,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                return Err(e.into());
            }
        };
        let _ = std::fs::remove_file(&tmp);
        Ok(created)
    }

    fn delete(&self, key: &str) -> Result<()> {
        match std::fs::remove_file(self.path(key)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
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
        assert_eq!(
            b.get_many(&["events/a/y.jsonl".into(), "missing".into(), "events/a/x.jsonl".into()])
                .unwrap(),
            vec![Some(b"yo".to_vec()), None, Some(b"hi".to_vec())],
            "batched reads preserve requested order and missing entries"
        );
        let keys = b.list("events").unwrap();
        assert_eq!(keys, vec!["events/a/x.jsonl", "events/a/y.jsonl"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // The backend contract the read-model + lease rely on: overwrite, size,
    // delete-missing, and that half-written `.part` temp files never read as
    // real objects (a killed writer must not corrupt a `list`).
    #[test]
    fn local_fs_backend_contract() {
        let dir = std::env::temp_dir().join(format!("synty-bucket-contract-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let b = LocalFs::new(&dir);

        // put overwrites; size tracks the latest bytes.
        b.put("k", b"v1").unwrap();
        assert_eq!(b.size("k").unwrap(), Some(2));
        b.put("k", b"v-longer").unwrap();
        assert_eq!(b.get("k").unwrap().as_deref(), Some(&b"v-longer"[..]), "put overwrites");
        assert_eq!(b.size("k").unwrap(), Some(8), "size reflects the overwrite");
        assert_eq!(b.size("absent").unwrap(), None);

        // delete is idempotent; a missing key is not an error.
        b.delete("k").unwrap();
        assert!(!b.exists("k").unwrap());
        b.delete("k").unwrap();

        // put_if_absent creates once, then refuses.
        assert!(b.put_if_absent("once", b"a").unwrap());
        assert!(!b.put_if_absent("once", b"b").unwrap(), "second creation refused");
        assert_eq!(b.get("once").unwrap().as_deref(), Some(&b"a"[..]), "first writer's bytes win");
        assert_eq!(
            b.put_many_if_absent(&[("once".into(), b"b".to_vec()), ("twice".into(), b"c".to_vec())])
                .unwrap(),
            1,
            "a batch creates only absent objects"
        );
        assert_eq!(b.get("twice").unwrap().as_deref(), Some(&b"c"[..]));

        // A leftover `.part` temp file is invisible to list (and get).
        std::fs::create_dir_all(dir.join("blobs")).unwrap();
        std::fs::write(dir.join("blobs/real"), b"x").unwrap();
        std::fs::write(dir.join("blobs/real.part.123-0"), b"half").unwrap();
        assert_eq!(b.list("blobs").unwrap(), vec!["blobs/real"], "part files never read as objects");
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
