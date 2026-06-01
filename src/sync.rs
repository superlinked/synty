// Publish the query read-model to a bucket and pull it back, so one build
// serves many readers: a worker (or whoever runs `index`) uploads the index
// under `index/` plus `docs.jsonl` (rendering needs the text), and any client
// downloads them and queries locally. The per-doc embeddings (large,
// build-side, already in the content-addressed store) are not published.
// `doc_hashes.json` doubles as the staleness manifest — a client whose copy
// matches the bucket's is current.

use crate::bucket;
use anyhow::Result;
use std::path::Path;

const PREFIX: &str = "index";
const MANIFEST: &str = "doc_hashes.json";
const DOCS_KEY: &str = "docs.jsonl";
const SKIP: &[&str] = &["embeddings.npy", "embeddings_lengths.json"];

/// Upload the index (minus build-side embeddings) and the docs to the bucket,
/// unless the bucket already holds this exact build (same manifest).
pub fn publish(bucket_uri: &str, index_dir: &str, docs_path: &str) -> Result<usize> {
    let b = bucket::open(bucket_uri)?;
    let local_manifest = std::fs::read(Path::new(index_dir).join(MANIFEST)).ok();
    if let Some(lm) = &local_manifest {
        if b.get(&format!("{PREFIX}/{MANIFEST}"))?.as_deref() == Some(lm.as_slice()) {
            return Ok(0); // bucket already current
        }
    }
    let mut n = 0;
    for entry in walkdir::WalkDir::new(index_dir).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy();
        if SKIP.iter().any(|s| *s == name) {
            continue;
        }
        let rel = entry.path().strip_prefix(index_dir)?.to_string_lossy().replace('\\', "/");
        b.put(&format!("{PREFIX}/{rel}"), &std::fs::read(entry.path())?)?;
        n += 1;
    }
    if let Ok(docs) = std::fs::read(docs_path) {
        b.put(DOCS_KEY, &docs)?;
        n += 1;
    }
    Ok(n)
}

/// Download the read-model into place when the local copy is missing or its
/// manifest differs from the bucket's. Returns whether it pulled.
pub fn pull_if_stale(bucket_uri: &str, index_dir: &str, docs_path: &str) -> Result<bool> {
    let b = bucket::open(bucket_uri)?;
    let Some(remote_manifest) = b.get(&format!("{PREFIX}/{MANIFEST}"))? else {
        return Ok(false); // nothing published yet
    };
    let local_manifest = std::fs::read(Path::new(index_dir).join(MANIFEST)).ok();
    if local_manifest.as_deref() == Some(remote_manifest.as_slice()) {
        return Ok(false); // already current
    }
    for key in b.list(PREFIX)? {
        let Some(bytes) = b.get(&key)? else { continue };
        let rel = key.strip_prefix(&format!("{PREFIX}/")).unwrap_or(&key);
        let dest = Path::new(index_dir).join(rel);
        if let Some(p) = dest.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::write(dest, bytes)?;
    }
    if let Some(docs) = b.get(DOCS_KEY)? {
        if let Some(p) = Path::new(docs_path).parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::write(docs_path, docs)?;
    }
    Ok(true)
}
