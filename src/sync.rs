// Publish the query read-model to a bucket and pull it back, so one build
// serves many readers. Builds are immutable: every file in a build directory
// uploads once as a content-addressed blob (`blobs/<fnv>` — consecutive builds
// share unchanged files, so an incremental append publishes only new chunks),
// a per-(build,rev) manifest (`builds/<build>.<rev>.json`) maps filenames to
// blobs, and a tiny `current.json` pointer is PUT LAST — a reader can never
// see a torn build, and concurrent publishers only race the pointer, each
// pointing at a complete build. The per-doc embeddings (large, build-side,
// already in the content-addressed store) are not published.

use crate::{bucket, readmodel};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

const POINTER_KEY: &str = "current.json";
const EVENTS: &str = "events";

/// Filename → blob hash for one (build, rev).
#[derive(Serialize, Deserialize)]
struct BuildManifest {
    files: BTreeMap<String, String>,
}

fn manifest_key(build: &str, rev: u64) -> String {
    format!("builds/{build}.{rev}.json")
}

/// Upload this device's event files to the bucket under `events/`, skipping any
/// whose bucket copy is already the same size (append-only files only grow).
/// With many trackers pointed at one bucket, each device's events converge here
/// under its own stream path, ready for a single build to process them all.
pub fn push_events(bucket_uri: &str, local_dir: &str) -> Result<usize> {
    let b = bucket::open(bucket_uri)?;
    let mut n = 0;
    for entry in walkdir::WalkDir::new(local_dir).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file()
            || entry.path().extension().and_then(|s| s.to_str()) != Some("jsonl")
        {
            continue;
        }
        let rel = entry.path().strip_prefix(local_dir)?.to_string_lossy().replace('\\', "/");
        let key = format!("{EVENTS}/{rel}");
        let bytes = std::fs::read(entry.path())?;
        if b.size(&key)? == Some(bytes.len() as u64) {
            continue;
        }
        b.put(&key, &bytes)?;
        n += 1;
    }
    Ok(n)
}

/// Download all devices' event files from the bucket into `local_dir`, skipping
/// any already present at the same size. This is how a build sees every
/// device's sessions, not just the local machine's.
pub fn pull_events(bucket_uri: &str, local_dir: &str) -> Result<usize> {
    let b = bucket::open(bucket_uri)?;
    let mut n = 0;
    for key in b.list(EVENTS)? {
        let rel = key.strip_prefix(&format!("{EVENTS}/")).unwrap_or(&key);
        let dest = Path::new(local_dir).join(rel);
        let local_size = std::fs::metadata(&dest).ok().map(|m| m.len());
        if local_size.is_some() && local_size == b.size(&key)? {
            continue;
        }
        let Some(bytes) = b.get(&key)? else { continue };
        if let Some(p) = dest.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::write(&dest, bytes)?;
        n += 1;
    }
    Ok(n)
}

/// The GitHub corpus manifest: per-file content hashes plus when the last
/// scrape ran. Written by `synty github`, shared through the bucket so one
/// tokened machine scrapes for the fleet — and so a builder that never
/// scraped can't publish a read-model with the org's PRs missing.
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct GithubManifest {
    pub scraped_at: String,
    pub files: BTreeMap<String, String>, // filename → fnv64 of contents
}

pub const GH_MANIFEST: &str = ".manifest.json";
const GITHUB_PREFIX: &str = "github";

pub fn load_github_manifest(dir: &str) -> GithubManifest {
    std::fs::read_to_string(Path::new(dir).join(GH_MANIFEST))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Push the local GitHub corpus (files whose hash the bucket manifest doesn't
/// have) and then the manifest. Returns objects uploaded.
pub fn push_github(bucket_uri: &str, dir: &str) -> Result<usize> {
    let local = load_github_manifest(dir);
    if local.files.is_empty() {
        return Ok(0); // never scraped here
    }
    let b = bucket::open(bucket_uri)?;
    let remote: GithubManifest = b
        .get(&format!("{GITHUB_PREFIX}/{GH_MANIFEST}"))?
        .and_then(|raw| serde_json::from_slice(&raw).ok())
        .unwrap_or_default();
    if remote.scraped_at >= local.scraped_at {
        return Ok(0); // bucket is as fresh or fresher
    }
    let mut n = 0;
    for (name, hash) in &local.files {
        if remote.files.get(name) != Some(hash) {
            b.put(&format!("{GITHUB_PREFIX}/{name}"), &std::fs::read(Path::new(dir).join(name))?)?;
            n += 1;
        }
    }
    b.put(&format!("{GITHUB_PREFIX}/{GH_MANIFEST}"), &serde_json::to_vec(&local)?)?;
    Ok(n + 1)
}

/// Pull the fleet's GitHub corpus when the bucket's manifest is fresher:
/// changed files are fetched, files dropped from the scrape set are removed,
/// and the local manifest adopts the remote one. Returns files changed.
pub fn pull_github(bucket_uri: &str, dir: &str) -> Result<usize> {
    let b = bucket::open(bucket_uri)?;
    let Some(remote) = b
        .get(&format!("{GITHUB_PREFIX}/{GH_MANIFEST}"))?
        .and_then(|raw| serde_json::from_slice::<GithubManifest>(&raw).ok())
    else {
        return Ok(0); // nothing published yet
    };
    let local = load_github_manifest(dir);
    if local.scraped_at >= remote.scraped_at {
        return Ok(0);
    }
    std::fs::create_dir_all(dir)?;
    let mut n = 0;
    for (name, hash) in &remote.files {
        if local.files.get(name) == Some(hash) {
            continue;
        }
        let Some(bytes) = b.get(&format!("{GITHUB_PREFIX}/{name}"))? else { continue };
        crate::write_atomic(&Path::new(dir).join(name).to_string_lossy(), &bytes)?;
        n += 1;
    }
    for name in local.files.keys() {
        if !remote.files.contains_key(name) {
            let _ = std::fs::remove_file(Path::new(dir).join(name));
            n += 1;
        }
    }
    crate::write_atomic(
        &Path::new(dir).join(GH_MANIFEST).to_string_lossy(),
        &serde_json::to_vec(&remote)?,
    )?;
    Ok(n)
}

/// Upload the current build: blobs for files the bucket doesn't have yet, the
/// (build, rev) manifest, then the pointer — strictly last, so readers either
/// see the previous complete build or this complete build, never a mix.
/// Returns the number of objects uploaded (0 = bucket already current).
pub fn publish(bucket_uri: &str) -> Result<usize> {
    let Some(cur) = readmodel::current() else { return Ok(0) };
    let b = bucket::open(bucket_uri)?;
    if b.get(POINTER_KEY)?.and_then(|raw| serde_json::from_slice::<readmodel::Current>(&raw).ok())
        == Some(cur.clone())
    {
        return Ok(0);
    }

    let mut n = 0;
    let mkey = manifest_key(&cur.build, cur.rev);
    if !b.exists(&mkey)? {
        // The manifest covers exactly the files this (build, rev) needs: the
        // index, the docs snapshot, and the rev-selected clusters file.
        let mut files = BTreeMap::new();
        let cluster_name = format!("unit_clusters.{}.json", cur.rev);
        for entry in walkdir::WalkDir::new(cur.dir()).into_iter().filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("unit_clusters.") && name != cluster_name {
                continue; // other revs' clusters
            }
            let bytes = std::fs::read(entry.path())?;
            let blob = format!("{:016x}", crate::index::fnv1a(&bytes));
            let bkey = format!("blobs/{blob}");
            if !b.exists(&bkey)? {
                b.put(&bkey, &bytes)?;
                n += 1;
            }
            files.insert(name, blob);
        }
        b.put(&mkey, &serde_json::to_vec(&BuildManifest { files })?)?;
        n += 1;
    }
    b.put(POINTER_KEY, &serde_json::to_vec(&cur)?)?;
    Ok(n + 1)
}

/// Download the read-model when the bucket's pointer differs from the local
/// one: blobs land in a fresh build dir (atomic per file), the new build is
/// verified loadable, and only then does the local pointer move. Returns
/// whether it pulled.
pub fn pull_if_stale(bucket_uri: &str) -> Result<bool> {
    let b = bucket::open(bucket_uri)?;
    let Some(remote) =
        b.get(POINTER_KEY)?.and_then(|raw| serde_json::from_slice::<readmodel::Current>(&raw).ok())
    else {
        return Ok(false); // nothing published yet
    };
    if remote.format > readmodel::FORMAT {
        eprintln!(
            "bucket read-model is format {} (written by synty {}) — newer than this binary understands; upgrade synty",
            remote.format, remote.writer
        );
        return Ok(false);
    }
    let local = readmodel::current();
    if local.as_ref() == Some(&remote) {
        return Ok(false);
    }

    let raw = b
        .get(&manifest_key(&remote.build, remote.rev))?
        .ok_or_else(|| anyhow!("bucket pointer names a missing manifest (publish in flight?)"))?;
    let manifest: BuildManifest = serde_json::from_slice(&raw)?;
    let dir = readmodel::build_dir(&remote.build);
    std::fs::create_dir_all(&dir)?;
    for (name, blob) in &manifest.files {
        let dest = dir.join(name);
        if dest.exists() {
            continue; // build files are immutable; present = complete (atomic writes)
        }
        let bytes = b.get(&format!("blobs/{blob}"))?.ok_or_else(|| anyhow!("missing blob {blob} for {name}"))?;
        crate::write_atomic(&dest.to_string_lossy(), &bytes)?;
    }
    // Verify before repointing — a bad pull must never become current.
    next_plaid::MmapIndex::load(&dir.to_string_lossy())
        .map_err(|e| anyhow!("pulled build does not load: {e}"))?;
    let keep: Vec<String> = local.iter().map(|c| c.build.clone()).collect();
    readmodel::repoint(&remote.build, remote.rev)?;
    readmodel::gc(&keep);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The GitHub corpus travels through the bucket by manifest: a scraping
    // machine pushes changed files; a token-less machine pulls them (and drops
    // files for repos that left the scrape set) — so any builder publishes
    // with the org's PRs present.
    #[test]
    fn github_corpus_roundtrips_through_the_bucket() {
        let root = std::env::temp_dir().join(format!("synty-ghsync-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (a, b, bucket) = (root.join("a"), root.join("b"), root.join("bucket"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        let write_manifest = |dir: &std::path::Path, m: &GithubManifest| {
            std::fs::write(dir.join(GH_MANIFEST), serde_json::to_vec(m).unwrap()).unwrap();
        };

        // Machine A scraped two repos.
        std::fs::write(a.join("prs-web.json"), b"[\"web\"]").unwrap();
        std::fs::write(a.join("prs-api.json"), b"[\"api\"]").unwrap();
        let mut m = GithubManifest { scraped_at: "2026-06-11T10:00:00Z".into(), ..Default::default() };
        for f in ["prs-web.json", "prs-api.json"] {
            let h = crate::index::fnv1a(&std::fs::read(a.join(f)).unwrap());
            m.files.insert(f.into(), format!("{h:016x}"));
        }
        write_manifest(&a, &m);
        let bu = bucket.to_str().unwrap();
        assert!(push_github(bu, a.to_str().unwrap()).unwrap() > 0);
        assert_eq!(push_github(bu, a.to_str().unwrap()).unwrap(), 0, "re-push is a no-op");

        // Machine B (no token, never scraped) pulls the corpus.
        assert!(pull_github(bu, b.to_str().unwrap()).unwrap() > 0);
        assert_eq!(std::fs::read(b.join("prs-web.json")).unwrap(), b"[\"web\"]");
        assert_eq!(pull_github(bu, b.to_str().unwrap()).unwrap(), 0, "re-pull is a no-op");

        // A re-scrapes later: api dropped from the set, web changed.
        std::fs::write(a.join("prs-web.json"), b"[\"web2\"]").unwrap();
        let mut m2 = GithubManifest { scraped_at: "2026-06-11T11:00:00Z".into(), ..Default::default() };
        let h = crate::index::fnv1a(&std::fs::read(a.join("prs-web.json")).unwrap());
        m2.files.insert("prs-web.json".into(), format!("{h:016x}"));
        write_manifest(&a, &m2);
        push_github(bu, a.to_str().unwrap()).unwrap();
        pull_github(bu, b.to_str().unwrap()).unwrap();
        assert_eq!(std::fs::read(b.join("prs-web.json")).unwrap(), b"[\"web2\"]");
        assert!(!b.join("prs-api.json").exists(), "dropped repo removed on pull");
        let _ = std::fs::remove_dir_all(&root);
    }
}
