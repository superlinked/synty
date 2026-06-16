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

/// One `[metrics sync] phase=… objects=… bytes=…` line, emitted only when a
/// transfer actually moved bytes — so an idle poll (the common case) stays
/// quiet. The bytes figure is what tells a team whether sync is practical on
/// their link; redirect stderr (`2>> sync.log`) to keep a history.
fn sync_metric(phase: &str, objects: usize, bytes: u64) {
    if objects == 0 && bytes == 0 {
        return;
    }
    crate::metrics::Run::new("sync").set("phase", phase).set("objects", objects).set("bytes", bytes).emit();
}

/// Per-build copy of the bucket manifest (filename → blob), written into the
/// build dir so a later `pull_if_stale` can map a wanted blob to a local file
/// and reuse it by hardlink instead of re-downloading. Skipped by `publish`.
const LOCAL_MANIFEST: &str = ".blobs.json";

fn write_local_manifest(dir: &Path, files: &BTreeMap<String, String>) {
    if let Ok(raw) = serde_json::to_vec(files) {
        let _ = std::fs::write(dir.join(LOCAL_MANIFEST), raw);
    }
}

/// {blob → a local file that already holds it}, gathered from the `.blobs.json`
/// of every build dir on disk — so the unchanged bulk of a new build (the
/// 500 MB residual segments) is hardlinked in, not pulled over the wire.
fn local_blob_index_in(builds_root: &Path) -> BTreeMap<String, std::path::PathBuf> {
    let mut map = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(builds_root) else { return map };
    for e in entries.filter_map(|e| e.ok()) {
        let dir = e.path();
        let Ok(raw) = std::fs::read(dir.join(LOCAL_MANIFEST)) else { continue };
        let Ok(files) = serde_json::from_slice::<BTreeMap<String, String>>(&raw) else { continue };
        for (name, blob) in files {
            let p = dir.join(&name);
            if p.exists() {
                map.entry(blob).or_insert(p);
            }
        }
    }
    map
}

/// Materialize a build's files into `dir`: hardlink-reuse blobs already on disk
/// (the unchanged bulk of an incremental build, found among sibling build dirs),
/// fetch only the rest from the bucket, and record the name→blob map locally so
/// the next pull can reuse these too. Returns (reused, fetched, bytes_down).
/// The blob-transfer core of `pull_if_stale`, factored out so the delta logic is
/// testable without a pointer, an index to verify, or a chdir.
fn fetch_build(b: &dyn bucket::Bucket, dir: &Path, files: &BTreeMap<String, String>) -> Result<(usize, usize, u64)> {
    std::fs::create_dir_all(dir)?;
    let have = local_blob_index_in(dir.parent().unwrap_or(dir));
    let (mut reused, mut fetched, mut bytes_down) = (0usize, 0usize, 0u64);
    for (name, blob) in files {
        let dest = dir.join(name);
        if dest.exists() {
            continue; // build files are immutable; present = complete (atomic writes)
        }
        // Reuse a blob already on disk — the unchanged bulk of an incremental
        // build — by hardlink (instant, no network, shared storage), copy as a
        // cross-filesystem fallback; only genuinely-new blobs cross the wire.
        if let Some(src) = have.get(blob) {
            if std::fs::hard_link(src, &dest).is_ok() || std::fs::copy(src, &dest).is_ok() {
                reused += 1;
                continue;
            }
        }
        let bytes = b.get(&format!("blobs/{blob}"))?.ok_or_else(|| anyhow!("missing blob {blob} for {name}"))?;
        bytes_down += bytes.len() as u64;
        fetched += 1;
        crate::write_atomic(&dest.to_string_lossy(), &bytes)?;
    }
    write_local_manifest(dir, files); // so the next pull can reuse these blobs
    Ok((reused, fetched, bytes_down))
}

/// Upload this device's event files to the bucket under `events/`, skipping any
/// whose bucket copy is already the same size (append-only files only grow).
/// With many trackers pointed at one bucket, each device's events converge here
/// under its own stream path, ready for a single build to process them all.
pub fn push_events(bucket_uri: &str, local_dir: &str) -> Result<usize> {
    let b = bucket::open(bucket_uri)?;
    let (mut n, mut bytes_up) = (0, 0u64);
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
        bytes_up += bytes.len() as u64;
        n += 1;
    }
    sync_metric("events_up", n, bytes_up);
    Ok(n)
}

/// Download all devices' event files from the bucket into `local_dir`, skipping
/// any already present at the same size. This is how a build sees every
/// device's sessions, not just the local machine's.
pub fn pull_events(bucket_uri: &str, local_dir: &str) -> Result<usize> {
    let b = bucket::open(bucket_uri)?;
    let (mut n, mut bytes_down) = (0, 0u64);
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
        bytes_down += bytes.len() as u64;
        std::fs::write(&dest, bytes)?;
        n += 1;
    }
    sync_metric("events_down", n, bytes_down);
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

    let (mut n, mut bytes_up) = (0, 0u64);
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
            if name == LOCAL_MANIFEST || (name.starts_with("unit_clusters.") && name != cluster_name) {
                continue; // the local blob index, and other revs' clusters
            }
            let bytes = std::fs::read(entry.path())?;
            let blob = format!("{:016x}", crate::index::fnv1a(&bytes));
            let bkey = format!("blobs/{blob}");
            if !b.exists(&bkey)? {
                b.put(&bkey, &bytes)?;
                bytes_up += bytes.len() as u64;
                n += 1;
            }
            files.insert(name, blob);
        }
        // Persist the name→blob map locally too, so a later `pull_if_stale` can
        // reuse these blobs by hardlink instead of re-downloading them.
        write_local_manifest(&cur.dir(), &files);
        b.put(&mkey, &serde_json::to_vec(&BuildManifest { files })?)?;
        n += 1;
    }
    b.put(POINTER_KEY, &serde_json::to_vec(&cur)?)?;
    sync_metric("publish_up", n + 1, bytes_up);
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
    // Transfer the build — hardlink-reuse on-disk blobs, fetch only the changed
    // ones (and record the local manifest for the next pull).
    let (reused, fetched, bytes_down) = fetch_build(&*b, &dir, &manifest.files)?;
    // Verify before repointing — a bad pull must never become current.
    next_plaid::MmapIndex::load(&dir.to_string_lossy())
        .map_err(|e| anyhow!("pulled build does not load: {e}"))?;
    let keep: Vec<String> = local.iter().map(|c| c.build.clone()).collect();
    readmodel::repoint(&remote.build, remote.rev)?;
    readmodel::gc(&keep);
    crate::metrics::Run::new("sync")
        .set("phase", "pull_down")
        .set("blobs_fetched", fetched)
        .set("blobs_reused", reused)
        .set("bytes", bytes_down)
        .emit();
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bucket::Bucket; // trait methods (put/get) on LocalFs in tests

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

    // The delta-pull's reuse source: every build dir's .blobs.json contributes
    // its files to a {blob → on-disk path} map, so an unchanged blob (the same
    // hash across builds) is found locally and never re-downloaded.
    #[test]
    fn local_blob_index_maps_blobs_to_on_disk_files() {
        let root = std::env::temp_dir().join(format!("synty-blobidx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // Build A: residuals (blob h1) + metadata (blob h2).
        let a = root.join("buildA");
        std::fs::create_dir_all(&a).unwrap();
        std::fs::write(a.join("residuals.npy"), b"big-vectors").unwrap();
        std::fs::write(a.join("metadata.db"), b"meta-1").unwrap();
        let files_a = BTreeMap::from([
            ("residuals.npy".to_string(), "h1".to_string()),
            ("metadata.db".to_string(), "h2".to_string()),
        ]);
        write_local_manifest(&a, &files_a);
        // Build B reused the residuals (h1) and changed metadata (h3) — but B's
        // dir need not exist for the index; A alone provides h1's location.
        let idx = local_blob_index_in(&root);
        assert_eq!(idx.get("h1"), Some(&a.join("residuals.npy")), "unchanged residual found locally");
        assert_eq!(idx.get("h2"), Some(&a.join("metadata.db")));
        assert_eq!(idx.get("h3"), None, "a never-seen blob is not on disk → must be fetched");
        let _ = std::fs::remove_dir_all(&root);
    }

    // A blob whose backing file was deleted (gc) must not be offered for reuse,
    // or the pull would hardlink a phantom.
    #[test]
    fn local_blob_index_skips_missing_files() {
        let root = std::env::temp_dir().join(format!("synty-blobidx2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let a = root.join("buildA");
        std::fs::create_dir_all(&a).unwrap();
        // Manifest claims a file that isn't actually on disk.
        write_local_manifest(&a, &BTreeMap::from([("gone.npy".to_string(), "h9".to_string())]));
        assert_eq!(local_blob_index_in(&root).get("h9"), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    // Scenario: a teammate's incremental update propagates as a DELTA. A new
    // machine joins cold (fetches the whole build), then a second build that
    // shares the heavy residual blob and changes only metadata reuses the
    // unchanged blob by hardlink and pulls only the changed one over the wire.
    #[test]
    fn fetch_build_cold_then_delta_reuses_unchanged_blobs() {
        let root = std::env::temp_dir().join(format!("synty-fetchbuild-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let b = crate::bucket::LocalFs::new(root.join("bucket"));
        let builds = root.join("builds");
        // Content-addressed blobs in the bucket (keyed by fnv1a, like publish).
        let put = |bytes: &[u8]| -> String {
            let h = format!("{:016x}", crate::index::fnv1a(bytes));
            b.put(&format!("blobs/{h}"), bytes).unwrap();
            h
        };
        let residual = put(b"the-heavy-residual-vectors"); // unchanged across builds
        let meta_v1 = put(b"metadata-v1");

        // Cold join: build A, nothing on disk → fetch everything.
        let files_a =
            BTreeMap::from([("residuals.npy".into(), residual.clone()), ("metadata.db".into(), meta_v1)]);
        let (reused, fetched, _) = fetch_build(&b, &builds.join("buildA"), &files_a).unwrap();
        assert_eq!((reused, fetched), (0, 2), "cold join fetches the whole build");
        assert!(builds.join("buildA").join(LOCAL_MANIFEST).exists(), "records blobs for the next pull");

        // Delta: build B keeps the residual, changes only the metadata.
        let meta_v2 = put(b"metadata-v2-after-an-incremental-build");
        let files_b = BTreeMap::from([
            ("residuals.npy".into(), residual),
            ("metadata.db".into(), meta_v2),
        ]);
        let (reused, fetched, bytes) = fetch_build(&b, &builds.join("buildB"), &files_b).unwrap();
        assert_eq!((reused, fetched), (1, 1), "the heavy blob is reused; only the changed one is fetched");
        assert_eq!(bytes, b"metadata-v2-after-an-incremental-build".len() as u64, "only the delta crossed the wire");
        // And the reused file is byte-identical (hardlinked from build A).
        assert_eq!(std::fs::read(builds.join("buildB").join("residuals.npy")).unwrap(), b"the-heavy-residual-vectors");
        let _ = std::fs::remove_dir_all(&root);
    }
}
