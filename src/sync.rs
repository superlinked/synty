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
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const POINTER_KEY: &str = "current.json";
const EVENTS: &str = "events";
const EVENT_STREAMS: &str = "event-streams";
const EVENT_CHUNK_BYTES: usize = 1 << 20;

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

/// Per-bucket byte offsets already converted from local append-only tracker
/// files into immutable cloud chunks. Bucket-scoping matters: switching a
/// machine to a new team bucket must replay its eligible local events there.
#[derive(Serialize, Deserialize, Default)]
struct UploadState {
    #[serde(default)]
    buckets: BTreeMap<String, BTreeMap<String, u64>>,
    /// Pre-boundary session_start lines staged until that session produces an
    /// eligible event. This preserves actor/repo metadata for a session that
    /// straddles the boundary without publishing starts for old-only sessions.
    #[serde(default)]
    pending_starts: BTreeMap<String, BTreeMap<String, String>>,
    /// Stream markers already created in each bucket. Readers list this bounded
    /// registry, then continue each stream from a persisted key cursor.
    #[serde(default)]
    registered_streams: BTreeMap<String, BTreeSet<String>>,
    /// The irreversible transform used for each bucket's uploaded chunks.
    /// Changing it after advancing offsets would silently mix policies while
    /// making the original local bytes impossible to replay.
    #[serde(default)]
    redaction_profiles: BTreeMap<String, String>,
    /// Repository policy and the session ids it admitted. Persisting this
    /// avoids rescanning the full append-only stream on every upload poll.
    #[serde(default)]
    capture_repo_policies: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    allowed_sessions: BTreeMap<String, BTreeMap<String, BTreeSet<String>>>,
}

/// Default event upload path used by builds outside the long-lived tracker.
/// Only stream directories owned by this resolved machine id are eligible;
/// pulled fleet streams below the same local corpus remain read-only.
pub fn push_events(bucket_uri: &str, local_dir: &str, machine: &str) -> Result<usize> {
    push_events_for_machine(
        bucket_uri,
        local_dir,
        ".synty/uploads.json",
        crate::config::capture_since_ms(),
        machine,
    )
}

fn push_events_for_machine(
    bucket_uri: &str,
    local_dir: &str,
    state_path: &str,
    capture_since_ms: Option<i64>,
    machine: &str,
) -> Result<usize> {
    let owned: BTreeSet<String> = std::fs::read_dir(local_dir)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| {
            crate::identity::stream_parts(name).is_some_and(|(owner, _)| owner == machine)
        })
        .collect();
    push_events_for_streams(
        bucket_uri,
        local_dir,
        state_path,
        capture_since_ms,
        &owned,
    )
}

/// Convert only the new complete bytes of local per-day tracker files into
/// deterministic, immutable objects. A failed/retried pass addresses the same
/// keys and put_if_absent makes it idempotent; no historical object is HEADed
/// on every poll, and the growing current-day file is never re-uploaded whole.
/// The absolute capture bound is applied here as a second privacy gate, so
/// joining a bucket does not silently publish older local history.
#[cfg(test)]
pub(crate) fn push_events_with(
    bucket_uri: &str,
    local_dir: &str,
    state_path: &str,
    capture_since_ms: Option<i64>,
) -> Result<usize> {
    push_events_scoped(bucket_uri, local_dir, state_path, capture_since_ms, None)
}

/// Production upload path with the tracker-owned streams made explicit.
pub fn push_events_for_streams(
    bucket_uri: &str,
    local_dir: &str,
    state_path: &str,
    capture_since_ms: Option<i64>,
    owned_streams: &BTreeSet<String>,
) -> Result<usize> {
    push_events_scoped(
        bucket_uri,
        local_dir,
        state_path,
        capture_since_ms,
        Some(owned_streams),
    )
}

fn push_events_scoped(
    bucket_uri: &str,
    local_dir: &str,
    state_path: &str,
    capture_since_ms: Option<i64>,
    owned_streams: Option<&BTreeSet<String>>,
) -> Result<usize> {
    let b = bucket::open(bucket_uri)?;
    let mut state: UploadState = std::fs::read_to_string(state_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let profile = crate::config::upload_redaction();
    let config = crate::config::load();
    let capture_repos = config.capture_repos;
    let known_repos: std::collections::HashSet<String> = config.repos.into_iter().collect();
    let profile_name = profile.as_str();
    ensure_upload_policy_compatible(
        &state,
        bucket_uri,
        profile_name,
        &capture_repos,
    )?;
    let profile_changed = state
        .redaction_profiles
        .insert(bucket_uri.to_string(), profile_name.to_string())
        .as_deref()
        != Some(profile_name);
    let repo_policy_changed = state.capture_repo_policies.get(bucket_uri) != Some(&capture_repos);
    if repo_policy_changed {
        state.capture_repo_policies.insert(bucket_uri.to_string(), capture_repos.clone());
        state.allowed_sessions.remove(bucket_uri);
    }
    let offsets = state.buckets.entry(bucket_uri.to_string()).or_default();
    let pending_starts = state
        .pending_starts
        .entry(bucket_uri.to_string())
        .or_default();
    let registered_streams = state
        .registered_streams
        .entry(bucket_uri.to_string())
        .or_default();
    let allowed_by_stream = state.allowed_sessions.entry(bucket_uri.to_string()).or_default();
    let (mut n, mut bytes_up) = (0, 0u64);
    let mut changed = profile_changed || repo_policy_changed;
    let mut files: Vec<std::path::PathBuf> = walkdir::WalkDir::new(local_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("jsonl"))
        .filter(|path| {
            let Ok(rel) = path.strip_prefix(local_dir) else { return false };
            // Pulled immutable chunks also live below corpus/local and must
            // never be re-chunked.
            rel.components().count() == 2
                && path.file_name().is_some_and(|n| n.to_string_lossy().starts_with("track"))
        })
        .collect();
    // Chunk keys sort by stream/day/offset. Publishing files in that same order
    // makes a reader's per-stream key cursor safe even while an upload is live.
    files.sort();
    for path in files {
        let rel_path = path.strip_prefix(local_dir)?;
        let rel = rel_path.to_string_lossy().replace('\\', "/");
        let (stream, file) = rel.split_once('/').unwrap_or(("local", rel.as_str()));
        if owned_streams.is_some_and(|owned| !owned.contains(stream)) {
            continue;
        }
        if registered_streams.insert(stream.to_string()) {
            b.put_if_absent(&format!("{EVENT_STREAMS}/{stream}"), b"")?;
            changed = true;
        }
        let bytes = std::fs::read(&path)?;
        let mut start = offsets
            .get(&rel)
            .copied()
            .unwrap_or(0)
            .min(bytes.len() as u64);
        if start == 0 {
            // One-time upgrade compatibility: an older Synty may already have
            // uploaded this exact mutable daily object. Do not duplicate it as
            // chunks merely because the local upload ledger is new.
            let legacy = format!("{EVENTS}/{rel}");
            if b.size(&legacy)? == Some(bytes.len() as u64) {
                offsets.insert(rel.clone(), bytes.len() as u64);
                changed = true;
                continue;
            }
        }
        if start >= bytes.len() as u64 {
            continue;
        }
        let pending = &bytes[start as usize..];
        let Some(last_nl) = pending.iter().rposition(|b| *b == b'\n') else {
            continue;
        };
        let end = start + last_nl as u64 + 1;
        let filtered =
            filter_events_since(&pending[..=last_nl], capture_since_ms, stream, pending_starts);
        let repo_filtered = if capture_repos.is_empty() {
            filtered
        } else {
            if !allowed_by_stream.contains_key(stream) {
                let allowed = allowed_sessions_for_repos(
                    &Path::new(local_dir).join(stream),
                    &capture_repos,
                    &known_repos,
                );
                allowed_by_stream.insert(stream.to_string(), allowed);
            }
            let allowed = allowed_by_stream.get_mut(stream).expect("stream policy initialized");
            update_allowed_sessions(&filtered, &capture_repos, &known_repos, allowed);
            filter_events_for_sessions(&filtered, allowed)
        };
        let redacted = redact_event_lines(&repo_filtered, profile);
        for (part, chunk) in event_chunks(&redacted, EVENT_CHUNK_BYTES)
            .into_iter()
            .enumerate()
        {
            let hash = Sha256::digest(&chunk);
            let short_hash = hash[..8]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            let stem = file.strip_suffix(".jsonl").unwrap_or(file);
            let key = format!(
                "{EVENTS}/{stream}/chunks/{stem}/{start:016x}-{end:016x}-{part:04}-{short_hash}.jsonl"
            );
            if b.put_if_absent(&key, &chunk)? {
                bytes_up += chunk.len() as u64;
                n += 1;
            }
        }
        start = end;
        offsets.insert(rel, start);
        changed = true;
    }
    if changed {
        if let Some(parent) = Path::new(state_path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        crate::write_atomic(state_path, &serde_json::to_vec(&state)?)?;
    }
    sync_metric("event_chunks_up", n, bytes_up);
    Ok(n)
}

fn ensure_upload_policy_compatible(
    state: &UploadState,
    bucket_uri: &str,
    profile: &str,
    capture_repos: &[String],
) -> Result<()> {
    let has_uploaded = state
        .buckets
        .get(bucket_uri)
        .is_some_and(|offsets| offsets.values().any(|offset| *offset > 0));
    if !has_uploaded {
        return Ok(());
    }
    // Ledgers written before these fields existed represent the historical
    // behavior: raw uploads with no repository allowlist.
    let previous_profile = state
        .redaction_profiles
        .get(bucket_uri)
        .map(String::as_str)
        .unwrap_or("off");
    anyhow::ensure!(
        previous_profile == profile,
        "upload redaction for {bucket_uri} changed from {previous_profile} to {profile} after upload offsets advanced; use a new bucket prefix or reset the upload ledger deliberately"
    );
    let previous_repos = state
        .capture_repo_policies
        .get(bucket_uri)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    anyhow::ensure!(
        previous_repos == capture_repos,
        "upload repository policy for {bucket_uri} changed after upload offsets advanced; use a new bucket prefix or reset the upload ledger deliberately"
    );
    Ok(())
}

fn update_allowed_sessions(
    raw: &[u8],
    capture_repos: &[String],
    known_repos: &std::collections::HashSet<String>,
    allowed: &mut BTreeSet<String>,
) {
    for line in raw.split_inclusive(|byte| *byte == b'\n') {
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(line) else { continue };
        if value["kind"].as_str() != Some("session_start") {
            continue;
        }
        let Some(session) = value["session_id"].as_str().filter(|value| !value.is_empty()) else { continue };
        let raw_repo = value["payload"]["repo"]
            .as_str()
            .filter(|value| !value.is_empty())
            .or_else(|| value["payload"]["cwd"].as_str())
            .unwrap_or("");
        let repo = crate::units::resolve_repo(raw_repo, known_repos);
        if capture_repos.iter().any(|candidate| candidate == &repo) {
            allowed.insert(session.to_string());
        }
    }
}

fn allowed_sessions_for_repos(
    stream_dir: &Path,
    capture_repos: &[String],
    known_repos: &std::collections::HashSet<String>,
) -> BTreeSet<String> {
    let mut allowed = BTreeSet::new();
    for path in crate::units::jsonl_files(stream_dir) {
        let Ok(raw) = std::fs::read_to_string(path) else { continue };
        for line in raw.lines() {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else { continue };
            if value["kind"].as_str() != Some("session_start") {
                continue;
            }
            let Some(session) = value["session_id"].as_str().filter(|value| !value.is_empty()) else { continue };
            let raw_repo = value["payload"]["repo"]
                .as_str()
                .filter(|value| !value.is_empty())
                .or_else(|| value["payload"]["cwd"].as_str())
                .unwrap_or("");
            let repo = crate::units::resolve_repo(raw_repo, known_repos);
            if capture_repos.iter().any(|candidate| candidate == &repo) {
                allowed.insert(session.to_string());
            }
        }
    }
    allowed
}

/// A configured repository allowlist is an upload boundary. Unknown sessions
/// fail closed: their start metadata may be in an older file, so the complete
/// stream is scanned above before any pending bytes are accepted.
fn filter_events_for_sessions(raw: &[u8], allowed: &BTreeSet<String>) -> Vec<u8> {
    let mut out = Vec::new();
    for line in raw.split_inclusive(|byte| *byte == b'\n') {
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(line) else { continue };
        if value["session_id"]
            .as_str()
            .is_some_and(|session| allowed.contains(session))
        {
            out.extend_from_slice(line);
        }
    }
    out
}

fn redact_event_lines(raw: &[u8], profile: crate::redact::Profile) -> Vec<u8> {
    if profile == crate::redact::Profile::Off {
        return raw.to_vec();
    }
    let mut out = Vec::new();
    for line in raw.split_inclusive(|byte| *byte == b'\n') {
        if line.iter().all(|byte| byte.is_ascii_whitespace()) {
            continue;
        }
        if let Some(redacted) = crate::redact::event_line(line, profile) {
            out.extend_from_slice(&redacted);
        } else {
            let text = crate::redact::text(&String::from_utf8_lossy(line), profile);
            out.extend_from_slice(text.trim_end_matches(['\r', '\n']).as_bytes());
            out.push(b'\n');
        }
    }
    out
}

/// Retain post-boundary events plus the session_start metadata for any session
/// that eventually crosses the boundary, staging starts across upload batches.
/// Unknown/future envelope lines are retained: a privacy cutoff may discard
/// something proven old, never data it cannot understand.
fn filter_events_since(
    raw: &[u8],
    capture_since_ms: Option<i64>,
    stream: &str,
    pending_starts: &mut BTreeMap<String, String>,
) -> Vec<u8> {
    let Some(cutoff) = capture_since_ms else {
        return raw.to_vec();
    };
    let rows: Vec<(&[u8], Option<serde_json::Value>)> = raw
        .split_inclusive(|b| *b == b'\n')
        .filter(|line| !line.iter().all(|b| b.is_ascii_whitespace()))
        .map(|line| (line, serde_json::from_slice(line).ok()))
        .collect();
    let active: BTreeSet<String> = rows
        .iter()
        .filter_map(|(_, v)| {
            let v = v.as_ref()?;
            (event_time_ms(v)? >= cutoff)
                .then(|| v["session_id"].as_str().map(ToString::to_string))
                .flatten()
        })
        .collect();
    for (line, value) in &rows {
        let Some(v) = value else { continue };
        let Some(sid) = v["session_id"].as_str().filter(|s| !s.is_empty()) else {
            continue;
        };
        if v["kind"].as_str() == Some("session_start")
            && event_time_ms(v).is_some_and(|ts| ts < cutoff)
        {
            pending_starts.insert(
                format!("{stream}\0{sid}"),
                String::from_utf8_lossy(line).into_owned(),
            );
        }
        if v["kind"].as_str() == Some("session_end") && !active.contains(sid) {
            pending_starts.remove(&format!("{stream}\0{sid}"));
        }
    }
    let mut out = Vec::new();
    for sid in &active {
        if let Some(start) = pending_starts.remove(&format!("{stream}\0{sid}")) {
            out.extend_from_slice(start.as_bytes());
        }
    }
    for (line, value) in rows {
        let keep = match value.as_ref() {
            None => true,
            Some(v) => match event_time_ms(v) {
                None => true,
                Some(ts) if ts >= cutoff => true,
                Some(_) => false,
            },
        };
        if keep {
            out.extend_from_slice(line);
        }
    }
    out
}

fn event_time_ms(v: &serde_json::Value) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(v["ts"].as_str()?)
        .ok()
        .map(|t| t.timestamp_millis())
}

/// Split at line boundaries, keeping an over-size single envelope intact.
fn event_chunks(raw: &[u8], target: usize) -> Vec<Vec<u8>> {
    if raw.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut cur = Vec::new();
    for line in raw.split_inclusive(|b| *b == b'\n') {
        if !cur.is_empty() && cur.len() + line.len() > target {
            out.push(std::mem::take(&mut cur));
        }
        cur.extend_from_slice(line);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Last immutable chunk key applied per (bucket, stream). The file lives below
/// `local_dir`, so removing a local corpus also resets its cursors and forces a
/// complete reconstruction.
#[derive(Serialize, Deserialize, Default)]
struct DownloadState {
    #[serde(default)]
    buckets: BTreeMap<String, BTreeMap<String, String>>,
}

/// Download all devices' event files from the bucket into `local_dir`. Writers
/// publish one immutable marker per stream; readers list that bounded registry
/// and ask each stream only for chunk keys after its persisted cursor. Legacy
/// buckets without markers retain the full-list path until a writer upgrades.
pub fn pull_events(bucket_uri: &str, local_dir: &str) -> Result<usize> {
    let b = bucket::open(bucket_uri)?;
    let state_path = Path::new(local_dir).join(".downloads.json");
    pull_events_from(b.as_ref(), bucket_uri, local_dir, &state_path)
}

fn pull_events_from(
    b: &dyn bucket::Bucket,
    bucket_uri: &str,
    local_dir: &str,
    state_path: &Path,
) -> Result<usize> {
    let mut state: DownloadState = std::fs::read_to_string(&state_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let cursors = state.buckets.entry(bucket_uri.to_string()).or_default();
    let (mut n, mut bytes_down) = (0, 0u64);
    let markers = b.list(EVENT_STREAMS)?;
    let mut cursor_changed = false;
    if markers.is_empty() {
        // Upgrade compatibility: old writers have no registry. Their mutable
        // files require size checks and the only safe discovery is a full list.
        for key in b.list(EVENTS)? {
            let (fetched, bytes) = pull_event_key(b, &key, local_dir)?;
            n += usize::from(fetched);
            bytes_down += bytes;
        }
    } else {
        for marker in markers {
            let Some(stream) = marker.strip_prefix(&format!("{EVENT_STREAMS}/")) else {
                continue;
            };
            if stream.is_empty() || stream.contains('/') {
                continue;
            }
            let chunk_prefix = format!("{EVENTS}/{stream}/chunks/");
            let mut cursor = cursors.get(stream).cloned().unwrap_or_default();
            if !cursor.is_empty() && !event_dest(local_dir, &cursor)?.is_file() {
                cursor.clear(); // local corpus was pruned; reconstruct it
            }
            for key in b.list_after(&chunk_prefix, &cursor)? {
                let (fetched, bytes) = pull_event_key(b, &key, local_dir)?;
                n += usize::from(fetched);
                bytes_down += bytes;
                cursor = key;
                cursors.insert(stream.to_string(), cursor.clone());
                cursor_changed = true;
            }
            // Old mutable daily files sit beside `chunks/`, so this bounded
            // compatibility list never walks the immutable chunk history.
            for key in b.list(&format!("{EVENTS}/{stream}/track"))? {
                let (fetched, bytes) = pull_event_key(b, &key, local_dir)?;
                n += usize::from(fetched);
                bytes_down += bytes;
            }
        }
    }
    if cursor_changed {
        if let Some(parent) = state_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        crate::write_atomic(&state_path.to_string_lossy(), &serde_json::to_vec(&state)?)?;
    }
    sync_metric("events_down", n, bytes_down);
    Ok(n)
}

fn event_dest(local_dir: &str, key: &str) -> Result<std::path::PathBuf> {
    let rel = key.strip_prefix(&format!("{EVENTS}/")).unwrap_or(key);
    if !Path::new(rel)
        .components()
        .all(|part| matches!(part, std::path::Component::Normal(_)))
    {
        return Err(anyhow!("unsafe event object key: {key}"));
    }
    Ok(Path::new(local_dir).join(rel))
}

/// Fetch one immutable chunk (presence is enough to skip) or legacy mutable
/// file (size comparison), always through an atomic local rename.
fn pull_event_key(b: &dyn bucket::Bucket, key: &str, local_dir: &str) -> Result<(bool, u64)> {
    let dest = event_dest(local_dir, key)?;
    let local_size = std::fs::metadata(&dest).ok().map(|m| m.len());
    if key.contains("/chunks/") && local_size.is_some() {
        return Ok((false, 0));
    }
    if local_size.is_some() && local_size == b.size(key)? {
        return Ok((false, 0));
    }
    let Some(bytes) = b.get(key)? else { return Err(anyhow!("event object disappeared during pull: {key}")) };
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let size = bytes.len() as u64;
    crate::write_atomic(&dest.to_string_lossy(), &bytes)?;
    Ok((true, size))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReaderPull {
    ReadModel,
    Events,
}

// A large raw history must never delay the small pointer + published query
// model that makes the MCP service useful. Events continue in the background.
const READER_PULL_ORDER: [ReaderPull; 2] = [ReaderPull::ReadModel, ReaderPull::Events];

/// Pull only the published query model. MCP performs this bounded startup step
/// before serving, while raw trace history syncs independently in the
/// background.
pub fn pull_read_model_for_read(bucket_uri: &str) {
    match pull_if_stale(bucket_uri) {
        Ok(true) => eprintln!("pulled published read-model from {bucket_uri}"),
        Ok(false) => {}
        Err(e) => eprintln!("read-model pull skipped ({e})"),
    }
}

/// Pull raw events needed by status/stats/trace and unpublished-delta notices.
/// Errors remain advisory so the last complete local snapshot stays usable.
pub fn pull_events_for_read(bucket_uri: &str) {
    match pull_events(bucket_uri, "corpus/local") {
        Ok(n) if n > 0 => eprintln!("pulled {n} fleet event chunks from {bucket_uri}"),
        Ok(_) => {}
        Err(e) => eprintln!("event pull skipped ({e})"),
    }
}

/// Bring a reader onto the fleet's published query model first, then its raw
/// event snapshot. Interactive MCP uses the same ordering on a background
/// thread after the initial read-model pull.
pub fn pull_for_read(bucket_uri: &str) {
    for step in READER_PULL_ORDER {
        match step {
            ReaderPull::ReadModel => pull_read_model_for_read(bucket_uri),
            ReaderPull::Events => pull_events_for_read(bucket_uri),
        }
    }
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

    #[test]
    fn published_read_model_precedes_unbounded_raw_history() {
        assert_eq!(READER_PULL_ORDER, [ReaderPull::ReadModel, ReaderPull::Events]);
    }

    struct RecordingBucket {
        inner: crate::bucket::LocalFs,
        calls: std::sync::Mutex<Vec<String>>,
    }

    impl RecordingBucket {
        fn new(root: &Path) -> Self {
            Self { inner: crate::bucket::LocalFs::new(root), calls: Default::default() }
        }
    }

    impl Bucket for RecordingBucket {
        fn put(&self, key: &str, bytes: &[u8]) -> Result<()> { self.inner.put(key, bytes) }
        fn get(&self, key: &str) -> Result<Option<Vec<u8>>> { self.inner.get(key) }
        fn exists(&self, key: &str) -> Result<bool> { self.inner.exists(key) }
        fn size(&self, key: &str) -> Result<Option<u64>> { self.inner.size(key) }
        fn list(&self, prefix: &str) -> Result<Vec<String>> {
            self.calls.lock().unwrap().push(format!("list:{prefix}"));
            self.inner.list(prefix)
        }
        fn list_after(&self, prefix: &str, offset: &str) -> Result<Vec<String>> {
            self.calls.lock().unwrap().push(format!("after:{prefix}:{offset}"));
            self.inner.list_after(prefix, offset)
        }
        fn put_if_absent(&self, key: &str, bytes: &[u8]) -> Result<bool> {
            self.inner.put_if_absent(key, bytes)
        }
        fn delete(&self, key: &str) -> Result<()> { self.inner.delete(key) }
    }

    #[test]
    fn repository_upload_gate_drops_denied_and_unknown_sessions() {
        let raw = concat!(
            "{\"session_id\":\"allowed\",\"kind\":\"user_prompt\"}\n",
            "{\"session_id\":\"denied\",\"kind\":\"user_prompt\"}\n",
            "{\"kind\":\"agent_meta\"}\n",
            "not-json\n",
        );
        let allowed = BTreeSet::from(["allowed".to_string()]);
        let filtered = String::from_utf8(filter_events_for_sessions(raw.as_bytes(), &allowed)).unwrap();
        assert!(filtered.contains("allowed"));
        assert!(!filtered.contains("denied"));
        assert!(!filtered.contains("agent_meta"));
        assert!(!filtered.contains("not-json"));
    }

    #[test]
    fn advanced_uploads_refuse_irreversible_policy_changes() {
        let bucket = "s3://team/events";
        let mut state = UploadState::default();
        state.buckets.insert(
            bucket.into(),
            BTreeMap::from([("edge/track.jsonl".into(), 42)]),
        );
        assert!(ensure_upload_policy_compatible(&state, bucket, "off", &[]).is_ok());
        assert!(ensure_upload_policy_compatible(&state, bucket, "standard", &[]).is_err());
        assert!(ensure_upload_policy_compatible(&state, bucket, "off", &["synty".into()]).is_err());
    }

    fn event(id: &str, sid: &str, kind: &str, ts: &str, text: &str) -> String {
        serde_json::json!({
            "v": 1, "event_id": id, "stream": "edge-m-codex", "seq": 0,
            "ts": ts, "source": "codex_cli", "session_id": sid, "kind": kind,
            "payload": {"text": text}
        })
        .to_string()
            + "\n"
    }

    #[test]
    fn event_uploads_are_immutable_incremental_chunks() {
        let root = std::env::temp_dir().join(format!("synty-event-chunks-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let local = root.join("local/edge-m-codex");
        let bucket = root.join("bucket");
        let state = root.join("uploads.json");
        std::fs::create_dir_all(&local).unwrap();
        let file = local.join("track.2026-07-21.jsonl");
        std::fs::write(
            &file,
            event("a", "s", "session_start", "2026-07-21T10:00:00Z", "")
                + &event("b", "s", "user_prompt", "2026-07-21T10:01:00Z", "first"),
        )
        .unwrap();
        let bu = bucket.to_str().unwrap();
        assert_eq!(
            push_events_with(
                bu,
                root.join("local").to_str().unwrap(),
                state.to_str().unwrap(),
                None
            )
            .unwrap(),
            1
        );
        let b = crate::bucket::LocalFs::new(&bucket);
        let first = b.list("events").unwrap();
        assert_eq!(first.len(), 1);
        assert!(first[0].contains("/chunks/"));
        assert!(
            !first[0].ends_with("track.2026-07-21.jsonl"),
            "growing daily file is never the cloud object"
        );
        let first_bytes = b.get(&first[0]).unwrap().unwrap();

        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(&file)
            .unwrap()
            .write_all(
                event(
                    "c",
                    "s",
                    "assistant_message",
                    "2026-07-21T10:02:00Z",
                    "second",
                )
                .as_bytes(),
            )
            .unwrap();
        assert_eq!(
            push_events_with(
                bu,
                root.join("local").to_str().unwrap(),
                state.to_str().unwrap(),
                None
            )
            .unwrap(),
            1
        );
        assert_eq!(
            b.list("events").unwrap().len(),
            2,
            "append creates one new object"
        );
        assert_eq!(
            b.get(&first[0]).unwrap().unwrap(),
            first_bytes,
            "published chunk stays immutable"
        );
        assert_eq!(
            push_events_with(
                bu,
                root.join("local").to_str().unwrap(),
                state.to_str().unwrap(),
                None
            )
            .unwrap(),
            0
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn owned_stream_upload_never_republishes_a_pulled_legacy_stream() {
        let root = std::env::temp_dir().join(format!("synty-event-owned-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let local = root.join("local");
        let mine = local.join("edge-mine-codex");
        let pulled = local.join("edge-other-codex");
        std::fs::create_dir_all(&mine).unwrap();
        std::fs::create_dir_all(&pulled).unwrap();
        std::fs::write(
            mine.join("track.2026-07-21.jsonl"),
            event("mine", "s1", "user_prompt", "2026-07-21T10:00:00Z", "mine"),
        )
        .unwrap();
        std::fs::write(
            pulled.join("track.2026-07-21.jsonl"),
            event(
                "theirs",
                "s2",
                "user_prompt",
                "2026-07-21T10:00:00Z",
                "theirs",
            ),
        )
        .unwrap();

        let bucket = root.join("bucket");
        let owned = BTreeSet::from(["edge-mine-codex".to_string()]);
        assert_eq!(
            push_events_for_streams(
                bucket.to_str().unwrap(),
                local.to_str().unwrap(),
                root.join("uploads.json").to_str().unwrap(),
                None,
                &owned,
            )
            .unwrap(),
            1
        );

        let b = crate::bucket::LocalFs::new(&bucket);
        assert_eq!(
            b.list(EVENT_STREAMS).unwrap(),
            vec!["event-streams/edge-mine-codex"],
            "only this tracker instance's stream is registered"
        );
        let uploaded = b.list(EVENTS).unwrap();
        assert_eq!(uploaded.len(), 1);
        assert!(
            uploaded[0].starts_with("events/edge-mine-codex/chunks/"),
            "pulled fleet data must stay read-only: {uploaded:?}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn public_event_upload_does_not_confuse_colliding_machine_prefixes() {
        let root = std::env::temp_dir().join(format!("synty-event-prefix-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let local = root.join("local");
        let mine = local.join("edge-a-codex");
        let pulled = local.join("edge-a-b-codex");
        std::fs::create_dir_all(&mine).unwrap();
        std::fs::create_dir_all(&pulled).unwrap();
        for (dir, id) in [(&mine, "mine"), (&pulled, "theirs")] {
            std::fs::write(
                dir.join("track.2026-07-21.jsonl"),
                event(id, id, "user_prompt", "2026-07-21T10:00:00Z", id),
            )
            .unwrap();
        }

        let bucket = root.join("bucket");
        assert_eq!(
            push_events_for_machine(
                bucket.to_str().unwrap(),
                local.to_str().unwrap(),
                root.join("uploads.json").to_str().unwrap(),
                None,
                "a",
            )
            .unwrap(),
            1
        );
        let b = crate::bucket::LocalFs::new(&bucket);
        assert_eq!(
            b.list(EVENT_STREAMS).unwrap(),
            vec!["event-streams/edge-a-codex"],
            "machine a must not claim machine a-b's stream"
        );
        assert!(
            b.list(EVENTS)
                .unwrap()
                .iter()
                .all(|key| key.starts_with("events/edge-a-codex/chunks/")),
            "only the exact machine identity is uploaded"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn event_upload_boundary_keeps_metadata_across_daily_files() {
        let root = std::env::temp_dir().join(format!("synty-event-since-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let local = root.join("local/edge-m-codex");
        let bucket = root.join("bucket");
        std::fs::create_dir_all(&local).unwrap();
        let file = local.join("track.2026-07-20.jsonl");
        std::fs::write(
            &file,
            event(
                "a",
                "s",
                "session_start",
                "2026-07-20T23:50:00Z",
                "metadata",
            ) + &event("b", "s", "user_prompt", "2026-07-20T23:55:00Z", "too old"),
        )
        .unwrap();
        let cutoff = chrono::DateTime::parse_from_rfc3339("2026-07-21T00:00:00Z")
            .unwrap()
            .timestamp_millis();
        let state = root.join("uploads.json");
        assert_eq!(
            push_events_with(
                bucket.to_str().unwrap(),
                root.join("local").to_str().unwrap(),
                state.to_str().unwrap(),
                Some(cutoff),
            )
            .unwrap(),
            0,
            "old-only batch publishes nothing"
        );
        std::fs::write(
            local.join("track.2026-07-21.jsonl"),
            event(
                "c",
                "s",
                "assistant_message",
                "2026-07-21T00:05:00Z",
                "keep me",
            ),
        ).unwrap();
        assert_eq!(
            push_events_with(
                bucket.to_str().unwrap(),
                root.join("local").to_str().unwrap(),
                state.to_str().unwrap(),
                Some(cutoff),
            )
            .unwrap(),
            1,
            "first eligible delta carries staged metadata"
        );
        let b = crate::bucket::LocalFs::new(&bucket);
        let body = b
            .list("events")
            .unwrap()
            .into_iter()
            .flat_map(|k| b.get(&k).unwrap().unwrap())
            .collect::<Vec<_>>();
        let body = String::from_utf8(body).unwrap();
        assert!(body.contains("metadata") && body.contains("keep me"));
        assert!(!body.contains("too old"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn chunking_never_splits_an_event_line() {
        let raw = b"aaaa\nbbbb\ncccc\n";
        assert_eq!(
            event_chunks(raw, 6),
            vec![b"aaaa\n".to_vec(), b"bbbb\n".to_vec(), b"cccc\n".to_vec()]
        );
    }

    #[test]
    fn event_object_keys_cannot_escape_the_local_corpus() {
        assert!(event_dest("corpus/local", "events/edge-ok/chunks/a.jsonl").is_ok());
        assert!(event_dest("corpus/local", "events/../../config.json").is_err());
    }

    #[test]
    fn repeated_event_pulls_continue_each_registered_stream_by_key() {
        let root = std::env::temp_dir().join(format!("synty-event-cursor-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let source = root.join("source/edge-m-codex");
        let bucket = root.join("bucket");
        let reader = root.join("reader");
        std::fs::create_dir_all(&source).unwrap();
        let file = source.join("track.2026-07-21.jsonl");
        std::fs::write(&file, event("a", "s", "user_prompt", "2026-07-21T10:00:00Z", "first")).unwrap();
        let upload_state = root.join("uploads.json");
        push_events_with(
            bucket.to_str().unwrap(),
            root.join("source").to_str().unwrap(),
            upload_state.to_str().unwrap(),
            None,
        ).unwrap();

        let store = RecordingBucket::new(&bucket);
        let download_state = reader.join(".downloads.json");
        assert_eq!(pull_events_from(
            &store,
            bucket.to_str().unwrap(),
            reader.to_str().unwrap(),
            &download_state,
        ).unwrap(), 1);
        store.calls.lock().unwrap().clear();
        assert_eq!(pull_events_from(
            &store,
            bucket.to_str().unwrap(),
            reader.to_str().unwrap(),
            &download_state,
        ).unwrap(), 0);
        let calls = store.calls.lock().unwrap().clone();
        assert!(!calls.iter().any(|c| c == "list:events"), "historical event namespace was relisted: {calls:?}");
        assert!(calls.iter().any(|c| c.starts_with("after:events/edge-m-codex/chunks/:events/edge-m-codex/chunks/")), "missing persisted stream cursor: {calls:?}");

        use std::io::Write;
        std::fs::OpenOptions::new().append(true).open(&file).unwrap()
            .write_all(event("b", "s", "assistant_message", "2026-07-21T10:01:00Z", "second").as_bytes())
            .unwrap();
        assert_eq!(push_events_with(
            bucket.to_str().unwrap(),
            root.join("source").to_str().unwrap(),
            upload_state.to_str().unwrap(),
            None,
        ).unwrap(), 1);
        assert_eq!(pull_events_from(
            &store,
            bucket.to_str().unwrap(),
            reader.to_str().unwrap(),
            &download_state,
        ).unwrap(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

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
