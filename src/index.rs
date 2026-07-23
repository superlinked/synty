// Encode docs and build the next-plaid index. Encoding is content-addressed:
// each doc's embedding is fetched from (or stored to) a shared EmbStore keyed
// by a hash of its text, so a message is encoded exactly once across all runs
// and devices. Builds are immutable directories under index/builds/<build>
// (see readmodel): an unchanged corpus skips everything, a tail-extended
// corpus CLONES the previous build and appends only the new docs, anything
// else rebuilds — and readers only ever see complete builds via the pointer.

use crate::store::EmbStore;
use crate::{encode::Encoder, load_docs, readmodel};
use anyhow::{Context, Result};
use ndarray::Array2;
use next_plaid::{IndexConfig, MmapIndex, UpdateConfig};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;
use std::time::Instant;

/// Bounds decoded ColBERT matrices plus next-plaid's temporary flat copy to a
/// workstation-sized resident set while still giving each disk update enough
/// work to amortize centroid and metadata maintenance.
const INDEX_BATCH_DOCS: usize = 2048;

fn index_batches(start: usize, end: usize) -> Vec<std::ops::Range<usize>> {
    (start..end)
        .step_by(INDEX_BATCH_DOCS)
        .map(|batch_start| batch_start..(batch_start + INDEX_BATCH_DOCS).min(end))
        .collect()
}

pub fn run(docs_path: &str, model_id: &str, bucket: &str) -> Result<()> {
    // Ingest holds the same lock across all three atomic file replacements.
    // Keep it through the pointer move so the build identity, embeddings,
    // metadata, and copied projections all come from that single generation.
    let _generation = crate::generation::exclusive(docs_path)?;
    let docs = load_docs(docs_path)?;
    anyhow::ensure!(!docs.is_empty(), "no docs at {docs_path}; run `ingest` first");
    let analysis_path = Path::new(docs_path).with_file_name("analysis.json");
    let trace_path = Path::new(docs_path).with_file_name("trace.json");
    anyhow::ensure!(
        analysis_path.is_file() && trace_path.is_file(),
        "raw-derived read snapshots are missing; run `ingest` first"
    );

    let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
    let hashes: Vec<u64> = texts.iter().map(|t| fnv1a(t.as_bytes())).collect();
    let metas: Vec<serde_json::Value> = docs
        .iter()
        .map(|d| serde_json::to_value(&d.meta))
        .collect::<std::result::Result<_, _>>()?;
    let projection_hashes = [file_hash(&analysis_path)?, file_hash(&trace_path)?];
    let build = readmodel::build_id(&hashes, &projection_hashes);

    // Unchanged corpus → the current build already matches; skip the rebuild
    // but still ensure the bucket has it (publish is a no-op if current).
    let prev = readmodel::current();
    let prev_hashes = prev.as_ref().and_then(|c| load_manifest(&c.dir()));
    if let Some(cur) = &prev {
        if cur.format == readmodel::FORMAT
            && cur.build == build
            && cur.analysis().is_file()
            && cur.trace().is_file()
            && prev_hashes.as_deref() == Some(hashes.as_slice())
            && MmapIndex::load(&cur.dir().to_string_lossy()).is_ok()
        {
            eprintln!("index up to date ({} docs, corpus unchanged)", docs.len());
            let published = crate::sync::publish(bucket)?;
            if published > 0 {
                eprintln!("published {published} read-model objects → {bucket}");
            }
            return Ok(());
        }
    }

    // ingest keeps docs.jsonl prefix-stable: existing docs hold their
    // positions, removals leave gaps, and new work appends at the tail (see
    // ingest::stable_order). So the previous build reconciles incrementally:
    // clone it (CoW), DELETE the docs that left the corpus (next-plaid
    // rewrites only their chunks and renumbers in order, which keeps index
    // ids aligned with docs.jsonl positions), and append the tail. Only a
    // genuinely foreign corpus (a build pulled from another machine, or churn
    // past RECONCILE_MAX) pays the full k-means + re-quantization rebuild.
    let dir = readmodel::build_dir(&build);
    let dir_str = dir.to_string_lossy().into_owned();
    let start = match (&prev, &prev_hashes) {
        (Some(cur), Some(p)) if MmapIndex::load(&cur.dir().to_string_lossy()).is_ok() => {
            match reconcile(p, &hashes) {
                Some((deletions, survivors)) => {
                    readmodel::clone_build(&cur.dir(), &dir)?;
                    if !deletions.is_empty() {
                        let mut idx = MmapIndex::load(&dir_str).context("load cloned build for patch")?;
                        let n = idx.delete(&deletions).context("patch out departed docs")?;
                        eprintln!("index: patched out {n} departed doc(s)");
                    }
                    survivors
                }
                None => {
                    let _ = std::fs::remove_dir_all(&dir);
                    std::fs::create_dir_all(&dir)?;
                    0
                }
            }
        }
        _ => {
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir)?;
            0
        }
    };
    let path = dir.as_path();

    // Inventory once, then fetch/encode/index bounded slices. ColBERT matrices
    // are large enough that retaining a 30-day corpus as f32 until the final
    // index call exceeds workstation memory; next-plaid's streaming entrypoint
    // keeps only this slice resident while its completed chunks stay on disk.
    let store = EmbStore::open(bucket, model_id)?;
    let n_new = texts.len() - start;
    let known = store.known_hashes()?;
    let listed_reused = hashes[start..].iter().filter(|hash| known.contains(hash)).count();
    let expected_miss = n_new - listed_reused;
    let t0 = Instant::now();
    if expected_miss > 0 {
        eprintln!("reusing {listed_reused} from store, encoding {expected_miss} new/changed in bounded batches");
    }

    let t1 = Instant::now();
    let idx = if n_new == 0 && start > 0 {
        // Pure patch: docs only departed, nothing to append — the cloned and
        // patched build is already complete.
        MmapIndex::load(&dir_str).context("load patched build")?
    } else {
        if start > 0 {
            eprintln!("index: appending {n_new} new docs ({start} already indexed)");
        }
        let index_config = IndexConfig::default();
        let update_config = UpdateConfig::default();
        let mut enc: Option<Encoder> = None;
        let mut encoded_done = 0usize;
        let mut indexed_done = 0usize;
        for range in index_batches(start, texts.len()) {
            let mut batch = store.get_listed(&hashes[range.clone()], &known)?;
            let miss: Vec<usize> = (0..batch.len()).filter(|&i| batch[i].is_none()).collect();
            if !miss.is_empty() {
                if enc.is_none() {
                    eprintln!("loading model {model_id} (first run downloads from HF, then offline)...");
                    enc = Some(Encoder::load(model_id)?);
                }
                for chunk in miss.chunks(64) {
                    let chunk_texts: Vec<String> =
                        chunk.iter().map(|&i| texts[range.start + i].clone()).collect();
                    let encoded = enc.as_mut().unwrap().encode_docs(&chunk_texts)?;
                    let entries: Vec<(u64, Array2<f32>)> = chunk
                        .iter()
                        .zip(encoded)
                        .map(|(&i, emb)| (hashes[range.start + i], emb))
                        .collect();
                    store.put_many(&entries)?; // share one write-once batch with the fleet
                    for (&i, (_, emb)) in chunk.iter().zip(entries) {
                        batch[i] = Some(emb);
                    }
                    encoded_done += chunk.len();
                    eprint!("\rencoded {encoded_done}/{expected_miss}");
                    crate::progress::phase("encoding", encoded_done, expected_miss.max(encoded_done));
                }
            }
            let embeddings: Vec<Array2<f32>> =
                batch.into_iter().map(|o| o.expect("every batch doc filled")).collect();
            let (next, ids) = MmapIndex::update_or_create_with_metadata(
                &embeddings,
                &dir_str,
                &index_config,
                &update_config,
                Some(&metas[range.clone()]),
            )
            .context("build next-plaid index batch")?;
            anyhow::ensure!(
                ids.len() == range.len()
                    && ids.first().copied() == Some(range.start as i64)
                    && ids.last().copied() == Some(range.end as i64 - 1),
                "next-plaid assigned non-contiguous ids for docs {}..{}",
                range.start,
                range.end
            );
            indexed_done += range.len();
            crate::progress::phase("indexing", indexed_done, n_new);
            drop(next);
        }
        if encoded_done == 0 {
            eprintln!("all {listed_reused} embeddings in store; no encode needed");
        } else {
            eprintln!("\nencoded {encoded_done} new docs in {:?}", t0.elapsed());
        }
        MmapIndex::load(&dir_str).context("load completed streamed build")?
    };

    // Complete the build dir: the manifest, the docs snapshot readers render
    // from (ids match this index, not a later corpus), and the previous
    // clusters carried forward so topics keep displaying until `cluster` runs.
    std::fs::write(path.join("doc_hashes.json"), serde_json::to_string(&hashes)?)?;
    std::fs::copy(docs_path, path.join("docs.jsonl"))?;
    std::fs::copy(&analysis_path, path.join("analysis.json"))?;
    std::fs::copy(&trace_path, path.join("trace.json"))?;
    if let Some(cur) = &prev {
        let _ = std::fs::copy(cur.clusters(), path.join("unit_clusters.0.json"));
    }

    // The pointer move is the publish: only now do readers see this build.
    let keep: Vec<String> = prev.iter().map(|c| c.build.clone()).collect();
    readmodel::repoint(&build, 0)?;
    readmodel::gc(&keep);

    eprintln!(
        "indexed {} docs / {} embeddings in {:?} → {}",
        idx.num_documents(),
        idx.num_embeddings(),
        t1.elapsed(),
        dir.display()
    );

    // Publish the read-model so other devices can query without rebuilding.
    let published = crate::sync::publish(bucket)?;
    if published > 0 {
        eprintln!("published {published} read-model objects → {bucket}");
    }
    Ok(())
}

/// Stream a projection fingerprint without retaining its serialized bytes.
fn file_hash(path: &Path) -> Result<u64> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        digest.update(&buf[..read]);
    }
    Ok(u64::from_le_bytes(digest.finalize()[..8].try_into().unwrap()))
}

/// More than 1/PATCH_FRAC of the previous index departing → full rebuild:
/// chunk-rewriting most of an index is slower and messier than replanning it.
const PATCH_FRAC: usize = 5;

/// Two-pointer diff between the previous build's doc hashes and the current
/// corpus. `ingest::stable_order` only removes docs or appends at the tail —
/// never reorders — so the survivors of `prev` appear as the head of `new`,
/// in order. Returns (prev positions to delete, survivor count == where the
/// append tail starts in `new`); a strict prefix yields no deletions, and
/// churn past the PATCH_FRAC guard yields None (a foreign or heavily
/// rewritten corpus — rebuild). An in-place edit, which stable_order never
/// produces, still degrades safely: it reads as delete-here + append-at-tail.
fn reconcile(prev: &[u64], new: &[u64]) -> Option<(Vec<i64>, usize)> {
    let mut deletions = Vec::new();
    let mut j = 0usize;
    for (i, h) in prev.iter().enumerate() {
        if j < new.len() && new[j] == *h {
            j += 1;
        } else {
            deletions.push(i as i64);
        }
    }
    if deletions.len() * PATCH_FRAC > prev.len().max(1) {
        return None;
    }
    Some((deletions, j))
}

/// The doc content hashes (in order) a build was built from.
fn load_manifest(dir: &Path) -> Option<Vec<u64>> {
    let raw = std::fs::read_to_string(dir.join("doc_hashes.json")).ok()?;
    serde_json::from_str::<Vec<u64>>(&raw).ok()
}

/// FNV-1a 64-bit — a small, deterministic content hash. A collision only costs
/// a needless re-encode (the index re-derives everything), so it need not be
/// cryptographic.
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x00000100000001B3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::{INDEX_BATCH_DOCS, fnv1a, index_batches, reconcile};

    #[test]
    fn fnv1a_is_stable_and_distinguishing() {
        assert_eq!(fnv1a(b"generation isolation"), fnv1a(b"generation isolation"));
        assert_ne!(fnv1a(b"abc"), fnv1a(b"abd"));
        assert_eq!(fnv1a(b""), 0xcbf29ce484222325);
    }

    #[test]
    fn large_corpus_is_covered_in_order_by_bounded_index_batches() {
        let start = 17;
        let end = start + INDEX_BATCH_DOCS * 2 + 9;
        let batches = index_batches(start, end);
        assert_eq!(batches, vec![17..2065, 2065..4113, 4113..4122]);
        assert!(batches.iter().all(|range| range.len() <= INDEX_BATCH_DOCS));
        assert_eq!(batches.iter().map(|range| range.len()).sum::<usize>(), end - start);
    }

    // The reconcile sees every corpus change stable_order can produce as an
    // incremental patch: appends, departures, and the move-to-tail shape of
    // an edit — never paying a full re-quantization for them.
    #[test]
    fn reconcile_patches_appends_and_departures() {
        // Strict prefix: nothing to delete, append from the survivor count.
        assert_eq!(reconcile(&[1, 2, 3], &[1, 2, 3, 4]), Some((vec![], 3)));
        // One departure mid-list plus a new tail.
        assert_eq!(reconcile(&[1, 2, 3, 4, 5], &[1, 2, 4, 5, 6]), Some((vec![2], 4)));
        // Pure departure, nothing appended (the "6 docs vanished" case).
        assert_eq!(reconcile(&[1, 2, 3, 4, 5], &[1, 2, 4, 5]), Some((vec![2], 4)));
        // An edit as stable_order emits it: old version departs, new at tail.
        assert_eq!(reconcile(&[1, 2, 3, 4, 5], &[1, 3, 4, 5, 9]), Some((vec![1], 4)));
        // Duplicate hashes match as a multiset, in order.
        assert_eq!(reconcile(&[7, 7, 8, 1, 2, 3], &[7, 8, 1, 2, 3]), Some((vec![1], 5)));
    }

    // Churn past 1/PATCH_FRAC of the index means chunk-rewriting most of it —
    // a full rebuild is cheaper and cleaner, so the reconcile abstains.
    #[test]
    fn reconcile_abstains_on_heavy_churn() {
        let prev: Vec<u64> = (0..100).collect();
        let mut new: Vec<u64> = (30..100).collect(); // 30 departures of 100
        new.push(1000);
        assert_eq!(reconcile(&prev, &new), None);
        // ...but 10 departures of 100 is a patch.
        let new: Vec<u64> = (10..100).collect();
        assert_eq!(reconcile(&prev, &new), Some(((0..10).collect(), 90)));
    }
}
