// Encode docs and build the next-plaid index. Encoding is content-addressed:
// each doc's embedding is fetched from (or stored to) a shared EmbStore keyed
// by a hash of its text, so a message is encoded exactly once across all runs
// and devices. The PLAID index is rebuilt from the assembled embeddings (cheap
// next to encoding); an unchanged corpus skips even that.

use crate::store::EmbStore;
use crate::{encode::Encoder, load_docs};
use anyhow::{Context, Result};
use ndarray::Array2;
use next_plaid::{IndexConfig, MmapIndex, UpdateConfig};
use std::path::Path;
use std::time::Instant;

pub fn run(docs_path: &str, index_path: &str, model_id: &str, bucket: &str) -> Result<()> {
    let docs = load_docs(docs_path)?;
    anyhow::ensure!(!docs.is_empty(), "no docs at {docs_path}; run `ingest` first");
    let path = Path::new(index_path);

    let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
    let hashes: Vec<u64> = texts.iter().map(|t| fnv1a(t.as_bytes())).collect();
    let metas: Vec<serde_json::Value> = docs
        .iter()
        .map(|d| serde_json::to_value(&d.meta))
        .collect::<std::result::Result<_, _>>()?;

    // Unchanged corpus → the local index already matches; skip the rebuild but
    // still ensure the bucket has it (publish is a no-op if already current).
    let prev = load_manifest(path);
    if prev.as_deref() == Some(hashes.as_slice()) && MmapIndex::load(index_path).is_ok() {
        eprintln!("index up to date ({} docs, corpus unchanged)", docs.len());
        let published = crate::sync::publish(bucket, index_path, docs_path)?;
        if published > 0 {
            eprintln!("published {published} read-model files → {bucket}/index/");
        }
        return Ok(());
    }

    // docs.jsonl is oldest-first, so new work extends the tail: when the
    // previous build is a strict prefix, append only the tail to the existing
    // index. Anything else (recency-cap drop, edited history) → full rebuild.
    let start = match &prev {
        Some(p)
            if p.len() < hashes.len()
                && p[..] == hashes[..p.len()]
                && MmapIndex::load(index_path).is_ok() =>
        {
            p.len()
        }
        _ => {
            let _ = std::fs::remove_dir_all(path);
            std::fs::create_dir_all(path)?;
            0
        }
    };

    // Pull every known embedding from the store; encode only the rest.
    let store = EmbStore::open(bucket)?;
    let n_new = texts.len() - start;
    let mut embeddings: Vec<Option<Array2<f32>>> = vec![None; n_new];
    let mut miss: Vec<usize> = Vec::new();
    for i in start..texts.len() {
        match store.get(hashes[i])? {
            Some(e) => embeddings[i - start] = Some(e),
            None => miss.push(i),
        }
    }
    let reused = n_new - miss.len();

    let t0 = Instant::now();
    if miss.is_empty() {
        eprintln!("all {reused} embeddings in store; no encode needed");
    } else {
        eprintln!("loading model {model_id} (first run downloads from HF, then offline)...");
        let mut enc = Encoder::load(model_id)?;
        eprintln!("reusing {reused} from store, encoding {} new/changed", miss.len());
        let mut done = 0;
        for chunk in miss.chunks(64) {
            let chunk_texts: Vec<String> = chunk.iter().map(|&i| texts[i].clone()).collect();
            for (&i, e) in chunk.iter().zip(enc.encode_docs(&chunk_texts)?) {
                store.put(hashes[i], &e)?; // share to the fleet
                embeddings[i - start] = Some(e);
            }
            done += chunk.len();
            eprint!("\rencoded {done}/{}", miss.len());
        }
        eprintln!("\nencoded {} new docs in {:?}", miss.len(), t0.elapsed());
    }
    let embeddings: Vec<Array2<f32>> =
        embeddings.into_iter().map(|o| o.expect("every doc filled")).collect();

    if start > 0 {
        eprintln!("index: appending {n_new} new docs ({start} already indexed)");
    }
    let t1 = Instant::now();
    let (idx, _ids) = MmapIndex::update_or_create_with_metadata(
        &embeddings,
        index_path,
        &IndexConfig::default(),
        &UpdateConfig::default(),
        Some(&metas[start..]),
    )
    .context("build next-plaid index")?;

    // doc_hashes.json is the build manifest the fast/append paths above read.
    std::fs::write(path.join("doc_hashes.json"), serde_json::to_string(&hashes)?)?;

    eprintln!(
        "indexed {} docs / {} embeddings in {:?} → {index_path}",
        idx.num_documents(),
        idx.num_embeddings(),
        t1.elapsed()
    );

    // Publish the read-model so other devices can query without rebuilding.
    let published = crate::sync::publish(bucket, index_path, docs_path)?;
    eprintln!("published {published} read-model files → {bucket}/index/");
    Ok(())
}

/// The doc content hashes (in order) the current index was built from.
fn load_manifest(path: &Path) -> Option<Vec<u64>> {
    let raw = std::fs::read_to_string(path.join("doc_hashes.json")).ok()?;
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
    use super::fnv1a;

    #[test]
    fn fnv1a_is_stable_and_distinguishing() {
        assert_eq!(fnv1a(b"generation isolation"), fnv1a(b"generation isolation"));
        assert_ne!(fnv1a(b"abc"), fnv1a(b"abd"));
        assert_eq!(fnv1a(b""), 0xcbf29ce484222325);
    }
}
