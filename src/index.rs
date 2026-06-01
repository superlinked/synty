// Encode docs with pylate-rs and build the next-plaid index (+ metadata for
// WHERE-filtering and FTS). Encoding is incremental: per-doc embeddings are
// cached by content hash (embeddings.npy + emb_hashes.json), so a re-index
// after editing the corpus only encodes the new/changed texts, and an
// unchanged corpus skips the model load entirely. The index itself is rebuilt
// from scratch each run (cheap relative to encoding).

use crate::{encode::Encoder, load_docs};
use anyhow::{Context, Result};
use ndarray::Array2;
use next_plaid::{IndexConfig, MmapIndex, UpdateConfig};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

pub fn run(docs_path: &str, index_path: &str, model_id: &str) -> Result<()> {
    let docs = load_docs(docs_path)?;
    anyhow::ensure!(!docs.is_empty(), "no docs at {docs_path}; run `ingest` first");
    let path = Path::new(index_path);

    let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
    let hashes: Vec<u64> = texts.iter().map(|t| fnv1a(t.as_bytes())).collect();
    let metas: Vec<serde_json::Value> = docs
        .iter()
        .map(|d| serde_json::to_value(&d.meta))
        .collect::<std::result::Result<_, _>>()?;

    // Fast path: if the corpus is byte-identical to what the current index was
    // built from (same content hashes in the same order) and that index still
    // loads, there is nothing to do — skip the re-encode and the rebuild.
    if let Some(stored) = load_hashes(path) {
        if stored == hashes && MmapIndex::load(index_path).is_ok() {
            eprintln!("index up to date ({} docs, corpus unchanged)", docs.len());
            return Ok(());
        }
    }

    // Reuse prior embeddings for unchanged texts — load before the dir is wiped.
    let cache = load_emb_cache(path);
    let mut embeddings: Vec<Option<Array2<f32>>> = vec![None; texts.len()];
    let mut miss: Vec<usize> = Vec::new();
    for i in 0..texts.len() {
        match cache.get(&hashes[i]) {
            Some(e) => embeddings[i] = Some(e.clone()),
            None => miss.push(i),
        }
    }
    let reused = texts.len() - miss.len();

    let t0 = Instant::now();
    if miss.is_empty() {
        eprintln!("all {reused} embeddings cached; no encode needed");
    } else {
        eprintln!("loading model {model_id} (first run downloads from HF, then offline)...");
        let mut enc = Encoder::load(model_id)?;
        eprintln!("reusing {reused} cached, encoding {} new/changed", miss.len());
        let mut done = 0;
        for chunk in miss.chunks(64) {
            let chunk_texts: Vec<String> = chunk.iter().map(|&i| texts[i].clone()).collect();
            for (&i, e) in chunk.iter().zip(enc.encode_docs(&chunk_texts)?) {
                embeddings[i] = Some(e);
            }
            done += chunk.len();
            eprint!("\rencoded {done}/{}", miss.len());
        }
        eprintln!("\nencoded {} new docs in {:?}", miss.len(), t0.elapsed());
    }
    let embeddings: Vec<Array2<f32>> =
        embeddings.into_iter().map(|o| o.expect("every doc filled")).collect();

    let _ = std::fs::remove_dir_all(path);
    std::fs::create_dir_all(path)?;

    let t1 = Instant::now();
    let (idx, _ids) = MmapIndex::update_or_create_with_metadata(
        &embeddings,
        index_path,
        &IndexConfig::default(),
        &UpdateConfig::default(),
        Some(&metas),
    )
    .context("build next-plaid index")?;

    // Persist embeddings + their content hashes so the next `index` reuses them
    // and `cluster` reads them instead of re-encoding the corpus.
    next_plaid::update::save_embeddings_npy(path, &embeddings)
        .map_err(|e| anyhow::anyhow!("save embeddings cache: {e}"))?;
    std::fs::write(path.join("emb_hashes.json"), serde_json::to_string(&hashes)?)?;

    eprintln!(
        "indexed {} docs / {} embeddings in {:?} → {index_path}",
        idx.num_documents(),
        idx.num_embeddings(),
        t1.elapsed()
    );
    Ok(())
}

/// The content hashes the current index was built from, in doc order.
fn load_hashes(path: &Path) -> Option<Vec<u64>> {
    let raw = std::fs::read_to_string(path.join("emb_hashes.json")).ok()?;
    serde_json::from_str::<Vec<u64>>(&raw).ok()
}

/// Map content-hash → cached embedding from the prior index, or empty if the
/// cache is absent or inconsistent.
fn load_emb_cache(path: &Path) -> HashMap<u64, Array2<f32>> {
    let Ok(embs) = next_plaid::update::load_embeddings_npy(path) else {
        return HashMap::new();
    };
    let Some(hashes) = load_hashes(path) else {
        return HashMap::new();
    };
    if hashes.len() != embs.len() {
        return HashMap::new();
    }
    hashes.into_iter().zip(embs).collect()
}

/// FNV-1a 64-bit — a small, deterministic, dependency-free content hash. A
/// collision just costs one needless re-encode, never a wrong embedding (the
/// rebuilt index re-derives everything), so cryptographic strength is not
/// needed.
fn fnv1a(bytes: &[u8]) -> u64 {
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

    // Same bytes hash the same; different bytes differ. (Stable across runs so
    // the embedding cache hits on an unchanged corpus.)
    #[test]
    fn fnv1a_is_stable_and_distinguishing() {
        assert_eq!(fnv1a(b"generation isolation"), fnv1a(b"generation isolation"));
        assert_ne!(fnv1a(b"abc"), fnv1a(b"abd"));
        assert_eq!(fnv1a(b""), 0xcbf29ce484222325); // FNV offset basis
    }
}
