// Encode docs with pylate-rs and build the next-plaid index (+ metadata for
// WHERE-filtering and FTS). Rebuilds from scratch each run (idempotent).

use crate::{encode::Encoder, load_docs};
use anyhow::{Context, Result};
use ndarray::Array2;
use next_plaid::{IndexConfig, MmapIndex, UpdateConfig};
use std::time::Instant;

pub fn run(docs_path: &str, index_path: &str, model_id: &str) -> Result<()> {
    let docs = load_docs(docs_path)?;
    anyhow::ensure!(!docs.is_empty(), "no docs at {docs_path}; run `ingest` first");

    eprintln!(
        "loading model {model_id} (first run downloads from HF, then offline)..."
    );
    let mut enc = Encoder::load(model_id)?;

    let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
    let metas: Vec<serde_json::Value> = docs
        .iter()
        .map(|d| serde_json::to_value(&d.meta))
        .collect::<std::result::Result<_, _>>()?;

    let t0 = Instant::now();
    let mut embeddings: Vec<Array2<f32>> = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(64) {
        embeddings.extend(enc.encode_docs(chunk)?);
        eprint!("\rencoded {}/{}", embeddings.len(), texts.len());
    }
    eprintln!("\nencoded {} docs in {:?}", embeddings.len(), t0.elapsed());

    let _ = std::fs::remove_dir_all(index_path);
    std::fs::create_dir_all(index_path)?;

    let t1 = Instant::now();
    let (idx, _ids) = MmapIndex::update_or_create_with_metadata(
        &embeddings,
        index_path,
        &IndexConfig::default(),
        &UpdateConfig::default(),
        Some(&metas),
    )
    .context("build next-plaid index")?;

    // Persist the raw per-doc embeddings alongside the index so `cluster` (and
    // incremental re-index) can reuse them instead of re-encoding the corpus.
    next_plaid::update::save_embeddings_npy(std::path::Path::new(index_path), &embeddings)
        .map_err(|e| anyhow::anyhow!("save embeddings cache: {e}"))?;

    eprintln!(
        "indexed {} docs / {} embeddings in {:?} → {index_path}",
        idx.num_documents(),
        idx.num_embeddings(),
        t1.elapsed()
    );
    Ok(())
}
