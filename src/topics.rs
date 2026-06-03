// Unit-level topics. Instead of clustering the 4k-doc message firehose, cluster
// the *units of work* (sessions, PRs, issues) by the multi-vector ColBERT
// embedding of their one-line summary — MaxSim kNN + Louvain, the same
// late-interaction substrate as retrieval, one level up. A topic is then a
// coherent set of units, so its members/facets/label/summary are consistent by
// construction (no doc-vs-unit reconciliation). Writes unit_clusters.json.

use crate::community::{louvain, modularity, Graph};
use crate::store::EmbStore;
use crate::{encode::Encoder, units};
use anyhow::{anyhow, ensure, Result};
use ndarray::Array2;
use next_plaid::{IndexConfig, MmapIndex, SearchParameters, UpdateConfig};
use std::collections::HashMap;

const K: usize = 6;
const FLOOR: f64 = 0.6; // keep a neighbor only if ≥60% of the best neighbor's score
const MIN_CLUSTER: usize = 3;
const INDEX_DIR: &str = "index/topics";

pub fn run(resolution: f64, model_id: &str, bucket: &str) -> Result<()> {
    let units = units::cluster_units()?;
    ensure!(!units.is_empty(), "no unit summaries; run `summarize` first");
    let n = units.len();
    eprintln!("topics: clustering {n} units by summary embedding");

    // Encode summaries, content-addressed in the shared store (encode-once per
    // summary text, reused across runs/devices like doc embeddings).
    let store = EmbStore::open(bucket)?;
    let hashes: Vec<u64> = units.iter().map(|u| crate::index::fnv1a(u.summary.as_bytes())).collect();
    let mut emb: Vec<Option<Array2<f32>>> = vec![None; n];
    let mut miss = Vec::new();
    for i in 0..n {
        match store.get(hashes[i])? {
            Some(e) => emb[i] = Some(e),
            None => miss.push(i),
        }
    }
    if !miss.is_empty() {
        eprintln!("topics: encoding {} new summaries", miss.len());
        let mut enc = Encoder::load(model_id)?;
        for chunk in miss.chunks(64) {
            let texts: Vec<String> = chunk.iter().map(|&i| units[i].summary.clone()).collect();
            for (&i, e) in chunk.iter().zip(enc.encode_docs(&texts)?) {
                store.put(hashes[i], &e)?;
                emb[i] = Some(e);
            }
        }
    }
    let emb: Vec<Array2<f32>> = emb.into_iter().map(|o| o.expect("every summary encoded")).collect();

    // A small PLAID index over the summaries gives MaxSim kNN, same as `cluster`.
    let _ = std::fs::remove_dir_all(INDEX_DIR);
    std::fs::create_dir_all(INDEX_DIR)?;
    let (idx, _) = MmapIndex::update_or_create_with_metadata(
        &emb,
        INDEX_DIR,
        &IndexConfig::default(),
        &UpdateConfig::default(),
        None,
    )
    .map_err(|e| anyhow!("build summary index: {e}"))?;

    let edges = build_edges(&idx, &emb)?;
    let comm = louvain(Graph::from_edges(n, &edges), resolution);
    let q = modularity(&Graph::from_edges(n, &edges), &comm, resolution);

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &c) in comm.iter().enumerate() {
        groups.entry(c).or_default().push(i);
    }
    let mut clusters: Vec<Vec<usize>> = groups.into_values().filter(|v| v.len() >= MIN_CLUSTER).collect();
    clusters.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));

    // c-TF-IDF labels over the member summaries.
    let texts: Vec<Vec<&str>> = clusters.iter().map(|c| c.iter().map(|&m| units[m].summary.as_str()).collect()).collect();
    let phrases = crate::keyphrase::labels(&texts, 4);

    let mut assign: Vec<serde_json::Value> = Vec::new();
    for (ci, c) in clusters.iter().enumerate() {
        let label = phrases[ci].join(", ");
        for &m in c {
            assign.push(serde_json::json!({"key": units[m].key, "cluster": ci, "label": label}));
        }
    }
    std::fs::write("unit_clusters.json", serde_json::to_string(&assign)?)?;
    eprintln!(
        "topics: {} units in {} clusters (resolution {resolution}, modularity {q:.3}) → unit_clusters.json",
        assign.len(),
        clusters.len()
    );
    Ok(())
}

/// kNN edges from MaxSim: normalized per-unit (÷ best neighbor), floored, summed
/// over both directions so mutual neighbors weigh more. Same as `cluster`.
fn build_edges(idx: &MmapIndex, emb: &[Array2<f32>]) -> Result<HashMap<(usize, usize), f64>> {
    let params = SearchParameters { top_k: K + 1, n_full_scores: 256, n_ivf_probe: 4, ..Default::default() };
    let results = idx.search_batch(emb, &params, true, None).map_err(|e| anyhow!("search: {e}"))?;
    let mut edges: HashMap<(usize, usize), f64> = HashMap::new();
    for (i, r) in results.iter().enumerate() {
        let pairs: Vec<(usize, f32)> = r
            .passage_ids
            .iter()
            .zip(r.scores.iter())
            .map(|(id, s)| (*id as usize, *s))
            .filter(|(x, _)| *x != i)
            .collect();
        let best = pairs.first().map(|(_, s)| *s).unwrap_or(0.0);
        if best <= 0.0 {
            continue;
        }
        for (j, s) in pairs.into_iter().take(K) {
            let d = (s / best) as f64;
            if d < FLOOR {
                continue;
            }
            *edges.entry((i.min(j), i.max(j))).or_insert(0.0) += d;
        }
    }
    Ok(edges)
}
