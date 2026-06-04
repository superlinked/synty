// Unit-level topics. Instead of clustering the 4k-doc message firehose, cluster
// the *units of work* (sessions, PRs, issues) by a multi-vector ColBERT
// embedding of a compact per-unit text (its summary plus the session keyphrases
// or the PR/issue title — a lone one-liner is too thin to separate) — MaxSim kNN
// + Louvain, the same late-interaction substrate as retrieval, one level up. A
// topic is then a coherent set of units, so its members/facets/label/summary are
// consistent by construction. Writes unit_clusters.json; reports silhouette.

use crate::community::{louvain, modularity, Graph};
use crate::store::EmbStore;
use crate::{encode::Encoder, units};
use anyhow::{anyhow, ensure, Result};
use ndarray::Array2;
use next_plaid::{IndexConfig, MmapIndex, SearchParameters, UpdateConfig};
use std::collections::HashMap;

const K: usize = 6;
const EVAL_K: usize = 16; // neighbors fetched for the silhouette eval (graph uses top-K)
const FLOOR: f64 = 0.6; // keep a neighbor only if ≥60% of the best neighbor's score
const MIN_CLUSTER: usize = 3;
const INDEX_DIR: &str = "index/topics";

pub fn run(resolution: f64, model_id: &str, bucket: &str) -> Result<()> {
    let units = units::cluster_units()?;
    ensure!(!units.is_empty(), "no unit summaries; run `summarize` first");
    let n = units.len();
    eprintln!("topics: clustering {n} units by summary embedding");

    // Encode the per-unit text, content-addressed in the shared store
    // (encode-once per text, reused across runs/devices like doc embeddings).
    let store = EmbStore::open(bucket)?;
    let hashes: Vec<u64> = units.iter().map(|u| crate::index::fnv1a(u.embed.as_bytes())).collect();
    let mut emb: Vec<Option<Array2<f32>>> = vec![None; n];
    let mut miss = Vec::new();
    for i in 0..n {
        match store.get(hashes[i])? {
            Some(e) => emb[i] = Some(e),
            None => miss.push(i),
        }
    }
    if !miss.is_empty() {
        eprintln!("topics: encoding {} units", miss.len());
        let mut enc = Encoder::load(model_id)?;
        for chunk in miss.chunks(64) {
            let texts: Vec<String> = chunk.iter().map(|&i| units[i].embed.clone()).collect();
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

    // One kNN search feeds both the graph (top-K) and the quality eval (full).
    let params = SearchParameters { top_k: EVAL_K + 1, n_full_scores: 256, n_ivf_probe: 4, ..Default::default() };
    eprintln!("topics: kNN over {n} summaries");
    let results = idx.search_batch(&emb, &params, true, None).map_err(|e| anyhow!("search: {e}"))?;

    let edges = build_edges(&results);
    let comm = louvain(Graph::from_edges(n, &edges), resolution);
    let q = modularity(&Graph::from_edges(n, &edges), &comm, resolution);

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &c) in comm.iter().enumerate() {
        groups.entry(c).or_default().push(i);
    }
    let mut clusters: Vec<Vec<usize>> = groups.into_values().filter(|v| v.len() >= MIN_CLUSTER).collect();
    clusters.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));

    // unit → cluster index from Louvain (None if its community was < MIN_CLUSTER).
    let mut of: Vec<Option<usize>> = vec![None; n];
    for (ci, c) in clusters.iter().enumerate() {
        for &m in c {
            of[m] = Some(ci);
        }
    }

    // Outlier reassignment: move each unit that's nearer another cluster into it
    // (one simultaneous pass over the Louvain assignment — no oscillation).
    let moved = reassign(&results, &mut of);
    if moved > 0 {
        eprintln!("topics: reassigned {moved} outlier units to their nearest cluster");
    }

    // Member lists + c-TF-IDF labels over the *reassigned* membership.
    let ncl = clusters.len();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); ncl];
    for (i, o) in of.iter().enumerate() {
        if let Some(ci) = o {
            members[*ci].push(i);
        }
    }
    let texts: Vec<Vec<&str>> = members.iter().map(|c| c.iter().map(|&m| units[m].summary.as_str()).collect()).collect();
    let phrases = crate::keyphrase::labels(&texts, 4);

    let mut assign: Vec<serde_json::Value> = Vec::new();
    for (i, o) in of.iter().enumerate() {
        if let Some(ci) = o {
            assign.push(serde_json::json!({"key": units[i].key, "cluster": ci, "label": phrases[*ci].join(", ")}));
        }
    }
    std::fs::write("unit_clusters.json", serde_json::to_string(&assign)?)?;
    eprintln!(
        "topics: {} units in {ncl} clusters (resolution {resolution}, modularity {q:.3}) → unit_clusters.json",
        assign.len(),
    );
    report_quality(&results, &of, &phrases, &units);
    diag(&units, &results, &of, &phrases);
    Ok(())
}

/// Optional per-unit neighbor dump (`SYNTY_DIAG=<key substring>`): for each
/// matching unit print its assigned cluster and its top MaxSim neighbors with
/// scores and their clusters — to see *why* a unit landed where it did.
fn diag(units: &[units::UnitClusterInput], results: &[next_plaid::QueryResult], of: &[Option<usize>], phrases: &[Vec<String>]) {
    let Ok(want) = std::env::var("SYNTY_DIAG") else { return };
    let label = |ci: Option<usize>| ci.map(|c| phrases.get(c).map(|p| p.join(", ")).unwrap_or_default()).unwrap_or_else(|| "—".into());
    for (i, u) in units.iter().enumerate() {
        if !u.key.contains(&want) {
            continue;
        }
        eprintln!("\ndiag {} → cluster {:?} [{}]", u.key, of[i], label(of[i]));
        eprintln!("  embed: {}", crate::excerpt(&u.embed, 160));
        eprintln!("  top MaxSim neighbors (score · cluster · key · embed):");
        for (id, s) in results[i].passage_ids.iter().zip(results[i].scores.iter()).take(10) {
            let j = *id as usize;
            if j == i {
                continue;
            }
            eprintln!("    {s:.3}  c{:<3?} {}  {}", of[j], crate::short(&units[j].key), crate::excerpt(&units[j].embed, 70));
        }
    }
}

/// kNN edges from MaxSim: normalized per-unit (÷ best neighbor), floored, summed
/// over both directions so mutual neighbors weigh more. Top-K per unit.
fn build_edges(results: &[next_plaid::QueryResult]) -> HashMap<(usize, usize), f64> {
    let n = results.len();
    // Directed normalized weights + each unit's top-K neighbor set.
    let mut dir: HashMap<(usize, usize), f64> = HashMap::new();
    let mut topset: Vec<std::collections::HashSet<usize>> = vec![std::collections::HashSet::new(); n];
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
            dir.insert((i, j), d);
            topset[i].insert(j);
        }
    }
    // Mutual k-NN: keep an edge only if each is in the other's top-K — symmetric
    // by construction, and it strips the one-way hub/spurious edges MaxSim's
    // asymmetry produces. Weight is both directions summed.
    let mut edges: HashMap<(usize, usize), f64> = HashMap::new();
    for (&(i, j), &w) in &dir {
        if topset[j].contains(&i) {
            *edges.entry((i.min(j), i.max(j))).or_insert(0.0) += w;
        }
    }
    edges
}

/// Reassign outliers to their nearest cluster, iterating a few simultaneous
/// passes so units left stranded when their neighbors move get cleaned up too.
/// Capped, so it always terminates. Returns how many units ended up moved.
fn reassign(results: &[next_plaid::QueryResult], of: &mut [Option<usize>]) -> usize {
    let orig = of.to_vec();
    for _ in 0..5 {
        if reassign_once(results, of) == 0 {
            break;
        }
    }
    (0..of.len()).filter(|&i| of[i] != orig[i]).count()
}

/// One simultaneous pass. A *clustered* unit nearer another cluster moves there;
/// an *orphan* (left unplaced by the mutual-kNN graph) joins its single nearest
/// cluster — so the tight core is kept while coverage is restored. Decisions
/// read the current assignment and apply at once. Returns how many changed.
fn reassign_once(results: &[next_plaid::QueryResult], of: &mut [Option<usize>]) -> usize {
    let orig = of.to_vec();
    let mut moved = 0;
    for (i, r) in results.iter().enumerate() {
        let (mut a, mut b, mut other) = (0.0f32, 0.0f32, None);
        for (id, s) in r.passage_ids.iter().zip(r.scores.iter()) {
            let j = *id as usize;
            if j == i {
                continue;
            }
            match orig[j] {
                Some(cj) if Some(cj) == orig[i] => {
                    if a == 0.0 {
                        a = *s; // first same-cluster neighbor is the best (sorted)
                    }
                }
                Some(cj) => {
                    if other.is_none() {
                        b = *s;
                        other = Some(cj);
                    }
                }
                None => {}
            }
        }
        // orphan with a placeable neighbor, or a member nearer another cluster
        if let Some(o) = other {
            if (orig[i].is_none() || b > a) && of[i] != Some(o) {
                of[i] = Some(o);
                moved += 1;
            }
        }
    }
    moved
}

/// Cluster-quality report. For each clustered unit, silhouette = (a − b)/max(a,b)
/// where a is its best same-cluster neighbor's MaxSim and b its best
/// other-cluster neighbor's. Negative means the unit is nearer a *different*
/// cluster — a likely misplacement. Reports the mean, the misplaced count, and
/// the worst offenders (where they sit vs where they'd rather be).
fn report_quality(results: &[next_plaid::QueryResult], of: &[Option<usize>], labels: &[Vec<String>], units: &[units::UnitClusterInput]) {
    let label = |ci: usize| labels.get(ci).map(|p| p.join(", ")).unwrap_or_default();
    let mut sils: Vec<(usize, usize, usize, f32)> = Vec::new(); // (unit, own ci, nearest other ci, silhouette)
    for (i, r) in results.iter().enumerate() {
        let Some(ci) = of[i] else { continue };
        let (mut a, mut b, mut other) = (None, None, ci);
        for (id, s) in r.passage_ids.iter().zip(r.scores.iter()) {
            let j = *id as usize;
            if j == i {
                continue;
            }
            match of[j] {
                Some(cj) if cj == ci => a = a.or(Some(*s)),
                Some(cj) => {
                    if b.is_none() {
                        b = Some(*s);
                        other = cj;
                    }
                }
                None => {}
            }
        }
        let (a, b) = (a.unwrap_or(0.0), b.unwrap_or(0.0));
        let sil = if a.max(b) > 0.0 { (a - b) / a.max(b) } else { 0.0 };
        sils.push((i, ci, other, sil));
    }
    if sils.is_empty() {
        return;
    }
    let mean = sils.iter().map(|x| x.3).sum::<f32>() / sils.len() as f32;
    let misplaced = sils.iter().filter(|x| x.3 < 0.0).count();
    eprintln!(
        "quality: mean silhouette {mean:.3} · {misplaced}/{} units nearer another cluster ({:.0}%)",
        sils.len(),
        100.0 * misplaced as f32 / sils.len() as f32
    );
    sils.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("  worst-placed units (in → would prefer):");
    for (i, ci, other, sil) in sils.iter().take(8) {
        eprintln!(
            "    [{sil:+.2}] {} — in “{}” → “{}”",
            crate::short(&units[*i].key),
            label(*ci),
            label(*other),
        );
    }
}
