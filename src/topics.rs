// Unit-level topics. Instead of clustering the 4k-doc message firehose, cluster
// the *units of work* (sessions, PRs, issues) by a multi-vector ColBERT
// embedding of a compact per-unit text (its summary plus the session's repo and
// touched files, or the PR/issue body — a lone one-liner is too thin to
// separate) — MaxSim kNN
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

    // Member lists, plus a readable label per cluster: its most concise member
    // summary. A provisional identifier for reports and the unit_clusters.json
    // fallback — the topic's own LLM name/summary replaces it once `summarize`
    // runs. (Clustering itself is LLM-free; this just borrows the unit summaries.)
    let ncl = clusters.len();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); ncl];
    for (i, o) in of.iter().enumerate() {
        if let Some(ci) = o {
            members[*ci].push(i);
        }
    }

    // I6 (experimental, enable with SYNTY_SPLIT=1): split grab-bag clusters into
    // their sub-themes via a local Louvain on each flagged cluster's induced
    // subgraph, gated so only genuinely-separable sub-themes split. It works —
    // breaks grab-bags into coherent sub-topics (the anchor membership eval goes
    // 3/5 → 5/5) — but at this calibration it also fragments coherent clusters,
    // and silhouette structurally penalizes the extra clusters, so it can't yet be
    // the validating metric or the default. Off until a better keep-criterion lands.
    if std::env::var("SYNTY_SPLIT").is_ok() {
        let mut next_id = members.len();
        let mut splits = 0;
        for ci in 0..members.len() {
            if members[ci].len() < GRABBAG_MIN {
                continue;
            }
            let comm = subgraph_split(&members[ci], &edges, resolution * SPLIT_RES);
            if count_big(&comm) < 2 || sub_silhouette(&members[ci], &comm, &results) < SPLIT_FLOOR {
                continue; // not splittable, or the sub-themes are too similar (coherent)
            }
            let mut sizes: HashMap<usize, usize> = HashMap::new();
            for &c in &comm {
                *sizes.entry(c).or_default() += 1;
            }
            let gid: HashMap<usize, usize> = sizes
                .iter()
                .filter(|(_, sz)| **sz >= MIN_CLUSTER)
                .map(|(c, _)| {
                    let g = next_id;
                    next_id += 1;
                    (*c, g)
                })
                .collect();
            for (k, &mem) in members[ci].iter().enumerate() {
                if let Some(&g) = gid.get(&comm[k]) {
                    of[mem] = Some(g);
                }
            }
            splits += 1;
        }
        if splits > 0 {
            eprintln!("topics: split {splits} grab-bag clusters into sub-themes");
            members = vec![Vec::new(); next_id];
            for (i, o) in of.iter().enumerate() {
                if let Some(ci) = o {
                    members[*ci].push(i);
                }
            }
        }
    }

    let labels: Vec<String> = members
        .iter()
        .map(|c| {
            c.iter()
                .map(|&m| units[m].summary.as_str())
                .filter(|s| !s.is_empty())
                .min_by_key(|s| s.len())
                .map(|s| crate::excerpt(s, 60))
                .unwrap_or_default()
        })
        .collect();

    // Stable content-addressed key per cluster, so the summary/name cache survives
    // renumbering. Read the PREVIOUS clusters (stable key → member set) before
    // overwriting unit_clusters.json. A new cluster inherits the previous key it
    // overlaps most (Jaccard ≥ 0.5, robust to membership drift); otherwise it gets
    // a fresh key hashed from its medoid. Greedy match — exact at this cluster count.
    let prev: Vec<(String, std::collections::HashSet<String>)> = std::fs::read_to_string("unit_clusters.json")
        .ok()
        .and_then(|s| serde_json::from_str::<Vec<serde_json::Value>>(&s).ok())
        .map(|a| {
            let mut m: HashMap<String, std::collections::HashSet<String>> = HashMap::new();
            for it in &a {
                if let (Some(t), Some(k)) = (it["topic"].as_str(), it["key"].as_str()) {
                    m.entry(t.to_string()).or_default().insert(k.to_string());
                }
            }
            m.into_iter().collect()
        })
        .unwrap_or_default();
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut stable_keys: Vec<String> = Vec::with_capacity(members.len());
    let mut inherited = 0usize;
    for (ci, m) in members.iter().enumerate() {
        if m.is_empty() {
            stable_keys.push(format!("e{ci}"));
            continue;
        }
        let cur: std::collections::HashSet<&str> = m.iter().map(|&i| units[i].key.as_str()).collect();
        let best = prev
            .iter()
            .filter(|(k, _)| !used.contains(k.as_str()))
            .map(|(k, s)| {
                let inter = s.iter().filter(|x| cur.contains(x.as_str())).count();
                (k, inter as f64 / (s.len() + cur.len() - inter).max(1) as f64)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        match best {
            Some((k, j)) if j >= 0.5 => {
                used.insert(k.clone());
                stable_keys.push(k.clone());
                inherited += 1;
            }
            _ => stable_keys.push(format!("{:016x}", crate::index::fnv1a(units[medoid(m, &results, &of)].key.as_bytes()))),
        }
    }
    let live = members.iter().filter(|m| !m.is_empty()).count();
    let id_continuity = if prev.is_empty() || live == 0 { 0.0 } else { 100.0 * inherited as f64 / live as f64 };

    let mut assign: Vec<serde_json::Value> = Vec::new();
    for (i, o) in of.iter().enumerate() {
        if let Some(ci) = o {
            assign.push(serde_json::json!({"key": units[i].key, "cluster": ci, "topic": stable_keys[*ci], "label": labels[*ci]}));
        }
    }
    std::fs::write("unit_clusters.json", serde_json::to_string(&assign)?)?;
    eprintln!("topics: wrote unit_clusters.json");
    let qual = report_quality(&results, &of, &labels, &units);
    diag(&units, &results, &of, &labels);

    // Standardized health/quality metrics. silhouette_macro is the headline (the
    // micro mean is size-inflated); cohesion_med/grabbags flag incoherent clusters.
    let mut sizes: Vec<usize> = members.iter().map(|m| m.len()).filter(|&l| l > 0).collect();
    sizes.sort_unstable();
    let docs = units.iter().filter(|u| u.key.starts_with("gh:")).count();
    let tiny = sizes.iter().filter(|&&l| l < MIN_CLUSTER).count();
    // splittable = sizeable clusters whose induced subgraph splits into ≥2
    // sub-themes — the I6 split candidates. (Not all are grab-bags: the global
    // resolution limit hides sub-structure even in coherent clusters; I6 keeps a
    // split only if it raises macro-silhouette.)
    let splittable = (0..members.len())
        .filter(|&ci| members[ci].len() >= GRABBAG_MIN && count_big(&subgraph_split(&members[ci], &edges, resolution)) >= 2)
        .count();
    crate::metrics::Run::new("cluster")
        .set("resolution", resolution)
        .set("units", n)
        .set("clustered", assign.len())
        .set("unclustered", n - assign.len())
        .set("clusters", sizes.len())
        .set("id_continuity", id_continuity)
        .set("modularity", q)
        .set("silhouette_macro", qual.macro_sil as f64)
        .set("silhouette", qual.micro_sil as f64)
        .set("cohesion_med", qual.cohesion_med as f64)
        .set("splittable", splittable)
        .set("misplaced", qual.misplaced)
        .set("misplaced_pct", if qual.scored > 0 { 100.0 * qual.misplaced as f64 / qual.scored as f64 } else { 0.0 })
        .set("vote_disagree", qual.vote_disagree)
        .set("size_min", sizes.first().copied().unwrap_or(0))
        .set("size_med", sizes.get(sizes.len() / 2).copied().unwrap_or(0))
        .set("size_max", sizes.last().copied().unwrap_or(0))
        .set("tiny", tiny)
        .set("sessions", n - docs)
        .set("docs", docs)
        .emit();
    Ok(())
}

/// Optional per-unit neighbor dump (`SYNTY_DIAG=<key substring>`): for each
/// matching unit print its assigned cluster and its top MaxSim neighbors with
/// scores and their clusters — to see *why* a unit landed where it did.
fn diag(units: &[units::UnitClusterInput], results: &[next_plaid::QueryResult], of: &[Option<usize>], phrases: &[String]) {
    let Ok(want) = std::env::var("SYNTY_DIAG") else { return };
    let label = |ci: Option<usize>| ci.map(|c| phrases.get(c).cloned().unwrap_or_default()).unwrap_or_else(|| "—".into());
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

/// The cluster's medoid — the member best connected to its co-members (max summed
/// same-cluster MaxSim). Its key seeds the cluster's stable id: a central member
/// persists across re-clusterings even as the periphery shifts.
fn medoid(members: &[usize], results: &[next_plaid::QueryResult], of: &[Option<usize>]) -> usize {
    *members
        .iter()
        .max_by(|&&a, &&b| same_cluster_score(a, results, of).partial_cmp(&same_cluster_score(b, results, of)).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(&members[0])
}

fn same_cluster_score(i: usize, results: &[next_plaid::QueryResult], of: &[Option<usize>]) -> f32 {
    results[i]
        .passage_ids
        .iter()
        .zip(results[i].scores.iter())
        .filter(|(id, _)| **id as usize != i && of[**id as usize] == of[i])
        .map(|(_, s)| *s)
        .sum()
}

/// Cluster quality, computed once from the kNN results. Headline is the MACRO
/// silhouette (per-cluster mean of means; the micro mean is inflated up to 41%
/// by one large cluster — arXiv:2401.05831). `cohesion_med`/`grabbags` flag
/// incoherent clusters via the cohesion ratio ρ_C = within ÷ global mean MaxSim
/// (arXiv:2511.19350). `vote_disagree` is a rescale-invariant placement check.
struct Quality {
    micro_sil: f32,
    macro_sil: f32,
    misplaced: usize,
    scored: usize,
    cohesion_med: f32,
    vote_disagree: usize,
}

/// Louvain on a cluster's induced subgraph — returns the sub-community per member
/// (parallel to `members`). The global resolution limit hides sub-structure even
/// in coherent clusters, so a local re-run with its own 2m resolves the themes a
/// grab-bag fused (the signal silhouette/cohesion miss). I0 counts ≥MIN sub-
/// communities; I6 reassigns members to them.
fn subgraph_split(members: &[usize], edges: &HashMap<(usize, usize), f64>, resolution: f64) -> Vec<usize> {
    let idx: HashMap<usize, usize> = members.iter().enumerate().map(|(local, &g)| (g, local)).collect();
    let mut sub: HashMap<(usize, usize), f64> = HashMap::new();
    for (&(i, j), &w) in edges {
        if let (Some(&li), Some(&lj)) = (idx.get(&i), idx.get(&j)) {
            sub.insert((li, lj), w);
        }
    }
    louvain(Graph::from_edges(members.len(), &sub), resolution)
}

/// Mean silhouette of a cluster's members against their SUB-communities (best
/// same-sub vs best different-sub neighbor, both within the parent). High → the
/// sub-themes are separable (a genuine grab-bag, splitting helps); near zero →
/// the sub-topics are mutually similar (a coherent cluster, splitting just
/// fragments it). This is the keep-the-split gate.
fn sub_silhouette(members: &[usize], comm: &[usize], results: &[next_plaid::QueryResult]) -> f32 {
    let sub_of: HashMap<usize, usize> = members.iter().zip(comm).map(|(&g, &c)| (g, c)).collect();
    let mut sum = 0.0f32;
    for (k, &mem) in members.iter().enumerate() {
        let (mut a, mut b) = (None, None);
        for (id, s) in results[mem].passage_ids.iter().zip(results[mem].scores.iter()) {
            let j = *id as usize;
            if j == mem {
                continue;
            }
            match sub_of.get(&j) {
                Some(&jc) if jc == comm[k] => a = a.or(Some(*s)),
                Some(_) => b = b.or(Some(*s)),
                None => {}
            }
        }
        let (a, b) = (a.unwrap_or(0.0), b.unwrap_or(0.0));
        sum += if a.max(b) > 0.0 { (a - b) / a.max(b) } else { 0.0 };
    }
    if members.is_empty() { 0.0 } else { sum / members.len() as f32 }
}

/// Number of sub-communities of at least `MIN_CLUSTER` members.
fn count_big(comm: &[usize]) -> usize {
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for &c in comm {
        *sizes.entry(c).or_default() += 1;
    }
    sizes.values().filter(|&&s| s >= MIN_CLUSTER).count()
}

/// Min cluster size for a per-cluster aggregate to count — tiny clusters give
/// one-pair noise that would set the floor (silhouette macro / cohesion median).
const QGATE: usize = 5;
/// A grab-bag must also be sizeable to be worth flagging/splitting.
const GRABBAG_MIN: usize = 8;
/// Resolution multiplier for the per-cluster re-split — a local Louvain with its
/// own 2m sidesteps the global resolution limit that fused the sub-themes.
const SPLIT_RES: f64 = 1.5;
/// Split a cluster only if its sub-themes are this separable (mean sub-silhouette)
/// — keeps coherent clusters whole while breaking up true grab-bags.
const SPLIT_FLOOR: f32 = 0.10;

/// Mean of per-cluster mean silhouettes over clusters of at least `min_size`.
fn macro_silhouette(per_cluster: &[Vec<f32>], min_size: usize) -> f32 {
    let means: Vec<f32> = per_cluster
        .iter()
        .filter(|c| c.len() >= min_size)
        .map(|c| c.iter().sum::<f32>() / c.len() as f32)
        .collect();
    if means.is_empty() {
        0.0
    } else {
        means.iter().sum::<f32>() / means.len() as f32
    }
}

fn report_quality(results: &[next_plaid::QueryResult], of: &[Option<usize>], labels: &[String], units: &[units::UnitClusterInput]) -> Quality {
    let label = |ci: usize| labels.get(ci).cloned().unwrap_or_default();
    let mut sils: Vec<(usize, usize, usize, f32)> = Vec::new(); // (unit, own ci, nearest other ci, silhouette)
    let mut by_cluster: HashMap<usize, (Vec<f32>, Vec<f32>)> = HashMap::new(); // ci -> (silhouettes, same-cluster best a)
    let (mut top1_sum, mut top1_n) = (0.0f32, 0usize);
    for (i, r) in results.iter().enumerate() {
        let Some(ci) = of[i] else { continue };
        let (mut a, mut b, mut other, mut top1) = (None, None, ci, None);
        for (id, s) in r.passage_ids.iter().zip(r.scores.iter()) {
            let j = *id as usize;
            if j == i {
                continue;
            }
            top1.get_or_insert(*s);
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
        top1_sum += top1.unwrap_or(0.0);
        top1_n += 1;
        let e = by_cluster.entry(ci).or_default();
        e.0.push(sil);
        e.1.push(a);
        sils.push((i, ci, other, sil));
    }
    if sils.is_empty() {
        return Quality { micro_sil: 0.0, macro_sil: 0.0, misplaced: 0, scored: 0, cohesion_med: 0.0, vote_disagree: 0 };
    }
    let micro_sil = sils.iter().map(|x| x.3).sum::<f32>() / sils.len() as f32;
    let misplaced = sils.iter().filter(|x| x.3 < 0.0).count();

    // Per cluster: mean silhouette S_C (headline, scale-free) and cohesion ratio
    // ρ_C = within ÷ global mean MaxSim (secondary signal).
    let per_cluster_sils: Vec<Vec<f32>> = by_cluster.values().map(|(s, _)| s.clone()).collect();
    let macro_sil = macro_silhouette(&per_cluster_sils, QGATE);
    let global = (top1_sum / top1_n as f32).max(1e-6);
    let mut per: Vec<(usize, usize, f32, f32)> = by_cluster // (ci, size, S_C, ρ_C)
        .iter()
        .map(|(ci, (s, a))| {
            let sc = s.iter().sum::<f32>() / s.len() as f32;
            (*ci, s.len(), sc, (a.iter().sum::<f32>() / a.len() as f32) / global)
        })
        .collect();
    let mut rho_gate: Vec<f32> = per.iter().filter(|(_, n, _, _)| *n >= QGATE).map(|(_, _, _, r)| *r).collect();
    rho_gate.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cohesion_med = rho_gate.get(rho_gate.len() / 2).copied().unwrap_or(0.0);

    // Rescale-invariant placement: a unit whose kNN-majority cluster differs from
    // its assignment is likely misplaced regardless of the score scale.
    let mut vote_disagree = 0;
    for (i, r) in results.iter().enumerate() {
        let Some(ci) = of[i] else { continue };
        let mut votes: HashMap<usize, usize> = HashMap::new();
        for id in &r.passage_ids {
            let j = *id as usize;
            if j != i {
                if let Some(cj) = of[j] {
                    *votes.entry(cj).or_default() += 1;
                }
            }
        }
        if let Some((&mode, _)) = votes.iter().max_by_key(|(_, c)| **c) {
            if mode != ci {
                vote_disagree += 1;
            }
        }
    }

    sils.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("  worst-placed units (in → would prefer):");
    for (i, ci, other, sil) in sils.iter().take(6) {
        eprintln!("    [{sil:+.2}] {} — in “{}” → “{}”", crate::short(&units[*i].key), label(*ci), label(*other));
    }
    per.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("  lowest-coherence clusters (S_C · ρ_C · size · label):");
    for (ci, sz, sc, r) in per.iter().filter(|(_, n, _, _)| *n >= GRABBAG_MIN).take(6) {
        eprintln!("    c{ci} [S_C {sc:+.2} ρ {r:.2}] n={sz} — {}", label(*ci));
    }
    Quality { micro_sil, macro_sil, misplaced, scored: sils.len(), cohesion_med, vote_disagree }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Macro silhouette = mean of per-cluster mean silhouettes, ignoring clusters
    // below the size gate. Hand-computable cross-check (not the size-guaranteed
    // "macro < micro" inequality, which proves nothing).
    #[test]
    fn macro_silhouette_averages_per_cluster_means() {
        // cluster A (size 5) mean = 1.0; cluster B (size 5) mean = 0.0;
        // a size-1 cluster of 1.0 must be ignored → macro = (1.0 + 0.0)/2 = 0.5,
        // whereas the micro mean would be (5·1 + 5·0 + 1·1)/11 ≈ 0.545.
        let per_cluster = vec![vec![1.0; 5], vec![0.0; 5], vec![1.0]];
        assert!((macro_silhouette(&per_cluster, QGATE) - 0.5).abs() < 1e-6);
        // gating: nothing meets the size floor → 0.0
        assert_eq!(macro_silhouette(&[vec![1.0], vec![0.0]], QGATE), 0.0);
    }
}
