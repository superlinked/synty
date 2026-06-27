// Unit-level topics. Instead of clustering the 4k-doc message firehose, cluster
// the *units of work* (sessions, PRs, issues) by a multi-vector ColBERT
// embedding of a compact per-unit text (for a session, repo and touched files
// leading and the summary appended; for a PR/issue, summary plus body — a lone
// one-liner is too thin to separate) — MaxSim kNN + Louvain, the same
// late-interaction substrate as retrieval, one level up. Near-duplicate units
// (the same work item re-run) collapse onto one representative before any
// graph work. A topic is then a coherent set of units, so its
// members/facets/label/summary are consistent by construction. Writes the
// clusters as a new rev in the current build (see readmodel); reports
// anchor-validated coherence.

use crate::community::{louvain, modularity, Graph};
use crate::store::EmbStore;
use crate::{encode::Encoder, units};
use anyhow::{ensure, Result};
use ndarray::{s, Array2, Axis};
use std::collections::HashMap;

const K: usize = 6;
const EVAL_K: usize = 16; // neighbors fetched for the quality eval (graph uses top-K)
const FLOOR: f64 = 0.6; // keep a neighbor only if ≥60% of the best neighbor's score
const MIN_CLUSTER: usize = 3;
/// Stage-1 candidates per unit for the exact-MaxSim re-rank. Pooled cosine is a
/// high-recall filter on summary-length texts; 4×EVAL_K headroom keeps the true
/// MaxSim top-EVAL_K inside the candidate set.
const CANDIDATES: usize = 64;
/// Louvain resolution scale. The base default was too coarse and produced
/// incoherent grab-bags (the resolution limit fusing weakly-connected sub-themes);
/// a finer resolution yields coherent topics. Calibrated against the anchor
/// membership eval, NOT silhouette (which prefers coarser clusters → grab-bags).
const RES_SCALE: f64 = 2.5;
/// Per-token symmetrized MaxSim at or above this is a near-duplicate — the
/// same work item re-run, not related work. Live non-dup same-topic pairs top
/// out ≈0.85, so 0.95 keeps a wide margin against union-find chaining.
const DUP: f32 = 0.95;

pub fn run(resolution: f64, model_id: &str, bucket: &str) -> Result<()> {
    let units = units::cluster_units()?;
    ensure!(
        !units.is_empty(),
        "no unit summaries yet — clustering groups units by their summaries.\nRun `synty summarize` first (or `synty build` for the whole pipeline)."
    );
    let n = units.len();
    eprintln!("topics: clustering {n} units by summary embedding");

    // Encode the per-unit text, content-addressed in the shared store
    // (encode-once per text, reused across runs/devices like doc embeddings).
    let store = EmbStore::open(bucket, model_id)?;
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

    // One kNN pass feeds both the graph (top-K) and the quality eval (full).
    eprintln!("topics: kNN over {n} summaries");
    crate::progress::phase("clustering", 0, 1);
    let mut results = maxsim_knn(&emb, EVAL_K);

    // Collapse near-duplicate units (the same work item re-run) onto one
    // representative before any graph work — a duplicate clique otherwise
    // consumes its members' edge budgets, skews every cohesion number, and
    // pads the topic prompts with the same line over and over.
    let rep = dup_groups(&emb, &results, &units);
    collapse_dups(&mut results, &rep);
    let dup_units = rep.iter().enumerate().filter(|&(i, r)| *r != i).count();
    if dup_units > 0 {
        eprintln!("topics: collapsed {dup_units} near-duplicate units");
    }

    let edges = build_edges(&results);
    // Degree in the mutual-kNN graph: a unit with no edge is an outlier we abstain
    // on rather than force-assign.
    let mut has_edge = vec![false; n];
    for &(i, j) in edges.keys() {
        has_edge[i] = true;
        has_edge[j] = true;
    }
    // Resolution is scaled UP (RES_SCALE): the default was too coarse, so the global
    // resolution limit fused weakly-connected sub-themes into incoherent grab-bags.
    // A finer resolution breaks them into coherent topics at the natural granularity
    // — judged by anchor membership (silhouette misleads here: it always prefers
    // fewer, coarser clusters, which is exactly the grab-bag failure). Agglomerative
    // re-merging was tried and dropped: coherent and grab-bag sub-themes merge at the
    // same threshold, so it can't re-coarsen selectively.
    let comm = louvain(Graph::from_edges(n, &edges), resolution * RES_SCALE);
    let q = modularity(&Graph::from_edges(n, &edges), &comm, resolution * RES_SCALE);

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &c) in comm.iter().enumerate() {
        groups.entry(c).or_default().push(i);
    }
    let mut clusters: Vec<Vec<usize>> = groups.into_values().filter(|v| v.len() >= MIN_CLUSTER).collect();
    clusters.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));

    // unit → cluster index (None if its community was < MIN_CLUSTER).
    let mut of: Vec<Option<usize>> = vec![None; n];
    for (ci, c) in clusters.iter().enumerate() {
        for &m in c {
            of[m] = Some(ci);
        }
    }

    // Outlier reassignment: move each unit that's nearer another cluster into it
    // (one simultaneous pass — no oscillation).
    let moved = reassign(&results, &mut of, &has_edge);
    if moved > 0 {
        eprintln!("topics: reassigned {moved} outlier units to their nearest cluster");
    }
    let bridges = snap_to_prs(&mut of, &units, &rep);
    if bridges > 0 {
        eprintln!("topics: snapped {bridges} sessions to their produced PR's topic");
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
    // renumbering. Read the PREVIOUS clusters (stable key → member set) from the
    // current build before writing the next rev. A new cluster inherits the
    // previous key it overlaps most (Jaccard ≥ 0.5, robust to membership drift);
    // otherwise it gets a fresh key hashed from its medoid. Greedy match — exact
    // at this cluster count.
    let cur = crate::readmodel::current()
        .ok_or_else(|| anyhow::anyhow!("no index build yet — run `synty index` (or `synty build`) first"))?;
    let prev: Vec<(String, std::collections::HashSet<String>)> = std::fs::read_to_string(cur.clusters())
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
    // For the Jaccard match a cluster's identity includes its collapsed
    // duplicates — previous rows list every unit, so comparing reps-only
    // against them would undercount overlap and shed stable keys (and with
    // them the cached summaries/names) on the first post-collapse run.
    let mut dups_of: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &r) in rep.iter().enumerate() {
        if r != i {
            dups_of.entry(r).or_default().push(i);
        }
    }
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut stable_keys: Vec<String> = Vec::with_capacity(members.len());
    let mut inherited = 0usize;
    for (ci, m) in members.iter().enumerate() {
        if m.is_empty() {
            stable_keys.push(format!("e{ci}"));
            continue;
        }
        let cur: std::collections::HashSet<&str> = m
            .iter()
            .flat_map(|&i| std::iter::once(i).chain(dups_of.get(&i).into_iter().flatten().copied()))
            .map(|i| units[i].key.as_str())
            .collect();
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

    // Centrality rank within each cluster (0 = medoid). Persisted so the
    // summarizer can lead its prompts with the most central members — the
    // theme, not whatever happens to be most recent.
    let rank = centrality_ranks(&members, &results, &of);
    let mut assign: Vec<serde_json::Value> = Vec::new();
    for (i, o) in of.iter().enumerate() {
        if let Some(ci) = o {
            assign.push(serde_json::json!({"key": units[i].key, "cluster": ci, "topic": stable_keys[*ci], "label": labels[*ci], "rank": rank[i]}));
        }
    }
    // Collapsed duplicates follow their representative: same cluster and rank,
    // plus a `dup` pointer so readers can fold them (and the summarizer can
    // skip them). A duplicate of an unclustered representative stays rowless,
    // like any unclustered unit.
    for (i, &r) in rep.iter().enumerate() {
        if r == i {
            continue;
        }
        if let Some(ci) = of[r] {
            assign.push(serde_json::json!({"key": units[i].key, "cluster": ci, "topic": stable_keys[ci], "label": labels[ci], "rank": rank[r], "dup": units[r].key}));
        }
    }
    // Clusters are a derived artifact OF a build: write the next rev as a new
    // file in the build dir (additive — never rewrites what a reader holds)
    // and repoint.
    let dest = cur.dir().join(format!("unit_clusters.{}.json", cur.rev + 1)).to_string_lossy().into_owned();
    crate::write_atomic(&dest, serde_json::to_string(&assign)?.as_bytes())?;
    crate::readmodel::repoint(&cur.build, cur.rev + 1)?;
    eprintln!("topics: wrote {dest}");
    let qual = report_quality(&results, &of, &labels, &units);
    diag(&units, &results, &of, &labels);

    // Standardized health/quality metrics. Coherence is judged by the anchor eval;
    // cohesion_med/vote_disagree are diagnostics, not decision metrics.
    let mut sizes: Vec<usize> = members.iter().map(|m| m.len()).filter(|&l| l > 0).collect();
    sizes.sort_unstable();
    let docs = units.iter().filter(|u| u.key.starts_with("gh:")).count();
    let tiny = sizes.iter().filter(|&&l| l < MIN_CLUSTER).count();
    crate::metrics::Run::new("cluster")
        .set("resolution", resolution)
        .set("units", n)
        .set("clustered", assign.len())
        .set("unclustered", n - assign.len())
        .set("clusters", sizes.len())
        .set("bridges", bridges)
        .set("dup_units", dup_units)
        .set("id_continuity", id_continuity)
        .set("modularity", q)
        .set("cohesion_med", qual.cohesion_med as f64)
        .set("misplaced", qual.misplaced)
        .set("misplaced_pct", if qual.scored > 0 { 100.0 * qual.misplaced as f64 / qual.scored as f64 } else { 0.0 })
        .set("vote_disagree", qual.vote_disagree)
        .set("grabbags", qual.grabbags)
        .set("size_min", sizes.first().copied().unwrap_or(0))
        .set("size_med", sizes.get(sizes.len() / 2).copied().unwrap_or(0))
        .set("size_max", sizes.last().copied().unwrap_or(0))
        .set("tiny", tiny)
        .set("sessions", n - docs)
        .set("docs", docs)
        .emit();
    Ok(())
}

/// One unit's neighbor list, ids sorted by descending MaxSim score (self
/// excluded). The substrate for the graph, reassignment, and quality metrics.
pub(crate) struct Knn {
    ids: Vec<usize>,
    scores: Vec<f32>,
}

/// Exact MaxSim kNN in two stages: a mean-pooled, L2-normalized vector per unit
/// ranks CANDIDATES by cosine (one blocked matmul), then true MaxSim re-scores
/// only those candidates. This replaced a PLAID index + search_batch over the
/// summaries that made a fresh cluster build search-bound; the scores the graph
/// was calibrated on (exact MaxSim) are unchanged — only the candidate pruning
/// is approximate, with 4×EVAL_K headroom.
fn maxsim_knn(emb: &[Array2<f32>], k: usize) -> Vec<Knn> {
    let n = emb.len();
    if n == 0 {
        return Vec::new();
    }
    let d = emb[0].ncols();
    let mut pooled = Array2::<f32>::zeros((n, d));
    for (i, e) in emb.iter().enumerate() {
        if e.nrows() == 0 {
            continue;
        }
        let m = e.mean_axis(Axis(0)).expect("non-empty rows");
        let norm = m.dot(&m).sqrt();
        if norm > 1e-6 {
            pooled.row_mut(i).assign(&(&m / norm));
        }
    }

    // Row-blocked so the cosine matrix never exceeds block×n, and threaded —
    // each chunk of units is independent.
    let threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).min(n);
    let chunk = n.div_ceil(threads);
    let parts: Vec<Vec<Knn>> = std::thread::scope(|scope| {
        let pooled = &pooled;
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(n));
                    let mut part = Vec::with_capacity(hi.saturating_sub(lo));
                    if lo >= hi {
                        return part;
                    }
                    let sims = pooled.slice(s![lo..hi, ..]).dot(&pooled.t());
                    for (bi, i) in (lo..hi).enumerate() {
                        let row = sims.row(bi);
                        let mut cand: Vec<usize> = (0..n).filter(|&j| j != i).collect();
                        let c = CANDIDATES.min(cand.len());
                        if c == 0 {
                            part.push(Knn { ids: Vec::new(), scores: Vec::new() });
                            continue;
                        }
                        if c < cand.len() {
                            cand.select_nth_unstable_by(c - 1, |&a, &b| {
                                row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            cand.truncate(c);
                        }
                        let mut scored: Vec<(usize, f32)> =
                            cand.into_iter().map(|j| (j, maxsim(&emb[i], &emb[j]))).collect();
                        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        scored.truncate(k);
                        part.push(Knn {
                            ids: scored.iter().map(|(j, _)| *j).collect(),
                            scores: scored.iter().map(|(_, s)| *s).collect(),
                        });
                    }
                    part
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("knn worker")).collect()
    });
    parts.into_iter().flatten().collect()
}

/// Late-interaction score: each query token's best doc-token dot, summed.
/// Also the substrate of the name-faithfulness gate (qwen).
pub(crate) fn maxsim(q: &Array2<f32>, doc: &Array2<f32>) -> f32 {
    if q.nrows() == 0 || doc.nrows() == 0 {
        return 0.0;
    }
    q.dot(&doc.t())
        .axis_iter(Axis(0))
        .map(|r| r.fold(f32::NEG_INFINITY, |m, &v| m.max(v)))
        .sum()
}

/// Optional per-unit neighbor dump (`SYNTY_DIAG=<key substring>`): for each
/// matching unit print its assigned cluster and its top MaxSim neighbors with
/// scores and their clusters — to see *why* a unit landed where it did.
fn diag(units: &[units::UnitClusterInput], results: &[Knn], of: &[Option<usize>], phrases: &[String]) {
    let Ok(want) = std::env::var("SYNTY_DIAG") else { return };
    let label = |ci: Option<usize>| ci.map(|c| phrases.get(c).cloned().unwrap_or_default()).unwrap_or_else(|| "—".into());
    for (i, u) in units.iter().enumerate() {
        if !u.key.contains(&want) {
            continue;
        }
        eprintln!("\ndiag {} → cluster {:?} [{}]", u.key, of[i], label(of[i]));
        eprintln!("  embed: {}", crate::excerpt(&u.embed, 160));
        eprintln!("  top MaxSim neighbors (score · cluster · key · embed):");
        for (id, s) in results[i].ids.iter().zip(results[i].scores.iter()).take(10) {
            let j = *id as usize;
            if j == i {
                continue;
            }
            eprintln!("    {s:.3}  c{:<3?} {}  {}", of[j], crate::short(&units[j].key), crate::excerpt(&units[j].embed, 70));
        }
    }
}

/// Group near-duplicate units (per-token symmetrized MaxSim ≥ DUP, same repo
/// and same kind — collapsing never crosses either) under one representative:
/// rep[i] == i marks a representative. Candidate pairs come from the kNN
/// lists — duplicates are nearest neighbors by definition — with a cheap
/// prefilter (one direction ≥ 2·DUP−1 implies the symmetric mean can reach
/// DUP). The representative is the member that produced a PR (it anchors the
/// group via snap_to_prs), else the smallest key — stable as reruns accumulate,
/// where "most recent" would churn the rep, the medoid, and the stable id.
fn dup_groups(emb: &[Array2<f32>], results: &[Knn], units: &[units::UnitClusterInput]) -> Vec<usize> {
    let n = results.len();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let mut parent: Vec<usize> = (0..n).collect();
    for (i, r) in results.iter().enumerate() {
        let ri = emb[i].nrows().max(1) as f32;
        for (id, s) in r.ids.iter().zip(r.scores.iter()) {
            let j = *id;
            if j == i || s / ri < 2.0 * DUP - 1.0 {
                continue;
            }
            if units[i].repo != units[j].repo
                || units[i].key.starts_with("gh:") != units[j].key.starts_with("gh:")
            {
                continue;
            }
            // Reverse direction: reuse j's kNN entry for i when present.
            let rj = emb[j].nrows().max(1) as f32;
            let back = results[j]
                .ids
                .iter()
                .position(|&x| x == i)
                .map(|p| results[j].scores[p])
                .unwrap_or_else(|| maxsim(&emb[j], &emb[i]));
            if (s / ri + back / rj) / 2.0 >= DUP {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[a] = b;
                }
            }
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }
    let mut rep: Vec<usize> = (0..n).collect();
    for (_, g) in groups {
        if g.len() < 2 {
            continue;
        }
        let r = *g.iter().min_by_key(|&&i| (units[i].linked.is_none(), &units[i].key)).expect("non-empty group");
        for &i in &g {
            rep[i] = r;
        }
    }
    rep
}

/// Remove collapsed duplicates from the kNN geometry: a non-representative
/// loses its neighbor list (degree 0 keeps it out of the graph, reassignment
/// and the quality report), and no surviving list points at one. One pass here
/// gives every consumer the corrected geometry — per-consumer skipping could
/// not fix build_edges, whose top-K budget and ÷best normalization a duplicate
/// clique would otherwise consume.
fn collapse_dups(results: &mut [Knn], rep: &[usize]) {
    for (i, r) in results.iter_mut().enumerate() {
        if rep[i] != i {
            r.ids.clear();
            r.scores.clear();
            continue;
        }
        let keep: Vec<usize> = (0..r.ids.len()).filter(|&k| rep[r.ids[k]] == r.ids[k]).collect();
        r.ids = keep.iter().map(|&k| r.ids[k]).collect();
        r.scores = keep.iter().map(|&k| r.scores[k]).collect();
    }
}

/// kNN edges from MaxSim: normalized per-unit (÷ best neighbor), floored, summed
/// over both directions so mutual neighbors weigh more. Top-K per unit.
/// Snap each session to the topic of the PR it produced — they're one unit of
/// work, so the GitHub artifact (clustered by its own content) anchors the
/// session. A hard override after reassignment, since a soft edge loses to the
/// kNN-based reassign. Collapsed duplicates are skipped (they follow their
/// representative, which is chosen linked-PR-first for exactly this reason).
/// Returns the number of sessions moved.
fn snap_to_prs(of: &mut [Option<usize>], units: &[units::UnitClusterInput], rep: &[usize]) -> usize {
    let idx: HashMap<&str, usize> = units.iter().enumerate().map(|(i, u)| (u.key.as_str(), i)).collect();
    let mut snapped = 0;
    for i in 0..units.len() {
        if rep[i] != i {
            continue;
        }
        if let Some(&j) = units[i].linked.as_deref().and_then(|pr| idx.get(pr)) {
            if of[j].is_some() && of[i] != of[j] {
                of[i] = of[j];
                snapped += 1;
            }
        }
    }
    snapped
}

fn build_edges(results: &[Knn]) -> HashMap<(usize, usize), f64> {
    let n = results.len();
    // Directed normalized weights + each unit's top-K neighbor set.
    let mut dir: HashMap<(usize, usize), f64> = HashMap::new();
    let mut topset: Vec<std::collections::HashSet<usize>> = vec![std::collections::HashSet::new(); n];
    for (i, r) in results.iter().enumerate() {
        let pairs: Vec<(usize, f32)> = r
            .ids
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
fn reassign(results: &[Knn], of: &mut [Option<usize>], has_edge: &[bool]) -> usize {
    let orig = of.to_vec();
    for _ in 0..5 {
        if reassign_once(results, of, has_edge) == 0 {
            break;
        }
    }
    (0..of.len()).filter(|&i| of[i] != orig[i]).count()
}

/// One simultaneous pass. A *clustered* unit nearer another cluster moves there;
/// an *orphan* joins its single nearest cluster — UNLESS it has no mutual-kNN
/// neighbor (degree 0), a genuine outlier we abstain on rather than force into a
/// topic it doesn't belong to (precision over coverage). Decisions read the
/// current assignment and apply at once. Returns how many changed.
fn reassign_once(results: &[Knn], of: &mut [Option<usize>], has_edge: &[bool]) -> usize {
    let orig = of.to_vec();
    let mut moved = 0;
    for (i, r) in results.iter().enumerate() {
        let (mut a, mut b, mut other) = (0.0f32, 0.0f32, None);
        for (id, s) in r.ids.iter().zip(r.scores.iter()) {
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
        if let Some(o) = other {
            // adopt an orphan only if it has a mutual neighbor (else abstain), or
            // move a clustered member that's nearer another cluster.
            let adopt = orig[i].is_none() && has_edge[i];
            let relocate = orig[i].is_some() && b > a;
            if (adopt || relocate) && of[i] != Some(o) {
                of[i] = Some(o);
                moved += 1;
            }
        }
    }
    moved
}

/// Each clustered unit's centrality rank within its cluster: members ordered
/// by summed same-cluster MaxSim, descending — rank 0 is the medoid. Unclustered
/// units keep 0; they are never written to unit_clusters.json.
fn centrality_ranks(members: &[Vec<usize>], results: &[Knn], of: &[Option<usize>]) -> Vec<usize> {
    let mut rank = vec![0usize; results.len()];
    for m in members {
        let mut ord = m.clone();
        ord.sort_by(|&a, &b| {
            same_cluster_score(b, results, of)
                .partial_cmp(&same_cluster_score(a, results, of))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (r, &i) in ord.iter().enumerate() {
            rank[i] = r;
        }
    }
    rank
}

/// The cluster's medoid — the member best connected to its co-members (max summed
/// same-cluster MaxSim). Its key seeds the cluster's stable id: a central member
/// persists across re-clusterings even as the periphery shifts.
fn medoid(members: &[usize], results: &[Knn], of: &[Option<usize>]) -> usize {
    *members
        .iter()
        .max_by(|&&a, &&b| same_cluster_score(a, results, of).partial_cmp(&same_cluster_score(b, results, of)).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(&members[0])
}

fn same_cluster_score(i: usize, results: &[Knn], of: &[Option<usize>]) -> f32 {
    results[i]
        .ids
        .iter()
        .zip(results[i].scores.iter())
        .filter(|(id, _)| **id as usize != i && of[**id as usize] == of[i])
        .map(|(_, s)| *s)
        .sum()
}

/// Cluster quality from the kNN results, all judged against the anchor eval — no
/// silhouette (it structurally prefers coarser clusters, i.e. it rewards exactly
/// the grab-bags we fixed). `misplaced` = units whose nearest neighbor sits in
/// another cluster; `cohesion_med` = median cohesion ratio ρ_C = within ÷ global
/// mean MaxSim (arXiv:2511.19350); `vote_disagree` = units whose kNN-majority
/// cluster differs from their assignment (rescale-invariant placement).
struct Quality {
    misplaced: usize,
    scored: usize,
    cohesion_med: f32,
    vote_disagree: usize,
    grabbags: usize,
}

/// Min cluster size for the cohesion median to count a cluster — tiny clusters
/// give one-pair noise that would set the floor.
const QGATE: usize = 5;
/// Min size for a cluster to appear in the lowest-cohesion debug.
const COHERENCE_MIN: usize = 8;
/// A gated cluster whose cohesion falls below this fraction of the run median
/// counts as a grab-bag — run-relative, like every placement metric here,
/// because MaxSim's scale doesn't transfer across corpora or models.
const GRABBAG_FRAC: f32 = 0.8;

fn report_quality(results: &[Knn], of: &[Option<usize>], labels: &[String], units: &[units::UnitClusterInput]) -> Quality {
    let label = |ci: usize| labels.get(ci).cloned().unwrap_or_default();
    let mut by_a: HashMap<usize, Vec<f32>> = HashMap::new(); // ci -> each member's best same-cluster MaxSim
    let mut margins: Vec<(usize, usize, usize, f32, f32)> = Vec::new(); // (unit, own ci, nearest other ci, a, b)
    let (mut top1_sum, mut top1_n, mut misplaced) = (0.0f32, 0usize, 0usize);
    for (i, r) in results.iter().enumerate() {
        let Some(ci) = of[i] else { continue };
        let (mut a, mut b, mut other, mut top1) = (None, None, ci, None);
        for (id, s) in r.ids.iter().zip(r.scores.iter()) {
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
        top1_sum += top1.unwrap_or(0.0);
        top1_n += 1;
        by_a.entry(ci).or_default().push(a);
        if b > a {
            misplaced += 1;
        }
        margins.push((i, ci, other, a, b));
    }
    let scored = margins.len();
    if scored == 0 {
        return Quality { misplaced: 0, scored: 0, cohesion_med: 0.0, vote_disagree: 0, grabbags: 0 };
    }

    // Cohesion ratio ρ_C = mean within-cluster best-neighbor MaxSim ÷ global mean.
    let global = (top1_sum / top1_n as f32).max(1e-6);
    let mut rho: Vec<(usize, usize, f32)> = by_a
        .iter()
        .map(|(ci, a)| (*ci, a.len(), (a.iter().sum::<f32>() / a.len() as f32) / global))
        .collect();
    let mut rho_gate: Vec<f32> = rho.iter().filter(|(_, n, _)| *n >= QGATE).map(|(_, _, r)| *r).collect();
    rho_gate.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cohesion_med = rho_gate.get(rho_gate.len() / 2).copied().unwrap_or(0.0);
    // Grab-bags: gated clusters well below the run median. (With a single
    // gated cluster the floor is a fraction of its own ρ, so it can never
    // self-flag — irrelevant at real corpus sizes.)
    let grabbags = rho_gate.iter().filter(|r| **r < GRABBAG_FRAC * cohesion_med).count();

    // Rescale-invariant placement: a unit whose kNN-majority cluster differs from
    // its assignment is likely misplaced regardless of the score scale.
    let mut vote_disagree = 0;
    let mut vote_mode: HashMap<usize, usize> = HashMap::new();
    for (i, r) in results.iter().enumerate() {
        let Some(ci) = of[i] else { continue };
        let mut votes: HashMap<usize, usize> = HashMap::new();
        for id in &r.ids {
            let j = *id as usize;
            if j != i {
                if let Some(cj) = of[j] {
                    *votes.entry(cj).or_default() += 1;
                }
            }
        }
        if let Some((&mode, _)) = votes.iter().max_by_key(|(_, c)| **c) {
            vote_mode.insert(i, mode);
            if mode != ci {
                vote_disagree += 1;
            }
        }
    }

    // Optional per-unit quality dump (`SYNTY_QDUMP=<path>`): one JSON row per
    // scored unit — key, embed hash (to fetch its vectors from the store),
    // embed token count, own/nearest-other cluster, same/other-cluster best
    // MaxSim, top-1 score, kNN-majority cluster. The raw material for offline
    // eval of candidate interventions (margin distributions by unit kind,
    // length bias, splitability) without bloating the metrics block.
    if let Ok(path) = std::env::var("SYNTY_QDUMP") {
        let rows: Vec<serde_json::Value> = margins
            .iter()
            .map(|(i, ci, other, a, b)| {
                serde_json::json!({
                    "key": units[*i].key,
                    "hash": format!("{:016x}", crate::index::fnv1a(units[*i].embed.as_bytes())),
                    "tokens": units[*i].embed.split_whitespace().count(),
                    "ci": ci,
                    "other": other,
                    "a": a,
                    "b": b,
                    "top1": results[*i].scores.first().copied().unwrap_or(0.0),
                    "vote": vote_mode.get(i),
                })
            })
            .collect();
        match serde_json::to_string(&rows).map_err(anyhow::Error::from).and_then(|s| crate::write_atomic(&path, s.as_bytes())) {
            Ok(()) => eprintln!("  qdump: wrote {} rows to {path}", rows.len()),
            Err(e) => eprintln!("  qdump failed: {e}"),
        }
    }

    margins.sort_by(|x, y| (x.3 - x.4).partial_cmp(&(y.3 - y.4)).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("  worst-placed units (in → would prefer):");
    for (i, ci, other, a, b) in margins.iter().take(6) {
        eprintln!("    [{:+.2}] {} — in “{}” → “{}”", a - b, crate::short(&units[*i].key), label(*ci), label(*other));
    }
    rho.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("  lowest-cohesion clusters (ρ_C · size · label):");
    for (ci, sz, r) in rho.iter().filter(|(_, n, _)| *n >= COHERENCE_MIN).take(6) {
        eprintln!("    c{ci} [ρ {r:.2}] n={sz} — {}", label(*ci));
    }
    Quality { misplaced, scored, cohesion_med, vote_disagree, grabbags }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    // Two tight groups of units: each unit's nearest neighbors must be its own
    // group, scored by exact MaxSim, nearest first.
    #[test]
    fn knn_finds_the_group_neighbors() {
        let a = arr2(&[[1.0, 0.0], [0.0, 1.0]]); // tokens on both axes
        let a2 = arr2(&[[0.9, 0.1], [0.1, 0.9]]);
        let b = arr2(&[[-1.0, 0.0], [0.0, -1.0]]);
        let b2 = arr2(&[[-0.9, -0.1], [-0.1, -0.9]]);
        let knn = maxsim_knn(&[a, a2, b, b2], 2);
        assert_eq!(knn.len(), 4);
        assert_eq!(knn[0].ids[0], 1, "unit 0's nearest is its twin");
        assert_eq!(knn[2].ids[0], 3, "unit 2's nearest is its twin");
        assert!(knn[0].scores[0] > knn[0].scores[1]);
    }

    // The pooled-candidate stage must not change the result when the candidate
    // set covers everything: two-stage kNN == brute-force MaxSim ranking.
    #[test]
    fn two_stage_matches_brute_force_maxsim() {
        // Deterministic pseudo-random token matrices (no RNG in tests).
        let units: Vec<Array2<f32>> = (0..30)
            .map(|u| {
                Array2::from_shape_fn((3 + u % 4, 8), |(r, c)| {
                    let x = ((u * 31 + r * 7 + c * 13) % 17) as f32 / 17.0 - 0.5;
                    x
                })
            })
            .collect();
        let knn = maxsim_knn(&units, 5);
        for i in 0..units.len() {
            let mut brute: Vec<(usize, f32)> = (0..units.len())
                .filter(|&j| j != i)
                .map(|j| (j, maxsim(&units[i], &units[j])))
                .collect();
            brute.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let want: Vec<usize> = brute.iter().take(5).map(|(j, _)| *j).collect();
            assert_eq!(knn[i].ids, want, "unit {i} neighbor ranking diverged");
        }
    }

    fn dup_unit(key: &str, repo: &str, linked: Option<&str>) -> units::UnitClusterInput {
        units::UnitClusterInput {
            key: key.to_string(),
            summary: String::new(),
            embed: String::new(),
            repo: repo.to_string(),
            linked: linked.map(String::from),
        }
    }

    // Re-runs of the same work item (near-identical embeds, same repo and
    // kind) collapse under one representative; merely-similar units do not.
    #[test]
    fn dup_groups_union_only_above_threshold() {
        let a = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let a2 = a.clone(); // identical → sym 1.0
        let b = arr2(&[[0.8, 0.6], [0.0, 1.0]]); // sym vs a = 0.9 < DUP
        let emb = vec![a, a2, b];
        let results = maxsim_knn(&emb, 2);
        let units: Vec<_> = ["s0", "s1", "s2"].iter().map(|k| dup_unit(k, "", None)).collect();
        assert_eq!(dup_groups(&emb, &results, &units), vec![0, 0, 2]);
    }

    // The guardrail: identical text in different repos, or a session vs a
    // GitHub doc, never collapses.
    #[test]
    fn dup_groups_respect_repo_and_kind() {
        let a = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let emb = vec![a.clone(), a.clone(), a.clone(), a];
        let results = maxsim_knn(&emb, 3);
        let units = vec![
            dup_unit("s0", "alpha", None),
            dup_unit("s1", "beta", None), // other repo
            dup_unit("gh:alpha#1", "alpha", None), // other kind
            dup_unit("s3", "alpha", None), // genuine dup of s0
        ];
        assert_eq!(dup_groups(&emb, &results, &units), vec![0, 1, 2, 0]);
    }

    // The representative anchors the group: the member that produced a PR
    // wins (snap_to_prs follows it), else the smallest key — stable as new
    // reruns accumulate.
    #[test]
    fn dup_rep_prefers_linked_pr_then_min_key() {
        let a = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let emb = vec![a.clone(), a.clone(), a.clone()];
        let results = maxsim_knn(&emb, 2);
        let linked: Vec<_> = vec![
            dup_unit("s0", "", None),
            dup_unit("s1", "", None),
            dup_unit("s2", "", Some("gh:r#1")),
        ];
        assert_eq!(dup_groups(&emb, &results, &linked), vec![2, 2, 2]);
        let unlinked: Vec<_> = ["s2", "s1", "s0"].iter().map(|k| dup_unit(k, "", None)).collect();
        assert_eq!(dup_groups(&emb, &results, &unlinked), vec![2, 2, 2]); // "s0" is at index 2
    }

    // Collapsing erases duplicates from the geometry: their lists empty, and
    // no surviving list points at one.
    #[test]
    fn collapse_dups_strips_non_reps() {
        let a = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let b = arr2(&[[0.6, 0.8], [0.8, -0.6]]);
        let emb = vec![a.clone(), a.clone(), b];
        let mut results = maxsim_knn(&emb, 2);
        let rep = vec![0, 0, 2];
        collapse_dups(&mut results, &rep);
        assert!(results[1].ids.is_empty() && results[1].scores.is_empty());
        for r in &results {
            assert!(!r.ids.contains(&1), "no list may point at a collapsed dup");
            assert_eq!(r.ids.len(), r.scores.len());
        }
    }

    // A cluster whose members barely cohere relative to the run's median is
    // flagged as a grab-bag; a tight cluster is not, and an empty run flags
    // nothing.
    #[test]
    fn grabbags_counts_low_cohesion_clusters() {
        let unit = |k: &str| units::UnitClusterInput {
            key: k.to_string(),
            summary: String::new(),
            embed: String::new(),
            repo: String::new(),
            linked: None,
        };
        let units: Vec<units::UnitClusterInput> = (0..10).map(|i| unit(&format!("u{i}"))).collect();
        // Cluster 0 (units 0-4) is tight: each member's neighbor scores 1.0.
        // Cluster 1 (units 5-9) is loose: each member's neighbor scores 0.1.
        let results: Vec<Knn> = (0..10)
            .map(|i| {
                let twin = if i < 5 { (i + 1) % 5 } else { 5 + (i - 4) % 5 };
                Knn { ids: vec![twin], scores: vec![if i < 5 { 1.0 } else { 0.1 }] }
            })
            .collect();
        let of: Vec<Option<usize>> = (0..10).map(|i| Some(if i < 5 { 0 } else { 1 })).collect();
        let q = report_quality(&results, &of, &[String::new(), String::new()], &units);
        assert_eq!(q.scored, 10);
        assert_eq!(q.grabbags, 1, "only the loose cluster is a grab-bag");
        assert_eq!(report_quality(&[], &[], &[], &[]).grabbags, 0);
    }

    // The hub of a cluster (best-connected member) gets rank 0 — its summary
    // leads the topic prompts, so the rank must reflect connectivity, not
    // insertion order.
    #[test]
    fn centrality_rank_puts_the_hub_first() {
        // Unit 1 is everyone's strong neighbor; 0 and 2 connect mainly to 1.
        let results = vec![
            Knn { ids: vec![1, 2], scores: vec![0.9, 0.2] },
            Knn { ids: vec![0, 2], scores: vec![0.9, 0.8] },
            Knn { ids: vec![1, 0], scores: vec![0.8, 0.2] },
        ];
        let of = vec![Some(0), Some(0), Some(0)];
        let rank = centrality_ranks(&[vec![0, 1, 2]], &results, &of);
        assert_eq!(rank[1], 0, "the hub is the medoid");
        assert!(rank[0] < rank[2], "stronger connectivity ranks earlier");
    }

    // maxsim: per query token, best doc-token dot, summed.
    #[test]
    fn maxsim_is_sum_of_per_token_best() {
        let q = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let d = arr2(&[[0.8, 0.0], [0.0, 0.5]]);
        assert!((maxsim(&q, &d) - 1.3).abs() < 1e-6);
        assert_eq!(maxsim(&q, &Array2::zeros((0, 2))), 0.0);
    }
}
