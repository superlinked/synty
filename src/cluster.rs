// Emergent topics with no LLM. A weighted similarity graph drives Louvain
// community detection (`community.rs`):
//   - kNN edges: each doc's cached embedding is queried against the index;
//     neighbor scores are normalized per-doc (÷ best neighbor) to strip the
//     length bias of token-sum MaxSim, floored, and summed across both
//     directions so mutual neighbors weigh more than one-way ones.
//   - GitHub "#<num>" references within a repo add a fixed-weight edge — a
//     *signal*, not the hard transitive union that previously merged a whole
//     homogeneous repo into one blob.
// Louvain then optimizes modularity with a `--resolution` knob (higher → more,
// smaller topics). Embeddings are reused from the index, so re-clustering (e.g.
// a resolution sweep) never re-encodes. Writes clusters.json.

use crate::community::{louvain, Graph};
use crate::{first_line, load_docs, short, Doc, DOCS_PATH, INDEX_PATH};
use anyhow::{anyhow, Result};
use next_plaid::{MmapIndex, SearchParameters};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

const K: usize = 6;
const FLOOR: f64 = 0.6; // keep a neighbor only if its normalized score ≥ 60% of the best
const LINK_WEIGHT: f64 = 1.0; // weight of one GitHub #-reference edge
const MIN_CLUSTER: usize = 3;

pub fn run(resolution: f64) -> Result<()> {
    let docs = load_docs(DOCS_PATH)?;
    let idx = MmapIndex::load(INDEX_PATH).map_err(|e| anyhow!("load index: {e} (run `index` first)"))?;
    let emb = next_plaid::update::load_embeddings_npy(Path::new(INDEX_PATH))
        .map_err(|e| anyhow!("load embeddings cache: {e} (re-run `index`)"))?;
    anyhow::ensure!(
        emb.len() == docs.len(),
        "embeddings cache ({}) != docs ({}); re-run `index`",
        emb.len(),
        docs.len()
    );
    let n = docs.len();

    let edges = load_or_build_edges(&idx, &emb, &docs)?;
    let comm = louvain(Graph::from_edges(n, &edges), resolution);
    let q = crate::community::modularity(&Graph::from_edges(n, &edges), &comm, resolution);

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &c) in comm.iter().enumerate() {
        groups.entry(c).or_default().push(i);
    }
    let mut clusters: Vec<Vec<usize>> =
        groups.into_values().filter(|v| v.len() >= MIN_CLUSTER).collect();
    clusters.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));

    // Extractive c-TF-IDF labels need every cluster's text at once.
    let cluster_texts: Vec<Vec<&str>> =
        clusters.iter().map(|c| c.iter().map(|&m| docs[m].text.as_str()).collect()).collect();
    let phrases = crate::keyphrase::labels(&cluster_texts, 4);

    let mut out = format!(
        "# clusters: {} (>={MIN_CLUSTER} members · resolution {resolution} · modularity {q:.3})\n\n",
        clusters.len()
    );
    let mut assign: Vec<Value> = Vec::new();
    for (ci, c) in clusters.iter().enumerate() {
        let (fallback, repos, kinds) = describe(&docs, c);
        let label = if phrases[ci].is_empty() { fallback } else { phrases[ci].join(", ") };
        out.push_str(&format!("## C{ci} — {label}  ({} docs)\n", c.len()));
        out.push_str(&format!("repos: {repos} · kinds: {kinds}\n"));
        for &m in c.iter().take(6) {
            out.push_str(&format!("- {}\n", title_of(&docs[m])));
        }
        out.push('\n');
        for &m in c {
            assign.push(serde_json::json!({"id": docs[m].id, "cluster": ci, "label": label}));
        }
    }
    print!("{out}");
    std::fs::write("clusters.json", serde_json::to_string(&assign)?)?;
    eprintln!(
        "wrote clusters.json ({} docs in {} clusters)",
        assign.len(),
        clusters.len()
    );
    Ok(())
}

/// The weighted graph is a function only of (docs, embeddings), both fixed once
/// `index` has run — so cache it next to the index (which `index` wipes on
/// rebuild, invalidating the cache for free). A resolution sweep then re-runs
/// only Louvain (milliseconds), never the ~1.8k index searches.
#[derive(serde::Serialize, serde::Deserialize)]
struct EdgeCache {
    n: usize,
    edges: Vec<(usize, usize, f64)>,
}

fn cache_path() -> std::path::PathBuf {
    Path::new(INDEX_PATH).join("knn_edges.json")
}

fn load_or_build_edges(
    idx: &MmapIndex,
    emb: &[ndarray::Array2<f32>],
    docs: &[Doc],
) -> Result<HashMap<(usize, usize), f64>> {
    if let Ok(raw) = std::fs::read_to_string(cache_path()) {
        if let Ok(c) = serde_json::from_str::<EdgeCache>(&raw) {
            if c.n == docs.len() {
                eprintln!("cluster: reusing cached graph ({} edges)", c.edges.len());
                return Ok(c.edges.into_iter().map(|(a, b, w)| ((a, b), w)).collect());
            }
        }
    }
    let edges = build_graph_edges(idx, emb, docs)?;
    let c = EdgeCache {
        n: docs.len(),
        edges: edges.iter().map(|(&(a, b), &w)| (a, b, w)).collect(),
    };
    let _ = std::fs::write(cache_path(), serde_json::to_string(&c)?);
    Ok(edges)
}

/// Build the weighted edge map feeding Louvain: normalized+floored kNN
/// similarity summed over both directions, plus fixed-weight GitHub links.
/// kNN runs as one parallel batched search rather than 1.8k sequential ones.
fn build_graph_edges(
    idx: &MmapIndex,
    emb: &[ndarray::Array2<f32>],
    docs: &[Doc],
) -> Result<HashMap<(usize, usize), f64>> {
    let n = docs.len();
    // Coarse topic kNN only needs the few nearest neighbors, so approximate
    // aggressively: probe fewer IVF cells and fully MaxSim-score far fewer
    // candidates than retrieval's default (4096). This is the dominant cost of
    // a fresh build; 256/4 cuts it ~16× with negligible effect on the top-K.
    let params = SearchParameters {
        top_k: K + 1,
        n_full_scores: 256,
        n_ivf_probe: 4,
        ..Default::default()
    };
    eprintln!("cluster: kNN over {n} docs (parallel batched search)…");
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
            let d = (s / best) as f64; // normalized directed weight in (0,1]
            if d < FLOOR {
                continue;
            }
            *edges.entry((i.min(j), i.max(j))).or_insert(0.0) += d;
        }
    }

    for (a, b) in github_links(docs) {
        *edges.entry((a.min(b), a.max(b))).or_insert(0.0) += LINK_WEIGHT;
    }
    Ok(edges)
}

/// GitHub "#<num>" references between docs in the same repo → edges.
fn github_links(docs: &[Doc]) -> Vec<(usize, usize)> {
    let mut bynum: HashMap<(&str, i64), usize> = HashMap::new();
    for (i, d) in docs.iter().enumerate() {
        if let Some(nu) = d.meta.number {
            bynum.insert((d.meta.repo.as_str(), nu), i);
        }
    }
    let mut edges = Vec::new();
    for (i, d) in docs.iter().enumerate() {
        if d.meta.number.is_none() {
            continue;
        }
        for nu in hash_refs(&d.text) {
            if let Some(&j) = bynum.get(&(d.meta.repo.as_str(), nu)) {
                if j != i {
                    edges.push((i, j));
                }
            }
        }
    }
    edges
}

fn hash_refs(s: &str) -> Vec<i64> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'#' {
            let mut j = i + 1;
            let mut num: i64 = 0;
            let mut any = false;
            while j < b.len() && b[j].is_ascii_digit() {
                num = num * 10 + (b[j] - b'0') as i64;
                j += 1;
                any = true;
            }
            if any {
                out.push(num);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

fn describe(docs: &[Doc], c: &[usize]) -> (String, String, String) {
    let mut repo: HashMap<String, usize> = HashMap::new();
    let mut kind: HashMap<String, usize> = HashMap::new();
    let mut label: HashMap<String, usize> = HashMap::new();
    for &i in c {
        *repo.entry(docs[i].meta.repo.clone()).or_default() += 1;
        *kind.entry(docs[i].meta.kind.clone()).or_default() += 1;
        for l in &docs[i].meta.labels {
            *label.entry(l.clone()).or_default() += 1;
        }
    }
    let top = |m: &HashMap<String, usize>, k: usize| {
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        v.into_iter().take(k).map(|(s, n)| format!("{s}({n})")).collect::<Vec<_>>().join(", ")
    };
    let lbl = if label.is_empty() { top(&repo, 1) } else { top(&label, 2) };
    (lbl, top(&repo, 3), top(&kind, 3))
}

fn title_of(d: &Doc) -> String {
    match d.meta.kind.as_str() {
        "pull_request" | "issue" => {
            format!("{} {}#{} {}", d.meta.kind, d.meta.repo, d.meta.number.unwrap_or(0), first_line(&d.text))
        }
        _ => format!(
            "{} {} \"{}\"",
            d.meta.kind,
            short(&d.meta.session_id),
            first_line(&d.text).chars().take(80).collect::<String>()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // GitHub-style references are picked up; prose numbers and bare # are not.
    #[test]
    fn hash_refs_finds_issue_numbers() {
        assert_eq!(hash_refs("Replaces #698, closes #53."), vec![698, 53]);
        assert_eq!(hash_refs("bumped to v0.4.1 today"), Vec::<i64>::new());
        assert_eq!(hash_refs("a # b"), Vec::<i64>::new());
        assert_eq!(hash_refs("see #12 and again #12"), vec![12, 12]);
    }

    // Same-repo #-references become edges; cross-repo numbers and self-refs do
    // not. (Two PRs in sie-internal referencing each other → one edge.)
    #[test]
    fn github_links_same_repo_only() {
        let mk = |id: i64, num: i64, repo: &str, text: &str| Doc {
            id,
            text: text.into(),
            meta: crate::Meta {
                source: "github".into(),
                kind: "pull_request".into(),
                repo: repo.into(),
                author: String::new(),
                session_id: String::new(),
                ts: String::new(),
                number: Some(num),
                url: None,
                state: None,
                labels: vec![],
            },
        };
        let docs = vec![
            mk(0, 10, "sie-internal", "closes #11"),     // → doc 1 (same repo)
            mk(1, 11, "sie-internal", "follow-up"),       // referenced
            mk(2, 11, "sie-web", "different repo same #"), // not linked (other repo)
        ];
        let mut e = github_links(&docs);
        e.sort();
        assert_eq!(e, vec![(0, 1)]);
    }
}
