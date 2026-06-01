// Emergent topics with no LLM: a mutual-kNN graph over late-interaction
// similarity (each doc re-queried against the index), unioned with GitHub
// "#<num>" references within a repo, then connected components (>=3 members)
// labeled by dominant repo / labels / kind. Writes clusters.json.

use crate::{encode::Encoder, first_line, load_docs, short, Doc, DOCS_PATH, INDEX_PATH};
use anyhow::{anyhow, Result};
use next_plaid::{MmapIndex, SearchParameters};
use std::collections::{HashMap, HashSet};

const K: usize = 6;
const FLOOR: f32 = 0.6; // keep a neighbor only if its score is >=60% of the best neighbor's
const MIN_CLUSTER: usize = 3;

pub fn run(model_id: &str) -> Result<()> {
    let docs = load_docs(DOCS_PATH)?;
    let idx = MmapIndex::load(INDEX_PATH).map_err(|e| anyhow!("load index: {e}"))?;
    let mut enc = Encoder::load(model_id)?;
    let n = docs.len();

    let params = SearchParameters { top_k: K + 1, ..Default::default() };
    let mut neigh: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, d) in docs.iter().enumerate() {
        let q = enc.encode_query(&d.text)?;
        let r = idx.search(&q, &params, None).map_err(|e| anyhow!("search: {e}"))?;
        let pairs: Vec<(usize, f32)> = r
            .passage_ids
            .iter()
            .zip(r.scores.iter())
            .map(|(id, s)| (*id as usize, *s))
            .filter(|(x, _)| *x != i)
            .collect();
        let best = pairs.first().map(|(_, s)| *s).unwrap_or(0.0);
        neigh[i] = pairs.into_iter().filter(|(_, s)| *s >= FLOOR * best).take(K).map(|(x, _)| x).collect();
        if i % 500 == 0 {
            eprint!("\rknn {i}/{n}");
        }
    }
    eprintln!("\rknn {n}/{n} done");

    let links = github_links(&docs);
    let clusters = components(&neigh, &links, MIN_CLUSTER);

    let mut out = format!("# clusters: {} (>={MIN_CLUSTER} members)\n\n", clusters.len());
    let mut assign: Vec<Value> = Vec::new();
    for (ci, c) in clusters.iter().enumerate() {
        let (label, repos, kinds) = describe(&docs, c);
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
    eprintln!("wrote clusters.json ({} docs in {} clusters)", assign.len(), clusters.len());
    Ok(())
}

use serde_json::Value;

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

/// Mutual-kNN edges (i→j kept only if j→i too) unioned with `links`, then
/// connected components with at least `min` members, largest first.
fn components(neigh: &[Vec<usize>], links: &[(usize, usize)], min: usize) -> Vec<Vec<usize>> {
    let n = neigh.len();
    let nset: Vec<HashSet<usize>> = neigh.iter().map(|v| v.iter().copied().collect()).collect();
    let mut dsu = Dsu::new(n);
    for i in 0..n {
        for &j in &neigh[i] {
            if j < n && nset[j].contains(&i) {
                dsu.union(i, j);
            }
        }
    }
    for &(a, b) in links {
        if a < n && b < n {
            dsu.union(a, b);
        }
    }
    let mut comp: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        comp.entry(dsu.find(i)).or_default().push(i);
    }
    let mut out: Vec<Vec<usize>> = comp.into_values().filter(|v| v.len() >= min).collect();
    out.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));
    out
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

struct Dsu {
    p: Vec<usize>,
}
impl Dsu {
    fn new(n: usize) -> Self {
        Self { p: (0..n).collect() }
    }
    fn find(&mut self, x: usize) -> usize {
        if self.p[x] != x {
            self.p[x] = self.find(self.p[x]);
        }
        self.p[x]
    }
    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.p[a] = b;
        }
    }
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

    // A user expects only mutually-similar docs to group: 0<->1<->2 are
    // mutual, {3,4} mutual but separate, 5 is a one-way neighbor of 0 so it
    // stays out. min=3 keeps only the first group.
    #[test]
    fn mutual_knn_groups_only_reciprocated_neighbors() {
        let neigh = vec![
            vec![1, 2, 5], // 0
            vec![0, 2],    // 1
            vec![0, 1],    // 2
            vec![4],       // 3
            vec![3],       // 4
            vec![0],       // 5 — points at 0, but 0 does not reciprocate to 5? it does (5 in 0's list)
        ];
        // 5 is in 0's list AND 0 is in 5's list → mutual, so 5 joins {0,1,2}.
        let cs = components(&neigh, &[], 3);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].len(), 4); // {0,1,2,5}
        assert!(cs[0].contains(&0) && cs[0].contains(&5));
    }

    // One-way neighbors do not create an edge.
    #[test]
    fn one_way_neighbor_is_not_an_edge() {
        let neigh = vec![vec![1], vec![2], vec![1], vec![]]; // 0->1 (1 doesn't list 0)
        // mutual pairs: 1->2? 2 lists 1 and 1 lists 2 → mutual {1,2}. 0->1 one-way.
        let cs = components(&neigh, &[], 2);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0], vec![1, 2]);
    }

    // A GitHub link bridges two otherwise-separate mutual groups.
    #[test]
    fn github_link_bridges_groups() {
        let neigh = vec![vec![1], vec![0], vec![3], vec![2]]; // {0,1} and {2,3}
        let with_link = components(&neigh, &[(1, 2)], 3);
        assert_eq!(with_link.len(), 1);
        assert_eq!(with_link[0].len(), 4);
        let without = components(&neigh, &[], 3);
        assert!(without.is_empty()); // both groups are size 2 < 3
    }
}
