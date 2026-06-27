// Louvain community detection over a weighted undirected graph — the no-LLM
// topic engine. Connected-components (the kernel's first cut) transitively
// merged a homogeneous repo into one giant blob: any `#`-reference chain or
// dense semantic neighborhood unioned everything it touched. Modularity
// optimization instead asks whether a node is *more densely* tied to a
// community than chance predicts, so it resists over-merging and exposes a
// `resolution` knob: higher → more, smaller communities; lower → fewer, larger.
//
// Graph representation: `adj[i]` lists `(neighbor, weight)` for every incident
// edge. A normal undirected edge {i,j} is stored in both `adj[i]` and `adj[j]`;
// a self-loop is stored once in `adj[i]`. Weighted degree `k_i = Σ adj[i]` then
// satisfies `Σ_i k_i = 2m`, and the aggregation step below preserves it.

use std::collections::HashMap;

/// Weighted undirected graph for community detection.
pub struct Graph {
    pub n: usize,
    adj: Vec<Vec<(usize, f64)>>,
}

impl Graph {
    pub fn new(n: usize) -> Self {
        Self { n, adj: vec![Vec::new(); n] }
    }

    /// Build from an accumulated edge map keyed by an unordered `(a,b)` pair
    /// (`a <= b`); `a == b` is a self-loop. Weights are taken verbatim.
    /// Adjacency lists are sorted: the map iterates in hash order, which varies
    /// per process, and float accumulation downstream (to_comm sums, aggregate)
    /// follows adjacency order — sorting makes a partition reproducible across
    /// machines from the same edges.
    pub fn from_edges(n: usize, edges: &HashMap<(usize, usize), f64>) -> Self {
        let mut g = Graph::new(n);
        for (&(a, b), &w) in edges {
            if w == 0.0 {
                continue;
            }
            if a == b {
                g.adj[a].push((a, w));
            } else {
                g.adj[a].push((b, w));
                g.adj[b].push((a, w));
            }
        }
        for adj in &mut g.adj {
            adj.sort_by(|x, y| x.0.cmp(&y.0));
        }
        g
    }

    fn degrees(&self) -> Vec<f64> {
        self.adj.iter().map(|e| e.iter().map(|&(_, w)| w).sum()).collect()
    }

    /// 2m — total weighted degree (sum of all `k_i`).
    fn two_m(&self) -> f64 {
        self.adj.iter().flat_map(|e| e.iter()).map(|&(_, w)| w).sum()
    }
}

/// One Louvain level: greedily move nodes to the neighbor community that most
/// improves modularity until no move helps. Returns `(community_of_node,
/// any_move_made)`. Node order is fixed (0..n) and ties prefer the current /
/// then lowest community id, so the result is deterministic.
fn one_level(g: &Graph, resolution: f64) -> (Vec<usize>, bool) {
    let n = g.n;
    let k = g.degrees();
    let two_m = g.two_m();
    let mut comm: Vec<usize> = (0..n).collect();
    let mut sigma_tot = k.clone(); // each node alone → Σ_tot[c] = k[c]
    if two_m == 0.0 {
        return (comm, false);
    }

    let mut any_move = false;
    loop {
        let mut moved = false;
        for i in 0..n {
            let ci = comm[i];
            let ki = k[i];
            // Tentatively isolate i from its community.
            sigma_tot[ci] -= ki;

            // Sum edge weight from i into each neighboring community (self-loop
            // excluded — it is internal to i, not an attraction to anyone).
            let mut to_comm: HashMap<usize, f64> = HashMap::new();
            for &(j, w) in &g.adj[i] {
                if j != i {
                    *to_comm.entry(comm[j]).or_insert(0.0) += w;
                }
            }

            // Gain of joining community c (relative to staying isolated, whose
            // gain is 0): w_{i→c} - resolution · Σ_tot[c] · k_i / 2m.
            let gain = |c: usize, w_ic: f64| w_ic - resolution * sigma_tot[c] * ki / two_m;
            let mut best_c = ci;
            let mut best_gain = gain(ci, to_comm.get(&ci).copied().unwrap_or(0.0));
            for (&c, &w_ic) in &to_comm {
                let gn = gain(c, w_ic);
                // Strict improvement, or equal gain but a lower community id —
                // keeps the partition deterministic and stable.
                if gn > best_gain || (gn == best_gain && c < best_c) {
                    best_gain = gn;
                    best_c = c;
                }
            }

            sigma_tot[best_c] += ki;
            comm[i] = best_c;
            if best_c != ci {
                moved = true;
                any_move = true;
            }
        }
        if !moved {
            break;
        }
    }
    (comm, any_move)
}

/// Relabel arbitrary community ids to a contiguous `0..k` range, in order of
/// first appearance (deterministic). Returns `(map_from_old_id, k)`.
fn contiguous(comm: &[usize]) -> (HashMap<usize, usize>, usize) {
    let mut map = HashMap::new();
    let mut next = 0;
    for &c in comm {
        map.entry(c).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
    }
    (map, next)
}

/// Collapse each community into a super-node. Edge weight between super-nodes C
/// and D is `Σ_{i∈C, j∈D} A_ij` (internal edges fold into a C self-loop). This
/// preserves every super-node's weighted degree as `Σ_{i∈C} k_i`, which is what
/// lets Louvain recurse correctly.
fn aggregate(g: &Graph, comm: &[usize], relabel: &HashMap<usize, usize>, k: usize) -> Graph {
    let mut acc: Vec<HashMap<usize, f64>> = vec![HashMap::new(); k];
    for i in 0..g.n {
        let ci = relabel[&comm[i]];
        for &(j, w) in &g.adj[i] {
            let cj = relabel[&comm[j]];
            *acc[ci].entry(cj).or_insert(0.0) += w;
        }
    }
    // `acc` is a directed accumulation. For C≠D it is symmetric (acc[C][D] ==
    // acc[D][C] == the undirected weight A_CD), so take each off-diagonal pair
    // once from the c<d side; the diagonal acc[C][C] is the super-node's
    // self-loop and is kept whole. This keeps degree[C] == Σ_{i∈C} k_i.
    let mut edges: HashMap<(usize, usize), f64> = HashMap::new();
    for c in 0..k {
        for (&d, &w) in &acc[c] {
            if d >= c {
                edges.insert((c, d), w);
            }
        }
    }
    Graph::from_edges(k, &edges)
}

/// Run Louvain to convergence. Returns one community id per original node,
/// contiguous `0..num_communities`. `resolution` defaults to 1.0 at the call
/// site; >1 yields finer communities, <1 coarser.
pub fn louvain(graph: Graph, resolution: f64) -> Vec<usize> {
    let n = graph.n;
    if n == 0 {
        return Vec::new();
    }
    // labels[orig] = current super-node id of orig in the working graph `g`.
    let mut labels: Vec<usize> = (0..n).collect();
    let mut g = graph;
    loop {
        let (comm, improved) = one_level(&g, resolution);
        let (relabel, k) = contiguous(&comm);
        for orig in 0..n {
            labels[orig] = relabel[&comm[labels[orig]]];
        }
        if !improved || k == g.n {
            break;
        }
        g = aggregate(&g, &comm, &relabel, k);
    }
    labels
}

/// Modularity of a partition (resolution-aware). Used by tests to confirm a
/// split beats the trivial one-community partition.
pub fn modularity(g: &Graph, comm: &[usize], resolution: f64) -> f64 {
    let two_m = g.two_m();
    if two_m == 0.0 {
        return 0.0;
    }
    let k = g.degrees();
    // Σ_in per community (internal weight, both directions) and Σ_tot.
    let mut internal: HashMap<usize, f64> = HashMap::new();
    let mut tot: HashMap<usize, f64> = HashMap::new();
    for i in 0..g.n {
        *tot.entry(comm[i]).or_insert(0.0) += k[i];
        for &(j, w) in &g.adj[i] {
            if comm[j] == comm[i] {
                *internal.entry(comm[i]).or_insert(0.0) += w; // self-loops + both dirs
            }
        }
    }
    let mut q = 0.0;
    for (&c, &in_w) in &internal {
        let st = tot.get(&c).copied().unwrap_or(0.0);
        q += in_w / two_m - resolution * (st / two_m) * (st / two_m);
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(n: usize, es: &[(usize, usize, f64)]) -> Graph {
        let mut m = HashMap::new();
        for &(a, b, w) in es {
            *m.entry((a.min(b), a.max(b))).or_insert(0.0) += w;
        }
        Graph::from_edges(n, &m)
    }

    // Two cliques joined by a single weak edge are two communities, not one —
    // the failure mode connected-components had (the weak bridge unioned them).
    #[test]
    fn two_cliques_one_bridge_split() {
        let g = graph(
            6,
            &[
                (0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), // clique {0,1,2}
                (3, 4, 1.0), (4, 5, 1.0), (3, 5, 1.0), // clique {3,4,5}
                (2, 3, 0.1),                            // weak bridge
            ],
        );
        let c = louvain(graph(6, &edges_of(&g)), 1.0);
        assert_eq!(c[0], c[1]);
        assert_eq!(c[1], c[2]);
        assert_eq!(c[3], c[4]);
        assert_eq!(c[4], c[5]);
        assert_ne!(c[0], c[3], "the two cliques must not merge over a weak bridge");
    }

    // A single clique is one community.
    #[test]
    fn one_clique_is_one_community() {
        let c = louvain(graph(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 2, 1.0), (1, 3, 1.0), (0, 3, 1.0)]), 1.0);
        assert!(c.iter().all(|&x| x == c[0]));
    }

    // The resolution knob does something monotone: very high resolution
    // fragments the two-clique-bridge graph into more communities than very low.
    #[test]
    fn resolution_controls_granularity() {
        let es = [
            (0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0),
            (3, 4, 1.0), (4, 5, 1.0), (3, 5, 1.0),
            (2, 3, 0.5),
        ];
        let low = num_communities(&louvain(graph(6, &es), 0.2));
        let high = num_communities(&louvain(graph(6, &es), 3.0));
        assert!(high >= low, "higher resolution should not coarsen ({high} < {low})");
        assert!(high >= 2);
    }

    // Found partition beats the trivial all-in-one partition on modularity.
    #[test]
    fn split_beats_trivial_on_modularity() {
        let g = graph(6, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), (3, 4, 1.0), (4, 5, 1.0), (3, 5, 1.0), (2, 3, 0.1)]);
        let trivial = vec![0usize; 6];
        let found = louvain(graph(6, &edges_of(&g)), 1.0);
        assert!(modularity(&g, &found, 1.0) > modularity(&g, &trivial, 1.0));
    }

    // Disconnected components never share a community.
    #[test]
    fn disconnected_nodes_separate() {
        let c = louvain(graph(4, &[(0, 1, 1.0), (2, 3, 1.0)]), 1.0);
        assert_eq!(c[0], c[1]);
        assert_eq!(c[2], c[3]);
        assert_ne!(c[0], c[2]);
    }

    fn num_communities(c: &[usize]) -> usize {
        let mut s: Vec<usize> = c.to_vec();
        s.sort_unstable();
        s.dedup();
        s.len()
    }

    // Recover an edge list from a graph (each undirected edge once) for rebuilds
    // in tests.
    fn edges_of(g: &Graph) -> Vec<(usize, usize, f64)> {
        let mut out = Vec::new();
        for i in 0..g.n {
            for &(j, w) in &g.adj[i] {
                if i <= j {
                    out.push((i, j, w));
                }
            }
        }
        out
    }
}
