// Retrieval probe harness, two parts:
//  - the fixed, corpus-grounded query set (some with metadata filters), top-5
//    per query to evals/runs.md + stdout for human scoring;
//  - derived gold pairs: every session whose tracker recorded the PR it
//    produced is a free relevance judgment — querying with the session's ask
//    must retrieve that PR. Scored automatically (hit@5), grows with the
//    corpus instead of being frozen to one snapshot.

use crate::{encode::Encoder, load_docs, readmodel, search, units};
use anyhow::{anyhow, Result};
use next_plaid::{MmapIndex, SearchParameters};
use serde::Deserialize;

const QUERIES_PATH: &str = "evals/queries.json";

/// One probe: a query and an optional `col=value` metadata filter. The gold set
/// is corpus-specific, so it lives in `evals/queries.json` (gitignored), not the
/// binary — each corpus brings its own.
#[derive(Deserialize)]
struct Probe {
    query: String,
    #[serde(default)]
    filter: Option<String>,
}

pub fn run(model_id: &str) -> Result<()> {
    let probes: Vec<Probe> = match std::fs::read_to_string(QUERIES_PATH) {
        Ok(s) => serde_json::from_str(&s).map_err(|e| anyhow!("{QUERIES_PATH}: {e}"))?,
        Err(_) => {
            eprintln!(r#"eval: no {QUERIES_PATH} — create it as [{{"query": "…", "filter": "repo=foo"}}, …] (filter optional)"#);
            Vec::new()
        }
    };
    let docs = load_docs(readmodel::docs_path())?;
    let idx = MmapIndex::load(&readmodel::index_dir().to_string_lossy())
        .map_err(|e| anyhow!("load index: {e}"))?;
    let mut enc = Encoder::load(model_id)?;
    let mut out = String::from("# retrieval probe runs\n\n");
    for p in &probes {
        let filt = p.filter.as_deref();
        let qe = enc.encode_query(&p.query)?;
        let subset = search::subset_for(filt)?;
        let params = SearchParameters { top_k: 5, ..Default::default() };
        let r = idx.search(&qe, &params, subset.as_deref()).map_err(|e| anyhow!("search: {e}"))?;
        out.push_str(&search::render(&docs, &p.query, filt, &r));
        out.push('\n');
    }

    // Derived session→PR pairs, scored without a human in the loop.
    let pairs = derived_pairs()?;
    let (mut hits, mut misses) = (0usize, Vec::new());
    for (query, key) in &pairs {
        let qe = enc.encode_query(query)?;
        let params = SearchParameters { top_k: 5, ..Default::default() };
        let r = idx.search(&qe, &params, None).map_err(|e| anyhow!("search: {e}"))?;
        let hit = r.passage_ids.iter().any(|id| {
            docs.get(*id as usize)
                .is_some_and(|d| units::gh_key(&d.meta.repo, d.meta.number.unwrap_or(0)) == *key)
        });
        if hit {
            hits += 1;
        } else {
            misses.push((query.clone(), key.clone()));
        }
    }
    if !pairs.is_empty() {
        out.push_str(&format!("## derived session→PR pairs: {hits}/{} hit@5\n", pairs.len()));
        for (q, key) in &misses {
            out.push_str(&format!("- MISS {key} ← “{}”\n", crate::excerpt(q, 100)));
        }
        out.push('\n');
    }

    std::fs::create_dir_all("evals").ok();
    std::fs::write("evals/runs.md", &out)?;
    print!("{out}");
    let mut m = crate::metrics::Run::new("eval");
    m.set("probes", probes.len()).set("derived_pairs", pairs.len()).set("derived_hits", hits).set(
        "derived_hit_rate",
        if pairs.is_empty() { 1.0 } else { hits as f64 / pairs.len() as f64 },
    );
    m.emit();
    eprintln!("wrote evals/runs.md ({} probes, {} derived pairs)", probes.len(), pairs.len());
    Ok(())
}

/// (query, gold gh: key) pairs from sessions that produced a PR: the query is
/// the session's opening ask — the user's own words, the realistic probe.
fn derived_pairs() -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    for s in units::sessions()? {
        let Some(key) = s.linked_pr.as_deref().and_then(units::linked_pr_key) else { continue };
        if s.ask.len() >= 12 {
            out.push((s.ask.clone(), key));
        }
    }
    Ok(out)
}
