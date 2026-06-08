// Retrieval probe harness: run a fixed, corpus-grounded query set (some with
// metadata filters) and emit the top-5 per query to eval_runs.md + stdout for
// scoring. Queries span GitHub topics and session topics.

use crate::{encode::Encoder, load_docs, search, DOCS_PATH, INDEX_PATH};
use anyhow::{anyhow, Result};
use next_plaid::{MmapIndex, SearchParameters};
use serde::Deserialize;

const QUERIES_PATH: &str = "eval_queries.json";

/// One probe: a query and an optional `col=value` metadata filter. The gold set
/// is corpus-specific, so it lives in `eval_queries.json` (gitignored), not the
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
            return Ok(());
        }
    };
    let docs = load_docs(DOCS_PATH)?;
    let idx = MmapIndex::load(INDEX_PATH).map_err(|e| anyhow!("load index: {e}"))?;
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
    std::fs::write("eval_runs.md", &out)?;
    print!("{out}");
    eprintln!("wrote eval_runs.md ({} queries)", probes.len());
    Ok(())
}
