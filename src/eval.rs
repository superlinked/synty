// Retrieval probe harness: run a fixed, corpus-grounded query set (some with
// metadata filters) and emit the top-5 per query to eval_runs.md + stdout for
// scoring. Queries span GitHub topics and session topics.

use crate::{encode::Encoder, load_docs, search, DOCS_PATH, INDEX_PATH};
use anyhow::{anyhow, Result};
use next_plaid::{MmapIndex, SearchParameters};

const QUERIES: &[(&str, Option<&str>)] = &[
    ("OCR document parsing adapter MinerU docling", None),
    ("close security vulnerabilities dependabot CodeQL alerts", None),
    ("generation benchmark model matrix quality eval", None),
    ("gateway generation isolation guardrails", None),
    ("GCP monitoring terraform SOC2 firestore alert policies", None),
    ("readme docker quickstart deployment dead links", None),
    ("dense model loader passes dim to adapters", Some("repo=sie-internal")),
    ("fix docs search after client navigation", Some("repo=sie-web")),
    ("VLM cache clears on uncovered paths", Some("kind=pull_request")),
    ("kind smoke test regression config service", None),
    ("synty edge agent event envelope tailer", Some("source=agent")),
    ("late interaction embeddings pylate next-plaid rust", Some("source=agent")),
];

pub fn run(model_id: &str) -> Result<()> {
    let docs = load_docs(DOCS_PATH)?;
    let idx = MmapIndex::load(INDEX_PATH).map_err(|e| anyhow!("load index: {e}"))?;
    let mut enc = Encoder::load(model_id)?;
    let mut out = String::from("# retrieval probe runs\n\n");
    for (q, filt) in QUERIES {
        let qe = enc.encode_query(q)?;
        let subset = search::subset_for(*filt)?;
        let params = SearchParameters { top_k: 5, ..Default::default() };
        let r = idx
            .search(&qe, &params, subset.as_deref())
            .map_err(|e| anyhow!("search: {e}"))?;
        out.push_str(&search::render(&docs, q, *filt, &r));
        out.push('\n');
    }
    std::fs::write("eval_runs.md", &out)?;
    print!("{out}");
    eprintln!("wrote eval_runs.md ({} queries)", QUERIES.len());
    Ok(())
}
