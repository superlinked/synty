// Filtered semantic search over the next-plaid index. `--filter` is a SQL
// WHERE clause over metadata columns (repo/source/kind/author/state/...),
// resolved to a doc-id subset via next-plaid's filtering module, then passed
// to MaxSim search. Renders Markdown to stdout — the agent-facing surface.

use crate::{encode::Encoder, excerpt, first_line, load_docs, short, Doc, DOCS_PATH, INDEX_PATH};
use anyhow::{anyhow, Result};
use next_plaid::{MmapIndex, QueryResult, SearchParameters};

/// Filter is `column=value` (e.g. `repo=sie-web`); bound as a parameter so the
/// next-plaid WHERE validator accepts it.
pub fn subset_for(filter: Option<&str>) -> Result<Option<Vec<i64>>> {
    let Some(f) = filter else { return Ok(None) };
    let (col, val) = f.split_once('=').ok_or_else(|| anyhow!("filter must be col=value: {f}"))?;
    let ids = next_plaid::filtering::where_condition(
        INDEX_PATH,
        &format!("{} = ?", col.trim()),
        &[serde_json::json!(val.trim())],
    )
    .map_err(|e| anyhow!("filter `{f}`: {e}"))?;
    Ok(Some(ids))
}

pub fn run(query: &str, filter: Option<&str>, k: usize, model_id: &str) -> Result<()> {
    let docs = load_docs(DOCS_PATH)?;
    let idx = MmapIndex::load(INDEX_PATH)
        .map_err(|e| anyhow!("load index: {e} (run `index` first)"))?;
    let mut enc = Encoder::load(model_id)?;
    let q = enc.encode_query(query)?;
    let subset = subset_for(filter)?;
    let params = SearchParameters { top_k: k, ..Default::default() };
    let res = idx
        .search(&q, &params, subset.as_deref())
        .map_err(|e| anyhow!("search: {e}"))?;
    print!("{}", render(&docs, query, filter, &res));
    Ok(())
}

/// Render a result set as Markdown. Shared with the eval harness.
pub fn render(docs: &[Doc], query: &str, filter: Option<&str>, res: &QueryResult) -> String {
    let mut o = format!("## {query}\n");
    if let Some(c) = filter {
        o.push_str(&format!("_filter: `{c}`_\n"));
    }
    o.push('\n');
    if res.passage_ids.is_empty() {
        o.push_str("_(no results)_\n");
        return o;
    }
    for (rank, (id, score)) in res.passage_ids.iter().zip(res.scores.iter()).enumerate() {
        if let Some(d) = docs.get(*id as usize) {
            o.push_str(&card(d, rank + 1, *score));
        }
    }
    o
}

fn card(d: &Doc, rank: usize, score: f32) -> String {
    match d.meta.kind.as_str() {
        "pull_request" | "issue" => {
            let url = d.meta.url.clone().unwrap_or_default();
            format!(
                "{rank}. [{score:.1}] **{} {}#{}** — {}\n   {} · {}\n",
                d.meta.kind,
                d.meta.repo,
                d.meta.number.unwrap_or(0),
                first_line(&d.text),
                d.meta.state.clone().unwrap_or_default(),
                url
            )
        }
        _ => format!(
            "{rank}. [{score:.1}] _{} · {} · {}_\n   {}\n",
            d.meta.kind,
            if d.meta.repo.is_empty() { "local" } else { &d.meta.repo },
            short(&d.meta.session_id),
            excerpt(&d.text, 160)
        ),
    }
}
