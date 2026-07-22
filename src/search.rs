// Filtered semantic search over the next-plaid index. `--filter` is a SQL
// WHERE clause over metadata columns (repo/source/kind/author/state/...),
// resolved to a doc-id subset via next-plaid's filtering module, then passed
// to MaxSim search. Renders Markdown to stdout — the agent-facing surface.

use crate::{encode::Encoder, excerpt, first_line, load_docs, readmodel, short, Doc};
use anyhow::{anyhow, Result};
use next_plaid::{MmapIndex, QueryResult, SearchParameters};

/// Filter is `column=value` (e.g. `repo=sie-web`); bound as a parameter so the
/// next-plaid WHERE validator accepts it.
pub fn subset_for(filter: Option<&str>) -> Result<Option<Vec<i64>>> {
    subset_for_scope(filter, &crate::policy::ReadScope::default())
}

pub fn subset_for_scope(
    filter: Option<&str>,
    scope: &crate::policy::ReadScope,
) -> Result<Option<Vec<i64>>> {
    let mut clauses = Vec::new();
    let mut params = Vec::new();
    if let Some(f) = filter {
        let (col, val) =
            f.split_once('=').ok_or_else(|| anyhow!("filter must be col=value: {f}"))?;
        clauses.push(format!("{} = ?", col.trim()));
        params.push(serde_json::json!(val.trim()));
    }
    append_scope_clause(&mut clauses, &mut params, "repo", &scope.repos);
    append_scope_clause(&mut clauses, &mut params, "campaign_id", &scope.campaigns);
    append_scope_clause(&mut clauses, &mut params, "campaign_role", &scope.roles);
    append_source_scope_clause(&mut clauses, &mut params, &scope.sources);
    if clauses.is_empty() {
        return Ok(None);
    }
    let ids = next_plaid::filtering::where_condition(
        &readmodel::index_dir().to_string_lossy(),
        &clauses.join(" AND "),
        &params,
    )
    .map_err(|e| anyhow!("filter: {e}"))?;
    Ok(Some(ids))
}

fn append_source_scope_clause(
    clauses: &mut Vec<String>,
    params: &mut Vec<serde_json::Value>,
    values: &[String],
) {
    if values.is_empty() {
        return;
    }
    let placeholders = std::iter::repeat_n("?", values.len()).collect::<Vec<_>>().join(",");
    clauses.push(format!("(capture_source IN ({placeholders}) OR (capture_source = '' AND source IN ({placeholders})))"));
    params.extend(values.iter().map(|value| serde_json::json!(value)));
    params.extend(values.iter().map(|value| serde_json::json!(value)));
}

fn append_scope_clause(
    clauses: &mut Vec<String>,
    params: &mut Vec<serde_json::Value>,
    column: &str,
    values: &[String],
) {
    if values.is_empty() {
        return;
    }
    clauses.push(format!(
        "{column} IN ({})",
        std::iter::repeat_n("?", values.len()).collect::<Vec<_>>().join(",")
    ));
    params.extend(values.iter().map(|value| serde_json::json!(value)));
}

pub fn run(
    query: &str,
    filter: Option<&str>,
    k: usize,
    model_id: &str,
    bucket: &str,
    json: bool,
) -> Result<()> {
    crate::sync::pull_for_read(bucket);
    if let Some(note) = crate::view::stale_note() {
        eprintln!("{note}");
    }
    let docs = load_docs(readmodel::docs_path())?;
    let idx = MmapIndex::load(&readmodel::index_dir().to_string_lossy())
        .map_err(|e| anyhow!("load index: {e} (run `index` first)"))?;
    let mut enc = Encoder::load(model_id)?;
    let q = enc.encode_query(query)?;
    let subset = subset_for(filter)?;
    let params = SearchParameters { top_k: k, ..Default::default() };
    let res = idx
        .search(&q, &params, subset.as_deref())
        .map_err(|e| anyhow!("search: {e}"))?;
    if json {
        println!("{}", render_json(&docs, &res));
    } else {
        print!("{}", render(&docs, query, filter, &res));
    }
    Ok(())
}

/// Results as a JSON array (`--json`), for scripts and agents.
pub fn render_json(docs: &[Doc], res: &QueryResult) -> String {
    let arr: Vec<serde_json::Value> = res
        .passage_ids
        .iter()
        .zip(res.scores.iter())
        .filter_map(|(id, score)| {
            let d = docs.get(*id as usize)?;
            Some(serde_json::json!({
                "score": score,
                "id": d.id,
                "kind": d.meta.kind,
                "repo": d.meta.repo,
                "author": d.meta.author,
                "session_id": d.meta.session_id,
                "ts": d.meta.ts,
                "number": d.meta.number,
                "url": d.meta.url,
                "state": d.meta.state,
                "text": excerpt(&d.text, 400),
            }))
        })
        .collect();
    crate::view::envelope("search", serde_json::Value::Array(arr))
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
            if d.meta.repo.is_empty() { "—" } else { &d.meta.repo },
            short(&d.meta.session_id),
            excerpt(&d.text, 160)
        ),
    }
}
