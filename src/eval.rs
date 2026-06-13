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

/// The name-quality eval: score every topic's cached name against its own
/// cluster's coherence (the self-cal gate's verdict), dump the rateable rows to
/// evals/names.json, a worst-first report to evals/names.md, and a
/// `[metrics nameeval]` block with the BETA sensitivity sweep. Read-only:
/// scores cached names, never regenerates them.
pub fn run_names(bucket: &str) -> Result<()> {
    use crate::qwen::{eval_names, is_slug, rejects_at, NameRow};
    let mut rows = eval_names(bucket)?;

    // Carry forward any human `rating`s from a prior dump so re-running the eval
    // doesn't wipe the gold labels.
    if let Ok(prev) = std::fs::read_to_string("evals/names.json") {
        if let Ok(old) = serde_json::from_str::<Vec<NameRow>>(&prev) {
            let prior: std::collections::HashMap<String, String> =
                old.into_iter().filter_map(|r| r.rating.map(|x| (r.key, x))).collect();
            for r in &mut rows {
                r.rating = prior.get(&r.key).cloned();
            }
        }
    }

    let med = |xs: &mut Vec<f32>| {
        if xs.is_empty() {
            0.0
        } else {
            xs.sort_by(f32::total_cmp);
            xs[xs.len() / 2]
        }
    };
    let scored: Vec<&NameRow> = rows.iter().filter(|r| r.scored && !r.is_fallback).collect();
    let mut ns: Vec<f32> = scored.iter().map(|r| r.name_score).collect();
    let mut p10s: Vec<f32> = scored.iter().map(|r| r.coh_p10).collect();
    let mut ratios: Vec<f32> = scored.iter().filter(|r| r.coh_p10 > 0.0).map(|r| r.name_score / r.coh_p10).collect();
    let would_reject = scored.iter().filter(|r| r.would_reject).count();
    let fallback = rows.iter().filter(|r| r.is_fallback).count();
    let slugs = rows.iter().filter(|r| is_slug(&r.name)).count();
    let mut seen: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for r in &rows {
        *seen.entry(r.name.as_str()).or_default() += 1;
    }
    let dupes: usize = seen.values().filter(|&&c| c > 1).sum();
    let at = |beta: f32| scored.iter().filter(|r| rejects_at(r.name_score, r.coh_p10, r.member_n, beta)).count();

    // Precision/recall of the gate's verdict against human ratings, when present
    // (a "wrong" name SHOULD be rejected). Independent of the score the gate uses.
    let rated: Vec<&NameRow> = scored.iter().filter(|r| r.rating.is_some()).copied().collect();
    let pr = if !rated.is_empty() {
        let is_wrong = |r: &NameRow| r.rating.as_deref() == Some("wrong");
        let tp = rated.iter().filter(|r| r.would_reject && is_wrong(r)).count();
        let fp = rated.iter().filter(|r| r.would_reject && !is_wrong(r)).count();
        let fn_ = rated.iter().filter(|r| !r.would_reject && is_wrong(r)).count();
        Some((tp, fp, fn_))
    } else {
        None
    };

    // Report: worst-faithfulness first — the eyeball targets.
    let mut sorted = rows.clone();
    sorted.sort_by(|a, b| a.name_score.total_cmp(&b.name_score));
    let mut out = format!("# topic-name eval ({} topics, {} scored)\n\n", rows.len(), scored.len());
    out.push_str("name · score / coh_p10 · verdict — summary\n\n");
    for r in &sorted {
        let v = if r.is_fallback {
            "fallback"
        } else if !r.scored {
            "abstain"
        } else if r.would_reject {
            "REJECT"
        } else {
            "keep"
        };
        out.push_str(&format!(
            "- **{}** · {:.2}/{:.2} · {v}{} — {}\n",
            r.name,
            r.name_score,
            r.coh_p10,
            r.rating.as_deref().map(|x| format!(" [rated {x}]")).unwrap_or_default(),
            crate::excerpt(&r.summary, 90),
        ));
    }
    std::fs::create_dir_all("evals").ok();
    std::fs::write("evals/names.json", serde_json::to_string_pretty(&rows)?)?;
    std::fs::write("evals/names.md", &out)?;

    let mut m = crate::metrics::Run::new("nameeval");
    m.set("topics", rows.len())
        .set("scored", scored.len())
        .set("name_score_med", med(&mut ns) as f64)
        .set("coh_p10_med", med(&mut p10s) as f64)
        .set("ratio_med", med(&mut ratios) as f64)
        .set("count_below_alpha", would_reject)
        .set("fallback", fallback)
        .set("dupes", dupes)
        .set("slugs", slugs)
        .set("rej@0.7", at(0.7))
        .set("rej@0.8", at(0.8))
        .set("rej@0.9", at(0.9));
    if let Some((tp, fp, fn_)) = pr {
        m.set("rated", rated.len()).set("tp", tp).set("fp", fp).set("fn", fn_);
    }
    m.emit();
    eprintln!("wrote evals/names.json + evals/names.md ({} topics)", rows.len());
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
