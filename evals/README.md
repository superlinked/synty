# evals/

Model- and corpus-dependent evaluation and tuning. Kept **out of `cargo test`**:
the unit suite is pure (no model, corpus, or network — see AGENTS.md), so quality
evals that need the encoder, the index, and a real corpus live here and run
through the binary instead.

## What's tracked vs not

- **Tracked:** `README.md` (this file), `tuning.md` (append-only log of threshold
  choices + before/after numbers).
- **Gitignored** (per-corpus / rebuildable): `queries.json`, `names.json`,
  `runs.md`, `names.md`. Each machine brings its own gold set; the schemas below
  let you recreate them.

## The eval surface

| What | Command | Output | Key metric |
|---|---|---|---|
| Retrieval | `synty eval` | `evals/runs.md` | `[metrics eval]` `derived_hit_rate` (session-ask → its PR, hit@5) + probe top-5 for human scoring |
| Topic names | `synty eval --names` | `evals/names.md` + `evals/names.json` | `[metrics nameeval]` (name-faithfulness distribution, fallback/dupe/slug counts) |
| Clustering | `synty cluster` (+ `SYNTY_QDUMP=<path>`) | stderr + the qdump JSON | `[metrics cluster]` (modularity, cohesion_med, misplaced_pct, grabbags, dup_units) |

Metrics print to stderr as `[metrics <op>] k=v · …`; redirect with `2>> evals/runs.log`
to keep a history.

## Gold-file schemas

`queries.json` — retrieval probes (optional `filter` is a `col=value` metadata WHERE):
```json
[{"query": "rate limiting middleware", "filter": "repo=sie-web"}, {"query": "…"}]
```

`names.json` — one row per topic, emitted by `eval --names`; add an optional
`rating` by hand (`"good" | "weak" | "wrong"`) to turn it into a labelled set
(the report then prints precision/recall of the gate's `would_reject`):
```json
[{"key":"…","name":"…","summary":"…","members":["…"],
  "name_score":0.71,"coh_p10":0.66,"ratio":1.08,"would_reject":false,"rating":null}]
```

## Tuning discipline

When a change is meant to move quality, read the metric — don't eyeball it or
recompute in a throwaway script (AGENTS.md). Record the decision, the chosen
threshold, and the before/after numbers in `tuning.md`.
