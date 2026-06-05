# quality-roadmap.md — clustering quality interventions

Eval-driven interventions to raise clustering quality for synty's purpose: help
someone scan **coherent, faithfully-named topics** to get up to speed on what's
been happening. Every intervention follows one loop — design the eval, apply the
change with a scenario test, confirm the tracked metric moved without regressing
the guardrails — and every number flows through `metrics::Run` so runs are
comparable. Grounded in web/arxiv research (see References); code line refs are
as of authoring.

## The use case it serves

synty is a passive work memory. You come back after being away and scan the topic
list (name · summary · who/where · activity) to answer *"what's happening?"*, then
drill in. So a topic must be (1) **faithfully named** — the name is what you act
on, and a wrong name actively misleads; (2) **coherent** — one workstream, or the
summary degrades to a vague list; (3) **precise** — a stray unit pollutes the
theme. Failure modes, worst first: wrong names → grab-bag merges → misplaced
members → giant hubs.

## The loop (applied to every intervention)

1. **Eval first** — name the metric(s) that should move and the direction, record
   the baseline on the local corpus, and name the guardrail metrics that must not
   regress.
2. **Apply** — implement the change plus a scenario-style unit test written from
   the expected behavior (per AGENTS.md), never from the implementation.
3. **Verify** — re-run `cluster`/`summarize`, diff the `[metrics …]` block against
   baseline; keep the change only if the target improved and guardrails held —
   otherwise iterate or revert. Commit each kept intervention separately with the
   measured effect in the message.

## Shared metrics (the eval surface)

Today — `[metrics cluster]`: `silhouette misplaced(+pct) modularity
size_{min,med,max} sessions docs clusters unclustered`; `[metrics summarize]`:
`unit_coverage_pct topics_named`.

Added by **I0**, justified by the research:

- `silhouette_macro` — per-cluster mean-of-means; becomes the headline (micro
  silhouette is inflated up to 41% by the single largest cluster — 2401.05831,
  and our sizes already span 2..76).
- `cohesion_min` / `cohesion_med` + `grabbags` — per-cluster cohesion ratio
  ρ_C = within-cluster mean MaxSim ÷ global mean MaxSim; `grabbags` counts
  clusters below a run-relative floor (2511.19350), robust under size imbalance.
- `name_faithful_pct` — share of topic names that clear the embedding-faithfulness
  gate against their members (2502.18469); added with **I2**.
- `unclustered` is already emitted — read it next to `misplaced`, since several
  interventions trade coverage for precision on purpose.

## Interventions (priority order)

### I0 — Expand the metric framework · effort low · status pending
Measurement first, so every later change is judged on the same surface and we can
see the grab-bags/hubs the global silhouette hides.
- **Approach:** add `silhouette_macro`, per-cluster `cohesion_*` + `grabbags` to
  `[metrics cluster]`; `report_quality` (topics.rs) already computes per-unit
  silhouette into `sils` — group by cluster. Emit a per-cluster debug line
  (id · size · S_C · ρ_C · label).
- **Eval:** the new fields emit and are sane — `silhouette_macro` < the current
  micro silhouette given the size imbalance, and the known grab-bag (topic 2)
  shows a low ρ_C. No behavior change.
- **Guardrail:** existing metric fields unchanged.

### I1 — Stable content-addressed cluster ids · low · pending
Addresses root cause #2. Prerequisite — every name/summary fix is pointless if
ids renumber under it (topic 27's "Colpali" name is a stale-id symptom).
- **Approach:** after Louvain + reassign, Jaccard-match each new cluster to the
  previous `unit_clusters.json` on member keys; overlap ≥ 0.5 inherits the old
  stable id (and thus its cached name/summary). New clusters get an id hashed
  (FNV1a) from their **medoid** unit key (member with max mean MaxSim to
  co-members, from the EVAL_K results already fetched). Write the stable id, not
  the positional `ci`; re-key `topic_key`/`topic_name_key`.
- **Eval:** new `id_continuity_pct` (clusters that inherited an id). Run `cluster`
  twice on unchanged data → continuity 100% and no `topic:<id>` cache mismatch.
- **Guardrail:** `silhouette_macro`, `misplaced` unchanged — this is a post-hoc
  label-transfer layer, not a clustering change.

### I2 — Name faithfulness gate + keyword fallback · low · pending
Addresses root cause #1 — catches "Colpali Visual Document Retrieval" on
error-handling PRs and "Update Dependencies" on synty sessions, after generation.
- **Approach:** embed the generated name with the same ColBERT encoder; mean
  MaxSim to members; below a run-relative τ, reject and fall back to a c-TF-IDF
  keyword-join label (100%-grounded — it cannot say a token absent from members).
  Cheap LLM-free pre-check: reject if the name shares zero unigrams with the
  cluster's top-10 c-TF-IDF terms. Show the LLM name only when it passes.
- **Eval:** define `name_faithful_pct`; the known-bad names (27, 28) stop showing.
- **Guardrail:** the keyword-fallback share stays bounded (not over-rejecting good
  names) — track it.

### I3 — Ground the naming prompt · medium · pending
Addresses root cause #1 at the source, complementing I2's after-the-fact gate.
- **Approach:** prompt with the medoid summary as the first line + top-5 c-TF-IDF
  keywords + the titles/first-lines of the 3–5 most-central members, preferring PR
  titles over abstract session summaries; reorder the reduce inputs by centrality
  (the 0.6B attends to early tokens). Bump TOPIC_PROMPT_VERSION/TOPIC_NAME_VERSION.
- **Eval:** `name_faithful_pct` rises and I2's keyword-fallback share drops (more
  names pass the gate).
- **Guardrail:** names stay natural/readable; `topics_named` coverage unchanged.

### I4 — MaxSim length-norm + decouple summary + PR-bridge edges · low · pending
Addresses root causes #5 (session misplacement) and #6 (representation).
- **Approach:** divide each raw MaxSim by the probe unit's token count before the
  ÷best/FLOOR step in `build_edges` (the units.rs comment already names this bias);
  embed a type-prefixed structure (`Session: … / PR: …`) with the summary
  appended-not-leading (so a bad summary no longer corrupts placement); inject a
  strong session↔`linked_pr` bridge edge.
- **Eval:** `misplaced_pct` down; spot-check that synty/slide/RAG sessions leave
  the OCR hub (topic 0).
- **Guardrail:** `silhouette_macro` not worse. A re-encode pass is expected (the
  embed string changes → new content hash).

### I5 — Abstain on borderline outliers · low · pending
Addresses root cause #5 — precision over forced coverage, which the get-up-to-speed
use case values.
- **Approach:** mark degree-0 nodes (no mutual-kNN neighbor) unclustered before
  Louvain; in `reassign_once`, move a unit only when `sil < -0.10 && b/a > 1.1`
  and **abstain** (leave `None`) when `|sil| < 0.05` instead of force-assigning.
- **Eval:** `misplaced_pct` down; `unclustered` rises (intended — the trade).
  Track both.
- **Guardrail:** don't strand coherent units — `silhouette_macro` up or flat.

### I6 — Targeted re-split of flagged hubs · medium · pending
Addresses root cause #3 (under-splitting) without a global resolution change.
- **Approach:** for each cluster flagged grab-bag by I0 (low ρ_C/S_C **and** size
  > ~8), extract the induced sub-graph from the existing `edges` and re-run
  `louvain()` on it at resolution × 1.5–2.0; replace with the sub-clusters. A
  local re-run sidesteps the Louvain resolution limit (Fortunato–Barthélemy 2007).
  Optional merge pass for cluster pairs with c-TF-IDF cosine > 0.8.
- **Eval:** `grabbags` down, `silhouette_macro` up; topic 2 splits into coherent
  sub-topics (spot-check).
- **Guardrail:** don't over-fragment — bound the cluster-count rise; I1 keeps the
  surviving ids stable.

### Deferred (revisit if the cause persists)
- **I7** self-consistency name verification (research #8) — needs temperature
  sampling added to the greedy decoder.
- **I8** IDF-weighted MaxSim (research #9) — corpus IDF table over token ids;
  clustering benefit inferred, validate first.
- **I9** HDBSCAN/GLOSH outlier pre-filter (research #10) — adds a dependency; I5
  gets most of the benefit dependency-free.

## Cross-cutting tradeoffs

- **Precision vs coverage** — I2 and I5 deliberately raise `unclustered` / show
  keyword labels. Track `misplaced` **and** `unclustered` together; a drop in one
  with a rise in the other is the intended trade, not a regression.
- **Readability vs faithfulness** — show the natural LLM name only when it clears
  the gate; otherwise accept a less-pretty but grounded keyword label.
- **Stability vs optimality** — keep stability a post-hoc label layer (Jaccard on
  top of a fresh Louvain), never seed the clustering, so the partition stays free
  to change while labels stay stable.
- **MaxSim is non-metric** — it's asymmetric and breaks the triangle inequality,
  so literature thresholds (silhouette 0.5) don't transfer; use run-relative
  cutoffs (below-median / z-score) and symmetrize where a distance is assumed.

## References

- arXiv:2502.18469 — cluster-label faithfulness (embedding gate I2, grounding I3)
- arXiv:2401.05831 — silhouette macro-vs-micro aggregation (I0; the 41% inflation)
- arXiv:2511.19350 — Cohesion Ratio per-cluster coherence (I0)
- arXiv:2409.18254 — ABCDE / Jaccard label transfer for stable ids (I1)
- arXiv:2412.13678 — Clio (Anthropic): short purpose facet + grounded labels (I3, I4)
- arXiv:2403.15112 — summarization-as-preprocessing doesn't reliably help (I4)
- arXiv:2603.26259 — late-interaction length bias persists inside a cap (I4)
- Fortunato & Barthélemy, PNAS 2007 — modularity resolution limit (I6)
