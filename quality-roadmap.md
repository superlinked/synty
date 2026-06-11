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

For probes that need more than the metrics block, `SYNTY_QDUMP=<path> synty
cluster` writes one JSON row per scored unit (key, embed hash, token count,
own/other cluster, same/other best MaxSim, top-1, kNN-vote) — margin
distributions, length-bias checks, and per-cluster splitability analyses run
offline from that plus the embedding store, with no code changes.

## Shared metrics (the eval surface)

Today — `[metrics cluster]`: `misplaced(+pct) modularity cohesion_med
vote_disagree bridges id_continuity size_{min,med,max} tiny sessions docs
clusters unclustered` (no silhouette: it structurally prefers coarser clusters
— the grab-bag failure — so coherence is judged by the anchor eval instead);
`[metrics summarize]`: `unit_coverage_pct topics_named name_dupes
names_kw_fallback names_scored name_faithful_pct` (the name fields added with
I2/I3 — topics sharing an identical name, names that are the keyword fallback
rather than an accepted LLM title, and the share of LLM names clearing the
embedding gate).

How the I0 research played out:

- `silhouette_macro` was planned as the headline (micro silhouette is inflated
  up to 41% by the single largest cluster — 2401.05831) but silhouette was
  dropped wholesale once calibration showed it rewards exactly the grab-bag
  failure; the anchor membership eval is the headline instead.
- `cohesion_med` shipped — per-cluster cohesion ratio ρ_C = within-cluster mean
  MaxSim ÷ global mean MaxSim (2511.19350), robust under size imbalance — with
  the lowest-ρ_C clusters printed per run; a `grabbags` count is still open.
- `name_faithful_pct` shipped with **I2** — share of topic names that clear the
  embedding-faithfulness gate against their members (2502.18469).
- `unclustered` is emitted — read it next to `misplaced`, since several
  interventions trade coverage for precision on purpose.

## Interventions (priority order)

### I0 — Expand the metric framework · effort low · status mostly shipped (silhouette dropped by design)
Measurement first, so every later change is judged on the same surface and we can
see the grab-bags/hubs a global score hides.
- **Shipped (topics.rs):** per-cluster cohesion ratio ρ_C with `cohesion_med`
  in `[metrics cluster]`, the lowest-cohesion-clusters debug lines
  (id · ρ_C · size · label), `misplaced(+pct)`, and rescale-invariant
  `vote_disagree`. Silhouette (micro and macro) was deliberately dropped, not
  deferred: it structurally prefers coarser clusters — exactly the grab-bag
  failure — so coherence is judged by the anchor membership eval.
- **Open:** a `grabbags` count (clusters below a run-relative ρ_C floor) is not
  emitted; the debug lines carry that signal today.

### I1 — Stable content-addressed cluster ids · low · shipped
Addresses root cause #2. Prerequisite — every name/summary fix is pointless if
ids renumber under it (topic 27's "Colpali" name was a stale-id symptom).
- **Shipped (topics.rs):** after Louvain + reassign, each new cluster
  Jaccard-matches the previous `unit_clusters.json` on member keys; overlap
  ≥ 0.5 inherits the old stable key (and thus its cached name/summary), new
  clusters get a key hashed (FNV1a) from their medoid unit key.
  `topic_key`/`topic_name_key` are keyed by it, and `id_continuity` is emitted
  — it has held 100% across consecutive re-clusters on live data, including
  re-clusters that crossed a build replacement.
- **Guardrail held:** a post-hoc label-transfer layer; the partition itself is
  untouched.

### I2 — Name faithfulness gate + keyword fallback · low · shipped
Addresses root cause #1 — catches "Colpali Visual Document Retrieval" on
error-handling PRs and "Update Dependencies" on synty sessions, after generation.
- **Approach:** embed the generated name with the same ColBERT encoder; mean
  MaxSim to members; below a run-relative τ, reject and fall back to a c-TF-IDF
  keyword-join label (100%-grounded — it cannot say a token absent from members).
  Cheap LLM-free pre-check: reject if the name shares zero unigrams with the
  cluster's top-12 c-TF-IDF terms. Show the LLM name only when it passes.
- **Shipped (qwen.rs):** the unigram pre-check gates on genuinely *contrastive*
  terms (per-cluster df × smoothed inverse cluster frequency — plain frequency
  had let "SIE" pass as grounded in most clusters), plus a ban on names equal
  to a repo slug or fragment; any rejection falls back to the keyword-join
  label, so no topic is ever titled by its summary sentence. The embedding gate
  (`embed_gate_names`) scores every LLM name against its members' cluster-time
  embeddings, length-normalized, and replaces run-relative outliers (< 0.6 ×
  median, ≥8 scored) with the keyword label — a local-cache correction every
  machine reaches deterministically; the write-once store keeps the raw
  generation. Measured on the live corpus: duplicate names 14→0 topics, empty
  names 5→0, bare repo-slug names 27→0; with the grounded prompt in place the
  gate found a tight distribution (median 0.83, min 0.75 — the unfaithful tail
  was eliminated at the source) and stands as the regression guard.
- **Guardrail:** the keyword-fallback share stays bounded (not over-rejecting good
  names) — `names_kw_fallback` tracks it, `name_faithful_pct` the gate's pass
  share.

### I3 — Ground the naming prompt · medium · shipped
Addresses root cause #1 at the source, complementing I2's after-the-fact gate.
- **Approach:** prompt with the medoid summary as the first line + top c-TF-IDF
  keywords + the titles/first-lines of the 3–5 most-central members, preferring PR
  titles over abstract session summaries; reorder the reduce inputs by centrality
  (the 0.6B attends to early tokens). Bump TOPIC_PROMPT_VERSION/TOPIC_NAME_VERSION.
- **Shipped (qwen.rs, prompt versions t8/s5):** `cluster` persists each unit's
  centrality rank (0 = medoid) into unit_clusters.json; the reduce inputs and
  the name prompt's example items are ordered by it, medoid first, with the
  examples filtered to well-formed member summaries (≥5 words) — the previous
  shortest-first pick selected degenerate slug echoes ("sie-internal: #955")
  that primed the 0.6B to answer in slugs. The prompt's key terms are the
  contrastive c-TF-IDF list (top 8). Member texts stay LLM summaries
  throughout (raw conventional-commit PR titles would re-introduce the slug
  register the examples filter removes).
- **Guardrail:** names stay natural/readable; `topics_named` coverage unchanged
  (82/82 after the change).

### I4 — MaxSim length-norm + decouple summary + PR-bridge edges · low · partially shipped
Addresses root causes #5 (session misplacement) and #6 (representation).
- **Shipped (alternate forms):** the session↔`linked_pr` bridge ships as
  `snap_to_prs` — a hard post-reassignment snap to the produced PR's topic
  (a soft edge loses to kNN reassign), with `bridges` emitted; the length bias
  is contained by capping every embed text at 500 chars so units stay
  comparable, rather than per-token normalization in `build_edges`.
- **Open:** the embed text still leads with the summary
  (`"{summary} {repo} {files}"`, units.rs) — a type-prefixed structure with
  the summary appended-not-leading remains untried, and per-token length-norm
  inside the cap is still a candidate if misplacement resurfaces.
- **Validated (SYNTY_QDUMP probe, live corpus):** sessions place markedly worse
  than PRs/issues — misplaced 9.2% vs 3.0%, kNN-vote disagreement 50.6% vs
  27.8% — so the representation gap the open half targets is real, and this is
  the highest-leverage open clustering change. The length bias itself is
  mostly contained by the 500-char cap (best-neighbor MaxSim is flat across
  the upper three embed-length quartiles, depressed only in the shortest).
- **Eval for the open half:** the session-vs-doc misplacement gap closes;
  `misplaced_pct` down. A re-encode pass is expected (the embed string
  changes → new content hash).

### I5 — Abstain on borderline outliers · low · shipped
Addresses root cause #5 — precision over forced coverage, which the get-up-to-speed
use case values.
- **Shipped (topics.rs):** a unit with no mutual-kNN edge (degree 0) is a
  genuine outlier the reassignment refuses to adopt — it stays unclustered
  rather than being forced into a topic it doesn't belong to, and `unclustered`
  is emitted next to `misplaced` so the precision/coverage trade stays visible.
  The silhouette-threshold variants died with silhouette itself (see I0);
  placement quality is watched through `misplaced_pct` and `vote_disagree`.

### I6 — Targeted re-split of flagged hubs · medium · pending
Addresses root cause #3 (under-splitting). The global counterpart already
shipped: Louvain runs at resolution × RES_SCALE (2.5), calibrated against the
anchor eval, which broke the original resolution-limit grab-bags; agglomerative
re-merging was tried and dropped (coherent and grab-bag sub-themes merge at the
same threshold). What remains is the *local* version, for the few residual
low-ρ_C clusters the run report still flags.
- **Approach:** for each cluster flagged by the lowest-cohesion debug (low ρ_C
  **and** size > ~8), extract the induced sub-graph from the existing `edges`
  and re-run `louvain()` on it at resolution × 1.5–2.0; replace with the
  sub-clusters. A local re-run sidesteps the Louvain resolution limit
  (Fortunato–Barthélemy 2007).
- **Eval:** the flagged clusters' ρ_C rises and their summaries/names sharpen
  (today they read as grab-bag lists); cluster count rise stays bounded.
- **Guardrail:** don't over-fragment; I1 keeps the surviving ids stable.
- **Probe before building (SYNTY_QDUMP + pairwise member MaxSim offline):** the
  flagship flagged cluster measured *diffuse*, not bimodal — best 2-way split
  separation (within − between) only +0.03 on per-token-normalized MaxSim, so
  a local re-split would cut arbitrarily. Run the same probe on any newly
  flagged cluster; build I6 only when one shows real separation, and note the
  other flagged clusters were mono-theme with weak names (a naming residual)
  or near-duplicate piles (I10), not under-splits.

### I10 — Collapse near-duplicate units · low · pending (new, probe-driven)
The lowest-cohesion cluster turned out to be ~a dozen near-identical kickoff
sessions: every member pair above 0.8 per-token MaxSim, 39 of 66 pairs above
0.9 — the same canonical summary re-generated session after session. No
planned intervention touches this; it pollutes cohesion metrics, inflates
topic unit counts, and pads the topic view with repetition.
- **Approach:** collapse units whose pairwise per-token MaxSim exceeds ~0.95
  into one logical unit for clustering and topic facets (keep the count as a
  badge, like "×9"), or flag-and-fold in the views; embeddings are already in
  the store, so detection is a cheap pass over each cluster's members.
- **Eval:** the blurb cluster folds to a handful of logical units and its ρ_C
  normalizes; topic unit counts stop double-counting reruns.
- **Guardrail:** never collapse across repos or kinds; keep keys reachable
  (drill-in still lists the collapsed sessions).

### Deferred (revisit if the cause persists)
- **I7** self-consistency name verification (research #8) — needs temperature
  sampling added to the greedy decoder. Its target failure is currently
  absent (`name_faithful_pct` 100, `name_dupes` 0) and the embedding gate
  covers regressions; revisit only if those metrics slip.
- **I8** IDF-weighted MaxSim (research #9) — corpus IDF table over token ids;
  clustering benefit inferred, validate first. Designed probe: re-encode one
  flagged cluster's embed texts with the repo-token suffix stripped and
  compare within/between separation — implement only if separation jumps.
- **I9** HDBSCAN/GLOSH outlier pre-filter (research #10) — adds a dependency; I5
  gets most of the benefit dependency-free. Probe evidence against: 82.5% of
  clustered units sit within 10% relative margin of another cluster (MaxSim
  margins are inherently tight here), so a margin-keyed outlier filter would
  over-fire; placement uncertainty is better read from `vote_disagree`.

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
