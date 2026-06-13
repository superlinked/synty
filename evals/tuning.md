# Tuning log

Append-only. Each entry: the knob, why it changed (or didn't), and the
before/after numbers that justified it. Newest last.

## Embeddings are L2-normalized → MaxSim scores are cosines

pylate-rs normalizes every token embedding to unit length on every encode path
(`pylate-rs/utils.rs::normalize_l2`). Consequences the thresholds below rely on:

- `topics::maxsim` sums each query token's best dot with a doc token; with
  unit-norm vectors each dot is a cosine in [-1,1].
- `dup_groups` divides each direction by token count, so the symmetrized DUP
  score is a **mean-best-match cosine**, and `qwen::name_score` divides by name
  token count, so it's a **per-name-token mean of mean-over-members best-match
  cosine** — both bounded in [-1,1] and independent of the encoder's score scale.

So an *absolute* cosine floor is meaningful (not just a run-relative one), and
near-identity / faithfulness cutoffs port across normalized encoders.

## DUP = 0.95 — leave it (not the encoder-brittle absolute it looked like)

The heuristics audit flagged `DUP=0.95` (topics.rs) as an absolute magnitude
calibrated to one encoder. Per the normalization above it's a mean-best-match
**cosine**: 0.95 = "two same-repo, same-kind units whose tokens align at ≥0.95
cosine on average" = near-identical text, portable to any normalized ColBERT.
No change. Revisit only if a non-ColBERT or much coarser encoder lands (which
could compress the cosine range enough to over-collapse).

## Name gate: self-calibrating verdict — BETA=0.85, P=10, ABS_FLOOR=0.5

`synty eval --names` on the live corpus (77 topics, 51 scored LLM names, 26
fallbacks). name_score_med 0.82, coh_p10_med 0.80, ratio_med 1.016 — names are
~as faithful as their cluster's least-aligned member. Sensitivity sweep:
rej@0.7 = rej@0.8 = rej@0.9 = 0. The gate rejects nothing on the current set,
i.e. **zero false positives on good names** — it's a lenient safety net, not an
active filter, which is what we want now that the names are already clean.

Chosen ABS_FLOOR=0.5 (a name averaging <0.5 cosine to its best member matches
is off-theme on any normalized encoder), BETA=0.85 × p10, MIN_MEMBERS_FOR_COH=5
(below that, ABS_FLOOR alone). Validated:
- No good name is rejected (sweep above + eyeball of the 25 lowest-score names
  in evals/names.md — all on-theme).
- It still CATCHES off-theme: a name at 0.6 cosine on a cluster with coh_p10 0.8
  → floor 0.68 → rejected; an orthogonal/hallucinated-token name → below
  ABS_FLOOR → rejected (unit tests `score_name_*`).

Two things the gate canNOT do, by design / by limitation:
- It is NOT what rejects most bad names today — `name_grounded` (unigram
  overlap) is, and it OVER-rejects: 34% fallback rate. Dropping name_grounded
  (next) lets faithful paraphrases survive while the gate still catches the
  genuinely off-theme (those have low cosine).
- It can't catch an invented proper noun when the rest of the name is on-theme:
  "Stablebridge Dashboard" scores 0.77, indistinguishable from good names at the
  same level ("Mobile Support Enhancements" 0.76). No faithfulness threshold
  separates it without collateral. Accepted as a rare 0.6B limitation.

## Dropping name_grounded — eval-REJECTED (kept it)

Tried removing the unigram `name_grounded` gate (the self-cal embedding gate as
sole faithfulness authority) + selfcal default + regenerate. Measured:
names_fallback 26→19 (recovered ~7 good paraphrases name_grounded wrongly
rejected) — the intended win. But eyeballing the regenerated set showed a worse
regression: mashed-token garbage that name_grounded had been catching now
survived — `Deckslide`, `Berlinbuzzwordscluster`, `Ocradaptercluster`,
`Tlsmodecluster` (was the excellent "Migrate Ingress from Ingress-Nginx to
Kubernetes Gateway API"), `Lorarouting`, `Commercialintent`.

Why the embedding gate can't replace it: a mashed token like "lorarouting"
embeds ON-theme (near the LoRA cluster), so name_score is high → not rejected.
name_grounded rejects it because the mashed token equals no cluster c-TF-IDF
*term*. That unigram check is doing real structural work (catching word
concatenations) that no faithfulness threshold can, and replacing it with a
"mashed-token detector" would be a worse, dictionary-needing heuristic.

Decision: KEEP name_grounded. The self-cal gate stays as the default semantic
backstop (validated 0 false positives), but clean_name + name_grounded remain
the primary gates. The 26-topic fallback rate is acceptable — those fallbacks
are good summary headings, not soup (names_kw_last_resort=0). This is the eval
doing its job: it caught a regression that the proxy metrics (fallback down,
dupes 0) alone would have green-lit.
