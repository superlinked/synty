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
