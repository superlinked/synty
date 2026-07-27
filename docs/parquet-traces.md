# Parquet trace projection

The raw event lake is already physically partitioned:

```text
events/<stream>/chunks/track.<day>/<immutable-range>.jsonl
```

Glue projects those paths as injected `stream` and date `day` partitions. That
is why `raw_events` can query the existing bucket without a crawler or
migration. The remaining costs are the many small JSONL objects, parsing every
envelope through a single opaque `line` column, and returning full matching
lines to the Rust trace fold.

## Additive layout

The Parquet path is a derived copy, not a replacement:

```text
events/<stream>/chunks/track.<day>/*.jsonl
    authoritative, immutable, never moved

derived/trace-events/v1/stream=<stream>/day=<day>/*.parquet
    compact, rebuildable query projection
```

`deploy/aws/athena-trace.yaml` defines `trace_events_v1` over the derived
prefix. It deliberately preserves the current reader contract:

| Column | Purpose |
|---|---|
| `line string` | complete canonical event envelope |
| `stream string` | injected physical partition |
| `day string` | projected date partition |

Therefore an MCP deployment can switch from `raw_events` to
`trace_events_v1` through `mcp.athena.table` without a binary or protocol
change. The MCP role remains read-only and can fall back to `raw_events`.

## Backfill and incremental work

Creating the Glue table writes no bucket data. Making historical sessions
available through it does require a one-time derived backfill:

1. Read each existing closed `(stream, day)` JSONL partition.
2. Write larger compressed Parquet files under the derived prefix.
3. Validate event counts and deterministic event-id sets against the raw
   partition.
4. Publish the partition only after validation.

New data needs the same projection for newly closed days. The projector must be
separately authorized; the serving MCP must never receive its write identity.
It should be restart-safe, reject an already-published partition unless running
an explicit repair generation, and emit `metrics::Run` counts for source
objects/bytes, events, duplicates, rejected envelopes, output files/bytes, and
elapsed time.

No local CLI migration is required. Local CLI/TUI continues to build and read
the normal local trace projection. Raw JSONL remains sufficient to rebuild
every derived format.

## What line-compatible Parquet does not solve

This first contract reduces scanned bytes and small-file overhead, but broad
queries can still return more than 64 MiB because Rust still receives full
envelopes and constructs turns, spans, and jobs after Athena.

The target second projection materializes typed `trace_turns`, `trace_spans`,
and `trace_jobs` Parquet rows. List/filter/sort then happens inside Athena and
returns only the requested rows; `trace_show` fetches raw context only for the
selected session/id. That entity projection, rather than Parquet encoding
alone, removes the raw-envelope transfer bottleneck.
