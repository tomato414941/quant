# Run Index Compact Record Necessity

## Issue

`FileRunResultStore` stores `genericCompactRecord` in the run index.

This is a cached projection of a full run record, built by
`build_compact_run_record()`, so list/latest APIs can read summary rows without
loading full result payloads.

## Current State

Current flow:

```text
full run record
  -> build_compact_run_record()
  -> genericCompactRecord
  -> run index
```

`list_compact_records()` and `find_latest_compact_record()` read this projection
from the index. Existing tests cover diagnostic metrics, market data snapshot
IDs, and backfill for older compact records.

## Concern

It is not clear that this cached projection is necessary in the true product
sense.

The compact record is useful for faster listing, but it also adds another stored
shape to maintain. This creates index schema surface, backfill logic, and another
place where result summary fields can drift from the full record.

If run counts and payload sizes stay small, the simpler design may be:

```text
run index stores keys/fingerprints/minimal metadata
list views load full records and build summaries on demand
```

## Direction

Do not remove `genericCompactRecord` now. Current CLI/API paths and tests depend
on it.

Before changing this, measure or inspect expected run count, full record size,
and list API latency. If the cache is not needed, consider moving toward a
minimal index plus on-demand summary construction.
