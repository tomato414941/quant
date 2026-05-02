# Comparison Module Responsibility Boundary

## Issue

`comparison_payloads.py` and `comparison_serialization.py` have grown beyond
their likely original responsibilities.

`comparison_serialization.py` should mainly handle JSON-safe serialization and
deserialization of comparison specs, run specs, and strategy definitions. Today
it also contains market availability diagnostics, signal market data context
helpers, predictor grouping, and other comparison helper logic.

`comparison_payloads.py` should mainly assemble API/CLI response payloads from
run builders, store reads, market context, and serializers. Today it coordinates
multiple workflows: strategy runs, predictor runs, condition sweeps, ranking
evaluation, latest run lookup, catalog payloads, and detail payloads.

## Concern

The boundary between serialization, payload orchestration, diagnostics, and
store/index concerns is not obvious. This makes it easier for new helper logic to
land in whichever module is nearby.

Recent cleanup removed old compact record builders from
`comparison_serialization.py`, which shows that store/index responsibilities had
leaked into the serialization layer.

## Direction

Do not split these modules immediately.

When touching this area, prefer moving new logic toward clearer homes:

- pure JSON conversion -> `comparison_serialization.py`
- API/CLI response assembly -> `comparison_payloads.py`
- run construction -> `comparison_run_builders.py`
- market data context/availability -> `comparison_market_context.py` or a
  dedicated diagnostics module
- run index summaries -> `run_store.py`

A future refactor can split `comparison_serialization.py` into smaller modules
only if active changes require it.
