from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


RUN_STORE_LOGIC_VERSION = "v70"
RUN_STORE_INDEX_FILENAME = "_index.json"


@dataclass(frozen=True)
class RunStoreSummary:
    cached_run_count: int = 0
    computed_run_count: int = 0

    def to_payload(self) -> dict[str, int]:
        return {
            "cachedRunCount": self.cached_run_count,
            "computedRunCount": self.computed_run_count,
        }


class FileRunResultStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def load(self, run_spec: dict) -> dict | None:
        path = self._path_for(run_spec)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return payload["result"]

    def list_records(
        self,
        *,
        run_kind: str | None = None,
        generation_method: str | None = None,
        strategy_definition_fingerprint: str | None = None,
        market_data_fingerprint: str | None = None,
        evaluation_fingerprint: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        filtered_entries = self._filter_index_entries(
            self._load_index_entries(),
            run_kind=run_kind,
            generation_method=generation_method,
            strategy_definition_fingerprint=strategy_definition_fingerprint,
            market_data_fingerprint=market_data_fingerprint,
            evaluation_fingerprint=evaluation_fingerprint,
        )
        records: list[dict] = []
        for entry in filtered_entries:
            record = self.get_record(entry["runKey"])
            if record is None:
                continue
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
        return records

    def list_compact_records(
        self,
        *,
        run_kind: str | None = None,
        generation_method: str | None = None,
        strategy_definition_fingerprint: str | None = None,
        market_data_fingerprint: str | None = None,
        evaluation_fingerprint: str | None = None,
        limit: int | None = None,
        view: str = "generic",
    ) -> list[dict]:
        compact_key = self._resolve_compact_record_key(view)
        filtered_entries = self._filter_index_entries(
            self._load_index_entries(),
            run_kind=run_kind,
            generation_method=generation_method,
            strategy_definition_fingerprint=strategy_definition_fingerprint,
            market_data_fingerprint=market_data_fingerprint,
            evaluation_fingerprint=evaluation_fingerprint,
        )
        records: list[dict] = []
        for entry in filtered_entries:
            compact_record = entry.get(compact_key)
            if compact_record is None:
                continue
            records.append(compact_record)
            if limit is not None and len(records) >= limit:
                break
        return records

    def find_latest_compact_record(
        self,
        *,
        run_kind: str | None = None,
        generation_method: str | None = None,
        strategy_definition_fingerprint: str | None = None,
        market_data_fingerprint: str | None = None,
        evaluation_fingerprint: str | None = None,
        view: str = "generic",
    ) -> dict | None:
        compact_records = self.list_compact_records(
            run_kind=run_kind,
            generation_method=generation_method,
            strategy_definition_fingerprint=strategy_definition_fingerprint,
            market_data_fingerprint=market_data_fingerprint,
            evaluation_fingerprint=evaluation_fingerprint,
            limit=1,
            view=view,
        )
        if not compact_records:
            return None
        return compact_records[0]

    def rebuild_index(self) -> dict[str, int]:
        entries = self._rebuild_index_entries()
        self._write_index_entries(entries)
        return {"entryCount": len(entries)}

    def get_record(self, run_key: str) -> dict | None:
        path = self.root_dir / f"{run_key}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        run_spec = payload.get("runSpec", {})
        if run_spec.get("logicVersion") != RUN_STORE_LOGIC_VERSION:
            return None
        return {
            "runKey": run_key,
            "savedAtUtc": payload.get("savedAtUtc"),
            "runSpec": run_spec,
            "result": payload.get("result", {}),
        }

    def save(self, run_spec: dict, result: dict) -> None:
        path = self._path_for(run_spec)
        saved_at_utc = datetime.now(timezone.utc).isoformat()
        payload = {
            "savedAtUtc": saved_at_utc,
            "runSpec": run_spec,
            "result": result,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        self._upsert_index_entry(path.stem, saved_at_utc, run_spec, result)

    def _path_for(self, run_spec: dict) -> Path:
        digest = build_run_cache_key(run_spec)
        return self.root_dir / f"{digest}.json"

    def _index_path(self) -> Path:
        return self.root_dir / RUN_STORE_INDEX_FILENAME

    def _load_index_entries(self) -> list[dict]:
        index_path = self._index_path()
        if index_path.exists():
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
                entries = payload.get("entries")
                if isinstance(entries, list):
                    return entries
            except json.JSONDecodeError:
                pass
        entries = self._rebuild_index_entries()
        self._write_index_entries(entries)
        return entries

    def _write_index_entries(self, entries: list[dict]) -> None:
        payload = {
            "kind": "run_store_index",
            "schemaVersion": "v1",
            "logicVersion": RUN_STORE_LOGIC_VERSION,
            "entries": entries,
        }
        self._index_path().write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def _rebuild_index_entries(self) -> list[dict]:
        entries: list[dict] = []
        for path in self.root_dir.glob("*.json"):
            if path.name == RUN_STORE_INDEX_FILENAME:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            run_spec = payload.get("runSpec", {})
            if run_spec.get("logicVersion") != RUN_STORE_LOGIC_VERSION:
                continue
            entries.append(
                self._build_index_entry(
                    run_key=path.stem,
                    saved_at_utc=payload.get("savedAtUtc"),
                    run_spec=run_spec,
                    result=payload.get("result", {}),
                )
            )
        return self._sort_index_entries(entries)

    def _upsert_index_entry(
        self,
        run_key: str,
        saved_at_utc: str | None,
        run_spec: dict,
        result: dict,
    ) -> None:
        entries = [
            entry for entry in self._load_index_entries()
            if entry.get("runKey") != run_key
        ]
        entries.append(
            self._build_index_entry(
                run_key=run_key,
                saved_at_utc=saved_at_utc,
                run_spec=run_spec,
                result=result,
            )
        )
        self._write_index_entries(self._sort_index_entries(entries))

    def _build_index_entry(
        self,
        *,
        run_key: str,
        saved_at_utc: str | None,
        run_spec: dict,
        result: dict,
    ) -> dict:
        generation = run_spec.get("generation", {})
        fingerprints = run_spec.get("fingerprints", {})
        record = {
            "runKey": run_key,
            "savedAtUtc": saved_at_utc,
            "runSpec": run_spec,
            "result": result,
        }
        return {
            "runKey": run_key,
            "savedAtUtc": saved_at_utc,
            "logicVersion": run_spec.get("logicVersion"),
            "runKind": run_spec.get("runKind"),
            "generationMethod": generation.get("method"),
            "strategyDefinitionFingerprint": fingerprints.get("strategyDefinition"),
            "evaluationSubjectFingerprint": fingerprints.get("evaluationSubject"),
            "marketDataFingerprint": fingerprints.get("marketData"),
            "evaluationFingerprint": fingerprints.get("evaluation"),
            "genericCompactRecord": build_compact_run_record(record),
            "predictorCompactRecord": build_compact_predictor_run_record(record)
            if run_spec.get("runKind") == "predictor_run"
            else None,
        }

    def _filter_index_entries(
        self,
        entries: list[dict],
        *,
        run_kind: str | None,
        generation_method: str | None,
        strategy_definition_fingerprint: str | None,
        market_data_fingerprint: str | None,
        evaluation_fingerprint: str | None,
    ) -> list[dict]:
        filtered = []
        for entry in entries:
            if entry.get("logicVersion") != RUN_STORE_LOGIC_VERSION:
                continue
            if run_kind is not None and entry.get("runKind") != run_kind:
                continue
            if generation_method is not None and entry.get("generationMethod") != generation_method:
                continue
            if (
                strategy_definition_fingerprint is not None
                and entry.get("strategyDefinitionFingerprint") != strategy_definition_fingerprint
            ):
                continue
            if (
                market_data_fingerprint is not None
                and entry.get("marketDataFingerprint") != market_data_fingerprint
            ):
                continue
            if (
                evaluation_fingerprint is not None
                and entry.get("evaluationFingerprint") != evaluation_fingerprint
            ):
                continue
            filtered.append(entry)
        return self._sort_index_entries(filtered)

    def _sort_index_entries(self, entries: list[dict]) -> list[dict]:
        return sorted(
            entries,
            key=lambda entry: (
                entry.get("savedAtUtc") or "",
                entry.get("runKey") or "",
            ),
            reverse=True,
        )

    def _resolve_compact_record_key(self, view: str) -> str:
        if view == "generic":
            return "genericCompactRecord"
        if view == "predictor":
            return "predictorCompactRecord"
        raise ValueError(f"Unsupported compact record view: {view}")



def build_compact_run_record(record: dict) -> dict:
    run_spec = record["runSpec"]
    result = record["result"]
    strategy = run_spec.get("strategyDefinition") or run_spec.get("strategy", {})
    execution_assumptions = run_spec.get("executionAssumptions", {})
    market_slice = run_spec.get("marketSlice", {})
    evaluation = run_spec.get("evaluation", {})
    fingerprints = run_spec.get("fingerprints", {})
    summary = result.get("summary", {})
    portfolio_summary = summary.get("portfolio", summary)

    return {
        "runKey": record["runKey"],
        "savedAtUtc": record.get("savedAtUtc"),
        "runKind": run_spec.get("runKind"),
        "logicVersion": run_spec.get("logicVersion"),
        "generationMethod": run_spec.get("generation", {}).get("method"),
        "generationBatchKey": run_spec.get("generation", {}).get("batchKey"),
        "strategyId": strategy.get("strategyId"),
        "strategyVersion": strategy.get("version"),
        "strategyLabel": strategy.get("label"),
        "strategyHypothesis": strategy.get("hypothesis"),
        "investmentUniverseLabel": strategy.get("components", {}).get("core", {}).get("investmentUniverse", {}).get("label"),
        "investmentUniverseAssetCount": strategy.get("components", {}).get("core", {}).get("investmentUniverse", {}).get("assetCount"),
        "portfolioModelLabel": strategy.get("components", {}).get("core", {}).get("portfolioModel", {}).get("label"),
        "executionLabel": (
            strategy.get("components", {}).get("core", {}).get("executionPlan", {}).get("label")
            or strategy.get("components", {}).get("core", {}).get("executionPolicy", {}).get("label")
        ),
        "period": market_slice.get("period"),
        "timeframe": strategy.get("components", {}).get("core", {}).get("dataResolution", {}).get("key")
        or market_slice.get("timeframe", {}).get("key"),
        "maxInvestmentPct": strategy.get("components", {}).get("optional", {}).get("riskControls", {}).get("maxInvestmentPct"),
        "maxWeightPct": strategy.get("components", {}).get("optional", {}).get("riskControls", {}).get("maxWeightPct"),
        "costModelKind": execution_assumptions.get("costModel", {}).get("kind"),
        "commissionPct": execution_assumptions.get("costModel", {}).get("parameters", {}).get("commissionPct"),
        "capitalBase": run_spec.get("capitalBase"),
        "splitRatioPct": evaluation.get("evaluationSettings", {}).get("splitRatioPct"),
        "strategyDefinitionFingerprint": fingerprints.get("strategyDefinition"),
        "evaluationSubjectFingerprint": fingerprints.get("evaluationSubject"),
        "marketDataFingerprint": fingerprints.get("marketData"),
        "evaluationFingerprint": fingerprints.get("evaluation"),
        "sharpeRatio": portfolio_summary.get("sharpeRatio"),
        "totalReturnPct": portfolio_summary.get("totalReturnPct"),
        "maxDrawdownPct": portfolio_summary.get("maxDrawdownPct"),
    }


def build_compact_predictor_run_record(record: dict) -> dict:
    run_spec = record["runSpec"]
    result = record["result"]
    fingerprints = run_spec.get("fingerprints", {})
    evaluation_subject = run_spec.get("evaluationSubject", {})
    predictor = evaluation_subject.get("predictor") or run_spec.get("strategy", {}).get("predictor", {})
    signal = predictor.get("signalSpec", {})
    observation = signal.get("observationSpec", {})
    predicted_quantity = predictor.get("predictedQuantitySpec", {})
    target = predictor.get("targetSpec", {})
    output = signal.get("outputSpec", {})
    horizon = target.get("horizonSpec", {})
    feature = predictor.get("featureSpec", {})
    training = predictor.get("trainingSpec", {})
    engine = predictor.get("engineSpec", {})
    decision_use = signal.get("decisionUseSpec", {})
    signal_source = engine.get("signalSourceSpec") or {}
    learner = engine.get("learnerSpec") or {}
    combiner = engine.get("combinerSpec") or {}
    overall = result.get("overall", {})
    test = result.get("test", {})

    return {
        "runKey": record["runKey"],
        "savedAtUtc": record.get("savedAtUtc"),
        "runKind": run_spec.get("runKind"),
        "logicVersion": run_spec.get("logicVersion"),
        "strategyDefinitionFingerprint": fingerprints.get("strategyDefinition"),
        "evaluationSubjectFingerprint": fingerprints.get("evaluationSubject"),
        "marketDataFingerprint": fingerprints.get("marketData"),
        "evaluationFingerprint": fingerprints.get("evaluation"),
        "predictorKey": predictor.get("key"),
        "predictorLabel": predictor.get("label"),
        "observationKey": observation.get("key"),
        "observationLabel": observation.get("label"),
        "observationAssetCount": observation.get("assetCount"),
        "observationFields": observation.get("fields"),
        "signalEntityKind": signal.get("entityKind"),
        "signalEntityCount": len(signal.get("entityIdentifiers") or []),
        "decisionUseKind": decision_use.get("useKind"),
        "featureKey": feature.get("key"),
        "signalSourceKind": signal_source.get("signalSourceKind"),
        "signalSourceFeatureKey": signal_source.get("featureKey"),
        "learnerKind": learner.get("learnerKind"),
        "combinerKind": combiner.get("combinerKind"),
        "trainingFitMode": training.get("fitMode"),
        "trainingMinSamples": training.get("minTrainSamples"),
        "timeframe": predictor.get("timeframe", {}).get("key"),
        "targetKey": target.get("key"),
        "predictedQuantityKind": predicted_quantity.get("quantityKind"),
        "targetBaseline": target.get("baseline"),
        "targetTransform": target.get("transform"),
        "outputKind": output.get("outputKind"),
        "horizonUnit": horizon.get("unit"),
        "horizonValue": horizon.get("value"),
        "period": run_spec.get("marketSlice", {}).get("period"),
        "observationCount": overall.get("observationCount"),
        "testObservationCount": test.get("observationCount"),
        "overallRankIc": overall.get("meanRankIc"),
        "testRankIc": test.get("meanRankIc"),
        "overallTopMinusBottomPct": overall.get("meanTopMinusBottomPct"),
        "testTopMinusBottomPct": test.get("meanTopMinusBottomPct"),
        "testHitRatePct": test.get("hitRatePct"),
    }

def build_run_cache_key(run_spec: dict) -> str:
    serialized = json.dumps(run_spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_run_fingerprint(payload: object) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_run_spec_fingerprints(
    *,
    strategy: dict | None,
    strategy_definition: dict | None,
    evaluation_subject: dict | None,
    market_slice: dict,
    evaluation: dict,
    execution_assumptions: dict,
    portfolio_state: dict,
    capital_base: float,
    generation: dict | None,
) -> dict[str, str]:
    return {
        "strategyDefinition": build_run_fingerprint(strategy_definition or strategy),
        "evaluationSubject": build_run_fingerprint(evaluation_subject or {"kind": "strategy"}),
        "marketData": build_run_fingerprint(
            {
                "marketSlice": market_slice,
                "marketDataContexts": evaluation.get("marketDataContexts", []),
                "signalMarketDataContexts": evaluation.get("signalMarketDataContexts", []),
            }
        ),
        "evaluation": build_run_fingerprint(
            {
                "evaluationSettings": evaluation.get("evaluationSettings"),
                "executionAssumptions": execution_assumptions,
                "portfolioState": portfolio_state,
                "capitalBase": capital_base,
                "generation": generation,
            }
        ),
    }


def build_run_spec(
    *,
    run_kind: str,
    market_slice: dict,
    evaluation: dict,
    execution_assumptions: dict,
    portfolio_state: dict,
    capital_base: float,
    generation: dict | None = None,
    strategy: dict | None = None,
    strategy_definition: dict | None = None,
    evaluation_subject: dict | None = None,
) -> dict:
    if strategy is None and strategy_definition is None:
        raise ValueError("run spec must include strategy or strategy_definition.")
    run_spec = {
        "kind": "run_spec",
        "schemaVersion": "v1",
        "logicVersion": RUN_STORE_LOGIC_VERSION,
        "runKind": run_kind,
        "marketSlice": market_slice,
        "portfolioState": portfolio_state,
        "capitalBase": capital_base,
        "evaluation": evaluation,
        "executionAssumptions": execution_assumptions,
        "fingerprints": build_run_spec_fingerprints(
            strategy=strategy,
            strategy_definition=strategy_definition,
            evaluation_subject=evaluation_subject,
            market_slice=market_slice,
            evaluation=evaluation,
            execution_assumptions=execution_assumptions,
            portfolio_state=portfolio_state,
            capital_base=capital_base,
            generation=generation,
        ),
    }
    if strategy_definition is not None:
        run_spec["strategyDefinition"] = strategy_definition
    if evaluation_subject is not None:
        run_spec["evaluationSubject"] = evaluation_subject
    if strategy is not None:
        run_spec["strategy"] = strategy
    if generation is not None:
        run_spec["generation"] = generation
    return run_spec
