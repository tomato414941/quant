from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


RUN_STORE_LOGIC_VERSION = "v5"


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

    def load(self, run_definition: dict) -> dict | None:
        path = self._path_for(run_definition)
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
        limit: int | None = None,
    ) -> list[dict]:
        records: list[dict] = []
        for path in self.root_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            run_definition = payload.get("runDefinition", {})
            if run_definition.get("logicVersion") != RUN_STORE_LOGIC_VERSION:
                continue
            if run_kind is not None and run_definition.get("runKind") != run_kind:
                continue
            generation = run_definition.get("generation", {})
            if generation_method is not None and generation.get("method") != generation_method:
                continue
            records.append(
                {
                    "runKey": path.stem,
                    "savedAtUtc": payload.get("savedAtUtc"),
                    "runDefinition": run_definition,
                    "result": payload.get("result", {}),
                }
            )
        records.sort(
            key=lambda record: (
                record["savedAtUtc"] or "",
                record["runKey"],
            ),
            reverse=True,
        )
        if limit is not None:
            return records[:limit]
        return records

    def save(self, run_definition: dict, result: dict) -> None:
        path = self._path_for(run_definition)
        payload = {
            "savedAtUtc": datetime.now(timezone.utc).isoformat(),
            "runDefinition": run_definition,
            "result": result,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def _path_for(self, run_definition: dict) -> Path:
        digest = build_run_cache_key(run_definition)
        return self.root_dir / f"{digest}.json"


def build_run_cache_key(run_definition: dict) -> str:
    serialized = json.dumps(run_definition, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_run_definition(
    *,
    run_kind: str,
    candidate: dict,
    dataset_spec: dict,
    execution_model: dict,
    backtest_config: dict,
    portfolio_state: dict,
    dataset_metadata: dict,
    generation: dict | None = None,
) -> dict:
    run_definition = {
        "logicVersion": RUN_STORE_LOGIC_VERSION,
        "runKind": run_kind,
        "candidate": candidate,
        "datasetSpec": {
            "tickers": dataset_spec["tickers"],
            "period": dataset_spec["period"],
            "frequency": dataset_spec["frequency"],
            "alignedStartDate": dataset_metadata["aligned_start_date"],
            "alignedEndDate": dataset_metadata["aligned_end_date"],
            "rowCount": dataset_metadata["row_count"],
        },
        "executionModel": execution_model,
        "backtestConfig": backtest_config,
        "portfolioState": portfolio_state,
    }
    if generation is not None:
        run_definition["generation"] = generation
    return run_definition
