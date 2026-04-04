from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


RUN_STORE_LOGIC_VERSION = "v1"


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
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["result"]

    def save(self, run_definition: dict, result: dict) -> None:
        path = self._path_for(run_definition)
        payload = {
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
    candidate: dict,
    dataset_spec: dict,
    execution_model: dict,
    backtest_config: dict,
    portfolio_state: dict,
    dataset_metadata: dict,
) -> dict:
    return {
        "logicVersion": RUN_STORE_LOGIC_VERSION,
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
