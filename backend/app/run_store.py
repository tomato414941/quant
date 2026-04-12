from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


RUN_STORE_LOGIC_VERSION = "v48"


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
        limit: int | None = None,
    ) -> list[dict]:
        records: list[dict] = []
        for path in self.root_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            run_spec = payload.get("runSpec", {})
            if run_spec.get("logicVersion") != RUN_STORE_LOGIC_VERSION:
                continue
            if run_kind is not None and run_spec.get("runKind") != run_kind:
                continue
            generation = run_spec.get("generation", {})
            if generation_method is not None and generation.get("method") != generation_method:
                continue
            records.append(
                {
                    "runKey": path.stem,
                    "savedAtUtc": payload.get("savedAtUtc"),
                    "runSpec": run_spec,
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
        payload = {
            "savedAtUtc": datetime.now(timezone.utc).isoformat(),
            "runSpec": run_spec,
            "result": result,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def _path_for(self, run_spec: dict) -> Path:
        digest = build_run_cache_key(run_spec)
        return self.root_dir / f"{digest}.json"


def build_run_cache_key(run_spec: dict) -> str:
    serialized = json.dumps(run_spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_run_spec(
    *,
    run_kind: str,
    strategy: dict,
    market_slice: dict,
    evaluation: dict,
    execution_assumptions: dict,
    portfolio_state: dict,
    capital_base: float,
    generation: dict | None = None,
) -> dict:
    run_spec = {
        "kind": "run_spec",
        "schemaVersion": "v1",
        "logicVersion": RUN_STORE_LOGIC_VERSION,
        "runKind": run_kind,
        "strategy": strategy,
        "marketSlice": market_slice,
        "portfolioState": portfolio_state,
        "capitalBase": capital_base,
        "evaluation": evaluation,
        "executionAssumptions": execution_assumptions,
    }
    if generation is not None:
        run_spec["generation"] = generation
    return run_spec
