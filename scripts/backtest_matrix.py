from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


STRATEGY_KEYS = (
    "stg-fu-eq",
    "stg-fu-eq-month",
    "stg-fu-eq-week",
    "stg-fu-eq-day",
    "stg-fu-rb",
    "stg-fu-rb-month",
    "stg-fu-rb-week",
    "stg-fu-rb-day",
    "stg-fu-minvar",
    "stg-fu-minvar-month",
    "stg-fu-minvar-week",
    "stg-fu-minvar-day",
    "stg-fu-hrp",
    "stg-fu-hrp-month",
    "stg-fu-hrp-week",
    "stg-fu-hrp-day",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full-universe backtest-strategy matrix for one market snapshot."
    )
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--market-snapshot-dir", required=True)
    parser.add_argument("--transaction-cost", default="0.001")
    parser.add_argument("--initial-capital", default="10000")
    parser.add_argument("--bars-per-year", default="252")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--output")
    return parser.parse_args()


def run_strategy(args: argparse.Namespace, strategy_key: str) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "app",
        "backtest-strategy",
        "--strategy-key",
        strategy_key,
        "--snapshot-id",
        args.snapshot_id,
        "--market-snapshot-dir",
        args.market_snapshot_dir,
        "--transaction-cost",
        args.transaction_cost,
        "--initial-capital",
        args.initial_capital,
        "--bars-per-year",
        args.bars_per_year,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def build_result_id(result: dict[str, object], context: dict[str, object]) -> str:
    payload = {
        "kind": "backtest_strategy_result",
        "context": context,
        "result": result,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_result(args: argparse.Namespace) -> dict[str, object]:
    payloads = [run_strategy(args, strategy_key) for strategy_key in STRATEGY_KEYS]
    rows = []
    snapshot = payloads[0]["snapshot"] if payloads else {}
    context = {
        "snapshotId": snapshot.get("snapshotId"),
        "transactionCost": float(args.transaction_cost),
        "initialCapital": float(args.initial_capital),
        "barsPerYear": float(args.bars_per_year),
    }
    for payload in payloads:
        result = payload["result"]
        summary = result["summary"]
        row = {
            "strategyKey": result["strategyKey"],
            "totalReturnPct": summary["totalReturnPct"],
            "cagrPct": summary["cagrPct"],
            "sharpeRatio": summary["sharpeRatio"],
            "maxDrawdownPct": summary["maxDrawdownPct"],
            "turnoverPct": summary["turnoverPct"],
            "firstInvestedDate": result["firstInvestedDate"],
            "seriesCount": result["seriesCount"],
            "eventCount": result["eventCount"],
        }
        row["resultId"] = build_result_id(result, context)
        rows.append(row)
    return {
        "kind": "backtest_strategy_matrix",
        "snapshot": snapshot,
        "transactionCost": float(args.transaction_cost),
        "initialCapital": float(args.initial_capital),
        "barsPerYear": float(args.bars_per_year),
        "strategyCount": len(rows),
        "results": rows,
    }


def render_markdown(result: dict[str, object]) -> str:
    snapshot = result["snapshot"]
    lines = [
        "# Backtest Strategy Matrix",
        "",
        f"- Snapshot: `{snapshot['snapshotId']}`",
        f"- Source: {snapshot['source']}",
        f"- Period: {snapshot['startDate']} to {snapshot['endDate']}",
        f"- Timeframe: `{snapshot['timeframe']}`",
        f"- Rows: {snapshot['rowCount']}",
        f"- Transaction cost: {result['transactionCost']}",
        "",
        "| Result ID | Strategy | Return | CAGR | Sharpe | MDD | Turnover | Events |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["results"]:
        lines.append(
            "| {resultId} | {strategyKey} | {totalReturnPct:.2f}% | {cagrPct:.2f}% | "
            "{sharpeRatio:.2f} | {maxDrawdownPct:.2f}% | {turnoverPct:.2f}% | {eventCount} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    result = build_result(args)
    output = json.dumps(result, ensure_ascii=False, indent=2) if args.as_json else render_markdown(result)
    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
