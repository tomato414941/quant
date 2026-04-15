import copy
import json

import pandas as pd

from app import cli as cli_module
from app import main as main_module


def fake_fetch_market_universe_bundle(
    tickers: list[str],
    period: str,
    timeframe: str = "1d",
) -> tuple[dict, dict]:
    if period == "3y":
        closes = pd.DataFrame(
            {
                "SPY": [100, 102, 104, 103, 105, 107, 108],
                "QQQ": [100, 104, 107, 109, 111, 114, 116],
                "IWM": [100, 101, 102, 102, 103, 104, 105],
                "EFA": [100, 101, 103, 104, 105, 106, 108],
                "EEM": [100, 99, 101, 102, 104, 105, 106],
                "EWJ": [100, 100, 101, 102, 103, 104, 105],
                "EWZ": [100, 98, 100, 103, 105, 106, 108],
                "VNQ": [100, 101, 103, 102, 104, 105, 106],
                "TLT": [100, 99, 98, 99, 100, 101, 102],
                "IEF": [100, 100, 100, 101, 101, 102, 102],
                "LQD": [100, 100, 101, 102, 102, 103, 104],
                "HYG": [100, 101, 102, 103, 104, 105, 106],
                "TIP": [100, 100, 101, 101, 102, 103, 104],
                "GLD": [100, 100, 101, 102, 102, 103, 104],
                "SLV": [100, 101, 103, 104, 105, 107, 108],
                "DBC": [100, 101, 100, 102, 103, 104, 105],
                "USO": [100, 103, 101, 104, 106, 108, 109],
                "UUP": [100, 99, 99, 100, 101, 101, 102],
                "BTC-USD": [100, 106, 108, 111, 113, 117, 119],
                "ETH-USD": [100, 107, 109, 114, 118, 121, 124],
            },
            index=[
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
                "2025-01-06",
                "2025-01-07",
            ],
        )
    else:
        closes = pd.DataFrame(
            {
                "SPY": [100, 101, 103, 102, 104, 106, 107],
                "QQQ": [100, 103, 105, 107, 108, 110, 112],
                "IWM": [100, 99, 100, 101, 103, 102, 104],
                "EFA": [100, 101, 102, 103, 104, 105, 106],
                "EEM": [100, 98, 99, 100, 101, 102, 103],
                "EWJ": [100, 100, 101, 102, 102, 103, 104],
                "EWZ": [100, 97, 98, 100, 102, 104, 105],
                "VNQ": [100, 101, 102, 101, 103, 104, 105],
                "TLT": [100, 100, 99, 100, 101, 102, 101],
                "IEF": [100, 100, 100, 100, 101, 101, 102],
                "LQD": [100, 100, 101, 101, 102, 102, 103],
                "HYG": [100, 101, 102, 102, 103, 104, 105],
                "TIP": [100, 100, 100, 101, 101, 102, 103],
                "GLD": [100, 101, 100, 102, 103, 104, 105],
                "SLV": [100, 101, 102, 103, 104, 105, 106],
                "DBC": [100, 99, 100, 101, 102, 103, 104],
                "USO": [100, 101, 100, 102, 104, 105, 107],
                "UUP": [100, 99, 99, 100, 100, 101, 102],
                "BTC-USD": [100, 105, 107, 110, 112, 115, 118],
                "ETH-USD": [100, 106, 108, 112, 115, 119, 122],
            },
            index=[
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
                "2025-01-06",
                "2025-01-07",
            ],
        )

    aligned_closes = closes[tickers]
    volumes = pd.DataFrame(
        {
            ticker: [1_000_000 + row_index * 10_000 for row_index in range(len(aligned_closes))]
            for ticker in aligned_closes.columns
        },
        index=aligned_closes.index,
    )
    return {
        "closes": aligned_closes,
        "volumes": volumes,
    }, {
        "tickers": tickers,
        "period": period,
        "source": "test",
        "timeframe": timeframe,
        "aligned_start_date": aligned_closes.index[0],
        "aligned_end_date": aligned_closes.index[-1],
        "row_count": len(aligned_closes),
    }


def configure_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "fetch_market_universe_bundle", fake_fetch_market_universe_bundle)
    config = copy.deepcopy(main_module.DEFAULT_COMPARISON_SPEC)
    config.result_store_dir = str(tmp_path / "run_results")
    monkeypatch.setattr(cli_module, "DEFAULT_COMPARISON_SPEC", config)
    return config


def test_dashboard_summary_command(monkeypatch, tmp_path, capsys) -> None:
    config = configure_cli(monkeypatch, tmp_path)

    exit_code = cli_module.main(["dashboard-summary", "--top", "2"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert config.comparison_id in captured.out
    assert "Top 2 candidate runs:" in captured.out
    assert "Sharpe" in captured.out


def test_run_catalog_command_json(monkeypatch, tmp_path, capsys) -> None:
    config = configure_cli(monkeypatch, tmp_path)

    cli_module.main(["dashboard-summary", "--top", "1"])
    capsys.readouterr()

    exit_code = cli_module.main(["run-catalog", "--limit", "3", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["comparisonId"] == config.comparison_id
    assert payload["recordCount"] > 0
