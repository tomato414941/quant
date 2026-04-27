import pytest

from app.portfolio_metrics import summarize_metrics_from_bar_returns


def test_cagr_uses_bar_return_count_as_elapsed_periods() -> None:
    summary = summarize_metrics_from_bar_returns(
        dates=[f"2025-{month:02d}-28" for month in range(1, 13)],
        bar_returns=[0.01] * 12,
        bars_per_year=12,
        equity_key="portfolioEquity",
        turnover=0.0,
    )

    assert summary["totalReturnPct"] == pytest.approx(12.68, abs=0.01)
    assert summary["cagrPct"] == pytest.approx(12.68, abs=0.01)
