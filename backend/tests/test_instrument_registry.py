from app.comparison_models import scale_cost_model_spec
from app.execution_defaults import build_realistic_multi_asset_cost_model_spec
from app.instrument_registry import (
    ETF_ONLY_TICKERS,
    GLOBAL_MULTI_ASSET_TICKERS,
    build_instrument_diagnostics,
    get_instrument,
    resolve_universe_variant_excluded_tickers,
)


def test_instrument_registry_defines_multi_asset_universe_metadata() -> None:
    assert len(GLOBAL_MULTI_ASSET_TICKERS) == 20
    assert len(ETF_ONLY_TICKERS) == 18

    btc = get_instrument("BTC-USD")
    assert btc is not None
    assert btc.asset_class == "crypto"
    assert btc.market_calendar == "24_7"
    assert btc.default_venue == "crypto_spot"

    spy = get_instrument("SPY")
    assert spy is not None
    assert spy.asset_class == "equity_etf"
    assert spy.market_calendar == "nyse"


def test_universe_variants_resolve_from_instrument_metadata() -> None:
    assert resolve_universe_variant_excluded_tickers(GLOBAL_MULTI_ASSET_TICKERS, "crypto_included") == set()
    assert resolve_universe_variant_excluded_tickers(GLOBAL_MULTI_ASSET_TICKERS, "btc_only") == {"ETH-USD"}
    assert resolve_universe_variant_excluded_tickers(GLOBAL_MULTI_ASSET_TICKERS, "no_crypto") == {
        "BTC-USD",
        "ETH-USD",
    }


def test_cost_profile_builds_asset_overrides_from_instruments() -> None:
    cost_model = build_realistic_multi_asset_cost_model_spec()

    assert cost_model.parameters["commissionPct"] == 0.05
    assert cost_model.per_asset_overrides["SPY"]["commissionPct"] == 0.02
    assert cost_model.per_asset_overrides["BTC-USD"]["commissionPct"] == 0.10
    assert cost_model.per_asset_overrides["BTC-USD"]["slippagePct"] == 0.15


def test_cost_multiplier_scales_defaults_and_overrides() -> None:
    cost_model = build_realistic_multi_asset_cost_model_spec()

    scaled = scale_cost_model_spec(cost_model, 2.0)

    assert scaled.parameters["commissionPct"] == 0.10
    assert scaled.parameters["slippagePct"] == 0.04
    assert scaled.parameters["impactCoefficientPct"] == 0.16
    assert scaled.parameters["advWindowBars"] == cost_model.parameters["advWindowBars"]
    assert scaled.parameters["minAdvNotional"] == cost_model.parameters["minAdvNotional"]
    assert scaled.per_asset_overrides["BTC-USD"]["commissionPct"] == 0.20
    assert scaled.per_asset_overrides["BTC-USD"]["slippagePct"] == 0.30


def test_instrument_diagnostics_reports_mixed_market_calendars() -> None:
    diagnostics = build_instrument_diagnostics(
        ("SPY", "BTC-USD"),
        cost_profile_key="retail_multi_asset_default",
    )

    assert diagnostics["costProfileKey"] == "retail_multi_asset_default"
    assert diagnostics["mixedMarketCalendar"] is True
    assert diagnostics["marketCalendars"] == {"24_7": 1, "nyse": 1}
    assert diagnostics["assetClassCounts"] == {"crypto": 1, "equity_etf": 1}
