from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    asset_class: str
    market_calendar: str
    default_venue: str
    currency: str
    cost_group: str


@dataclass(frozen=True)
class CostProfileSpec:
    key: str
    label: str
    default_parameters: dict[str, float]
    group_overrides: dict[str, dict[str, float]]


@dataclass(frozen=True)
class UniverseVariantSpec:
    key: str
    label: str
    include_asset_classes: tuple[str, ...] = ()
    include_symbols: tuple[str, ...] = ()
    exclude_asset_classes: tuple[str, ...] = ()
    exclude_symbols: tuple[str, ...] = ()


INSTRUMENTS: tuple[InstrumentSpec, ...] = (
    InstrumentSpec("SPY", "equity_etf", "nyse", "us_brokerage_etf", "USD", "us_liquid_etf"),
    InstrumentSpec("QQQ", "equity_etf", "nyse", "us_brokerage_etf", "USD", "us_liquid_etf"),
    InstrumentSpec("IWM", "equity_etf", "nyse", "us_brokerage_etf", "USD", "us_core_etf"),
    InstrumentSpec("EFA", "equity_etf", "nyse", "us_brokerage_etf", "USD", "developed_market_etf"),
    InstrumentSpec("EEM", "equity_etf", "nyse", "us_brokerage_etf", "USD", "emerging_market_etf"),
    InstrumentSpec("EWJ", "equity_etf", "nyse", "us_brokerage_etf", "USD", "developed_market_etf"),
    InstrumentSpec("EWZ", "equity_etf", "nyse", "us_brokerage_etf", "USD", "higher_friction_etf"),
    InstrumentSpec("VNQ", "real_estate_etf", "nyse", "us_brokerage_etf", "USD", "us_core_etf"),
    InstrumentSpec("TLT", "bond_etf", "nyse", "us_brokerage_etf", "USD", "us_liquid_etf"),
    InstrumentSpec("IEF", "bond_etf", "nyse", "us_brokerage_etf", "USD", "us_liquid_etf"),
    InstrumentSpec("LQD", "bond_etf", "nyse", "us_brokerage_etf", "USD", "bond_credit_etf"),
    InstrumentSpec("HYG", "bond_etf", "nyse", "us_brokerage_etf", "USD", "high_yield_etf"),
    InstrumentSpec("TIP", "bond_etf", "nyse", "us_brokerage_etf", "USD", "bond_credit_etf"),
    InstrumentSpec("GLD", "commodity_etf", "nyse", "us_brokerage_etf", "USD", "commodity_core_etf"),
    InstrumentSpec("SLV", "commodity_etf", "nyse", "us_brokerage_etf", "USD", "commodity_higher_friction_etf"),
    InstrumentSpec("DBC", "commodity_etf", "nyse", "us_brokerage_etf", "USD", "higher_friction_etf"),
    InstrumentSpec("USO", "commodity_etf", "nyse", "us_brokerage_etf", "USD", "commodity_high_friction_etf"),
    InstrumentSpec("UUP", "currency_etf", "nyse", "us_brokerage_etf", "USD", "developed_market_etf"),
    InstrumentSpec("BTC-USD", "crypto", "24_7", "crypto_spot", "USD", "crypto_spot"),
    InstrumentSpec("ETH-USD", "crypto", "24_7", "crypto_spot", "USD", "crypto_spot"),
)

INSTRUMENTS_BY_SYMBOL = {instrument.symbol: instrument for instrument in INSTRUMENTS}
NORMALIZED_ASSET_CLASS_BY_INSTRUMENT_CLASS = {
    "equity_etf": "equity",
    "real_estate_etf": "real_estate",
    "bond_etf": "bond",
    "commodity_etf": "commodity",
    "currency_etf": "currency",
    "crypto": "crypto",
}

ETF_TICKERS = tuple(
    instrument.symbol
    for instrument in INSTRUMENTS
    if instrument.asset_class != "crypto"
)
CRYPTO_TICKERS = tuple(
    instrument.symbol
    for instrument in INSTRUMENTS
    if instrument.asset_class == "crypto"
)
ETF_PLUS_CRYPTO_TICKERS = (*ETF_TICKERS, *CRYPTO_TICKERS)

RETAIL_MULTI_ASSET_DEFAULT_COST_PROFILE = CostProfileSpec(
    key="retail_multi_asset_default",
    label="Retail multi-asset default",
    default_parameters={
        "commissionPct": 0.05,
        "slippagePct": 0.02,
        "impactCoefficientPct": 0.08,
        "advWindowBars": 20.0,
        "minAdvNotional": 1_000_000.0,
    },
    group_overrides={
        "us_liquid_etf": {"commissionPct": 0.02, "slippagePct": 0.01, "impactCoefficientPct": 0.02},
        "us_core_etf": {"commissionPct": 0.03, "slippagePct": 0.02, "impactCoefficientPct": 0.04},
        "developed_market_etf": {"commissionPct": 0.03, "slippagePct": 0.02, "impactCoefficientPct": 0.03},
        "emerging_market_etf": {"commissionPct": 0.04, "slippagePct": 0.03, "impactCoefficientPct": 0.06},
        "bond_credit_etf": {"commissionPct": 0.02, "slippagePct": 0.02, "impactCoefficientPct": 0.03},
        "high_yield_etf": {"commissionPct": 0.03, "slippagePct": 0.03, "impactCoefficientPct": 0.05},
        "commodity_core_etf": {"commissionPct": 0.03, "slippagePct": 0.03, "impactCoefficientPct": 0.04},
        "commodity_higher_friction_etf": {"commissionPct": 0.04, "slippagePct": 0.04, "impactCoefficientPct": 0.07},
        "higher_friction_etf": {"commissionPct": 0.05, "slippagePct": 0.05, "impactCoefficientPct": 0.10},
        "commodity_high_friction_etf": {"commissionPct": 0.06, "slippagePct": 0.06, "impactCoefficientPct": 0.12},
        "crypto_spot": {"commissionPct": 0.10, "slippagePct": 0.15, "impactCoefficientPct": 0.25},
    },
)

DEFAULT_COST_PROFILE = RETAIL_MULTI_ASSET_DEFAULT_COST_PROFILE

UNIVERSE_VARIANTS = {
    "crypto_included": UniverseVariantSpec(
        key="crypto_included",
        label="Crypto included",
    ),
    "only_etf": UniverseVariantSpec(
        key="only_etf",
        label="Only ETFs",
        exclude_asset_classes=("crypto",),
    ),
    "btc_only": UniverseVariantSpec(
        key="btc_only",
        label="BTC only",
        exclude_symbols=("ETH-USD",),
    ),
    "only_crypto": UniverseVariantSpec(
        key="only_crypto",
        label="Only crypto",
        include_asset_classes=("crypto",),
    ),
    "no_crypto": UniverseVariantSpec(
        key="no_crypto",
        label="No crypto",
        exclude_asset_classes=("crypto",),
    ),
    "only_equity": UniverseVariantSpec(
        key="only_equity",
        label="Only equity ETFs",
        include_asset_classes=("equity_etf",),
    ),
    "no_equity": UniverseVariantSpec(
        key="no_equity",
        label="No equity ETFs",
        exclude_asset_classes=("equity_etf",),
    ),
    "only_real_estate": UniverseVariantSpec(
        key="only_real_estate",
        label="Only real estate ETFs",
        include_asset_classes=("real_estate_etf",),
    ),
    "no_real_estate": UniverseVariantSpec(
        key="no_real_estate",
        label="No real estate ETFs",
        exclude_asset_classes=("real_estate_etf",),
    ),
    "only_bond": UniverseVariantSpec(
        key="only_bond",
        label="Only bond ETFs",
        include_asset_classes=("bond_etf",),
    ),
    "no_bond": UniverseVariantSpec(
        key="no_bond",
        label="No bond ETFs",
        exclude_asset_classes=("bond_etf",),
    ),
    "only_commodity": UniverseVariantSpec(
        key="only_commodity",
        label="Only commodity ETFs",
        include_asset_classes=("commodity_etf",),
    ),
    "no_commodity": UniverseVariantSpec(
        key="no_commodity",
        label="No commodity ETFs",
        exclude_asset_classes=("commodity_etf",),
    ),
    "only_currency": UniverseVariantSpec(
        key="only_currency",
        label="Only currency ETFs",
        include_asset_classes=("currency_etf",),
    ),
    "no_currency": UniverseVariantSpec(
        key="no_currency",
        label="No currency ETFs",
        exclude_asset_classes=("currency_etf",),
    ),
}
UNIVERSE_VARIANT_KEYS = tuple(UNIVERSE_VARIANTS)


def get_instrument(symbol: str) -> InstrumentSpec | None:
    return INSTRUMENTS_BY_SYMBOL.get(symbol)


def get_normalized_asset_class(symbol: str) -> str:
    instrument = get_instrument(symbol)
    if instrument is None:
        return "other"
    return NORMALIZED_ASSET_CLASS_BY_INSTRUMENT_CLASS.get(instrument.asset_class, instrument.asset_class)


def build_cost_overrides_for_profile(
    profile: CostProfileSpec = DEFAULT_COST_PROFILE,
    symbols: tuple[str, ...] = ETF_PLUS_CRYPTO_TICKERS,
) -> dict[str, dict[str, float]]:
    overrides: dict[str, dict[str, float]] = {}
    for symbol in symbols:
        instrument = get_instrument(symbol)
        if instrument is None:
            continue
        group_override = profile.group_overrides.get(instrument.cost_group)
        if group_override is not None:
            overrides[symbol] = dict(group_override)
    return overrides


def get_universe_variant(key: str) -> UniverseVariantSpec:
    try:
        return UNIVERSE_VARIANTS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown universe variant: {key}") from exc


def resolve_universe_variant_excluded_tickers(
    tickers: tuple[str, ...],
    variant_key: str,
) -> set[str]:
    variant = get_universe_variant(variant_key)
    excluded_symbols = set(variant.exclude_symbols)
    included_symbols = set(variant.include_symbols)
    included_asset_classes = set(variant.include_asset_classes)
    excluded_asset_classes = set(variant.exclude_asset_classes)
    for ticker in tickers:
        instrument = get_instrument(ticker)
        if included_symbols or included_asset_classes:
            if ticker in included_symbols:
                continue
            if instrument is not None and instrument.asset_class in included_asset_classes:
                continue
            excluded_symbols.add(ticker)
            continue
        if instrument is None:
            continue
        if instrument.asset_class in excluded_asset_classes:
            excluded_symbols.add(ticker)
    return excluded_symbols & set(tickers)


def build_instrument_diagnostics(
    symbols: tuple[str, ...],
    *,
    cost_profile_key: str,
) -> dict[str, object]:
    asset_class_counts: dict[str, int] = {}
    market_calendars: dict[str, int] = {}
    unknown_symbols = []
    for symbol in symbols:
        instrument = get_instrument(symbol)
        if instrument is None:
            unknown_symbols.append(symbol)
            continue
        asset_class_counts[instrument.asset_class] = asset_class_counts.get(instrument.asset_class, 0) + 1
        market_calendars[instrument.market_calendar] = market_calendars.get(instrument.market_calendar, 0) + 1
    return {
        "costProfileKey": cost_profile_key,
        "assetClassCounts": dict(sorted(asset_class_counts.items())),
        "marketCalendars": dict(sorted(market_calendars.items())),
        "mixedMarketCalendar": len(market_calendars) > 1,
        "unknownSymbols": sorted(unknown_symbols),
    }
