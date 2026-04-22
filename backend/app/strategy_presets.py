from __future__ import annotations

from app.execution_defaults import (
    DEFAULT_ANNUAL_EXECUTION_POLICY,
    DEFAULT_EVERY_BAR_EXECUTION_POLICY,
    DEFAULT_MONTH_END_EXECUTION_POLICY,
    REFERENCE_HOLD_EXECUTION_POLICY,
)
from app.portfolio import (
    build_investment_universe_spec,
    build_portfolio_model_spec,
    build_risk_controls_spec,
    build_selection_spec,
)
from app.timeframe_models import (
    DEFAULT_DAILY_TIMEFRAME,
    DEFAULT_MONTHLY_TIMEFRAME,
    DEFAULT_WEEKLY_TIMEFRAME,
)
from app.instrument_registry import GLOBAL_MULTI_ASSET_TICKERS, ETF_ONLY_TICKERS


def window_spec(*, unit: str, value: float) -> dict[str, float | str]:
    return {"unit": unit, "value": value}


FULL_UNIVERSE = build_selection_spec(
    strategy_type="full_universe",
    key="full_universe",
    label="全資産",
    description="全ETFを候補にして配分する",
)
FULL_UNIVERSE_MOMENTUM_TILT = build_selection_spec(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt",
    label="全資産モメンタム傾斜",
    description="全ETFを候補に残しつつ、モメンタムで重みだけを傾ける",
    score_parameters={"tilt_strength": 0.5, "windowSpec": window_spec(unit="months", value=12)},
)
FULL_UNIVERSE_MOMENTUM_TILT_WEAK = build_selection_spec(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_weak",
    label="全資産モメンタム傾斜 弱 12ヶ月",
    description="全ETFを候補に残しつつ、弱めの12ヶ月モメンタム傾斜で重みを調整する",
    score_parameters={"tilt_strength": 0.25, "tilt_shape": 0.0, "windowSpec": window_spec(unit="months", value=12)},
)
FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP = build_selection_spec(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_weak_top",
    label="全資産モメンタム傾斜 上位優遇 12ヶ月",
    description="全ETFを候補に残しつつ、12ヶ月モメンタムの上位優遇傾斜で重みを調整する",
    score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": window_spec(unit="months", value=12)},
)
FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_6M = build_selection_spec(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_weak_top_6m",
    label="全資産モメンタム傾斜 上位優遇 6ヶ月",
    description="全ETFを候補に残しつつ、6ヶ月モメンタムの上位優遇傾斜で重みを調整する",
    score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": window_spec(unit="months", value=6)},
)
FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_9M = build_selection_spec(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_weak_top_9m",
    label="全資産モメンタム傾斜 上位優遇 9ヶ月",
    description="全ETFを候補に残しつつ、9ヶ月モメンタムの上位優遇傾斜で重みを調整する",
    score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": window_spec(unit="months", value=9)},
)
FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_8M = build_selection_spec(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_weak_top_8m",
    label="全資産モメンタム傾斜 上位優遇 8ヶ月",
    description="全ETFを候補に残しつつ、8ヶ月モメンタムの上位優遇傾斜で重みを調整する",
    score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": window_spec(unit="months", value=8)},
)
FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_10M = build_selection_spec(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_weak_top_10m",
    label="全資産モメンタム傾斜 上位優遇 10ヶ月",
    description="全ETFを候補に残しつつ、10ヶ月モメンタムの上位優遇傾斜で重みを調整する",
    score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": window_spec(unit="months", value=10)},
)
FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_11M = build_selection_spec(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_weak_top_11m",
    label="全資産モメンタム傾斜 上位優遇 11ヶ月",
    description="全ETFを候補に残しつつ、11ヶ月モメンタムの上位優遇傾斜で重みを調整する",
    score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": window_spec(unit="months", value=11)},
)
FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_3M = build_selection_spec(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_weak_top_3m",
    label="全資産モメンタム傾斜 上位優遇 3ヶ月",
    description="全ETFを候補に残しつつ、3ヶ月モメンタムの上位優遇傾斜で重みを調整する",
    score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": window_spec(unit="months", value=3)},
)
FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_1M = build_selection_spec(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_weak_top_1m",
    label="全資産モメンタム傾斜 上位優遇 1ヶ月",
    description="全ETFを候補に残しつつ、1ヶ月モメンタムの上位優遇傾斜で重みを調整する",
    score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": window_spec(unit="months", value=1)},
)
FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M = build_selection_spec(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_weak_top_2m",
    label="全資産モメンタム傾斜 上位優遇 2ヶ月",
    description="全ETFを候補に残しつつ、2ヶ月モメンタムの上位優遇傾斜で重みを調整する",
    score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": window_spec(unit="months", value=2)},
)
FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_15M = build_selection_spec(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_weak_top_15m",
    label="全資産モメンタム傾斜 上位優遇 15ヶ月",
    description="全ETFを候補に残しつつ、15ヶ月モメンタムの上位優遇傾斜で重みを調整する",
    score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": window_spec(unit="months", value=15)},
)
FULL_UNIVERSE_MOMENTUM_LOW_VOL_TILT_WEAK_TOP = build_selection_spec(
    strategy_type="full_universe_momentum_low_vol_tilt",
    key="full_universe_momentum_low_vol_tilt_weak_top",
    label="全資産モメンタム低ボラ傾斜 弱 上位優遇 12ヶ月",
    description="全ETFを候補に残しつつ、12ヶ月モメンタムと低ボラの複合スコアで弱く上位優遇する",
    score_parameters={
        "tilt_strength": 0.25,
        "tilt_shape": 1.0,
        "windowSpec": window_spec(unit="months", value=12),
        "momentum_weight": 0.7,
        "low_vol_weight": 0.3,
    },
)
FULL_UNIVERSE_MOMENTUM_LOW_VOL_TILT_LIGHT_TOP = build_selection_spec(
    strategy_type="full_universe_momentum_low_vol_tilt",
    key="full_universe_momentum_low_vol_tilt_light_top",
    label="全資産モメンタム低ボラ傾斜 低ボラ弱め 上位優遇",
    description="全ETFを候補に残しつつ、12ヶ月モメンタムを主役に低ボラを弱く混ぜて上位優遇する",
    score_parameters={
        "tilt_strength": 0.25,
        "tilt_shape": 1.0,
        "windowSpec": window_spec(unit="months", value=12),
        "momentum_weight": 0.85,
        "low_vol_weight": 0.15,
    },
)
FULL_UNIVERSE_MOMENTUM_MACRO_TILT_LIGHT_TOP = build_selection_spec(
    strategy_type="full_universe_momentum_macro_tilt",
    key="full_universe_momentum_macro_tilt_light_top",
    label="全資産モメンタムマクロ傾斜 マクロ弱め 上位優遇",
    description="全ETFを候補に残しつつ、12ヶ月モメンタムを主役にマクロproxyを弱く混ぜて上位優遇する",
    score_parameters={
        "tilt_strength": 0.25,
        "tilt_shape": 1.0,
        "windowSpec": window_spec(unit="months", value=12),
        "momentum_weight": 0.85,
        "macro_weight": 0.15,
    },
)
FULL_UNIVERSE_MOMENTUM_TILT_WEAK_SOFTMAX = build_selection_spec(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_weak_softmax",
    label="全資産モメンタム傾斜 弱 softmax",
    description="全ETFを候補に残しつつ、弱めのsoftmax型モメンタム傾斜で重みを調整する",
    score_parameters={"tilt_strength": 0.25, "tilt_shape": 2.0, "windowSpec": window_spec(unit="months", value=12)},
)
FULL_UNIVERSE_MOMENTUM_TILT_STRONG = build_selection_spec(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_strong",
    label="全資産モメンタム傾斜 強",
    description="全ETFを候補に残しつつ、強めの12ヶ月モメンタム傾斜で重みを調整する",
    score_parameters={"tilt_strength": 1.0, "tilt_shape": 0.0, "windowSpec": window_spec(unit="months", value=12)},
)
MOMENTUM_TOP3 = build_selection_spec(
    strategy_type="momentum_top3",
    key="momentum_top3",
    label="モメンタム上位3",
    description="学習期間で強かった上位3ETFに絞って配分する",
)
DUAL_MOMENTUM_TOP3 = build_selection_spec(
    strategy_type="dual_momentum_top3",
    key="dual_momentum_top3",
    label="デュアルモメンタム上位3",
    description="上昇しているETFだけからモメンタム上位3を選び、弱い相場ではCASHへ逃がす",
)
POSITIVE_MOMENTUM_UNIVERSE = build_selection_spec(
    strategy_type="positive_momentum_universe",
    key="positive_momentum_universe",
    label="上昇資産",
    description="上昇しているETFだけを候補にして配分する",
)
POSITIVE_MOMENTUM_LOW_VOL_UNIVERSE = build_selection_spec(
    strategy_type="positive_momentum_low_vol_universe",
    key="positive_momentum_low_vol_universe",
    label="上昇低ボラ資産",
    description="上昇しているETFのうち、低ボラ群だけを候補にして配分する",
)
TRAILING_MOMENTUM_LOW_VOL_UNIVERSE = build_selection_spec(
    strategy_type="trailing_momentum_low_vol_universe",
    key="trailing_momentum_low_vol_universe",
    label="12ヶ月モメンタム低ボラ資産",
    description="12ヶ月モメンタムが正のETFを候補にし、低ボラ群だけをHRPへ渡す",
)
POSITIVE_MOMENTUM_HIGH_VOLUME_UNIVERSE = build_selection_spec(
    strategy_type="positive_momentum_high_volume_universe",
    key="positive_momentum_high_volume_universe",
    label="上昇出来高資産",
    description="上昇しているETFのうち、出来高が強い群だけを候補にして配分する",
)
POSITIVE_TREND_SHORT_REVERSAL = build_selection_spec(
    strategy_type="positive_trend_short_reversal",
    key="positive_trend_short_reversal",
    label="上昇トレンド短期リバーサル",
    description="長期上昇中で直近短期に売られたETFだけを候補にして配分する",
    score_parameters={
        "trendWindowSpec": window_spec(unit="days", value=60),
        "reversalWindowSpec": window_spec(unit="days", value=5),
        "assetCount": 5,
    },
)
RISK_REGIME_POSITIVE_MOMENTUM = build_selection_spec(
    strategy_type="risk_regime_positive_momentum",
    key="risk_regime_positive_momentum",
    label="リスク局面別上昇資産",
    description="株式リスクproxyが弱い局面では防御資産へ絞り、それ以外は上昇しているETFだけを候補にする",
    score_parameters={
        "windowSpec": window_spec(unit="bars", value=252),
        "riskWindowSpec": window_spec(unit="days", value=60),
        "riskProxyAssets": ("SPY", "QQQ"),
        "defensiveAssetClasses": ("bond_etf", "commodity_etf", "currency_etf"),
    },
)

EQUAL_WEIGHT = build_portfolio_model_spec(
    model_type="equal_weight",
    key="equal_weight",
    label="等金額配分",
    description="全ETFを同じ比率で持つ",
)
RISK_BUDGETING = build_portfolio_model_spec(
    model_type="risk_budgeting",
    key="risk_budgeting",
    label="リスク予算配分",
    description="各ETFのリスク寄与が近づくように配分する",
)
MINIMUM_VARIANCE = build_portfolio_model_spec(
    model_type="minimum_variance",
    key="minimum_variance",
    label="最小分散",
    description="全体の分散が最小になるように配分する",
)
HIERARCHICAL_RISK_PARITY = build_portfolio_model_spec(
    model_type="hierarchical_risk_parity",
    key="hierarchical_risk_parity",
    label="HRP",
    description="相関クラスタを使って階層的にリスクを分散する",
)
MEAN_RISK_UTILITY = build_portfolio_model_spec(
    model_type="mean_risk_utility",
    key="mean_risk_utility",
    label="MeanRisk効用最大化",
    description="モメンタム順位から作った期待リターン proxy とリスクの両方で配分する",
)
MEAN_RISK_UTILITY_CONSERVATIVE = build_portfolio_model_spec(
    model_type="mean_risk_utility_conservative",
    key="mean_risk_utility_conservative",
    label="MeanRisk効用最大化 弱",
    description="モメンタム順位の期待リターン proxy を弱めに使い、リスクをより強く見る",
)

DEFAULT_INVESTMENT_UNIVERSE = build_investment_universe_spec(
    key="global_multi_asset_v1",
    label="20資産マルチアセット",
    tickers=GLOBAL_MULTI_ASSET_TICKERS,
)

ETF_ONLY_INVESTMENT_UNIVERSE = build_investment_universe_spec(
    key="global_etf_only_v1",
    label="18資産ETF",
    tickers=ETF_ONLY_TICKERS,
)

DEFAULT_RISK_CONTROLS = build_risk_controls_spec(
    max_investment_ratio=1.0,
    max_weight=0.45,
)


__all__ = [
    "DEFAULT_ANNUAL_EXECUTION_POLICY",
    "DEFAULT_DAILY_TIMEFRAME",
    "DEFAULT_EVERY_BAR_EXECUTION_POLICY",
    "DEFAULT_INVESTMENT_UNIVERSE",
    "DEFAULT_MONTH_END_EXECUTION_POLICY",
    "DEFAULT_MONTHLY_TIMEFRAME",
    "DEFAULT_RISK_CONTROLS",
    "DEFAULT_WEEKLY_TIMEFRAME",
    "DUAL_MOMENTUM_TOP3",
    "EQUAL_WEIGHT",
    "ETF_ONLY_INVESTMENT_UNIVERSE",
    "FULL_UNIVERSE",
    "FULL_UNIVERSE_MOMENTUM_LOW_VOL_TILT_LIGHT_TOP",
    "FULL_UNIVERSE_MOMENTUM_LOW_VOL_TILT_WEAK_TOP",
    "FULL_UNIVERSE_MOMENTUM_MACRO_TILT_LIGHT_TOP",
    "FULL_UNIVERSE_MOMENTUM_TILT",
    "FULL_UNIVERSE_MOMENTUM_TILT_STRONG",
    "FULL_UNIVERSE_MOMENTUM_TILT_WEAK",
    "FULL_UNIVERSE_MOMENTUM_TILT_WEAK_SOFTMAX",
    "FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP",
    "FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_10M",
    "FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_11M",
    "FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_15M",
    "FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_1M",
    "FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M",
    "FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_3M",
    "FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_6M",
    "FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_8M",
    "FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_9M",
    "HIERARCHICAL_RISK_PARITY",
    "MEAN_RISK_UTILITY",
    "MEAN_RISK_UTILITY_CONSERVATIVE",
    "MINIMUM_VARIANCE",
    "MOMENTUM_TOP3",
    "POSITIVE_MOMENTUM_HIGH_VOLUME_UNIVERSE",
    "POSITIVE_MOMENTUM_LOW_VOL_UNIVERSE",
    "POSITIVE_MOMENTUM_UNIVERSE",
    "POSITIVE_TREND_SHORT_REVERSAL",
    "REFERENCE_HOLD_EXECUTION_POLICY",
    "RISK_REGIME_POSITIVE_MOMENTUM",
    "RISK_BUDGETING",
    "TRAILING_MOMENTUM_LOW_VOL_UNIVERSE",
    "window_spec",
]
