from __future__ import annotations

from app.portfolio import (
    build_execution_policy_spec,
    build_investment_universe_spec,
    build_portfolio_model_spec,
    build_portfolio_state,
    build_risk_controls_spec,
    build_selection_spec,
    build_strategy_spec,
)
from app.comparison_models import (
    ComparisonSpec,
    ConditionVariant,
    CostModelSpec,
    EvaluationSpec,
    EvaluationSettings,
    MarketSliceSpec,
    RunSpec,
    SelectionPolicy,
    build_asset_specific_adv_cost_model_spec,
    build_execution_assumptions_spec,
)
from app.timeframe_models import (
    DEFAULT_DAILY_TIMEFRAME,
    DEFAULT_MONTHLY_TIMEFRAME,
    DEFAULT_WEEKLY_TIMEFRAME,
)


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
    tickers=[
        "SPY",
        "QQQ",
        "IWM",
        "EFA",
        "EEM",
        "EWJ",
        "EWZ",
        "VNQ",
        "TLT",
        "IEF",
        "LQD",
        "HYG",
        "TIP",
        "GLD",
        "SLV",
        "DBC",
        "USO",
        "UUP",
        "BTC-USD",
        "ETH-USD",
    ],
)

ETF_ONLY_INVESTMENT_UNIVERSE = build_investment_universe_spec(
    key="global_etf_only_v1",
    label="18資産ETF",
    tickers=[
        "SPY",
        "QQQ",
        "IWM",
        "EFA",
        "EEM",
        "EWJ",
        "EWZ",
        "VNQ",
        "TLT",
        "IEF",
        "LQD",
        "HYG",
        "TIP",
        "GLD",
        "SLV",
        "DBC",
        "USO",
        "UUP",
    ],
)

def build_realistic_multi_asset_cost_model_spec() -> CostModelSpec:
    return build_asset_specific_adv_cost_model_spec(
        default_commission_pct=0.05,
        default_slippage_pct=0.02,
        default_impact_coefficient_pct=0.08,
        adv_window_bars=20,
        min_adv_notional=1_000_000.0,
        per_asset_overrides={
            "SPY": {"commissionPct": 0.02, "slippagePct": 0.01, "impactCoefficientPct": 0.02},
            "QQQ": {"commissionPct": 0.02, "slippagePct": 0.01, "impactCoefficientPct": 0.02},
            "IWM": {"commissionPct": 0.03, "slippagePct": 0.02, "impactCoefficientPct": 0.04},
            "EFA": {"commissionPct": 0.03, "slippagePct": 0.02, "impactCoefficientPct": 0.03},
            "EEM": {"commissionPct": 0.04, "slippagePct": 0.03, "impactCoefficientPct": 0.06},
            "EWJ": {"commissionPct": 0.03, "slippagePct": 0.02, "impactCoefficientPct": 0.03},
            "EWZ": {"commissionPct": 0.05, "slippagePct": 0.05, "impactCoefficientPct": 0.10},
            "VNQ": {"commissionPct": 0.03, "slippagePct": 0.02, "impactCoefficientPct": 0.04},
            "TLT": {"commissionPct": 0.02, "slippagePct": 0.01, "impactCoefficientPct": 0.02},
            "IEF": {"commissionPct": 0.02, "slippagePct": 0.01, "impactCoefficientPct": 0.02},
            "LQD": {"commissionPct": 0.02, "slippagePct": 0.02, "impactCoefficientPct": 0.03},
            "HYG": {"commissionPct": 0.03, "slippagePct": 0.03, "impactCoefficientPct": 0.05},
            "TIP": {"commissionPct": 0.02, "slippagePct": 0.02, "impactCoefficientPct": 0.03},
            "GLD": {"commissionPct": 0.03, "slippagePct": 0.03, "impactCoefficientPct": 0.04},
            "SLV": {"commissionPct": 0.04, "slippagePct": 0.04, "impactCoefficientPct": 0.07},
            "DBC": {"commissionPct": 0.05, "slippagePct": 0.05, "impactCoefficientPct": 0.10},
            "USO": {"commissionPct": 0.06, "slippagePct": 0.06, "impactCoefficientPct": 0.12},
            "UUP": {"commissionPct": 0.03, "slippagePct": 0.02, "impactCoefficientPct": 0.03},
            "BTC-USD": {"commissionPct": 0.10, "slippagePct": 0.15, "impactCoefficientPct": 0.25},
            "ETH-USD": {"commissionPct": 0.10, "slippagePct": 0.15, "impactCoefficientPct": 0.25},
        },
    )


def build_condition_variants() -> list[ConditionVariant]:
    variants: list[ConditionVariant] = []
    commission_values = [0.05, 0.1, 0.2]
    investment_values = [1.0, 0.9, 0.8]
    max_weight_values = [None, 0.45, 0.35]

    for commission_pct in commission_values:
        for max_investment_ratio in investment_values:
            for max_weight in max_weight_values:
                cash_pct = round((1 - max_investment_ratio) * 100)
                cap_label = "上限なし" if max_weight is None else f"{max_weight * 100:.0f}%上限"
                cap_key = "no_cap" if max_weight is None else f"cap_{int(max_weight * 100)}"
                variants.append(
                    ConditionVariant(
                        key=(
                            f"fee_{str(commission_pct).replace('.', '_')}"
                            f"__invest_{int(max_investment_ratio * 100)}"
                            f"__{cap_key}"
                        ),
                        label=(
                            f"手数料 {commission_pct:.2f}% / 投資 {int(max_investment_ratio * 100)}%"
                            f" / CASH {cash_pct}% / {cap_label}"
                        ),
                        commission_pct=commission_pct,
                        max_investment_ratio=max_investment_ratio,
                        max_weight=max_weight,
                    )
                )
    return variants


DEFAULT_ANNUAL_EXECUTION_POLICY = build_execution_policy_spec(
    key="year_end",
    label="年次",
    entry="train_once_then_periodic_rebalance",
    rebalance_schedule="year_end",
)

DEFAULT_EVERY_BAR_EXECUTION_POLICY = build_execution_policy_spec(
    key="every_bar",
    label="毎バー",
    entry="train_once_then_periodic_rebalance",
    rebalance_schedule="every_bar",
)

REFERENCE_HOLD_EXECUTION_POLICY = build_execution_policy_spec(
    key="hold",
    label="保有",
    entry="hold",
    rebalance_schedule="hold",
)

DEFAULT_EXECUTION_ASSUMPTIONS = build_execution_assumptions_spec(
    label="終値約定",
    parameters={
        "fillPrice": "close",
    },
    cost_model=build_realistic_multi_asset_cost_model_spec(),
)

DEFAULT_RISK_CONTROLS = build_risk_controls_spec(
    max_investment_ratio=1.0,
    max_weight=0.45,
)


REFERENCE_EQUAL_WEIGHT_WITH_CASH = build_strategy_spec(
    strategy_id="ref-fu-eq-cash",
    investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
    selection=FULL_UNIVERSE,
    portfolio_model=EQUAL_WEIGHT,
    execution_policy=REFERENCE_HOLD_EXECUTION_POLICY,
    risk_controls=build_risk_controls_spec(
        max_investment_ratio=0.85,
        max_weight=None,
    ),
    label="等金額買い持ち + CASH",
    description="全資産を等金額で買い持ちし、15% を CASH に残す参照用 Strategy",
)


DEFAULT_COMPARISON_SPEC = ComparisonSpec(
    comparison_id="etf_portfolio_models_10y",
    title="有望Strategyの探索",
    question="共通の評価前提で Strategy を比較し、現時点で最も有望な構成を見つける",
    run_spec=RunSpec(
        market_slice=MarketSliceSpec(
            period="10y",
            sanity_periods=["3y"],
        ),
        portfolio_state=build_portfolio_state(
            current_weights={
                "SPY": 0.0425,
                "QQQ": 0.0425,
                "IWM": 0.0425,
                "EFA": 0.0425,
                "EEM": 0.0425,
                "EWJ": 0.0425,
                "EWZ": 0.0425,
                "VNQ": 0.0425,
                "TLT": 0.0425,
                "IEF": 0.0425,
                "LQD": 0.0425,
                "HYG": 0.0425,
                "TIP": 0.0425,
                "GLD": 0.0425,
                "SLV": 0.0425,
                "DBC": 0.0425,
                "USO": 0.0425,
                "UUP": 0.0425,
                "BTC-USD": 0.0425,
                "ETH-USD": 0.0425,
            },
            cash_weight=0.15,
        ),
        capital_base=10_000,
        execution_assumptions=DEFAULT_EXECUTION_ASSUMPTIONS,
        evaluation=EvaluationSpec(
            evaluation_settings=EvaluationSettings(
                split_ratio=0.7,
            ),
        ),
    ),
    selection_policy=SelectionPolicy(
        primary_metric="sharpe_ratio",
        secondary_metric="total_return",
        tertiary_metric="max_drawdown",
    ),
    candidate_strategies=[
        build_strategy_spec(
            strategy_id="stg-fu-eq",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE,
            portfolio_model=EQUAL_WEIGHT,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-fu-rb",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE,
            portfolio_model=RISK_BUDGETING,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-fu-minvar",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE,
            portfolio_model=MINIMUM_VARIANCE,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-fu-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo12-lin025-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
            hypothesis="全資産を残した弱いモメンタム傾斜は、分散を保ちながら成績を改善しやすい",
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo12-top035-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
            hypothesis="全資産を残した上位優遇型のモメンタム傾斜は、分散を保ちながらSharpeを改善しやすい",
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo6-top035-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_6M,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
            hypothesis="全資産を残した6ヶ月モメンタムの上位優遇傾斜は、中期の強さを取り込みやすい",
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo9-top035-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_9M,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
            hypothesis="全資産を残した9ヶ月モメンタムの上位優遇傾斜は、中長期の強さを取り込みやすい",
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo9-top035-hrp-1w",
            timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_9M,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            execution_policy=DEFAULT_EVERY_BAR_EXECUTION_POLICY,
            risk_controls=DEFAULT_RISK_CONTROLS,
            label="全資産モメンタム傾斜 上位優遇 9ヶ月 × HRP × 毎バー × 週次",
            hypothesis="全資産を残した9ヶ月モメンタムを週次バーごとに反映すると、日次より低回転でトレンドを取り込みやすい",
            description="全ETFを候補に残しつつ、9ヶ月モメンタムの上位優遇傾斜を週次バーごとにHRPへ反映する",
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo9-top035-hrp-1mo",
            timeframe=DEFAULT_MONTHLY_TIMEFRAME,
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_9M,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            execution_policy=DEFAULT_EVERY_BAR_EXECUTION_POLICY,
            risk_controls=DEFAULT_RISK_CONTROLS,
            label="全資産モメンタム傾斜 上位優遇 9ヶ月 × HRP × 毎バー × 月次",
            hypothesis="全資産を残した9ヶ月モメンタムを月次バーごとに反映すると、さらに低回転で中長期トレンドを取り込みやすい",
            description="全ETFを候補に残しつつ、9ヶ月モメンタムの上位優遇傾斜を月次バーごとにHRPへ反映する",
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo8-top035-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_8M,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
            hypothesis="全資産を残した8ヶ月モメンタムの上位優遇傾斜は、中期寄りの強さを取り込みやすい",
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo10-top035-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_10M,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
            hypothesis="全資産を残した10ヶ月モメンタムの上位優遇傾斜は、中長期の強さを取り込みやすい",
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo11-top035-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_11M,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
            hypothesis="全資産を残した11ヶ月モメンタムの上位優遇傾斜は、中長期の強さを取り込みやすい",
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo3-top035-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_3M,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
            hypothesis="全資産を残した3ヶ月モメンタムの上位優遇傾斜は、短期の強さを取り込みやすい",
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo15-top035-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_15M,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
            hypothesis="全資産を残した15ヶ月モメンタムの上位優遇傾斜は、より長いトレンドを取り込みやすい",
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momolv7030-top025-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_LOW_VOL_TILT_WEAK_TOP,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momolv8515-top025-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_LOW_VOL_TILT_LIGHT_TOP,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momomac8515-top025-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_MACRO_TILT_LIGHT_TOP,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo12-top035-mru",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP,
            portfolio_model=MEAN_RISK_UTILITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo12-top035-mruc",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP,
            portfolio_model=MEAN_RISK_UTILITY_CONSERVATIVE,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo12-soft025-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_SOFTMAX,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo12-lin050-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-fu-momo12-lin100-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_STRONG,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-top3-eq",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=MOMENTUM_TOP3,
            portfolio_model=EQUAL_WEIGHT,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-top3-rb",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=MOMENTUM_TOP3,
            portfolio_model=RISK_BUDGETING,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-top3-minvar",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=MOMENTUM_TOP3,
            portfolio_model=MINIMUM_VARIANCE,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-top3-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=MOMENTUM_TOP3,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-dualtop3-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=DUAL_MOMENTUM_TOP3,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-poslowvol-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=POSITIVE_MOMENTUM_LOW_VOL_UNIVERSE,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-trailmomlowvol-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=TRAILING_MOMENTUM_LOW_VOL_UNIVERSE,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            strategy_id="stg-posvol-hrp",
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            selection=POSITIVE_MOMENTUM_HIGH_VOLUME_UNIVERSE,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
        ),
        build_strategy_spec(
            investment_universe=ETF_ONLY_INVESTMENT_UNIVERSE,
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP,
            portfolio_model=HIERARCHICAL_RISK_PARITY,
            risk_controls=DEFAULT_RISK_CONTROLS,
            strategy_id="stg-etf-momo12-top035-hrp",
            label="ETF限定モメンタム傾斜 上位優遇 12ヶ月 × HRP",
            hypothesis="暗号資産を外したETFユニバースでも、上位優遇型のモメンタム傾斜が有効に働く可能性がある",
            description="18資産ETFに限定して、12ヶ月モメンタムの上位優遇傾斜をHRPに載せる",
        ),
    ],
    reference_strategies=[REFERENCE_EQUAL_WEIGHT_WITH_CASH],
    condition_variants=build_condition_variants(),
)
