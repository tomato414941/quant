from __future__ import annotations

from app.portfolio import (
    build_portfolio_candidate_definition,
    build_portfolio_model_definition,
    build_portfolio_state,
    build_portfolio_strategy_definition,
)
from app.study_models import (
    BacktestConfig,
    ConditionVariant,
    DatasetSpec,
    ExecutionModelConfig,
    StudyDefinition,
)


FULL_UNIVERSE = build_portfolio_strategy_definition(
    strategy_type="full_universe",
    key="full_universe",
    label="全資産",
    description="全ETFを候補にして配分する",
)
FULL_UNIVERSE_MOMENTUM_TILT = build_portfolio_strategy_definition(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt",
    label="全資産モメンタム傾斜",
    description="全ETFを候補に残しつつ、12ヶ月モメンタムで重みだけを傾ける",
    score_parameters={"tilt_strength": 0.5},
)
FULL_UNIVERSE_MOMENTUM_TILT_WEAK = build_portfolio_strategy_definition(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_weak",
    label="全資産モメンタム傾斜 弱",
    description="全ETFを候補に残しつつ、弱めの12ヶ月モメンタム傾斜で重みを調整する",
    score_parameters={"tilt_strength": 0.25, "tilt_shape": 0.0},
)
FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP = build_portfolio_strategy_definition(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_weak_top",
    label="全資産モメンタム傾斜 弱 上位優遇",
    description="全ETFを候補に残しつつ、弱めの上位優遇モメンタム傾斜で重みを調整する",
    score_parameters={"tilt_strength": 0.25, "tilt_shape": 1.0},
)
FULL_UNIVERSE_MOMENTUM_LOW_VOL_TILT_WEAK_TOP = build_portfolio_strategy_definition(
    strategy_type="full_universe_momentum_low_vol_tilt",
    key="full_universe_momentum_low_vol_tilt_weak_top",
    label="全資産モメンタム低ボラ傾斜 弱 上位優遇",
    description="全ETFを候補に残しつつ、12ヶ月モメンタムと低ボラの複合スコアで弱く上位優遇する",
    score_parameters={
        "tilt_strength": 0.25,
        "tilt_shape": 1.0,
        "momentum_weight": 0.7,
        "low_vol_weight": 0.3,
    },
)
FULL_UNIVERSE_MOMENTUM_LOW_VOL_TILT_LIGHT_TOP = build_portfolio_strategy_definition(
    strategy_type="full_universe_momentum_low_vol_tilt",
    key="full_universe_momentum_low_vol_tilt_light_top",
    label="全資産モメンタム低ボラ傾斜 低ボラ弱め 上位優遇",
    description="全ETFを候補に残しつつ、12ヶ月モメンタムを主役に低ボラを弱く混ぜて上位優遇する",
    score_parameters={
        "tilt_strength": 0.25,
        "tilt_shape": 1.0,
        "momentum_weight": 0.85,
        "low_vol_weight": 0.15,
    },
)
FULL_UNIVERSE_MOMENTUM_MACRO_TILT_LIGHT_TOP = build_portfolio_strategy_definition(
    strategy_type="full_universe_momentum_macro_tilt",
    key="full_universe_momentum_macro_tilt_light_top",
    label="全資産モメンタムマクロ傾斜 マクロ弱め 上位優遇",
    description="全ETFを候補に残しつつ、12ヶ月モメンタムを主役にマクロproxyを弱く混ぜて上位優遇する",
    score_parameters={
        "tilt_strength": 0.25,
        "tilt_shape": 1.0,
        "momentum_weight": 0.85,
        "macro_weight": 0.15,
    },
)
FULL_UNIVERSE_MOMENTUM_TILT_WEAK_SOFTMAX = build_portfolio_strategy_definition(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_weak_softmax",
    label="全資産モメンタム傾斜 弱 softmax",
    description="全ETFを候補に残しつつ、弱めのsoftmax型モメンタム傾斜で重みを調整する",
    score_parameters={"tilt_strength": 0.25, "tilt_shape": 2.0},
)
FULL_UNIVERSE_MOMENTUM_TILT_STRONG = build_portfolio_strategy_definition(
    strategy_type="full_universe_momentum_tilt",
    key="full_universe_momentum_tilt_strong",
    label="全資産モメンタム傾斜 強",
    description="全ETFを候補に残しつつ、強めの12ヶ月モメンタム傾斜で重みを調整する",
    score_parameters={"tilt_strength": 1.0, "tilt_shape": 0.0},
)
MOMENTUM_TOP3 = build_portfolio_strategy_definition(
    strategy_type="momentum_top3",
    key="momentum_top3",
    label="モメンタム上位3",
    description="学習期間で強かった上位3ETFに絞って配分する",
)
DUAL_MOMENTUM_TOP3 = build_portfolio_strategy_definition(
    strategy_type="dual_momentum_top3",
    key="dual_momentum_top3",
    label="デュアルモメンタム上位3",
    description="上昇しているETFだけからモメンタム上位3を選び、弱い相場ではCASHへ逃がす",
)
POSITIVE_MOMENTUM_LOW_VOL_UNIVERSE = build_portfolio_strategy_definition(
    strategy_type="positive_momentum_low_vol_universe",
    key="positive_momentum_low_vol_universe",
    label="上昇低ボラ資産",
    description="上昇しているETFのうち、低ボラ群だけを候補にして配分する",
)
TRAILING_MOMENTUM_LOW_VOL_UNIVERSE = build_portfolio_strategy_definition(
    strategy_type="trailing_momentum_low_vol_universe",
    key="trailing_momentum_low_vol_universe",
    label="12ヶ月モメンタム低ボラ資産",
    description="12ヶ月モメンタムが正のETFを候補にし、低ボラ群だけをHRPへ渡す",
)
POSITIVE_MOMENTUM_HIGH_VOLUME_UNIVERSE = build_portfolio_strategy_definition(
    strategy_type="positive_momentum_high_volume_universe",
    key="positive_momentum_high_volume_universe",
    label="上昇出来高資産",
    description="上昇しているETFのうち、出来高が強い群だけを候補にして配分する",
)

EQUAL_WEIGHT = build_portfolio_model_definition(
    model_type="equal_weight",
    key="equal_weight",
    label="等金額配分",
    description="全ETFを同じ比率で持つ",
)
RISK_BUDGETING = build_portfolio_model_definition(
    model_type="risk_budgeting",
    key="risk_budgeting",
    label="リスク予算配分",
    description="各ETFのリスク寄与が近づくように配分する",
)
MINIMUM_VARIANCE = build_portfolio_model_definition(
    model_type="minimum_variance",
    key="minimum_variance",
    label="最小分散",
    description="全体の分散が最小になるように配分する",
)
HIERARCHICAL_RISK_PARITY = build_portfolio_model_definition(
    model_type="hierarchical_risk_parity",
    key="hierarchical_risk_parity",
    label="HRP",
    description="相関クラスタを使って階層的にリスクを分散する",
)
MEAN_RISK_UTILITY = build_portfolio_model_definition(
    model_type="mean_risk_utility",
    key="mean_risk_utility",
    label="MeanRisk効用最大化",
    description="モメンタム順位から作った期待リターン proxy とリスクの両方で配分する",
)
MEAN_RISK_UTILITY_CONSERVATIVE = build_portfolio_model_definition(
    model_type="mean_risk_utility_conservative",
    key="mean_risk_utility_conservative",
    label="MeanRisk効用最大化 弱",
    description="モメンタム順位の期待リターン proxy を弱めに使い、リスクをより強く見る",
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


DEFAULT_DASHBOARD_CONFIG = StudyDefinition(
    study_id="etf_portfolio_models_10y",
    title="マルチアセット戦略 x ポートフォリオ構築の比較",
    question="条件スイープで最良だった 年次 / 100%投資 / 45%上限 / 0.05%手数料 を固定し、10y を主期間に戦略と配分法を比較する",
    dataset_spec=DatasetSpec(
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
        period="10y",
        sanity_periods=["3y"],
        frequency="daily",
    ),
    execution_variants=[
        ExecutionModelConfig(
            key="annual",
            label="年次",
            entry="train_once_then_periodic_rebalance",
            commission_pct=0.05,
            slippage_pct=0.0,
            rebalance_frequency="annual",
        ),
    ],
    backtest_config=BacktestConfig(
        split_ratio=0.7,
        initial_capital=10_000,
        max_investment_ratio=1.0,
        benchmark="equal_weight_buy_and_hold_with_cash",
        max_weight=0.45,
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
    candidate_definitions=[
        build_portfolio_candidate_definition(FULL_UNIVERSE, EQUAL_WEIGHT),
        build_portfolio_candidate_definition(FULL_UNIVERSE, RISK_BUDGETING),
        build_portfolio_candidate_definition(FULL_UNIVERSE, MINIMUM_VARIANCE),
        build_portfolio_candidate_definition(FULL_UNIVERSE, HIERARCHICAL_RISK_PARITY),
        build_portfolio_candidate_definition(FULL_UNIVERSE_MOMENTUM_TILT_WEAK, HIERARCHICAL_RISK_PARITY),
        build_portfolio_candidate_definition(FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP, HIERARCHICAL_RISK_PARITY),
        build_portfolio_candidate_definition(FULL_UNIVERSE_MOMENTUM_LOW_VOL_TILT_WEAK_TOP, HIERARCHICAL_RISK_PARITY),
        build_portfolio_candidate_definition(FULL_UNIVERSE_MOMENTUM_LOW_VOL_TILT_LIGHT_TOP, HIERARCHICAL_RISK_PARITY),
        build_portfolio_candidate_definition(FULL_UNIVERSE_MOMENTUM_MACRO_TILT_LIGHT_TOP, HIERARCHICAL_RISK_PARITY),
        build_portfolio_candidate_definition(FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP, MEAN_RISK_UTILITY),
        build_portfolio_candidate_definition(FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP, MEAN_RISK_UTILITY_CONSERVATIVE),
        build_portfolio_candidate_definition(FULL_UNIVERSE_MOMENTUM_TILT_WEAK_SOFTMAX, HIERARCHICAL_RISK_PARITY),
        build_portfolio_candidate_definition(FULL_UNIVERSE_MOMENTUM_TILT, HIERARCHICAL_RISK_PARITY),
        build_portfolio_candidate_definition(FULL_UNIVERSE_MOMENTUM_TILT_STRONG, HIERARCHICAL_RISK_PARITY),
        build_portfolio_candidate_definition(MOMENTUM_TOP3, EQUAL_WEIGHT),
        build_portfolio_candidate_definition(MOMENTUM_TOP3, RISK_BUDGETING),
        build_portfolio_candidate_definition(MOMENTUM_TOP3, MINIMUM_VARIANCE),
        build_portfolio_candidate_definition(MOMENTUM_TOP3, HIERARCHICAL_RISK_PARITY),
        build_portfolio_candidate_definition(DUAL_MOMENTUM_TOP3, HIERARCHICAL_RISK_PARITY),
        build_portfolio_candidate_definition(
            POSITIVE_MOMENTUM_LOW_VOL_UNIVERSE,
            HIERARCHICAL_RISK_PARITY,
        ),
        build_portfolio_candidate_definition(
            TRAILING_MOMENTUM_LOW_VOL_UNIVERSE,
            HIERARCHICAL_RISK_PARITY,
        ),
        build_portfolio_candidate_definition(
            POSITIVE_MOMENTUM_HIGH_VOLUME_UNIVERSE,
            HIERARCHICAL_RISK_PARITY,
        ),
    ],
    condition_variants=build_condition_variants(),
)
