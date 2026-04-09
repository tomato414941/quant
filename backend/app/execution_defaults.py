from __future__ import annotations

from app.portfolio import build_execution_policy_spec
from app.comparison_models import (
    ConditionVariant,
    CostModelSpec,
    build_asset_specific_adv_cost_model_spec,
    build_execution_assumptions_spec,
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

DEFAULT_MONTH_END_EXECUTION_POLICY = build_execution_policy_spec(
    key="month_end",
    label="月次",
    entry="train_once_then_periodic_rebalance",
    rebalance_schedule="month_end",
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

DEFAULT_COMPARISON_CONDITION_VARIANTS = build_condition_variants()
