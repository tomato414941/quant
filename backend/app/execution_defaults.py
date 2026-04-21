from __future__ import annotations

from app.portfolio import build_execution_policy_spec
from app.comparison_models import (
    ConditionVariant,
    CostModelSpec,
    build_asset_specific_adv_cost_model_spec,
    build_execution_assumptions_spec,
)
from app.instrument_registry import (
    DEFAULT_COST_PROFILE,
    GLOBAL_MULTI_ASSET_TICKERS,
    build_cost_overrides_for_profile,
)


def build_realistic_multi_asset_cost_model_spec() -> CostModelSpec:
    profile = DEFAULT_COST_PROFILE
    return build_asset_specific_adv_cost_model_spec(
        default_commission_pct=profile.default_parameters["commissionPct"],
        default_slippage_pct=profile.default_parameters["slippagePct"],
        default_impact_coefficient_pct=profile.default_parameters["impactCoefficientPct"],
        adv_window_bars=int(profile.default_parameters["advWindowBars"]),
        min_adv_notional=profile.default_parameters["minAdvNotional"],
        per_asset_overrides=build_cost_overrides_for_profile(profile, GLOBAL_MULTI_ASSET_TICKERS),
    )


def build_condition_variants() -> list[ConditionVariant]:
    variants: list[ConditionVariant] = []
    cost_multipliers = [1.0, 2.0, 3.0]
    investment_values = [1.0, 0.9, 0.8]
    max_weight_values = [None, 0.45, 0.35]

    for cost_multiplier in cost_multipliers:
        for max_investment_ratio in investment_values:
            for max_weight in max_weight_values:
                cash_pct = round((1 - max_investment_ratio) * 100)
                cap_label = "上限なし" if max_weight is None else f"{max_weight * 100:.0f}%上限"
                cap_key = "no_cap" if max_weight is None else f"cap_{int(max_weight * 100)}"
                multiplier_key = str(cost_multiplier).replace(".", "_")
                variants.append(
                    ConditionVariant(
                        key=(
                            f"cost_x{multiplier_key}"
                            f"__invest_{int(max_investment_ratio * 100)}"
                            f"__{cap_key}"
                        ),
                        label=(
                            f"コスト x{cost_multiplier:.1f} / 投資 {int(max_investment_ratio * 100)}%"
                            f" / CASH {cash_pct}% / {cap_label}"
                        ),
                        cost_multiplier=cost_multiplier,
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
        "costProfileKey": DEFAULT_COST_PROFILE.key,
    },
    cost_model=build_realistic_multi_asset_cost_model_spec(),
)

DEFAULT_COMPARISON_CONDITION_VARIANTS = build_condition_variants()
