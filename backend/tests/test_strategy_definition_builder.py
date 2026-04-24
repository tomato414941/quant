import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.portfolio import (
    COST_AWARE_NO_TRADE_DECISION_POLICY,
    SIGNAL_RETURN_PROXY_EDGE_SOURCE,
    PREDICTION_FEATURE_NAMES,
    build_observation_spec,
    build_prediction_combiner_spec,
    build_derived_feature_spec,
    build_feature_spec,
    build_feature_input_spec,
    build_prediction_output_spec,
    build_alignment_policy_spec,
    build_decision_use_spec,
    build_prediction_engine_spec,
    build_prediction_learner_spec,
    build_predicted_quantity_spec,
    build_prediction_signal_source_spec,
    build_signal_spec,
    build_prediction_target_spec,
    build_predictor_spec,
    build_predictor_use_spec,
    build_ranking_feature_recipe_spec,
    build_strategy_definition_from_evaluator_strategy_spec,
    build_strategy_signal_execution_contexts_from_definition,
    build_strategy_definition,
    build_strategy_execution_plan_spec,
    build_strategy_signal_spec,
    build_training_spec,
    build_investment_universe_spec,
    build_portfolio_model_spec,
    build_portfolio_state,
    build_strategy_data_source_spec,
    build_strategy_feature_definition_spec,
    build_executable_evaluator_strategy_spec_from_definition,
    build_execution_policy_spec,
    build_risk_controls_spec,
    build_selection_spec,
    build_evaluator_strategy_spec,
    compare_portfolio_runs,
    compute_portfolio_allocation,
    compute_predictor_panel,
    convert_window_spec_to_bars,
    compute_trade_cost,
    compute_strategy_score_series,
    build_strategy_forecast_snapshot,
    evaluate_predictor_spec,
    get_direct_execution_strategy_definition_compatibility_issues,
    get_strategy_definition_signal_execution_contexts,
    get_strategy_signal_execution_contexts,
    is_direct_execution_compatible_strategy_definition,
    prepare_strategy_market_data,
    prepare_strategy_predictor_panel,
    prepare_strategy_signal_data,
    extract_predictor_signal_payload,
    prepare_signal_component_data,
    resample_market_frame_to_timeframe,
    resample_returns_frame_to_timeframe,
    resolve_decision_policy_kind,
    resolve_decision_schedule,
    resolve_strategy_market_data_timeframe_key,
    serialize_strategy_definition,
    serialize_evaluator_strategy_spec,
    serialize_strategy_signal_spec,
    summarize_portfolio_decision_events,
    select_assets,
    should_rebalance,
)
from app.strategy_definition_builder import (
    ExecutionVariantDefinition,
    PortfolioModelVariantDefinition,
    PredictorDefinitionDefinition,
    PredictorVariantDefinition,
    SelectionDefinitionDefinition,
    SelectionVariantDefinition,
    build_predictor_strategy_definition,
    build_predictor_strategy_definition_product,
    build_selection_strategy_definition,
    build_selection_strategy_definition_product,
)
from app.strategy_candidate_baselines import (
    BASELINE_CANDIDATE_DEFINITIONS,
)
from app.strategy_candidate_filtered import (
    FILTERED_CANDIDATE_DEFINITIONS,
)
from app.strategy_candidate_predictors import (
    PREDICTOR_CANDIDATE_DEFINITIONS,
)
from app.strategy_candidate_full_universe import (
    EXPERIMENTAL_FULL_UNIVERSE_CANDIDATE_DEFINITIONS,
    FULL_UNIVERSE_CANDIDATE_DEFINITIONS,
)
from app.strategy_candidate_timeframes import (
    TIMEFRAME_VARIANT_CANDIDATE_DEFINITIONS,
)
from app.strategy_candidate_universe_variants import (
    UNIVERSE_VARIANT_CANDIDATE_DEFINITIONS,
)
from app.strategy_catalog import CANONICAL_CANDIDATE_DEFINITIONS
from app.strategy_presets import FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M
from app.timeframe_models import DEFAULT_DAILY_TIMEFRAME, DEFAULT_MONTHLY_TIMEFRAME, DEFAULT_WEEKLY_TIMEFRAME


def make_execution_assumptions() -> dict:
    return {
        "kind": "close_execution_assumptions",
        "label": "終値約定",
        "parameters": {
            "fillPrice": "close",
        },
        "costModel": {
            "kind": "flat_cost",
            "parameters": {"commissionPct": 0.1, "slippagePct": 0.0},
            "perAssetOverrides": {},
        },
    }


def make_strategy(
    strategy_type: str,
    model_type: str,
    *,
    max_investment_ratio: float = 1.0,
    max_weight: float | None = None,
    score_parameters: dict[str, float | str | bool] | None = None,
    predictor_use=None,
    decision_policy: str = "direct_score_to_weight",
):
    return build_evaluator_strategy_spec(
        investment_universe=build_investment_universe_spec(
            tickers=[
                "SPY",
                "QQQ",
                "IWM",
                "EFA",
                "EEM",
                "TLT",
                "IEF",
                "LQD",
                "HYG",
                "GLD",
                "BTC-USD",
                "ETH-USD",
            ],
            key="test_universe",
            label="Test universe",
        ),
        selection=build_selection_spec(
            strategy_type,
            score_parameters=score_parameters,
        ),
        portfolio_model=build_portfolio_model_spec(model_type),
        risk_controls=build_risk_controls_spec(
            max_investment_ratio=max_investment_ratio,
            max_weight=max_weight,
        ),
        predictor_use=predictor_use,
        extensions={"decision_policy": decision_policy},
    )


def make_predictor_spec(
    strategy,
    *,
    predictor_key: str,
    label: str,
    horizon_bars: int,
    min_train_samples: int = 3,
    signal_source_feature_key: str | None = None,
    combiner_kind: str = "learner_only",
):
    signal_source_spec = None
    learner_spec = build_prediction_learner_spec(
        key=f"learner__{predictor_key}",
        label=f"{label} linear learner",
        kind="linear_regression",
    )
    if signal_source_feature_key is not None:
        signal_source_spec = build_prediction_signal_source_spec(
            key=f"signal-source__{predictor_key}",
            label=f"{label} signal source",
            kind="derived_feature",
            feature_key=signal_source_feature_key,
        )
    if combiner_kind == "weighted_blend":
        combiner_spec = build_prediction_combiner_spec(
            key=f"combiner__{predictor_key}",
            label=f"{label} weighted blend",
            kind="weighted_blend",
            signal_source_weight=0.7,
            learner_weight=0.3,
        )
    else:
        combiner_spec = build_prediction_combiner_spec(
            key=f"combiner__{predictor_key}",
            label=f"{label} learner only",
            kind="learner_only",
        )
    return build_predictor_spec(
        key=predictor_key,
        label=label,
        description=strategy.description,
        timeframe=strategy.timeframe,
        signal_spec=build_signal_spec(
            key=f"signal__{predictor_key}",
            label=f"{label} signal",
            observation_spec=build_observation_spec(
                key=f"observation__{predictor_key}",
                label=f"{label} observation",
                tickers=strategy.investment_universe.tickers,
                fields=strategy.selection.ranking_signal.feature_inputs,
            ),
            entity_kind="asset_set",
            entity_identifiers=strategy.investment_universe.tickers,
            output_spec=build_prediction_output_spec(
                key=f"output__{predictor_key}",
                label=f"{label} output",
                kind="score",
            ),
            decision_use_spec=build_decision_use_spec(strategy.selection),
        ),
        predicted_quantity_spec=build_predicted_quantity_spec(
            key=f"quantity__{predictor_key}",
            label=f"{label} quantity",
            kind="return",
        ),
        target_spec=build_prediction_target_spec(
            key=f"{predictor_key}__target",
            label=label,
            horizon_spec={"unit": "bars", "value": horizon_bars},
            baseline="cross_sectional_mean",
            transform="identity",
        ),
        feature_spec=build_feature_spec(
            key=f"features__{predictor_key}",
            label=f"{label} features",
            feature_inputs=tuple(
                build_feature_input_spec(key=feature_input)
                for feature_input in strategy.selection.ranking_signal.feature_inputs
            ),
            derived_features=tuple(
                build_derived_feature_spec(key=feature_key)
                for feature_key in PREDICTION_FEATURE_NAMES
            ),
            ranking_feature_recipe=build_ranking_feature_recipe_spec(strategy.selection),
        ),
        engine_spec=build_prediction_engine_spec(
            key=f"engine__{predictor_key}",
            label=f"{label} linear",
            signal_source_spec=signal_source_spec,
            learner_spec=learner_spec,
            combiner_spec=combiner_spec,
        ),
        training_spec=build_training_spec(
            key=f"training__{predictor_key}",
            label=f"{label} training",
            fit_mode="expanding",
            min_train_samples=min_train_samples,
        ),
    )


def test_build_strategy_definition_supports_multiple_signal_timeframes() -> None:
    universe = build_investment_universe_spec(
        tickers=["SPY", "QQQ", "TLT"],
        key="multi_tf_universe",
        label="Multi timeframe universe",
    )
    daily_signal = build_strategy_signal_spec(
        key="signal__daily_momentum",
        label="Daily momentum",
        description="Use daily prices for short-term momentum.",
        observation_spec=build_observation_spec(
            key="observation__daily",
            label="Daily observation",
            tickers=universe.tickers,
            fields=("close",),
        ),
        data_timeframe=DEFAULT_DAILY_TIMEFRAME,
        source_kind="ranking_signal",
        signal_parameters={"window": {"unit": "months", "value": 2}},
    )
    monthly_signal = build_strategy_signal_spec(
        key="signal__monthly_macro",
        label="Monthly macro",
        description="Use monthly regime data as a slower overlay.",
        observation_spec=build_observation_spec(
            key="observation__monthly",
            label="Monthly observation",
            tickers=universe.tickers,
            fields=("close",),
        ),
        data_timeframe=DEFAULT_MONTHLY_TIMEFRAME,
        signal_timeframe=DEFAULT_MONTHLY_TIMEFRAME,
        source_kind="macro_signal",
        weight=0.3,
        signal_parameters={"overlay": "regime"},
    )

    definition = build_strategy_definition(
        strategy_id="definition__multi_timeframe",
        label="Daily plus monthly definition",
        description="Combine fast and slow signals in one strategy definition.",
        investment_universe=universe,
        signals=[daily_signal, monthly_signal],
        portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
        execution_plan=build_strategy_execution_plan_spec(
            key="execution__daily_month_end",
            label="Daily decisions / month-end trades",
            decision_schedule="every_bar",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
    )

    assert definition.strategy_id == "definition__multi_timeframe"
    assert len(definition.signals) == 2
    assert definition.signals[0].data_timeframe.key == "1d"
    assert definition.signals[1].data_timeframe.key == "1mo"
    assert definition.execution_plan.decision_schedule == "every_bar"
    assert definition.execution_plan.rebalance_schedule == "month_end"


def test_build_strategy_definition_rejects_signal_outside_universe() -> None:
    universe = build_investment_universe_spec(
        tickers=["SPY", "QQQ"],
        key="small_universe",
        label="Small universe",
    )
    foreign_signal = build_strategy_signal_spec(
        key="signal__foreign",
        label="Foreign signal",
        description="Signal that references an out-of-universe ticker.",
        observation_spec=build_observation_spec(
            key="observation__foreign",
            label="Foreign observation",
            tickers=("SPY", "TLT"),
            fields=("close",),
        ),
        data_timeframe=DEFAULT_DAILY_TIMEFRAME,
        source_kind="ranking_signal",
    )

    try:
        build_strategy_definition(
            strategy_id="definition__invalid",
            investment_universe=universe,
            signals=[foreign_signal],
            portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            execution_plan=build_strategy_execution_plan_spec(
                key="execution__month_end",
                label="Month-end execution",
                decision_schedule="month_end",
                rebalance_schedule="month_end",
            ),
            risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        )
    except ValueError as exc:
        assert "investment universe" in str(exc)
    else:
        raise AssertionError("Expected build_strategy_definition to reject out-of-universe tickers.")


def test_build_strategy_definition_from_evaluator_strategy_spec_maps_predictor_overlay() -> None:
    predictor_use = build_predictor_use_spec(
        predictor_key="pred__overlay",
        signal_weight=0.6,
        predictor_weight=0.4,
    )
    strategy = build_evaluator_strategy_spec(
        strategy_id="strategy__with_predictor",
        label="Strategy with predictor",
        description="Legacy strategy with predictor overlay.",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            tickers=["SPY", "QQQ", "TLT"],
            key="adapter_universe",
            label="Adapter universe",
        ),
        selection=build_selection_spec(
            "full_universe_momentum_tilt",
            key="adapter_selection",
            score_parameters={
                "tilt_strength": 0.35,
                "tilt_shape": 1.0,
                "windowSpec": {"unit": "months", "value": 8},
            },
        ),
        portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
        execution_policy=build_execution_policy_spec(
            key="month_end",
            label="月次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
        predictor_use=predictor_use,
        signal_execution_contexts=[
            {
                "signalKey": "selection_signal",
                "signalLabel": "Adapter selection signal",
                "description": "Adapter selection signal.",
                "sourceKind": "selection_signal",
                "selectionKey": "adapter_selection",
                "strategyType": "full_universe_momentum_tilt",
                "scoreParameters": {
                    "tilt_strength": 0.35,
                    "tilt_shape": 1.0,
                    "windowSpec": {"unit": "months", "value": 8},
                },
                "dataTimeframe": "1d",
                "signalTimeframe": "1w",
                "alignmentPolicy": {"key": "weekly", "label": "Weekly", "method": "asof_last", "parameters": {}},
                "weight": 0.6,
            }
        ],
        predictor_signal_execution_context={
            "signalKey": "predictor_signal",
            "signalLabel": "Adapter predictor signal",
            "description": "Adapter predictor signal.",
            "sourceKind": "predictor_overlay",
            "predictorKey": "pred__overlay",
            "signalWeight": 0.6,
            "predictorWeight": 0.4,
            "dataTimeframe": "1d",
            "signalTimeframe": "1w",
            "alignmentPolicy": {"key": "weekly", "label": "Weekly", "method": "calendar_resample", "parameters": {}},
            "weight": 0.4,
        },
        decision_schedule="every_bar",
    )

    definition = build_strategy_definition_from_evaluator_strategy_spec(strategy)

    assert definition.strategy_id == strategy.strategy_id
    assert definition.execution_plan.decision_schedule == "every_bar"
    assert definition.execution_plan.rebalance_schedule == "month_end"
    assert len(definition.signals) == 2
    assert definition.signals[0].source_kind == "selection_signal"
    assert definition.signals[0].weight == 0.6
    assert definition.signals[0].data_timeframe.key == "1d"
    assert definition.signals[0].signal_timeframe.key == "1w"
    assert definition.signals[0].alignment_policy is not None
    assert definition.signals[0].alignment_policy.method == "asof_last"
    assert definition.signals[1].source_kind == "predictor_overlay"
    assert definition.signals[1].predictor_key == "pred__overlay"
    assert definition.signals[1].weight == 0.4
    assert definition.signals[1].data_timeframe.key == "1d"
    assert definition.signals[1].signal_timeframe.key == "1w"
    assert definition.signals[1].alignment_policy is not None
    assert definition.signals[1].alignment_policy.method == "calendar_resample"


def test_build_strategy_definition_from_evaluator_strategy_spec_without_predictor_keeps_single_signal() -> None:
    strategy = make_strategy(
        "full_universe",
        "equal_weight",
        max_investment_ratio=0.8,
    )

    definition = build_strategy_definition_from_evaluator_strategy_spec(strategy)

    assert definition.strategy_id == strategy.strategy_id
    assert len(definition.signals) == 1
    assert definition.signals[0].source_kind == "selection_signal"
    assert definition.signals[0].weight == 1.0
    assert definition.execution_plan.rebalance_schedule == strategy.execution_policy.rebalance_schedule


def test_build_executable_evaluator_strategy_spec_from_definition_round_trips_direct_strategy() -> None:
    predictor_use = build_predictor_use_spec(
        predictor_key="pred__overlay",
        signal_weight=0.6,
        predictor_weight=0.4,
    )
    strategy = build_evaluator_strategy_spec(
        strategy_id="strategy__round_trip",
        version="v7",
        hypothesis="Round-trip direct execution.",
        label="Round trip strategy",
        description="Direct execution round-trip test strategy.",
        investment_universe=build_investment_universe_spec(
            tickers=["SPY", "QQQ", "TLT"],
            key="round_trip_universe",
            label="Round-trip universe",
        ),
        selection=build_selection_spec(
            "full_universe_momentum_tilt",
            key="round_trip_selection",
            score_parameters={
                "tilt_strength": 0.35,
                "tilt_shape": 1.0,
                "windowSpec": {"unit": "months", "value": 8},
            },
        ),
        portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
        execution_policy=build_execution_policy_spec(
            key="month_end",
            label="月次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
        predictor_use=predictor_use,
        extensions={"source": "test"},
    )

    rebuilt = build_executable_evaluator_strategy_spec_from_definition(
        build_strategy_definition_from_evaluator_strategy_spec(strategy)
    )

    assert rebuilt.strategy_id == strategy.strategy_id
    assert rebuilt.version == strategy.version
    assert rebuilt.hypothesis == strategy.hypothesis
    assert rebuilt.timeframe.key == strategy.timeframe.key
    assert rebuilt.selection.strategy_type == strategy.selection.strategy_type
    assert rebuilt.selection.key == strategy.selection.key
    assert dict(rebuilt.selection.ranking_signal.score_parameters) == dict(
        strategy.selection.ranking_signal.score_parameters
    )
    assert rebuilt.execution_policy.rebalance_schedule == strategy.execution_policy.rebalance_schedule
    assert rebuilt.predictor_use is not None
    assert rebuilt.predictor_use.predictor_key == predictor_use.predictor_key
    assert rebuilt.predictor_use.signal_weight == predictor_use.signal_weight
    assert rebuilt.predictor_use.predictor_weight == predictor_use.predictor_weight
    assert rebuilt.decision_schedule == strategy.execution_policy.rebalance_schedule
    assert rebuilt.signal_execution_contexts[0]["dataTimeframe"] == strategy.timeframe.key
    assert rebuilt.signal_execution_contexts[0]["signalTimeframe"] == strategy.timeframe.key
    assert rebuilt.predictor_signal_execution_context is not None
    assert rebuilt.predictor_signal_execution_context["predictorKey"] == predictor_use.predictor_key
    assert dict(rebuilt.extensions) == dict(strategy.extensions)
    payload = serialize_strategy_definition(build_strategy_definition_from_evaluator_strategy_spec(strategy))
    support = payload["executionSupport"]
    assert support["directExecutionCompatible"] is True
    assert support["directExecutionIssues"] == []
    assert "strategySpecAdapterCompatible" not in support
    assert "strategySpecAdapterIssues" not in support
    assert "evaluatorAdapterCompatible" not in support
    assert "evaluatorAdapterIssues" not in support


def test_build_executable_evaluator_strategy_spec_from_definition_accepts_split_execution_plan() -> None:
    definition = build_strategy_definition(
        strategy_id="definition__split_execution",
        investment_universe=build_investment_universe_spec(
            tickers=["SPY", "QQQ"],
            key="split_execution_universe",
            label="Split execution universe",
        ),
        signals=[
            build_strategy_signal_spec(
                key="signal__selection",
                label="Selection signal",
                description="Selection signal.",
                observation_spec=build_observation_spec(
                    key="observation__selection",
                    label="Selection observation",
                    tickers=("SPY", "QQQ"),
                    fields=("close",),
                ),
                data_timeframe=DEFAULT_DAILY_TIMEFRAME,
                source_kind="selection_signal",
                signal_parameters={
                    "selectionKey": "full_universe",
                    "strategyType": "full_universe",
                    "scoreParameters": {},
                },
            )
        ],
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_plan=build_strategy_execution_plan_spec(
            key="execution__split",
            label="Split execution",
            decision_schedule="every_bar",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
    )

    rebuilt = build_executable_evaluator_strategy_spec_from_definition(definition)

    assert rebuilt.decision_schedule == "every_bar"
    assert rebuilt.execution_policy.rebalance_schedule == "month_end"
    assert rebuilt.execution_mode == "direct_signal_timeframe"


def test_build_executable_evaluator_strategy_spec_from_definition_accepts_mismatched_signal_timeframe() -> None:
    definition = build_strategy_definition(
        strategy_id="definition__mismatched_signal_timeframe",
        investment_universe=build_investment_universe_spec(
            tickers=["SPY", "QQQ"],
            key="mismatched_signal_timeframe_universe",
            label="Mismatched signal timeframe universe",
        ),
        signals=[
            build_strategy_signal_spec(
                key="signal__selection",
                label="Selection signal",
                description="Selection signal.",
                observation_spec=build_observation_spec(
                    key="observation__selection",
                    label="Selection observation",
                    tickers=("SPY", "QQQ"),
                    fields=("close",),
                ),
                data_timeframe=DEFAULT_DAILY_TIMEFRAME,
                signal_timeframe=DEFAULT_MONTHLY_TIMEFRAME,
                source_kind="selection_signal",
                signal_parameters={
                    "selectionKey": "full_universe",
                    "strategyType": "full_universe",
                    "scoreParameters": {},
                },
            )
        ],
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_plan=build_strategy_execution_plan_spec(
            key="execution__month_end",
            label="Month end",
            decision_schedule="month_end",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
    )

    rebuilt = build_executable_evaluator_strategy_spec_from_definition(definition)

    assert rebuilt.timeframe.key == "1mo"
    assert rebuilt.signal_execution_contexts[0]["dataTimeframe"] == "1d"
    assert rebuilt.signal_execution_contexts[0]["signalTimeframe"] == "1mo"


def test_full_universe_candidates_are_strategy_definitions() -> None:
    assert len({definition.strategy_id for definition in FULL_UNIVERSE_CANDIDATE_DEFINITIONS}) == len(FULL_UNIVERSE_CANDIDATE_DEFINITIONS)

    core_candidate = FULL_UNIVERSE_CANDIDATE_DEFINITIONS[8]
    assert core_candidate.execution_plan.decision_schedule == "year_end"
    assert core_candidate.execution_plan.rebalance_schedule == "year_end"
    assert core_candidate.signals[0].source_kind == "selection_signal"
    assert core_candidate.signals[0].data_timeframe.key == "1d"

    monthly_keys = {
        "stg-fu-momolv8515-top025-hrp-month",
        "stg-fu-momomac8515-top025-hrp-month",
        "stg-fu-momo12-soft025-hrp-month",
        "stg-fu-momo12-lin050-hrp-month",
    }
    monthly_candidates = {
        definition.strategy_id: definition
        for definition in FULL_UNIVERSE_CANDIDATE_DEFINITIONS
        if definition.strategy_id in monthly_keys
    }
    assert set(monthly_candidates) == monthly_keys
    assert all(
        definition.strategy_id != "stg-fu-momolv8515-top025-hrp-month-costaware"
        for definition in FULL_UNIVERSE_CANDIDATE_DEFINITIONS
    )
    for definition in monthly_candidates.values():
        assert definition.execution_plan.decision_schedule == "month_end"
        assert definition.execution_plan.rebalance_schedule == "month_end"
        assert definition.signals[0].source_kind == "selection_signal"
        assert definition.signals[0].data_timeframe.key == "1d"

    assert len(EXPERIMENTAL_FULL_UNIVERSE_CANDIDATE_DEFINITIONS) == 1
    cost_aware = EXPERIMENTAL_FULL_UNIVERSE_CANDIDATE_DEFINITIONS[0]
    assert cost_aware.strategy_id == "stg-fu-momolv8515-top025-hrp-month-costaware"
    assert dict(cost_aware.extensions)["decision_policy"] == COST_AWARE_NO_TRADE_DECISION_POLICY


def test_strategy_signal_builder_infers_data_source_and_feature_definition() -> None:
    signal = build_strategy_signal_spec(
        key="signal__metadata",
        label="Metadata signal",
        description="Signal metadata inference.",
        observation_spec=build_observation_spec(
            key="observation__metadata",
            label="Metadata observation",
            tickers=("SPY", "QQQ"),
            fields=("close", "volume"),
        ),
        data_timeframe=DEFAULT_DAILY_TIMEFRAME,
        source_kind="selection_signal",
    )

    assert signal.data_source_spec is not None
    assert signal.data_source_spec.kind == "market_observation"
    assert signal.feature_definition_spec is not None
    assert signal.feature_definition_spec.source_field_keys == ("close", "volume")
    assert signal.alignment_policy is None

    serialized = serialize_strategy_signal_spec(signal)
    assert serialized["dataSource"]["kind"] == "market_observation"
    assert serialized["featureDefinition"]["sourceFieldKeys"] == ["close", "volume"]
    assert serialized["alignmentPolicy"] is None


def test_strategy_signal_builder_infers_alignment_policy_for_mismatched_timeframes() -> None:
    signal = build_strategy_signal_spec(
        key="signal__alignment",
        label="Alignment signal",
        description="Signal alignment inference.",
        observation_spec=build_observation_spec(
            key="observation__alignment",
            label="Alignment observation",
            tickers=("SPY", "QQQ"),
            fields=("close",),
        ),
        data_timeframe=DEFAULT_DAILY_TIMEFRAME,
        signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        source_kind="selection_signal",
    )

    assert signal.alignment_policy is not None
    assert signal.alignment_policy.method == "asof_last"

    serialized = serialize_strategy_signal_spec(signal)
    assert serialized["alignmentPolicy"]["method"] == "asof_last"
    assert serialized["alignmentPolicy"]["parameters"]["fromTimeframe"] == "1d"
    assert serialized["alignmentPolicy"]["parameters"]["toTimeframe"] == "1w"


def test_selection_definition_builder_creates_direct_execution_compatible_definition() -> None:
    definition = build_selection_strategy_definition(
        SelectionDefinitionDefinition(
            strategy_id="builder__selection",
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            execution_policy=build_execution_policy_spec(
                key="month_end",
                label="月次",
                entry="train_once_then_periodic_rebalance",
                rebalance_schedule="month_end",
            ),
            investment_universe=build_investment_universe_spec(
                tickers=["SPY", "QQQ", "TLT"],
                key="builder_universe",
                label="Builder universe",
            ),
            risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
            label="Builder strategy",
            hypothesis="Builder hypothesis",
            description="Builder description",
        )
    )

    assert definition.strategy_id == "builder__selection"
    assert definition.signals[0].source_kind == "selection_signal"
    assert definition.signals[0].data_timeframe.key == "1d"

    rebuilt = build_executable_evaluator_strategy_spec_from_definition(definition)
    assert rebuilt.selection.strategy_type == FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M.strategy_type
    assert rebuilt.execution_policy.rebalance_schedule == "month_end"
    assert rebuilt.decision_schedule == "month_end"
    assert rebuilt.signal_execution_contexts[0]["dataTimeframe"] == "1d"
    assert rebuilt.signal_execution_contexts[0]["signalTimeframe"] == "1d"


def test_predictor_definition_builder_creates_overlay_signal() -> None:
    definition = build_predictor_strategy_definition(
        PredictorDefinitionDefinition(
            strategy_id="builder__predictor",
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            predictor_key="pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-5050",
            signal_weight=0.6,
            predictor_weight=0.4,
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            execution_policy=build_execution_policy_spec(
                key="every_bar",
                label="毎バー",
                entry="train_once_then_periodic_rebalance",
                rebalance_schedule="every_bar",
            ),
            investment_universe=build_investment_universe_spec(
                tickers=["SPY", "QQQ", "TLT"],
                key="builder_universe",
                label="Builder universe",
            ),
            risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
            label="Builder predictor strategy",
            hypothesis="Builder predictor hypothesis",
            description="Builder predictor description",
        )
    )

    assert len(definition.signals) == 2
    assert definition.signals[0].weight == 0.6
    assert definition.signals[1].source_kind == "predictor_overlay"
    assert definition.signals[1].predictor_key == "pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-5050"

    rebuilt = build_executable_evaluator_strategy_spec_from_definition(definition)
    assert rebuilt.predictor_use is not None
    assert rebuilt.predictor_use.predictor_weight == 0.4
    assert rebuilt.predictor_signal_execution_context is not None
    assert rebuilt.predictor_signal_execution_context["predictorKey"] == "pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-5050"


def test_selection_definition_builder_accepts_explicit_signal_metadata() -> None:
    observation_spec = build_observation_spec(
        key="observation__builder_metadata",
        label="Builder metadata observation",
        tickers=("SPY", "QQQ", "TLT"),
        fields=("close", "volume"),
    )
    definition = build_selection_strategy_definition(
        SelectionDefinitionDefinition(
            strategy_id="builder__selection__metadata",
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            execution_policy=build_execution_policy_spec(
                key="month_end",
                label="月次",
                entry="train_once_then_periodic_rebalance",
                rebalance_schedule="month_end",
            ),
            decision_schedule="every_bar",
            investment_universe=build_investment_universe_spec(
                tickers=["SPY", "QQQ", "TLT"],
                key="builder_universe",
                label="Builder universe",
            ),
            risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
            data_source_spec=build_strategy_data_source_spec(
                key="data_source__builder_selection",
                label="Selection market data",
                kind="market_panel",
                observation_spec=observation_spec,
            ),
            feature_definition_spec=build_strategy_feature_definition_spec(
                key="feature_definition__builder_selection",
                label="Selection features",
                source_field_keys=("close", "volume"),
                derived_feature_keys=("momentum", "volume_strength"),
            ),
            alignment_policy=build_alignment_policy_spec(
                key="alignment_policy__builder_selection",
                label="Weekly alignment",
                method="end_of_period",
                parameters={"anchor": "week_end"},
            ),
        )
    )

    signal = definition.signals[0]
    assert signal.data_source_spec is not None
    assert signal.data_source_spec.kind == "market_panel"
    assert signal.feature_definition_spec is not None
    assert signal.feature_definition_spec.derived_feature_keys == ("momentum", "volume_strength")
    assert signal.alignment_policy is not None
    assert signal.alignment_policy.method == "end_of_period"


def test_selection_definition_builder_supports_distinct_signal_timeframe() -> None:
    definition = build_selection_strategy_definition(
        SelectionDefinitionDefinition(
            strategy_id="builder__selection__multi_tf",
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            execution_policy=build_execution_policy_spec(
                key="month_end",
                label="月次",
                entry="train_once_then_periodic_rebalance",
                rebalance_schedule="month_end",
            ),
            decision_schedule="every_bar",
            investment_universe=build_investment_universe_spec(
                tickers=["SPY", "QQQ", "TLT"],
                key="builder_universe",
                label="Builder universe",
            ),
            risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
        )
    )

    assert definition.signals[0].data_timeframe.key == "1d"
    assert definition.signals[0].signal_timeframe.key == "1w"
    assert definition.execution_plan.decision_schedule == "every_bar"
    assert definition.execution_plan.rebalance_schedule == "month_end"


def test_predictor_definition_builder_accepts_explicit_signal_metadata() -> None:
    selection_observation_spec = build_observation_spec(
        key="observation__builder_predictor_selection_metadata",
        label="Predictor selection observation",
        tickers=("SPY", "QQQ", "TLT"),
        fields=("close",),
    )
    predictor_observation_spec = build_observation_spec(
        key="observation__builder_predictor_overlay_metadata",
        label="Predictor overlay observation",
        tickers=("SPY", "QQQ", "TLT"),
        fields=("close", "volume"),
    )
    definition = build_predictor_strategy_definition(
        PredictorDefinitionDefinition(
            strategy_id="builder__predictor__metadata",
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            predictor_key="pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-5050",
            signal_weight=0.6,
            predictor_weight=0.4,
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            execution_policy=build_execution_policy_spec(
                key="month_end",
                label="月次",
                entry="train_once_then_periodic_rebalance",
                rebalance_schedule="month_end",
            ),
            selection_signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            predictor_data_timeframe=DEFAULT_MONTHLY_TIMEFRAME,
            predictor_signal_timeframe=DEFAULT_MONTHLY_TIMEFRAME,
            decision_schedule="every_bar",
            investment_universe=build_investment_universe_spec(
                tickers=["SPY", "QQQ", "TLT"],
                key="builder_universe",
                label="Builder universe",
            ),
            risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
            label="Builder predictor strategy",
            hypothesis="Builder predictor hypothesis",
            description="Builder predictor description",
            selection_data_source_spec=build_strategy_data_source_spec(
                key="data_source__builder_predictor_selection",
                label="Predictor selection source",
                kind="selection_panel",
                observation_spec=selection_observation_spec,
            ),
            selection_feature_definition_spec=build_strategy_feature_definition_spec(
                key="feature_definition__builder_predictor_selection",
                label="Predictor selection features",
                source_field_keys=("close",),
                derived_feature_keys=("momentum",),
            ),
            predictor_data_source_spec=build_strategy_data_source_spec(
                key="data_source__builder_predictor_overlay",
                label="Predictor overlay source",
                kind="predictor_panel",
                observation_spec=predictor_observation_spec,
            ),
            predictor_feature_definition_spec=build_strategy_feature_definition_spec(
                key="feature_definition__builder_predictor_overlay",
                label="Predictor overlay features",
                source_field_keys=("close", "volume"),
                derived_feature_keys=("momentum", "score"),
            ),
            predictor_alignment_policy=build_alignment_policy_spec(
                key="alignment_policy__builder_predictor_overlay",
                label="Predictor month-end alignment",
                method="calendar_resample",
                parameters={"toTimeframe": "1mo"},
            ),
        )
    )

    selection_signal, predictor_signal = definition.signals
    assert selection_signal.data_source_spec is not None
    assert selection_signal.data_source_spec.kind == "selection_panel"
    assert predictor_signal.data_source_spec is not None
    assert predictor_signal.data_source_spec.kind == "predictor_panel"
    assert predictor_signal.feature_definition_spec is not None
    assert predictor_signal.feature_definition_spec.derived_feature_keys == ("momentum", "score")
    assert predictor_signal.alignment_policy is not None
    assert predictor_signal.alignment_policy.method == "calendar_resample"


def test_predictor_definition_builder_supports_distinct_predictor_timeframes() -> None:
    definition = build_predictor_strategy_definition(
        PredictorDefinitionDefinition(
            strategy_id="builder__predictor__multi_tf",
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            predictor_key="pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-5050",
            signal_weight=0.6,
            predictor_weight=0.4,
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            execution_policy=build_execution_policy_spec(
                key="month_end",
                label="月次",
                entry="train_once_then_periodic_rebalance",
                rebalance_schedule="month_end",
            ),
            selection_signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            predictor_data_timeframe=DEFAULT_MONTHLY_TIMEFRAME,
            predictor_signal_timeframe=DEFAULT_MONTHLY_TIMEFRAME,
            decision_schedule="every_bar",
            investment_universe=build_investment_universe_spec(
                tickers=["SPY", "QQQ", "TLT"],
                key="builder_universe",
                label="Builder universe",
            ),
            risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
            label="Builder predictor strategy",
            hypothesis="Builder predictor hypothesis",
            description="Builder predictor description",
        )
    )

    assert definition.signals[0].data_timeframe.key == "1d"
    assert definition.signals[0].signal_timeframe.key == "1w"
    assert definition.signals[1].data_timeframe.key == "1mo"
    assert definition.signals[1].signal_timeframe.key == "1mo"
    assert definition.execution_plan.decision_schedule == "every_bar"
    assert definition.execution_plan.rebalance_schedule == "month_end"


def test_direct_execution_definition_compatibility_helper_accepts_signal_timeframe_execution() -> None:
    definition = build_selection_strategy_definition(
        SelectionDefinitionDefinition(
            strategy_id="builder__selection__direct_execution",
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            execution_policy=build_execution_policy_spec(
                key="month_end",
                label="月次",
                entry="train_once_then_periodic_rebalance",
                rebalance_schedule="month_end",
            ),
            decision_schedule="every_bar",
            investment_universe=build_investment_universe_spec(
                tickers=["SPY", "QQQ", "TLT"],
                key="builder_universe",
                label="Builder universe",
            ),
            risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
        )
    )

    assert is_direct_execution_compatible_strategy_definition(definition) is True
    assert get_direct_execution_strategy_definition_compatibility_issues(definition) == []

    executable_strategy = build_executable_evaluator_strategy_spec_from_definition(definition)
    assert executable_strategy.timeframe.key == "1w"
    assert executable_strategy.execution_mode == "direct_signal_timeframe"
    assert executable_strategy.signal_execution_contexts[0]["dataTimeframe"] == "1d"
    assert executable_strategy.signal_execution_contexts[0]["signalTimeframe"] == "1w"
    assert executable_strategy.decision_schedule == "every_bar"
    assert executable_strategy.execution_policy.rebalance_schedule == "month_end"


def test_direct_execution_definition_compatibility_helper_accepts_multi_selection_blend() -> None:
    definition = build_selection_strategy_definition(
        SelectionDefinitionDefinition(
            strategy_id="builder__selection__multi_signal_direct_execution",
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            execution_policy=build_execution_policy_spec(
                key="month_end",
                label="月次",
                entry="train_once_then_periodic_rebalance",
                rebalance_schedule="month_end",
            ),
            decision_schedule="every_bar",
            investment_universe=build_investment_universe_spec(
                tickers=["SPY", "QQQ", "TLT"],
                key="builder_universe",
                label="Builder universe",
            ),
            risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
        )
    )
    secondary_signal = build_strategy_signal_spec(
        key="selection__secondary",
        label="Secondary momentum signal",
        description="Secondary momentum signal",
        observation_spec=definition.signals[0].observation_spec,
        data_timeframe=DEFAULT_DAILY_TIMEFRAME,
        signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        source_kind="selection_signal",
        weight=0.4,
        signal_parameters={
            "selectionKey": "secondary_momo6",
            "strategyType": "full_universe_momentum_tilt",
            "scoreParameters": {
                "tilt_strength": 0.35,
                "tilt_shape": 1.0,
                "windowSpec": {"unit": "months", "value": 6},
            },
        },
    )
    definition = replace(definition, signals=(definition.signals[0], secondary_signal))

    assert is_direct_execution_compatible_strategy_definition(definition) is True
    executable_strategy = build_executable_evaluator_strategy_spec_from_definition(definition)
    assert executable_strategy.timeframe.key == "1w"
    assert len(executable_strategy.signal_execution_contexts) == 2
    assert executable_strategy.signal_execution_contexts[1]["selectionKey"] == "secondary_momo6"
    assert executable_strategy.signal_execution_contexts[1]["signalTimeframe"] == "1w"


def test_direct_execution_definition_compatibility_helper_preserves_selection_alignment_policy_payloads() -> None:
    definition = build_selection_strategy_definition(
        SelectionDefinitionDefinition(
            strategy_id="builder__selection__alignment_payloads",
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            execution_policy=build_execution_policy_spec(
                key="month_end",
                label="月次",
                entry="train_once_then_periodic_rebalance",
                rebalance_schedule="month_end",
            ),
            decision_schedule="every_bar",
            investment_universe=build_investment_universe_spec(
                tickers=["SPY", "QQQ", "TLT"],
                key="builder_universe",
                label="Builder universe",
            ),
            risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
            alignment_policy=build_alignment_policy_spec(
                key="primary_asof_last",
                label="Primary as-of",
                method="asof_last",
            ),
        )
    )
    secondary_signal = build_strategy_signal_spec(
        key="selection__secondary",
        label="Secondary momentum signal",
        description="Secondary momentum signal",
        observation_spec=definition.signals[0].observation_spec,
        data_timeframe=DEFAULT_DAILY_TIMEFRAME,
        signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        source_kind="selection_signal",
        weight=0.4,
        alignment_policy=build_alignment_policy_spec(
            key="secondary_end_of_period",
            label="Secondary end-of-period",
            method="end_of_period",
        ),
        signal_parameters={
            "selectionKey": "secondary_momo6",
            "strategyType": "full_universe_momentum_tilt",
            "scoreParameters": {
                "tilt_strength": 0.35,
                "tilt_shape": 1.0,
                "windowSpec": {"unit": "months", "value": 6},
            },
        },
    )
    definition = replace(definition, signals=(definition.signals[0], secondary_signal))

    executable_strategy = build_executable_evaluator_strategy_spec_from_definition(definition)

    assert executable_strategy.signal_execution_contexts[0]["alignmentPolicy"]["method"] == "asof_last"
    assert executable_strategy.signal_execution_contexts[1]["alignmentPolicy"]["method"] == "end_of_period"
    assert executable_strategy.signal_execution_contexts[1]["dataTimeframe"] == "1d"
    assert executable_strategy.signal_execution_contexts[1]["signalTimeframe"] == "1w"


def test_build_strategy_signal_execution_contexts_from_definition_returns_selection_and_predictor_contexts() -> None:
    definition = build_predictor_strategy_definition(
        PredictorDefinitionDefinition(
            strategy_id="builder__predictor__execution_contexts",
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            predictor_key="pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-5050",
            signal_weight=0.6,
            predictor_weight=0.4,
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            selection_signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            predictor_data_timeframe=DEFAULT_DAILY_TIMEFRAME,
            predictor_signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            predictor_alignment_policy=build_alignment_policy_spec(
                key="predictor_calendar_resample",
                label="Predictor calendar resample",
                method="calendar_resample",
            ),
            execution_policy=build_execution_policy_spec(
                key="month_end",
                label="月次",
                entry="train_once_then_periodic_rebalance",
                rebalance_schedule="month_end",
            ),
            decision_schedule="every_bar",
            investment_universe=build_investment_universe_spec(
                tickers=["SPY", "QQQ", "TLT"],
                key="builder_universe",
                label="Builder universe",
            ),
            risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
            label="Builder predictor direct strategy",
            hypothesis="Builder predictor direct hypothesis",
            description="Builder predictor direct description",
        )
    )
    secondary_signal = build_strategy_signal_spec(
        key="selection__secondary",
        label="Secondary momentum signal",
        description="Secondary momentum signal",
        observation_spec=definition.signals[0].observation_spec,
        data_timeframe=DEFAULT_DAILY_TIMEFRAME,
        signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        source_kind="selection_signal",
        weight=0.4,
        alignment_policy=build_alignment_policy_spec(
            key="secondary_end_of_period",
            label="Secondary end-of-period",
            method="end_of_period",
        ),
        signal_parameters={
            "selectionKey": "secondary_momo6",
            "strategyType": "full_universe_momentum_tilt",
            "scoreParameters": {
                "tilt_strength": 0.35,
                "tilt_shape": 1.0,
                "windowSpec": {"unit": "months", "value": 6},
            },
        },
    )
    definition = replace(definition, signals=(definition.signals[0], secondary_signal, definition.signals[1]))

    selection_contexts, predictor_context = build_strategy_signal_execution_contexts_from_definition(definition)

    assert len(selection_contexts) == 2
    assert isinstance(selection_contexts[0]["selectionKey"], str)
    assert selection_contexts[1]["alignmentPolicy"]["method"] == "end_of_period"
    assert predictor_context is not None
    assert predictor_context["predictorKey"] == "pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-5050"
    assert predictor_context["alignmentPolicy"]["method"] == "calendar_resample"


def test_get_strategy_signal_execution_contexts_returns_selection_and_predictor_contexts() -> None:
    strategy = build_evaluator_strategy_spec(
        strategy_id="signal_execution_contexts",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            tickers=["AAA", "BBB"],
            key="signal_execution_contexts_universe",
            label="Signal execution contexts universe",
        ),
        selection=build_selection_spec(
            "full_universe_momentum_tilt",
            score_parameters={
                "tilt_strength": 0.35,
                "tilt_shape": 1.0,
                "windowSpec": {"unit": "bars", "value": 3},
            },
        ),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="month_end",
            label="月次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-signal-context",
            signal_weight=0.6,
            predictor_weight=0.4,
        ),
        signal_execution_contexts=[
            {
                "selectionKey": "full_universe_momentum_tilt",
                "strategyType": "full_universe_momentum_tilt",
                "label": "Primary momentum",
                "description": "Primary momentum",
                "scoreParameters": {
                    "tilt_strength": 0.35,
                    "tilt_shape": 1.0,
                    "windowSpec": {"unit": "bars", "value": 3},
                },
                "weight": 0.7,
                "dataTimeframe": "1d",
                "signalTimeframe": "1w",
                "alignmentPolicy": {"method": "end_of_period"},
            },
            {
                "selectionKey": "secondary_momo6",
                "strategyType": "full_universe_momentum_tilt",
                "label": "Secondary momentum",
                "description": "Secondary momentum",
                "scoreParameters": {
                    "tilt_strength": 0.35,
                    "tilt_shape": 1.0,
                    "windowSpec": {"unit": "bars", "value": 6},
                },
                "weight": 0.3,
                "dataTimeframe": "1d",
                "signalTimeframe": "1w",
                "alignmentPolicy": {"method": "calendar_resample"},
            },
        ],
        predictor_signal_execution_context={
            "predictorKey": "pred-signal-context",
            "signalWeight": 0.6,
            "predictorWeight": 0.4,
            "dataTimeframe": "1d",
            "signalTimeframe": "1w",
            "alignmentPolicy": {"method": "calendar_resample"},
        },
    )

    selection_contexts, predictor_context = get_strategy_signal_execution_contexts(strategy)

    assert len(selection_contexts) == 2
    assert selection_contexts[0]["source_kind"] == "selection_signal"
    assert selection_contexts[0]["weight"] == 0.7
    assert selection_contexts[1]["alignment_policy"]["method"] == "calendar_resample"
    assert predictor_context is not None
    assert predictor_context["source_kind"] == "predictor_overlay"
    assert predictor_context["predictor_key"] == "pred-signal-context"
    assert predictor_context["alignment_policy"]["method"] == "calendar_resample"


def test_serialize_evaluator_strategy_spec_includes_explicit_execution_contexts() -> None:
    strategy = build_evaluator_strategy_spec(
        strategy_id="serialized_context_strategy",
        version="v1",
        label="Serialized context strategy",
        description="Serialized context strategy",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            key="serialized_context_universe",
            label="Serialized context universe",
            tickers=["SPY", "QQQ", "TLT"],
        ),
        selection=build_selection_spec(
            "full_universe_momentum_tilt",
            score_parameters={
                "tilt_strength": 0.35,
                "tilt_shape": 1.0,
                "windowSpec": {"unit": "months", "value": 8},
            },
        ),
        portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
        execution_policy=build_execution_policy_spec(
            key="month_end",
            label="月次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=0.9, max_weight=0.45),
        predictor_use=build_predictor_use_spec(
            predictor_key="pred10mom5050",
            signal_weight=0.6,
            predictor_weight=0.4,
        ),
        signal_execution_contexts=[
            {
                "sourceKind": "selection_signal",
                "selectionKey": "full_universe_momentum_tilt",
                "strategyType": "full_universe_momentum_tilt",
                "scoreParameters": {
                    "tilt_strength": 0.35,
                    "tilt_shape": 1.0,
                    "windowSpec": {"unit": "months", "value": 8},
                },
                "dataTimeframe": "1d",
                "signalTimeframe": "1w",
                "alignmentPolicy": {"key": "weekly", "label": "Weekly", "method": "asof_last", "parameters": {}},
                "weight": 1.0,
            },
        ],
        predictor_signal_execution_context={
            "sourceKind": "predictor_overlay",
            "predictorKey": "pred10mom5050",
            "signalWeight": 0.6,
            "predictorWeight": 0.4,
            "dataTimeframe": "1d",
            "signalTimeframe": "1w",
            "alignmentPolicy": {"key": "weekly", "label": "Weekly", "method": "calendar_resample", "parameters": {}},
        },
        decision_schedule="every_bar",
    )

    payload = serialize_evaluator_strategy_spec(strategy)

    assert payload["components"]["core"]["decisionSchedule"] == "every_bar"
    assert payload["components"]["core"]["executionMode"] is None
    assert payload["components"]["optional"]["signalExecutionContexts"]["selectionSignals"][0]["dataTimeframe"] == "1d"
    assert payload["components"]["optional"]["signalExecutionContexts"]["predictorSignal"]["predictorKey"] == "pred10mom5050"


def test_get_strategy_definition_signal_execution_contexts_prefers_explicit_strategy_contexts() -> None:
    selection_contexts = [
        {
            "signalKey": "selection_signal",
            "signalLabel": "Selection signal",
            "description": "Selection signal",
            "sourceKind": "selection_signal",
            "selectionKey": "full_universe_momentum_tilt",
            "strategyType": "full_universe_momentum_tilt",
            "scoreParameters": {
                "tilt_strength": 0.35,
                "tilt_shape": 1.0,
                "windowSpec": {"unit": "months", "value": 8},
            },
            "dataTimeframe": "1d",
            "signalTimeframe": "1w",
            "alignmentPolicy": {"key": "weekly", "label": "Weekly", "method": "asof_last", "parameters": {}},
            "weight": 1.0,
        }
    ]
    predictor_context = {
        "sourceKind": "predictor_overlay",
        "predictorKey": "pred10mom5050",
        "signalWeight": 0.6,
        "predictorWeight": 0.4,
        "dataTimeframe": "1d",
        "signalTimeframe": "1w",
        "alignmentPolicy": {"key": "weekly", "label": "Weekly", "method": "calendar_resample", "parameters": {}},
    }
    strategy = build_evaluator_strategy_spec(
        strategy_id="explicit_context_strategy",
        version="v1",
        label="Explicit context strategy",
        description="Explicit context strategy",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            key="explicit_context_universe",
            label="Explicit context universe",
            tickers=["SPY", "QQQ", "TLT"],
        ),
        selection=build_selection_spec(
            "full_universe_momentum_tilt",
            score_parameters={
                "tilt_strength": 0.35,
                "tilt_shape": 1.0,
                "windowSpec": {"unit": "months", "value": 8},
            },
        ),
        portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
        execution_policy=build_execution_policy_spec(
            key="month_end",
            label="月次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=0.9, max_weight=0.45),
        predictor_use=build_predictor_use_spec(
            predictor_key="pred10mom5050",
            signal_weight=0.6,
            predictor_weight=0.4,
        ),
        signal_execution_contexts=selection_contexts,
        predictor_signal_execution_context=predictor_context,
        decision_schedule="every_bar",
    )

    resolved_selection_contexts, resolved_predictor_context = get_strategy_definition_signal_execution_contexts(strategy)

    assert resolved_selection_contexts == selection_contexts
    assert resolved_predictor_context == predictor_context
    assert resolve_decision_schedule(strategy) == "every_bar"
    assert resolve_strategy_market_data_timeframe_key(strategy) == "1d"


def test_evaluator_strategy_spec_extensions_do_not_rehydrate_execution_contexts() -> None:
    strategy = build_evaluator_strategy_spec(
        strategy_id="strategy_definition_signal_execution_contexts",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            tickers=["AAA", "BBB"],
            key="strategy_definition_signal_execution_contexts_universe",
            label="Strategy definition signal execution contexts universe",
        ),
        selection=build_selection_spec(
            "full_universe_momentum_tilt",
            score_parameters={
                "tilt_strength": 0.35,
                "tilt_shape": 1.0,
                "windowSpec": {"unit": "bars", "value": 3},
            },
        ),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="month_end",
            label="月次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-signal-context",
            signal_weight=0.6,
            predictor_weight=0.4,
        ),
        extensions={
            "signal_data_timeframe": "1d",
            "signal_timeframe": "1w",
            "selection_signal_weight": "0.7",
            "selection_signal_alignment_policy": json.dumps({"method": "end_of_period"}, sort_keys=True),
            "predictor_signal_data_timeframe": "1d",
            "predictor_signal_timeframe": "1w",
            "predictor_signal_alignment_policy": json.dumps({"method": "calendar_resample"}, sort_keys=True),
        },
    )

    selection_contexts, predictor_context = get_strategy_definition_signal_execution_contexts(strategy)

    assert len(selection_contexts) == 1
    assert selection_contexts[0]["sourceKind"] == "selection_signal"
    assert selection_contexts[0]["selectionKey"] == strategy.selection.key
    assert selection_contexts[0]["dataTimeframe"] == "1w"
    assert selection_contexts[0]["signalTimeframe"] == "1w"
    assert selection_contexts[0]["alignmentPolicy"] is None
    assert predictor_context is not None
    assert predictor_context["sourceKind"] == "predictor_overlay"
    assert predictor_context["predictorKey"] == "pred-signal-context"
    assert predictor_context["dataTimeframe"] == "1w"
    assert predictor_context["signalTimeframe"] == "1w"
    assert predictor_context["alignmentPolicy"] is None


def test_direct_execution_definition_compatibility_helper_preserves_predictor_alignment_policy_payload() -> None:
    definition = build_predictor_strategy_definition(
        PredictorDefinitionDefinition(
            strategy_id="builder__predictor__alignment_payload",
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            predictor_key="pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-5050",
            signal_weight=0.6,
            predictor_weight=0.4,
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            selection_signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            predictor_data_timeframe=DEFAULT_DAILY_TIMEFRAME,
            predictor_signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            predictor_alignment_policy=build_alignment_policy_spec(
                key="predictor_calendar_resample",
                label="Predictor calendar resample",
                method="calendar_resample",
            ),
            execution_policy=build_execution_policy_spec(
                key="month_end",
                label="月次",
                entry="train_once_then_periodic_rebalance",
                rebalance_schedule="month_end",
            ),
            decision_schedule="every_bar",
            investment_universe=build_investment_universe_spec(
                tickers=["SPY", "QQQ", "TLT"],
                key="builder_universe",
                label="Builder universe",
            ),
            risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
            label="Builder predictor direct strategy",
            hypothesis="Builder predictor direct hypothesis",
            description="Builder predictor direct description",
        )
    )

    executable_strategy = build_executable_evaluator_strategy_spec_from_definition(definition)
    predictor_payload = extract_predictor_signal_payload(executable_strategy)

    assert executable_strategy.predictor_signal_execution_context is not None
    assert executable_strategy.predictor_signal_execution_context["alignmentPolicy"]["method"] == "calendar_resample"
    assert predictor_payload is not None
    assert predictor_payload["alignmentPolicy"]["method"] == "calendar_resample"
    assert predictor_payload["signalTimeframe"] == "1w"


def test_direct_execution_definition_compatibility_helper_accepts_predictor_overlay_with_multi_selection_blend() -> None:
    definition = build_predictor_strategy_definition(
        PredictorDefinitionDefinition(
            strategy_id="builder__predictor__multi_selection_direct_execution",
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            predictor_key="pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-5050",
            signal_weight=0.6,
            predictor_weight=0.4,
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            selection_signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            predictor_data_timeframe=DEFAULT_DAILY_TIMEFRAME,
            predictor_signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            execution_policy=build_execution_policy_spec(
                key="month_end",
                label="月次",
                entry="train_once_then_periodic_rebalance",
                rebalance_schedule="month_end",
            ),
            decision_schedule="every_bar",
            investment_universe=build_investment_universe_spec(
                tickers=["SPY", "QQQ", "TLT"],
                key="builder_universe",
                label="Builder universe",
            ),
            risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
            label="Builder predictor direct strategy",
            hypothesis="Builder predictor direct hypothesis",
            description="Builder predictor direct description",
        )
    )
    secondary_signal = build_strategy_signal_spec(
        key="selection__secondary",
        label="Secondary momentum signal",
        description="Secondary momentum signal",
        observation_spec=definition.signals[0].observation_spec,
        data_timeframe=DEFAULT_DAILY_TIMEFRAME,
        signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        source_kind="selection_signal",
        weight=0.4,
        signal_parameters={
            "selectionKey": "secondary_momo6",
            "strategyType": "full_universe_momentum_tilt",
            "scoreParameters": {
                "tilt_strength": 0.35,
                "tilt_shape": 1.0,
                "windowSpec": {"unit": "months", "value": 6},
            },
        },
    )
    definition = replace(
        definition,
        signals=(definition.signals[0], secondary_signal, definition.signals[1]),
    )

    assert is_direct_execution_compatible_strategy_definition(definition) is True
    executable_strategy = build_executable_evaluator_strategy_spec_from_definition(definition)
    assert executable_strategy.predictor_use is not None
    assert executable_strategy.predictor_use.predictor_key == "pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-5050"
    assert executable_strategy.signal_execution_contexts[1]["selectionKey"] == "secondary_momo6"


def test_direct_execution_definition_compatibility_helper_accepts_predictor_overlay() -> None:
    definition = build_predictor_strategy_definition(
        PredictorDefinitionDefinition(
            strategy_id="builder__predictor__direct_execution",
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            predictor_key="pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-5050",
            signal_weight=0.6,
            predictor_weight=0.4,
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            selection_signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            predictor_data_timeframe=DEFAULT_DAILY_TIMEFRAME,
            predictor_signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            execution_policy=build_execution_policy_spec(
                key="month_end",
                label="月次",
                entry="train_once_then_periodic_rebalance",
                rebalance_schedule="month_end",
            ),
            decision_schedule="every_bar",
            investment_universe=build_investment_universe_spec(
                tickers=["SPY", "QQQ", "TLT"],
                key="builder_universe",
                label="Builder universe",
            ),
            risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
            label="Builder predictor direct strategy",
            hypothesis="Builder predictor direct hypothesis",
            description="Builder predictor direct description",
        )
    )

    assert is_direct_execution_compatible_strategy_definition(definition) is True
    assert get_direct_execution_strategy_definition_compatibility_issues(definition) == []

    executable_strategy = build_executable_evaluator_strategy_spec_from_definition(definition)
    assert executable_strategy.timeframe.key == "1w"
    assert executable_strategy.predictor_use is not None
    assert executable_strategy.predictor_use.predictor_key == "pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-5050"
    assert executable_strategy.predictor_use.predictor_weight == 0.4
    assert executable_strategy.execution_mode == "direct_signal_timeframe"


def test_direct_execution_definition_compatibility_helper_reports_blockers() -> None:
    definition = build_selection_strategy_definition(
        SelectionDefinitionDefinition(
            strategy_id="builder__selection__incompatible",
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            signal_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
            execution_policy=build_execution_policy_spec(
                key="month_end",
                label="月次",
                entry="train_once_then_periodic_rebalance",
                rebalance_schedule="month_end",
            ),
            decision_schedule="every_bar",
            investment_universe=build_investment_universe_spec(
                tickers=["SPY", "QQQ", "TLT"],
                key="builder_universe",
                label="Builder universe",
            ),
            risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
        )
    )

    definition = replace(
        definition,
        signals=(
            replace(
                definition.signals[0],
                data_timeframe=DEFAULT_WEEKLY_TIMEFRAME,
                signal_timeframe=DEFAULT_DAILY_TIMEFRAME,
            ),
        ),
        execution_plan=build_strategy_execution_plan_spec(
            key="execution__incompatible",
            label="Incompatible execution",
            decision_schedule="quarter_end",
            rebalance_schedule="month_end",
        ),
    )

    issues = get_direct_execution_strategy_definition_compatibility_issues(definition)
    assert is_direct_execution_compatible_strategy_definition(definition) is False
    assert "direct execution requires decision_schedule to match rebalance_schedule or be every_bar" in issues
    assert "direct execution requires signal_timeframe to be coarser than or equal to data_timeframe" in issues

    try:
        build_executable_evaluator_strategy_spec_from_definition(definition)
    except ValueError as exc:
        assert "Direct execution incompatibilities" in str(exc)
        assert "decision_schedule to match rebalance_schedule or be every_bar" in str(exc)
    else:
        raise AssertionError("Expected direct execution incompatibility error")


def test_selection_definition_product_builder_generates_cross_product() -> None:
    definitions = build_selection_strategy_definition_product(
        strategy_id_pattern="prod-{selection}-{portfolio_model}-{execution}",
        selection_variants=[
            SelectionVariantDefinition(
                key="momo2",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            ),
        ],
        portfolio_model_variants=[
            PortfolioModelVariantDefinition(
                key="eq",
                portfolio_model=build_portfolio_model_spec("equal_weight"),
            ),
            PortfolioModelVariantDefinition(
                key="hrp",
                portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            ),
        ],
        execution_variants=[
            ExecutionVariantDefinition(
                key="month",
                timeframe=DEFAULT_DAILY_TIMEFRAME,
                execution_policy=build_execution_policy_spec(
                    key="month_end",
                    label="月次",
                    entry="train_once_then_periodic_rebalance",
                    rebalance_schedule="month_end",
                ),
            ),
            ExecutionVariantDefinition(
                key="daily",
                timeframe=DEFAULT_DAILY_TIMEFRAME,
                execution_policy=build_execution_policy_spec(
                    key="every_bar",
                    label="毎バー",
                    entry="train_once_then_periodic_rebalance",
                    rebalance_schedule="every_bar",
                ),
            ),
        ],
        investment_universe=build_investment_universe_spec(
            tickers=["SPY", "QQQ", "TLT"],
            key="product_universe",
            label="Product universe",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
    )

    assert [definition.strategy_id for definition in definitions] == [
        "prod-momo2-eq-month",
        "prod-momo2-eq-daily",
        "prod-momo2-hrp-month",
        "prod-momo2-hrp-daily",
    ]


def test_predictor_definition_product_builder_generates_cross_product() -> None:
    definitions = build_predictor_strategy_definition_product(
        strategy_id_pattern="prod-{selection}-{predictor}-{portfolio_model}-{execution}",
        selection_variants=[
            SelectionVariantDefinition(
                key="momo2",
                selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            ),
        ],
        predictor_variants=[
            PredictorVariantDefinition(
                key="pred5050",
                predictor_key="pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-5050",
                signal_weight=0.6,
                predictor_weight=0.4,
                label="Predictor blend",
                hypothesis="Predictor hypothesis",
                description="Predictor description",
            ),
        ],
        portfolio_model_variants=[
            PortfolioModelVariantDefinition(
                key="hrp",
                portfolio_model=build_portfolio_model_spec("hierarchical_risk_parity"),
            ),
        ],
        execution_variants=[
            ExecutionVariantDefinition(
                key="month",
                timeframe=DEFAULT_DAILY_TIMEFRAME,
                execution_policy=build_execution_policy_spec(
                    key="month_end",
                    label="月次",
                    entry="train_once_then_periodic_rebalance",
                    rebalance_schedule="month_end",
                ),
                label="月次",
            ),
            ExecutionVariantDefinition(
                key="daily",
                timeframe=DEFAULT_DAILY_TIMEFRAME,
                execution_policy=build_execution_policy_spec(
                    key="every_bar",
                    label="毎バー",
                    entry="train_once_then_periodic_rebalance",
                    rebalance_schedule="every_bar",
                ),
                label="毎バー",
            ),
        ],
        investment_universe=build_investment_universe_spec(
            tickers=["SPY", "QQQ", "TLT"],
            key="product_universe",
            label="Product universe",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0, max_weight=0.45),
    )

    assert [definition.strategy_id for definition in definitions] == [
        "prod-momo2-pred5050-hrp-month",
        "prod-momo2-pred5050-hrp-daily",
    ]
    assert [definition.label for definition in definitions] == [
        "Predictor blend × 月次",
        "Predictor blend × 毎バー",
    ]


def test_filtered_candidates_are_strategy_definitions() -> None:
    assert len({definition.strategy_id for definition in FILTERED_CANDIDATE_DEFINITIONS}) == len(FILTERED_CANDIDATE_DEFINITIONS)
    assert all(definition.signals[0].source_kind == "selection_signal" for definition in FILTERED_CANDIDATE_DEFINITIONS)
    strategy_by_id = {definition.strategy_id: definition for definition in FILTERED_CANDIDATE_DEFINITIONS}
    assert strategy_by_id["stg-posmom-hrp-month"].execution_plan.decision_schedule == "month_end"
    assert strategy_by_id["stg-posrev5-trend60-hrp-month"].execution_plan.decision_schedule == "month_end"
    assert strategy_by_id["stg-riskoff-posmom-hrp-month"].execution_plan.decision_schedule == "month_end"


def test_universe_variant_candidates_are_strategy_definitions() -> None:
    assert len({definition.strategy_id for definition in UNIVERSE_VARIANT_CANDIDATE_DEFINITIONS}) == len(UNIVERSE_VARIANT_CANDIDATE_DEFINITIONS)
    assert all(definition.investment_universe.tickers for definition in UNIVERSE_VARIANT_CANDIDATE_DEFINITIONS)


def test_predictor_candidates_are_strategy_definitions() -> None:
    assert len({definition.strategy_id for definition in PREDICTOR_CANDIDATE_DEFINITIONS}) == len(PREDICTOR_CANDIDATE_DEFINITIONS)

    predictor_candidate = PREDICTOR_CANDIDATE_DEFINITIONS[1]
    assert predictor_candidate.execution_plan.decision_schedule == "month_end"
    assert predictor_candidate.signals[0].weight == 0.6
    assert predictor_candidate.signals[1].source_kind == "predictor_overlay"
    assert predictor_candidate.signals[1].predictor_key is not None


def test_baseline_definition_catalog_has_unique_strategy_ids() -> None:
    assert len({definition.strategy_id for definition in BASELINE_CANDIDATE_DEFINITIONS}) == len(BASELINE_CANDIDATE_DEFINITIONS)


def test_canonical_definition_catalog_has_unique_strategy_ids() -> None:
    assert len({definition.strategy_id for definition in CANONICAL_CANDIDATE_DEFINITIONS}) == len(CANONICAL_CANDIDATE_DEFINITIONS)


def test_canonical_definition_catalog_uses_etf_universe() -> None:
    universe_keys = {definition.investment_universe.key for definition in CANONICAL_CANDIDATE_DEFINITIONS}

    assert universe_keys == {"etf"}
    assert all("BTC-USD" not in definition.investment_universe.tickers for definition in CANONICAL_CANDIDATE_DEFINITIONS)
    assert all("ETH-USD" not in definition.investment_universe.tickers for definition in CANONICAL_CANDIDATE_DEFINITIONS)


def test_timeframe_variant_candidates_are_strategy_definitions() -> None:
    assert len({definition.strategy_id for definition in TIMEFRAME_VARIANT_CANDIDATE_DEFINITIONS}) == len(TIMEFRAME_VARIANT_CANDIDATE_DEFINITIONS)

    monthly_candidate = TIMEFRAME_VARIANT_CANDIDATE_DEFINITIONS[2]
    assert monthly_candidate.execution_plan.decision_schedule == "month_end"
    assert monthly_candidate.execution_plan.rebalance_schedule == "month_end"
    assert monthly_candidate.signals[0].source_kind == "selection_signal"
    assert monthly_candidate.signals[0].data_timeframe.key == "1d"
