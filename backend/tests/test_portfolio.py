import json
from dataclasses import replace
from unittest.mock import patch

import pandas as pd

from app.portfolio import (
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
    build_strategy_definition_from_strategy_spec,
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
    build_direct_execution_strategy_spec_from_definition,
    build_executable_strategy_spec_from_definition,
    build_execution_policy_spec,
    build_risk_controls_spec,
    build_selection_spec,
    build_strategy_spec_from_definition,
    build_strategy_spec,
    compare_portfolio_runs,
    compute_predictor_panel,
    convert_window_spec_to_bars,
    compute_trade_cost,
    compute_strategy_score_series,
    evaluate_predictor_spec,
    get_direct_execution_strategy_definition_compatibility_issues,
    get_legacy_strategy_definition_compatibility_issues,
    get_strategy_definition_signal_execution_contexts,
    get_strategy_signal_execution_contexts,
    is_direct_execution_compatible_strategy_definition,
    is_legacy_compatible_strategy_definition,
    prepare_strategy_market_data,
    prepare_strategy_predictor_panel,
    prepare_strategy_signal_data,
    extract_predictor_signal_payload,
    prepare_signal_component_data,
    resample_market_frame_to_timeframe,
    resample_returns_frame_to_timeframe,
    resolve_decision_schedule,
    resolve_strategy_market_data_timeframe_key,
    serialize_strategy_spec,
    serialize_strategy_signal_spec,
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
    build_strategy_specs_from_definitions,
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
):
    return build_strategy_spec(
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


def test_build_strategy_definition_from_strategy_spec_maps_predictor_overlay() -> None:
    predictor_use = build_predictor_use_spec(
        predictor_key="pred__overlay",
        signal_weight=0.6,
        predictor_weight=0.4,
    )
    strategy = build_strategy_spec(
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

    definition = build_strategy_definition_from_strategy_spec(strategy)

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


def test_build_strategy_definition_from_strategy_spec_without_predictor_keeps_single_signal() -> None:
    strategy = make_strategy(
        "full_universe",
        "equal_weight",
        max_investment_ratio=0.8,
    )

    definition = build_strategy_definition_from_strategy_spec(strategy)

    assert definition.strategy_id == strategy.strategy_id
    assert len(definition.signals) == 1
    assert definition.signals[0].source_kind == "selection_signal"
    assert definition.signals[0].weight == 1.0
    assert definition.execution_plan.rebalance_schedule == strategy.execution_policy.rebalance_schedule


def test_build_strategy_spec_from_definition_round_trips_legacy_strategy() -> None:
    predictor_use = build_predictor_use_spec(
        predictor_key="pred__overlay",
        signal_weight=0.6,
        predictor_weight=0.4,
    )
    strategy = build_strategy_spec(
        strategy_id="strategy__round_trip",
        version="v7",
        hypothesis="Round-trip legacy adapter.",
        label="Round trip strategy",
        description="Legacy strategy for definition round-trip tests.",
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

    rebuilt = build_strategy_spec_from_definition(
        build_strategy_definition_from_strategy_spec(strategy)
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


def test_build_strategy_spec_from_definition_rejects_split_execution_plan() -> None:
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
                description="Legacy-compatible selection signal.",
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

    try:
        build_strategy_spec_from_definition(definition)
    except ValueError as exc:
        assert "matching decision and rebalance schedules" in str(exc)
    else:
        raise AssertionError("Expected split execution plan to be rejected.")


def test_build_strategy_spec_from_definition_rejects_mismatched_signal_timeframe() -> None:
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
                description="Legacy-compatible selection signal.",
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

    try:
        build_strategy_spec_from_definition(definition)
    except ValueError as exc:
        assert "requires matching selection data and signal timeframes" in str(exc)
    else:
        raise AssertionError("Expected mismatched signal timeframe to be rejected.")


def test_full_universe_candidates_are_derived_from_definitions() -> None:
    derived_strategies = build_strategy_specs_from_definitions(FULL_UNIVERSE_CANDIDATE_DEFINITIONS)
    assert len(FULL_UNIVERSE_CANDIDATE_DEFINITIONS) == len(derived_strategies)
    assert [
        strategy.strategy_id for strategy in derived_strategies
    ] == [
        definition.strategy_id for definition in FULL_UNIVERSE_CANDIDATE_DEFINITIONS
    ]

    core_candidate = FULL_UNIVERSE_CANDIDATE_DEFINITIONS[8]
    assert core_candidate.execution_plan.decision_schedule == "year_end"
    assert core_candidate.execution_plan.rebalance_schedule == "year_end"
    assert core_candidate.signals[0].data_timeframe.key == "1d"

    rebuilt = build_strategy_spec_from_definition(core_candidate)
    assert rebuilt == derived_strategies[8]



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



def test_selection_definition_builder_creates_legacy_compatible_definition() -> None:
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

    rebuilt = build_strategy_spec_from_definition(definition)
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

    rebuilt = build_strategy_spec_from_definition(definition)
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

    assert is_legacy_compatible_strategy_definition(definition) is False
    assert is_direct_execution_compatible_strategy_definition(definition) is True
    assert get_direct_execution_strategy_definition_compatibility_issues(definition) == []

    direct_strategy = build_direct_execution_strategy_spec_from_definition(definition)
    executable_strategy = build_executable_strategy_spec_from_definition(definition)
    assert direct_strategy.timeframe.key == "1w"
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
    executable_strategy = build_executable_strategy_spec_from_definition(definition)
    assert executable_strategy.timeframe.key == "1w"
    assert len(executable_strategy.signal_execution_contexts) == 2
    assert executable_strategy.signal_execution_contexts[1]["selectionKey"] == "secondary_momo6"
    assert executable_strategy.signal_execution_contexts[1]["signalTimeframe"] == "1w"


def test_resample_market_frame_to_timeframe_supports_alignment_methods() -> None:
    frame = pd.DataFrame(
        {"SPY": [100.0, 101.0, 102.0, 103.0]},
        index=pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"]),
    )

    end_of_period = resample_market_frame_to_timeframe(
        frame,
        target_timeframe_key="1w",
        value_kind="close",
        alignment_method="end_of_period",
    )
    calendar_resample = resample_market_frame_to_timeframe(
        frame,
        target_timeframe_key="1w",
        value_kind="close",
        alignment_method="calendar_resample",
    )
    asof_last = resample_market_frame_to_timeframe(
        frame,
        target_timeframe_key="1w",
        value_kind="close",
        alignment_method="asof_last",
    )

    assert str(end_of_period.index[-1].date()) == "2025-01-09"
    assert str(calendar_resample.index[-1].date()) == "2025-01-10"
    assert str(asof_last.index[-1].date()) == "2025-01-10"
    assert float(end_of_period.iloc[-1, 0]) == 103.0
    assert float(calendar_resample.iloc[-1, 0]) == 103.0


def test_prepare_signal_component_data_uses_alignment_policy_for_signal_returns() -> None:
    closes = pd.DataFrame(
        {"SPY": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-13"]),
    )
    returns = closes.pct_change().dropna()
    volumes = pd.DataFrame(
        {"SPY": [10.0, 11.0, 12.0, 13.0]},
        index=returns.index,
    )

    end_of_period_returns, end_of_period_volumes, end_of_period_bars = prepare_signal_component_data(
        history_returns=returns,
        volume_history=volumes,
        data_timeframe_key="1d",
        signal_timeframe_key="1w",
        alignment_policy={"method": "end_of_period"},
    )
    calendar_returns, calendar_volumes, calendar_bars = prepare_signal_component_data(
        history_returns=returns,
        volume_history=volumes,
        data_timeframe_key="1d",
        signal_timeframe_key="1w",
        alignment_policy={"method": "calendar_resample"},
    )

    assert end_of_period_bars == DEFAULT_WEEKLY_TIMEFRAME.bars_per_year
    assert calendar_bars == DEFAULT_WEEKLY_TIMEFRAME.bars_per_year
    assert str(end_of_period_returns.index[-1].date()) == "2025-01-13"
    assert str(calendar_returns.index[-1].date()) == "2025-01-17"
    assert str(end_of_period_volumes.index[0].date()) == "2025-01-13"
    assert str(calendar_volumes.index[0].date()) == "2025-01-17"


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

    executable_strategy = build_executable_strategy_spec_from_definition(definition)

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
    strategy = build_strategy_spec(
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
        extensions={
            "signal_data_timeframe": "1d",
            "signal_timeframe": "1w",
            "selection_signal_weight": "0.7",
            "selection_signal_alignment_policy": json.dumps({"method": "end_of_period"}, sort_keys=True),
            "additional_selection_signals": json.dumps([
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
                }
            ], sort_keys=True),
            "predictor_signal_data_timeframe": "1d",
            "predictor_signal_timeframe": "1w",
            "predictor_signal_alignment_policy": json.dumps({"method": "calendar_resample"}, sort_keys=True),
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


def test_serialize_strategy_spec_includes_explicit_execution_contexts() -> None:
    strategy = build_strategy_spec(
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

    payload = serialize_strategy_spec(strategy)

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
    strategy = build_strategy_spec(
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


def test_get_strategy_definition_signal_execution_contexts_normalizes_legacy_strategy() -> None:
    strategy = build_strategy_spec(
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
    assert selection_contexts[0]["dataTimeframe"] == "1d"
    assert selection_contexts[0]["signalTimeframe"] == "1w"
    assert selection_contexts[0]["alignmentPolicy"]["method"] == "end_of_period"
    assert predictor_context is not None
    assert predictor_context["sourceKind"] == "predictor_overlay"
    assert predictor_context["predictorKey"] == "pred-signal-context"
    assert predictor_context["alignmentPolicy"]["method"] == "calendar_resample"



def test_compare_portfolio_runs_uses_explicit_signal_execution_contexts() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 102, 104, 103, 105, 107, 108, 110],
            "BBB": [100, 101, 102, 103, 104, 105, 106, 107],
            "CCC": [100, 99, 101, 100, 102, 101, 103, 104],
        },
        index=pd.date_range("2024-01-05", periods=8, freq="W-FRI"),
    )
    strategy = build_strategy_spec(
        strategy_id="explicit_signal_execution_contexts",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            tickers=["AAA", "BBB", "CCC"],
            key="explicit_signal_execution_contexts_universe",
            label="Explicit signal execution contexts universe",
        ),
        selection=build_selection_spec("full_universe"),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="month_end",
            label="月次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
    )
    signal_execution_contexts = get_strategy_definition_signal_execution_contexts(strategy)

    with patch(
        "app.portfolio.get_strategy_signal_execution_contexts",
        side_effect=AssertionError("explicit contexts should avoid strategy re-extraction"),
    ):
        runs = compare_portfolio_runs(
            closes=closes,
            volumes=None,
            strategies=[strategy],
            initial_capital=10000,
            split_ratio=0.6,
            transaction_cost=0.001,
            bars_per_year=DEFAULT_WEEKLY_TIMEFRAME.bars_per_year,
            strategy_signal_execution_contexts_by_key={strategy.key: signal_execution_contexts},
        )

    assert len(runs) == 1
    assert runs[0]["key"] == strategy.key

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

    executable_strategy = build_executable_strategy_spec_from_definition(definition)
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
    executable_strategy = build_executable_strategy_spec_from_definition(definition)
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

    assert is_legacy_compatible_strategy_definition(definition) is False
    assert is_direct_execution_compatible_strategy_definition(definition) is True
    assert get_direct_execution_strategy_definition_compatibility_issues(definition) == []

    executable_strategy = build_executable_strategy_spec_from_definition(definition)
    assert executable_strategy.timeframe.key == "1w"
    assert executable_strategy.predictor_use is not None
    assert executable_strategy.predictor_use.predictor_key == "pred-fu-momo2-supplement-10bar-linear-momentum-weighted_blend-5050"
    assert executable_strategy.predictor_use.predictor_weight == 0.4
    assert executable_strategy.execution_mode == "direct_signal_timeframe"


def test_legacy_definition_compatibility_helper_reports_blockers() -> None:
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

    issues = get_legacy_strategy_definition_compatibility_issues(definition)
    assert is_legacy_compatible_strategy_definition(definition) is False
    assert "requires matching decision and rebalance schedules" in issues
    assert "requires matching selection data and signal timeframes" in issues

    try:
        build_strategy_spec_from_definition(definition)
    except ValueError as exc:
        assert "Legacy strategy adapter incompatibilities" in str(exc)
        assert "requires matching decision and rebalance schedules" in str(exc)
    else:
        raise AssertionError("Expected legacy adapter incompatibility error")



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



def test_filtered_candidates_are_derived_from_definitions() -> None:
    derived_strategies = build_strategy_specs_from_definitions(FILTERED_CANDIDATE_DEFINITIONS)
    assert len(FILTERED_CANDIDATE_DEFINITIONS) == len(derived_strategies)
    assert [
        strategy.strategy_id for strategy in derived_strategies
    ] == [
        definition.strategy_id for definition in FILTERED_CANDIDATE_DEFINITIONS
    ]

    rebuilt = build_strategy_spec_from_definition(FILTERED_CANDIDATE_DEFINITIONS[0])
    assert rebuilt == derived_strategies[0]


def test_universe_variant_candidates_are_derived_from_definitions() -> None:
    derived_strategies = build_strategy_specs_from_definitions(UNIVERSE_VARIANT_CANDIDATE_DEFINITIONS)
    assert len(UNIVERSE_VARIANT_CANDIDATE_DEFINITIONS) == len(derived_strategies)
    assert [
        strategy.strategy_id for strategy in derived_strategies
    ] == [
        definition.strategy_id for definition in UNIVERSE_VARIANT_CANDIDATE_DEFINITIONS
    ]

    rebuilt = build_strategy_spec_from_definition(UNIVERSE_VARIANT_CANDIDATE_DEFINITIONS[0])
    assert rebuilt == derived_strategies[0]


def test_predictor_candidates_are_derived_from_definitions() -> None:
    derived_strategies = build_strategy_specs_from_definitions(PREDICTOR_CANDIDATE_DEFINITIONS)
    assert len(PREDICTOR_CANDIDATE_DEFINITIONS) == len(derived_strategies)
    assert [
        strategy.strategy_id for strategy in derived_strategies
    ] == [
        definition.strategy_id for definition in PREDICTOR_CANDIDATE_DEFINITIONS
    ]

    predictor_candidate = PREDICTOR_CANDIDATE_DEFINITIONS[1]
    assert predictor_candidate.execution_plan.decision_schedule == "month_end"
    assert predictor_candidate.signals[0].weight == 0.6
    assert predictor_candidate.signals[1].source_kind == "predictor_overlay"
    assert predictor_candidate.signals[1].predictor_key is not None

    rebuilt = build_strategy_spec_from_definition(predictor_candidate)
    assert rebuilt == derived_strategies[1]


def test_baseline_definition_catalog_matches_strategy_catalog() -> None:
    derived_strategies = build_strategy_specs_from_definitions(BASELINE_CANDIDATE_DEFINITIONS)
    assert len(BASELINE_CANDIDATE_DEFINITIONS) == len(derived_strategies)
    assert [
        definition.strategy_id for definition in BASELINE_CANDIDATE_DEFINITIONS
    ] == [
        strategy.strategy_id for strategy in derived_strategies
    ]


def test_canonical_definition_catalog_matches_strategy_catalog() -> None:
    derived_strategies = build_strategy_specs_from_definitions(CANONICAL_CANDIDATE_DEFINITIONS)
    assert len(CANONICAL_CANDIDATE_DEFINITIONS) == len(derived_strategies)
    assert [
        definition.strategy_id for definition in CANONICAL_CANDIDATE_DEFINITIONS
    ] == [
        strategy.strategy_id for strategy in derived_strategies
    ]



def test_timeframe_variant_candidates_are_derived_from_definitions() -> None:
    derived_strategies = build_strategy_specs_from_definitions(TIMEFRAME_VARIANT_CANDIDATE_DEFINITIONS)
    assert len(TIMEFRAME_VARIANT_CANDIDATE_DEFINITIONS) == len(derived_strategies)
    assert [
        strategy.strategy_id for strategy in derived_strategies
    ] == [
        definition.strategy_id for definition in TIMEFRAME_VARIANT_CANDIDATE_DEFINITIONS
    ]

    monthly_candidate = TIMEFRAME_VARIANT_CANDIDATE_DEFINITIONS[2]
    assert monthly_candidate.execution_plan.decision_schedule == "month_end"
    assert monthly_candidate.execution_plan.rebalance_schedule == "month_end"
    assert monthly_candidate.signals[0].data_timeframe.key == "1d"

    rebuilt = build_strategy_spec_from_definition(monthly_candidate)
    assert rebuilt == derived_strategies[2]



def test_compare_portfolio_runs_uses_explicit_decision_schedule() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 110, 121, 133, 146, 146, 146, 146],
            "BBB": [100, 108, 116, 125, 135, 148, 163, 179],
            "CCC": [100, 107, 114, 122, 130, 139, 149, 160],
            "DDD": [100, 90, 81, 73, 95, 124, 161, 209],
        },
        index=pd.date_range("2025-01-01", periods=8, freq="D"),
    )
    volumes = pd.DataFrame(
        {ticker: [1_000_000 + idx * 10_000 for idx in range(len(closes))] for ticker in closes.columns},
        index=closes.index,
    )
    hold_strategy = build_strategy_spec(
        strategy_id="decision_schedule__hold",
        investment_universe=build_investment_universe_spec(
            tickers=list(closes.columns),
            key="decision_schedule_universe",
            label="Decision schedule universe",
        ),
        selection=build_selection_spec(
            "momentum_top3",
            score_parameters={"windowSpec": {"unit": "days", "value": 2}},
        ),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="every_bar",
            label="毎バー",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="every_bar",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        decision_schedule="hold",
    )
    adaptive_strategy = build_strategy_spec(
        strategy_id="decision_schedule__every_bar",
        investment_universe=hold_strategy.investment_universe,
        selection=hold_strategy.selection,
        portfolio_model=hold_strategy.portfolio_model,
        execution_policy=hold_strategy.execution_policy,
        risk_controls=hold_strategy.risk_controls,
        decision_schedule="every_bar",
    )

    runs = compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=[hold_strategy, adaptive_strategy],
        initial_capital=1000.0,
        split_ratio=0.5,
        transaction_cost=0.001,
        bars_per_year=252.0,
    )

    hold_run, adaptive_run = runs
    assert "DDD" not in hold_run["selectedAssets"]
    assert "DDD" in adaptive_run["selectedAssets"]
    assert adaptive_run["summary"]["turnoverPct"] > hold_run["summary"]["turnoverPct"]


def test_prepare_strategy_market_data_resamples_daily_source_to_weekly_signal_timeframe() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105, 106],
            "BBB": [100, 100, 101, 101, 102, 103, 103],
        },
        index=pd.date_range("2025-01-01", periods=7, freq="D"),
    )
    volumes = pd.DataFrame(
        {
            "AAA": [10, 11, 12, 13, 14, 15, 16],
            "BBB": [20, 21, 22, 23, 24, 25, 26],
        },
        index=closes.index,
    )
    strategy = build_strategy_spec(
        strategy_id="weekly_signal_from_daily_source",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            tickers=["AAA", "BBB"],
            key="weekly_signal_source_universe",
            label="Weekly signal source universe",
        ),
        selection=build_selection_spec("full_universe"),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="month_end",
            label="月次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        signal_execution_contexts=[
            {
                "selectionKey": "full_universe",
                "strategyType": "full_universe",
                "label": "full_universe",
                "description": "full_universe",
                "scoreParameters": {},
                "weight": 1.0,
                "dataTimeframe": "1d",
                "signalTimeframe": "1w",
                "alignmentPolicy": None,
            }
        ],
    )

    prepared_closes, prepared_volumes = prepare_strategy_market_data(
        closes=closes,
        volumes=volumes,
        strategy=strategy,
    )

    assert list(prepared_closes.index) == [pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-07")]
    assert prepared_closes.loc[pd.Timestamp("2025-01-03"), "AAA"] == 102
    assert prepared_closes.loc[pd.Timestamp("2025-01-07"), "AAA"] == 106
    assert prepared_volumes.loc[pd.Timestamp("2025-01-03"), "AAA"] == 33
    assert prepared_volumes.loc[pd.Timestamp("2025-01-07"), "AAA"] == 58


def test_compare_portfolio_runs_keeps_daily_performance_with_weekly_signal_timeframe() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100 + idx for idx in range(24)],
            "BBB": [100 + (idx // 2) for idx in range(24)],
            "CCC": [100 - 4 + idx for idx in range(24)],
        },
        index=pd.date_range("2025-01-01", periods=24, freq="D"),
    )
    volumes = pd.DataFrame(
        {ticker: [1_000_000 + idx * 10_000 for idx in range(len(closes))] for ticker in closes.columns},
        index=closes.index,
    )
    weekly_signal_strategy = build_strategy_spec(
        strategy_id="daily_performance_weekly_signal",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            tickers=list(closes.columns),
            key="daily_performance_weekly_signal_universe",
            label="Daily performance weekly signal universe",
        ),
        selection=build_selection_spec(
            "momentum_top3",
            score_parameters={"windowSpec": {"unit": "weeks", "value": 1}},
        ),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="month_end",
            label="月次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        signal_execution_contexts=[
            {
                "selectionKey": "momentum_top3",
                "strategyType": "momentum_top3",
                "label": "momentum_top3",
                "description": "momentum_top3",
                "scoreParameters": {"windowSpec": {"unit": "weeks", "value": 1}},
                "weight": 1.0,
                "dataTimeframe": "1d",
                "signalTimeframe": "1w",
                "alignmentPolicy": None,
            }
        ],
        decision_schedule="every_bar",
    )

    run = compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=[weekly_signal_strategy],
        initial_capital=1000.0,
        split_ratio=0.5,
        transaction_cost=0.001,
        bars_per_year=252.0,
    )[0]

    assert len(run["series"]) == len(closes.pct_change().dropna())
    assert run["splitAnalysis"]["test"]["barCount"] > 0
    assert run["strategy"]["components"]["core"]["dataResolution"]["key"] == "1w"


def test_compare_portfolio_runs_returns_strategy_combinations() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106, 107],
            "QQQ": [100, 103, 105, 107, 108, 110, 112],
            "IWM": [100, 99, 100, 101, 103, 102, 104],
            "EFA": [100, 101, 102, 103, 104, 105, 106],
            "EEM": [100, 98, 99, 100, 101, 102, 103],
            "TLT": [100, 100, 99, 100, 101, 102, 101],
            "IEF": [100, 100, 100, 100, 101, 101, 102],
            "LQD": [100, 101, 101, 102, 103, 103, 104],
            "HYG": [100, 101, 102, 103, 104, 104, 105],
            "GLD": [100, 101, 100, 102, 103, 104, 105],
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

    payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[
            make_strategy("full_universe", "equal_weight", max_investment_ratio=0.8),
            make_strategy("full_universe", "risk_budgeting", max_investment_ratio=0.8),
            make_strategy("full_universe", "minimum_variance", max_investment_ratio=0.8),
            make_strategy(
                "full_universe",
                "hierarchical_risk_parity",
                max_investment_ratio=0.8,
            ),
            make_strategy("momentum_top3", "equal_weight", max_investment_ratio=0.8),
            make_strategy("momentum_top3", "risk_budgeting", max_investment_ratio=0.8),
            make_strategy(
                "momentum_top3",
                "minimum_variance",
                max_investment_ratio=0.8,
            ),
            make_strategy(
                "momentum_top3",
                "hierarchical_risk_parity",
                max_investment_ratio=0.8,
            ),
        ],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(
            current_weights={
                "SPY": 0.1,
                "QQQ": 0.1,
                "IWM": 0.1,
                "EFA": 0.1,
                "EEM": 0.1,
                "TLT": 0.1,
                "IEF": 0.1,
                "LQD": 0.1,
            },
            cash_weight=0.2,
        ),
    )

    assert [row["key"] for row in payload] == [
        "full_universe__equal_weight",
        "full_universe__risk_budgeting",
        "full_universe__minimum_variance",
        "full_universe__hierarchical_risk_parity",
        "momentum_top3__equal_weight",
        "momentum_top3__risk_budgeting",
        "momentum_top3__minimum_variance",
        "momentum_top3__hierarchical_risk_parity",
    ]
    assert any(row["asset"] == "CASH" and row["weightPct"] == 20.0 for row in payload[0]["weights"])
    assert payload[0]["summary"]["turnoverPct"] >= 80.0
    assert len(payload[0]["series"]) == 6


def test_compute_trade_cost_increases_when_liquidity_is_lower() -> None:
    weight_delta = pd.Series([0.2, 0.2], index=["SPY", "BTC-USD"]).to_numpy(dtype="float64")
    linear_cost_rates = pd.Series([0.0003, 0.0025], index=["SPY", "BTC-USD"]).to_numpy(dtype="float64")
    impact_cost_rates = pd.Series([0.0002, 0.0025], index=["SPY", "BTC-USD"]).to_numpy(dtype="float64")
    price_snapshot = pd.Series({"SPY": 500.0, "BTC-USD": 60_000.0}, dtype="float64")

    high_liquidity_volumes = pd.DataFrame(
        {
            "SPY": [10_000_000.0, 11_000_000.0, 9_500_000.0],
            "BTC-USD": [3_000.0, 3_200.0, 3_100.0],
        }
    )
    low_liquidity_volumes = pd.DataFrame(
        {
            "SPY": [2_000_000.0, 2_100_000.0, 1_900_000.0],
            "BTC-USD": [300.0, 320.0, 310.0],
        }
    )

    high_liquidity_cost = compute_trade_cost(
        weight_delta=weight_delta,
        linear_cost_rates=linear_cost_rates,
        impact_cost_rates=impact_cost_rates,
        portfolio_equity=10_000.0,
        price_snapshot=price_snapshot,
        volume_history=high_liquidity_volumes,
        adv_window_bars=20,
        min_adv_notional=1_000_000.0,
    )
    low_liquidity_cost = compute_trade_cost(
        weight_delta=weight_delta,
        linear_cost_rates=linear_cost_rates,
        impact_cost_rates=impact_cost_rates,
        portfolio_equity=10_000.0,
        price_snapshot=price_snapshot,
        volume_history=low_liquidity_volumes,
        adv_window_bars=20,
        min_adv_notional=1_000_000.0,
    )

    assert low_liquidity_cost > high_liquidity_cost


def test_dual_momentum_can_fall_back_to_cash() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 99, 98, 97, 96, 95, 94],
            "QQQ": [100, 99, 98, 97, 96, 95, 94],
            "TLT": [100, 99, 98, 97, 96, 95, 94],
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

    payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[
            make_strategy(
                "dual_momentum_top3",
                "hierarchical_risk_parity",
                max_investment_ratio=0.85,
            )
        ],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )

    assert payload[0]["selectedAssets"] == []
    assert payload[0]["summary"]["totalReturnPct"] == 0.0
    assert payload[0]["weights"][0]["asset"] == "CASH"
    assert payload[0]["weights"][0]["weightPct"] == 100.0


def test_compare_portfolio_runs_respects_max_weight_cap() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 102, 103, 104, 105, 106],
            "QQQ": [100, 102, 103, 105, 106, 108, 109],
            "TLT": [100, 100, 101, 101, 102, 102, 103],
            "GLD": [100, 101, 101, 102, 103, 103, 104],
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

    payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[
            make_strategy(
                "full_universe",
                "equal_weight",
                max_weight=0.2,
            )
        ],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )

    asset_weights = [row["weightPct"] for row in payload[0]["weights"] if row["asset"] != "CASH"]
    cash_rows = [row for row in payload[0]["weights"] if row["asset"] == "CASH"]

    assert max(asset_weights) == 20.0
    assert cash_rows[0]["weightPct"] == 20.0


def test_trailing_momentum_low_vol_strategy_prefers_recent_winners() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 102, 104, 106, 108, 110, 112],
            "QQQ": [100, 101, 102, 103, 104, 105, 106],
            "TLT": [100, 99, 98, 97, 96, 95, 94],
            "GLD": [100, 100.5, 101, 101.5, 102, 102.5, 103],
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

    payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[make_strategy("trailing_momentum_low_vol_universe", "hierarchical_risk_parity")],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )

    assert sorted(payload[0]["selectedAssets"]) == ["GLD", "QQQ"]


def test_full_universe_momentum_tilt_overweights_stronger_assets() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 103, 106, 109, 112, 115, 118],
            "QQQ": [100, 102, 104, 106, 108, 110, 112],
            "TLT": [100, 99.5, 99, 98.5, 98, 97.5, 97],
            "GLD": [100, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2],
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

    baseline_payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[make_strategy("full_universe", "hierarchical_risk_parity")],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )
    tilted_payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[make_strategy("full_universe_momentum_tilt", "hierarchical_risk_parity")],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )

    baseline_weight_map = {
        row["asset"]: row["weightPct"] for row in baseline_payload[0]["weights"] if row["asset"] != "CASH"
    }
    tilted_weight_map = {
        row["asset"]: row["weightPct"] for row in tilted_payload[0]["weights"] if row["asset"] != "CASH"
    }
    assert tilted_weight_map["TLT"] < baseline_weight_map["TLT"]


def test_momentum_window_spec_changes_ranking_scores() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 104, 108, 112, 116, 120, 124],
            "QQQ": [100, 90, 85, 84, 95, 105, 115],
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
    returns = closes.pct_change().dropna()
    short_window_strategy = build_selection_spec(
        "full_universe_momentum_tilt",
        score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": {"unit": "bars", "value": 2}},
    )
    long_window_strategy = build_selection_spec(
        "full_universe_momentum_tilt",
        score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": {"unit": "bars", "value": 6}},
    )

    short_scores = compute_strategy_score_series(
        returns,
        None,
        short_window_strategy,
        bars_per_year=252,
    )
    long_scores = compute_strategy_score_series(
        returns,
        None,
        long_window_strategy,
        bars_per_year=252,
    )

    assert short_scores is not None
    assert long_scores is not None
    assert short_scores["QQQ"] > short_scores["SPY"]
    assert long_scores["SPY"] > long_scores["QQQ"]


def test_prediction_supplement_changes_momentum_scores() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106, 107, 108, 109],
            "QQQ": [100, 103, 105, 107, 108, 110, 112, 113, 115],
            "TLT": [100, 100, 99, 100, 101, 102, 101, 101, 102],
        },
        index=[
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
        ],
    )
    volumes = pd.DataFrame(
        {
            "SPY": [1_000_000, 1_020_000, 1_010_000, 1_030_000, 1_040_000, 1_050_000, 1_045_000, 1_060_000, 1_070_000],
            "QQQ": [900_000, 940_000, 960_000, 970_000, 990_000, 1_000_000, 1_010_000, 1_030_000, 1_050_000],
            "TLT": [800_000, 805_000, 810_000, 815_000, 820_000, 825_000, 830_000, 835_000, 840_000],
        },
        index=closes.index,
    )
    returns = closes.pct_change().dropna()
    volume_history = volumes.loc[returns.index]

    baseline_strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={"tilt_strength": 0.35, "tilt_shape": 1.0, "windowSpec": {"unit": "bars", "value": 3}},
    )
    supplemented_strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={
            "tilt_strength": 0.35,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 3},
        },
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-test-supplement-2bar",
            signal_weight=0.8,
            predictor_weight=0.2,
        ),
    )
    predictor_panel = pd.DataFrame(
        {
            "SPY": [0.8],
            "QQQ": [0.1],
            "TLT": [-0.7],
        },
        index=[returns.index[-1]],
    )

    baseline_scores = compute_strategy_score_series(
        returns,
        volume_history,
        baseline_strategy,
        bars_per_year=252,
    )
    supplemented_scores = compute_strategy_score_series(
        returns,
        volume_history,
        supplemented_strategy,
        bars_per_year=252,
        current_date=str(returns.index[-1]),
        predictor_panel=predictor_panel,
    )

    assert baseline_scores is not None
    assert supplemented_scores is not None
    assert not baseline_scores.equals(supplemented_scores)




def test_select_assets_uses_explicit_selection_contexts_for_momentum_top3() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105, 106, 107],
            "BBB": [100, 102, 103, 104, 105, 106, 107, 108],
            "CCC": [100, 99, 98, 100, 103, 107, 112, 118],
            "DDD": [100, 101, 101, 102, 102, 103, 103, 104],
        },
        index=pd.date_range("2025-01-01", periods=8, freq="D"),
    )
    returns = closes.pct_change().dropna()
    strategy = build_strategy_spec(
        strategy_id="explicit_selection_assets_top3",
        investment_universe=build_investment_universe_spec(
            tickers=list(closes.columns),
            key="explicit_selection_assets_top3_universe",
            label="Explicit selection assets top3 universe",
        ),
        selection=build_selection_spec(
            "momentum_top3",
            score_parameters={"windowSpec": {"unit": "bars", "value": 2}},
        ),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="every_bar",
            label="毎バー",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="every_bar",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        signal_execution_contexts=[
            {
                "selectionKey": "momentum_top3",
                "strategyType": "momentum_top3",
                "label": "momentum_top3",
                "description": "momentum_top3",
                "scoreParameters": {"windowSpec": {"unit": "bars", "value": 2}},
                "weight": 0.4,
                "dataTimeframe": "1d",
                "signalTimeframe": "1d",
                "alignmentPolicy": None,
            },
            {
                "selectionKey": "secondary_momo4",
                "strategyType": "momentum_top3",
                "label": "Secondary momentum",
                "description": "Secondary momentum",
                "scoreParameters": {"windowSpec": {"unit": "bars", "value": 4}},
                "weight": 0.6,
                "dataTimeframe": "1d",
                "signalTimeframe": "1d",
                "alignmentPolicy": None,
            },
        ],
    )
    selection_contexts, _ = get_strategy_signal_execution_contexts(strategy)

    with patch(
        "app.portfolio.get_strategy_signal_execution_contexts",
        side_effect=AssertionError("explicit selection contexts should avoid strategy re-extraction"),
    ):
        selected_assets = select_assets(
            returns,
            None,
            strategy,
            bars_per_year=252.0,
            selection_contexts=selection_contexts,
        )

    assert "CCC" in selected_assets
def test_compare_portfolio_runs_blends_additional_selection_signals_for_momentum_top3() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105, 106, 107],
            "BBB": [100, 102, 103, 104, 105, 106, 107, 108],
            "CCC": [100, 99, 98, 100, 103, 107, 112, 118],
            "DDD": [100, 101, 101, 102, 102, 103, 103, 104],
        },
        index=pd.date_range("2025-01-01", periods=8, freq="D"),
    )
    strategy = build_strategy_spec(
        strategy_id="multi_selection_top3",
        investment_universe=build_investment_universe_spec(
            tickers=list(closes.columns),
            key="multi_selection_top3_universe",
            label="Multi selection top3 universe",
        ),
        selection=build_selection_spec(
            "momentum_top3",
            score_parameters={"windowSpec": {"unit": "bars", "value": 2}},
        ),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="every_bar",
            label="毎バー",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="every_bar",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        signal_execution_contexts=[
            {
                "selectionKey": "momentum_top3",
                "strategyType": "momentum_top3",
                "label": "momentum_top3",
                "description": "momentum_top3",
                "scoreParameters": {"windowSpec": {"unit": "bars", "value": 2}},
                "weight": 0.4,
                "dataTimeframe": "1d",
                "signalTimeframe": "1d",
                "alignmentPolicy": None,
            },
            {
                "selectionKey": "secondary_momo4",
                "strategyType": "momentum_top3",
                "label": "Secondary momentum",
                "description": "Secondary momentum",
                "scoreParameters": {"windowSpec": {"unit": "bars", "value": 4}},
                "weight": 0.6,
                "dataTimeframe": "1d",
                "signalTimeframe": "1d",
                "alignmentPolicy": None,
            },
        ],
    )

    run = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[strategy],
        initial_capital=1000.0,
        split_ratio=0.5,
        transaction_cost=0.001,
        bars_per_year=252.0,
    )[0]

    assert "CCC" in run["selectedAssets"]


def test_compute_strategy_score_series_blends_explicit_additional_selection_signals() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106, 108, 109],
            "QQQ": [100, 104, 105, 107, 109, 112, 114, 116],
            "TLT": [100, 99, 98, 99, 100, 101, 102, 103],
        },
        index=pd.date_range("2025-01-01", periods=8, freq="D"),
    )
    returns = closes.pct_change().dropna()
    primary_strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={
            "tilt_strength": 0.35,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 3},
        },
    )
    blended_strategy = build_strategy_spec(
        strategy_id="multi_selection_blend",
        timeframe=primary_strategy.timeframe,
        investment_universe=primary_strategy.investment_universe,
        selection=primary_strategy.selection,
        portfolio_model=primary_strategy.portfolio_model,
        execution_policy=primary_strategy.execution_policy,
        risk_controls=primary_strategy.risk_controls,
        signal_execution_contexts=[
            {
                "selectionKey": primary_strategy.selection.key,
                "strategyType": primary_strategy.selection.strategy_type,
                "label": primary_strategy.selection.label,
                "description": primary_strategy.selection.description,
                "scoreParameters": {
                    "tilt_strength": 0.35,
                    "tilt_shape": 1.0,
                    "windowSpec": {"unit": "bars", "value": 3},
                },
                "weight": 0.6,
                "dataTimeframe": primary_strategy.timeframe.key,
                "signalTimeframe": primary_strategy.timeframe.key,
                "alignmentPolicy": None,
            },
            {
                "selectionKey": "secondary_momo2",
                "strategyType": "full_universe_momentum_tilt",
                "label": "Secondary momentum",
                "description": "Secondary momentum",
                "scoreParameters": {
                    "tilt_strength": 0.35,
                    "tilt_shape": 1.0,
                    "windowSpec": {"unit": "bars", "value": 2},
                },
                "weight": 0.4,
                "dataTimeframe": primary_strategy.timeframe.key,
                "signalTimeframe": primary_strategy.timeframe.key,
                "alignmentPolicy": None,
            },
        ],
    )

    primary_scores = compute_strategy_score_series(
        returns,
        None,
        primary_strategy,
        bars_per_year=252,
    )
    blended_scores = compute_strategy_score_series(
        returns,
        None,
        blended_strategy,
        bars_per_year=252,
    )

    assert primary_scores is not None
    assert blended_scores is not None
    assert not primary_scores.equals(blended_scores)




def test_compute_strategy_score_series_uses_explicit_predictor_context() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106],
            "QQQ": [100, 103, 105, 107, 108, 110],
            "TLT": [100, 100, 99, 100, 101, 102],
        },
        index=pd.date_range("2025-01-01", periods=6, freq="D"),
    )
    returns = closes.pct_change().dropna()
    strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={
            "tilt_strength": 0.35,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 3},
        },
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-test-explicit-context-2bar",
            signal_weight=0.8,
            predictor_weight=0.2,
        ),
    )
    selection_contexts, predictor_context = get_strategy_signal_execution_contexts(strategy)
    predictor_panel = pd.DataFrame(
        {
            "SPY": [0.8],
            "QQQ": [0.1],
            "TLT": [-0.7],
        },
        index=[returns.index[-1]],
    )
    assert predictor_context is not None

    with patch(
        "app.portfolio.get_strategy_signal_execution_contexts",
        side_effect=AssertionError("explicit score contexts should avoid strategy re-extraction"),
    ):
        scores = compute_strategy_score_series(
            returns,
            None,
            strategy,
            bars_per_year=252,
            current_date=str(returns.index[-1]),
            predictor_panel=predictor_panel,
            selection_contexts=selection_contexts,
            predictor_context=predictor_context,
        )

    assert scores is not None
    assert scores["SPY"] > scores["TLT"]
def test_compute_strategy_score_series_uses_latest_predictor_snapshot_before_current_date() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106],
            "QQQ": [100, 103, 105, 107, 108, 110],
            "TLT": [100, 100, 99, 100, 101, 102],
        },
        index=pd.date_range("2025-01-01", periods=6, freq="D"),
    )
    returns = closes.pct_change().dropna()
    strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={
            "tilt_strength": 0.35,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 3},
        },
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-test-asof-2bar",
            signal_weight=0.8,
            predictor_weight=0.2,
        ),
    )
    predictor_panel = pd.DataFrame(
        {
            "SPY": [0.8],
            "QQQ": [0.1],
            "TLT": [-0.7],
        },
        index=[returns.index[-2]],
    )

    exact_scores = compute_strategy_score_series(
        returns,
        None,
        strategy,
        bars_per_year=252,
        current_date=str(returns.index[-2]),
        predictor_panel=predictor_panel,
    )
    asof_scores = compute_strategy_score_series(
        returns,
        None,
        strategy,
        bars_per_year=252,
        current_date=str(returns.index[-1]),
        predictor_panel=predictor_panel,
    )

    assert exact_scores is not None
    assert asof_scores is not None
    pd.testing.assert_series_equal(asof_scores.sort_index(), exact_scores.sort_index())




def test_prepare_strategy_signal_data_uses_explicit_selection_contexts() -> None:
    closes = pd.DataFrame(
        {
            "AAA": [100, 102, 104, 103, 105, 107, 108],
            "BBB": [100, 101, 102, 103, 104, 105, 106],
        },
        index=pd.date_range("2025-01-01", periods=7, freq="D"),
    )
    returns = closes.pct_change().dropna()
    strategy = build_strategy_spec(
        strategy_id="explicit_selection_signal_data",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            tickers=["AAA", "BBB"],
            key="explicit_selection_signal_data_universe",
            label="Explicit selection signal data universe",
        ),
        selection=build_selection_spec("full_universe"),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="month_end",
            label="月次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        extensions={"signal_data_timeframe": "1d", "signal_timeframe": "1w"},
    )
    selection_contexts, _ = get_strategy_signal_execution_contexts(strategy)

    with patch(
        "app.portfolio.get_strategy_signal_execution_contexts",
        side_effect=AssertionError("explicit selection contexts should avoid strategy re-extraction"),
    ):
        signal_returns, signal_volumes, signal_bars_per_year = prepare_strategy_signal_data(
            history_returns=returns,
            volume_history=None,
            selection_contexts=selection_contexts,
        )

    assert signal_volumes is None
    assert signal_bars_per_year == DEFAULT_WEEKLY_TIMEFRAME.bars_per_year
    assert list(signal_returns.index) == [pd.Timestamp("2025-01-07")]


def test_prepare_strategy_predictor_panel_uses_explicit_predictor_context() -> None:
    predictor_panel = pd.DataFrame(
        {
            "AAA": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "BBB": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        },
        index=pd.date_range("2025-01-01", periods=7, freq="D"),
    )
    strategy = build_strategy_spec(
        strategy_id="explicit_predictor_signal_data",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            tickers=["AAA", "BBB"],
            key="explicit_predictor_signal_data_universe",
            label="Explicit predictor signal data universe",
        ),
        selection=build_selection_spec("full_universe"),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="month_end",
            label="月次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-explicit-predictor-context",
            signal_weight=0.6,
            predictor_weight=0.4,
        ),
        predictor_signal_execution_context={
            "predictorKey": "pred-explicit-predictor-context",
            "signalWeight": 0.6,
            "predictorWeight": 0.4,
            "dataTimeframe": "1d",
            "signalTimeframe": "1w",
            "alignmentPolicy": None,
        },
    )
    _, predictor_context = get_strategy_signal_execution_contexts(strategy)
    assert predictor_context is not None

    with patch(
        "app.portfolio.get_strategy_signal_execution_contexts",
        side_effect=AssertionError("explicit predictor context should avoid strategy re-extraction"),
    ):
        prepared_panel = prepare_strategy_predictor_panel(
            predictor_panel=predictor_panel,
            predictor_context=predictor_context,
        )

    assert prepared_panel is not None
    assert list(prepared_panel.index) == [pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-07")]
    assert prepared_panel.loc[pd.Timestamp("2025-01-07"), "AAA"] == 0.7
def test_prepare_strategy_predictor_panel_resamples_daily_source_to_weekly_signal_timeframe() -> None:
    predictor_panel = pd.DataFrame(
        {
            "AAA": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "BBB": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        },
        index=pd.date_range("2025-01-01", periods=7, freq="D"),
    )
    strategy = build_strategy_spec(
        strategy_id="weekly_predictor_from_daily_source",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            tickers=["AAA", "BBB"],
            key="weekly_predictor_source_universe",
            label="Weekly predictor source universe",
        ),
        selection=build_selection_spec("full_universe"),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="month_end",
            label="月次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-weekly-predictor-source",
            signal_weight=0.6,
            predictor_weight=0.4,
        ),
        signal_execution_contexts=[
            {
                "selectionKey": "full_universe",
                "strategyType": "full_universe",
                "label": "full_universe",
                "description": "full_universe",
                "scoreParameters": {},
                "weight": 1.0,
                "dataTimeframe": "1d",
                "signalTimeframe": "1w",
                "alignmentPolicy": None,
            }
        ],
    )

    prepared_panel = prepare_strategy_predictor_panel(
        predictor_panel=predictor_panel,
        strategy=strategy,
    )

    calendar_strategy = build_strategy_spec(
        strategy_id="weekly_predictor_from_daily_source_calendar",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=strategy.investment_universe,
        selection=strategy.selection,
        portfolio_model=strategy.portfolio_model,
        execution_policy=strategy.execution_policy,
        risk_controls=strategy.risk_controls,
        predictor_use=strategy.predictor_use,
        signal_execution_contexts=[
            {
                "selectionKey": "full_universe",
                "strategyType": "full_universe",
                "label": "full_universe",
                "description": "full_universe",
                "scoreParameters": {},
                "weight": 1.0,
                "dataTimeframe": "1d",
                "signalTimeframe": "1w",
                "alignmentPolicy": None,
            }
        ],
        predictor_signal_execution_context={
            "predictorKey": "pred-weekly-predictor-source",
            "signalWeight": 0.6,
            "predictorWeight": 0.4,
            "dataTimeframe": "1d",
            "signalTimeframe": "1w",
            "alignmentPolicy": {"method": "calendar_resample"},
        },
    )
    calendar_panel = prepare_strategy_predictor_panel(
        predictor_panel=predictor_panel,
        strategy=calendar_strategy,
    )

    assert prepared_panel is not None
    assert list(prepared_panel.index) == [pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-07")]
    assert prepared_panel.loc[pd.Timestamp("2025-01-03"), "AAA"] == 0.3
    assert prepared_panel.loc[pd.Timestamp("2025-01-07"), "AAA"] == 0.7
    assert calendar_panel is not None
    assert list(calendar_panel.index) == [pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-10")]
    assert calendar_panel.loc[pd.Timestamp("2025-01-03"), "AAA"] == 0.3
    assert calendar_panel.loc[pd.Timestamp("2025-01-10"), "AAA"] == 0.7


def test_predictor_evaluation_includes_predictor_series() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106, 107, 108, 109],
            "QQQ": [100, 103, 105, 107, 108, 110, 112, 113, 115],
            "TLT": [100, 100, 99, 100, 101, 102, 101, 101, 102],
        },
        index=[
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
        ],
    )
    returns = closes.pct_change().dropna()
    strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={
            "tilt_strength": 0.35,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 3},
        },
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-test-eval-2bar",
            signal_weight=0.8,
            predictor_weight=0.2,
        ),
    )
    predictor_spec = make_predictor_spec(
        strategy,
        predictor_key="pred-test-eval-2bar",
        label="2bar補助予測",
        horizon_bars=2,
    )

    result = evaluate_predictor_spec(
        returns=returns,
        volumes=None,
        split_ratio=0.6,
        predictor_spec=predictor_spec,
        bars_per_year=252,
    )

    assert result["predictorSpec"]["kind"] == "predictor_spec"
    assert "predictorSeries" in result
    assert result["predictorSeries"]["assets"]


def test_strategy_run_can_reuse_predictor_panel() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106, 107, 108, 109],
            "QQQ": [100, 103, 105, 107, 108, 110, 112, 113, 115],
            "TLT": [100, 100, 99, 100, 101, 102, 101, 101, 102],
        },
        index=[
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
        ],
    )
    returns = closes.pct_change().dropna()
    volumes = pd.DataFrame(
        {
            "SPY": [1_000_000, 1_020_000, 1_010_000, 1_030_000, 1_040_000, 1_050_000, 1_045_000, 1_060_000, 1_070_000],
            "QQQ": [900_000, 940_000, 960_000, 970_000, 990_000, 1_000_000, 1_010_000, 1_030_000, 1_050_000],
            "TLT": [800_000, 805_000, 810_000, 815_000, 820_000, 825_000, 830_000, 835_000, 840_000],
        },
        index=closes.index,
    )
    strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={
            "tilt_strength": 0.35,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 3},
        },
        predictor_use=build_predictor_use_spec(
            predictor_key="pred-test-reuse-2bar",
            signal_weight=0.8,
            predictor_weight=0.2,
        ),
    )
    predictor_spec = make_predictor_spec(
        strategy,
        predictor_key="pred-test-reuse-2bar",
        label="2bar補助予測",
        horizon_bars=2,
    )

    predictor_panel = compute_predictor_panel(
        returns=returns,
        volumes=volumes.loc[returns.index],
        predictor_spec=predictor_spec,
        bars_per_year=252,
    )

    inline_payload = compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=[strategy],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )
    reused_payload = compare_portfolio_runs(
        closes=closes,
        volumes=volumes,
        strategies=[strategy],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions=make_execution_assumptions(),
        transaction_cost=0.001,
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
        predictor_panels_by_strategy={strategy.key: predictor_panel},
    )

    assert reused_payload[0]["summary"] == inline_payload[0]["summary"]


def test_predictor_panel_supports_momentum_signal_source_blend() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 103, 102, 104, 106, 107, 108, 109],
            "QQQ": [100, 103, 105, 107, 108, 110, 112, 113, 115],
            "TLT": [100, 100, 99, 100, 101, 102, 101, 101, 102],
        },
        index=[
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
        ],
    )
    volumes = pd.DataFrame(
        {
            "SPY": [1_000_000, 1_020_000, 1_010_000, 1_030_000, 1_040_000, 1_050_000, 1_045_000, 1_060_000, 1_070_000],
            "QQQ": [900_000, 940_000, 960_000, 970_000, 990_000, 1_000_000, 1_010_000, 1_030_000, 1_050_000],
            "TLT": [800_000, 805_000, 810_000, 815_000, 820_000, 825_000, 830_000, 835_000, 840_000],
        },
        index=closes.index,
    )
    returns = closes.pct_change().dropna()
    strategy = make_strategy(
        "full_universe_momentum_tilt",
        "hierarchical_risk_parity",
        score_parameters={
            "tilt_strength": 0.35,
            "tilt_shape": 1.0,
            "windowSpec": {"unit": "bars", "value": 3},
        },
    )
    predictor_spec = make_predictor_spec(
        strategy,
        predictor_key="pred-test-momentum-blend-2bar",
        label="2barモメンタム補助予測",
        horizon_bars=2,
        signal_source_feature_key="momentum",
        combiner_kind="weighted_blend",
    )

    predictor_panel = compute_predictor_panel(
        returns=returns,
        volumes=volumes.loc[returns.index],
        predictor_spec=predictor_spec,
        bars_per_year=252,
    )

    assert predictor_panel.notna().any().any()


def test_convert_window_spec_to_bars_supports_duration_units() -> None:
    assert convert_window_spec_to_bars({"unit": "bars", "value": 8}, bars_per_year=252) == 8
    assert convert_window_spec_to_bars({"unit": "days", "value": 5}, bars_per_year=252) == 5
    assert convert_window_spec_to_bars({"unit": "weeks", "value": 2}, bars_per_year=252) == 10
    assert convert_window_spec_to_bars({"unit": "months", "value": 3}, bars_per_year=252) == 63
    assert convert_window_spec_to_bars({"unit": "years", "value": 1}, bars_per_year=252) == 252


def test_should_rebalance_supports_every_bar_schedule() -> None:
    assert should_rebalance("2025-01-01", "2025-01-02", "every_bar") is True
    assert should_rebalance("2025-01-01", "2025-01-01", "every_bar") is False


def test_compare_portfolio_runs_supports_asset_specific_linear_cost() -> None:
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 102, 103, 104, 105, 106],
            "QQQ": [100, 101, 102, 103, 104, 105, 106],
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

    strategy = build_strategy_spec(
        investment_universe=build_investment_universe_spec(
            tickers=["SPY", "QQQ"],
            key="test_universe_small",
            label="Test universe small",
        ),
        selection=build_selection_spec("full_universe"),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
    )

    flat_payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[strategy],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions={
            "kind": "close_execution_assumptions",
            "label": "終値約定",
            "parameters": {"entry": "hold", "rebalanceSchedule": "hold"},
            "costModel": {
                "kind": "flat_cost",
                "parameters": {"commissionPct": 0.1, "slippagePct": 0.0},
                "perAssetOverrides": {},
            },
        },
        cost_model={
            "kind": "flat_cost",
            "parameters": {"commissionPct": 0.1, "slippagePct": 0.0},
            "perAssetOverrides": {},
        },
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )
    asset_specific_payload = compare_portfolio_runs(
        closes=closes,
        volumes=None,
        strategies=[strategy],
        initial_capital=10_000,
        split_ratio=0.6,
        execution_assumptions={
            "kind": "close_execution_assumptions",
            "label": "終値約定",
            "parameters": {"entry": "hold", "rebalanceSchedule": "hold"},
            "costModel": {
                "kind": "asset_specific_linear_cost",
                "parameters": {"commissionPct": 0.1, "slippagePct": 0.0},
                "perAssetOverrides": {
                    "SPY": {"commissionPct": 0.5, "slippagePct": 0.0},
                },
            },
        },
        cost_model={
            "kind": "asset_specific_linear_cost",
            "parameters": {"commissionPct": 0.1, "slippagePct": 0.0},
            "perAssetOverrides": {
                "SPY": {"commissionPct": 0.5, "slippagePct": 0.0},
            },
        },
        portfolio_state=build_portfolio_state(current_weights={}, cash_weight=1.0),
    )

    assert (
        asset_specific_payload[0]["summary"]["totalReturnPct"]
        < flat_payload[0]["summary"]["totalReturnPct"]
    )
