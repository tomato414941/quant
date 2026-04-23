import pandas as pd

from app.domain import BacktestArtifact, RunContext
from app.engine import run_strategy_backtest
from app.portfolio import (
    build_evaluator_strategy_spec,
    build_execution_policy_spec,
    build_investment_universe_spec,
    build_portfolio_model_spec,
    build_predictor_use_spec,
    build_risk_controls_spec,
    build_selection_spec,
    build_strategy_definition_from_evaluator_strategy_spec,
    evaluate_strategy_definition_run,
)
from app.spec import build_strategy_blueprint_from_definition, build_strategy_definition_from_blueprint
from app.timeframe_models import DEFAULT_WEEKLY_TIMEFRAME


def make_execution_assumptions() -> dict:
    return {
        "kind": "close_execution_assumptions",
        "label": "終値約定",
        "parameters": {"fillPrice": "close"},
        "costModel": {
            "kind": "flat_cost",
            "parameters": {"commissionPct": 0.1, "slippagePct": 0.0},
            "perAssetOverrides": {},
        },
    }


def build_roundtrip_strategy_definition():
    predictor_use = build_predictor_use_spec(
        predictor_key="pred__overlay",
        signal_weight=0.6,
        predictor_weight=0.4,
    )
    strategy = build_evaluator_strategy_spec(
        strategy_id="strategy__blueprint_roundtrip",
        label="Blueprint roundtrip strategy",
        description="Roundtrip coverage for blueprint adapter.",
        timeframe=DEFAULT_WEEKLY_TIMEFRAME,
        investment_universe=build_investment_universe_spec(
            tickers=["SPY", "QQQ", "TLT"],
            key="blueprint_universe",
            label="Blueprint universe",
        ),
        selection=build_selection_spec(
            "full_universe_momentum_tilt",
            key="blueprint_selection",
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
                "selectionKey": "blueprint_selection",
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
        extensions={"decision_policy": "direct_score_to_weight", "execution_mode": "direct_signal_timeframe"},
    )
    return build_strategy_definition_from_evaluator_strategy_spec(strategy)


def build_runtime_strategy_definition():
    strategy = build_evaluator_strategy_spec(
        strategy_id="strategy__blueprint_runtime",
        label="Blueprint runtime strategy",
        description="Runtime coverage for blueprint engine.",
        investment_universe=build_investment_universe_spec(
            tickers=["SPY", "QQQ", "TLT"],
            key="runtime_universe",
            label="Runtime universe",
        ),
        selection=build_selection_spec(
            "momentum_top3",
            score_parameters={"windowSpec": {"unit": "bars", "value": 3}},
        ),
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_policy=build_execution_policy_spec(
            key="every_bar",
            label="日次",
            entry="train_once_then_periodic_rebalance",
            rebalance_schedule="every_bar",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=1.0),
        extensions={"decision_policy": "direct_score_to_weight"},
    )
    return build_strategy_definition_from_evaluator_strategy_spec(strategy)


def build_market_bundle() -> dict:
    index = pd.date_range("2025-01-01", periods=10, freq="D")
    closes = pd.DataFrame(
        {
            "SPY": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "QQQ": [100, 102, 104, 105, 107, 109, 110, 112, 113, 115],
            "TLT": [100, 100, 99, 99, 98, 98, 97, 97, 96, 96],
        },
        index=index,
    )
    volumes = pd.DataFrame(
        {
            "SPY": [1_000_000 + row * 1_000 for row in range(len(index))],
            "QQQ": [1_100_000 + row * 1_000 for row in range(len(index))],
            "TLT": [900_000 + row * 1_000 for row in range(len(index))],
        },
        index=index,
    )
    return {"closes": closes, "volumes": volumes}


def test_strategy_blueprint_round_trips_strategy_definition() -> None:
    original = build_roundtrip_strategy_definition()

    blueprint = build_strategy_blueprint_from_definition(original)
    restored = build_strategy_definition_from_blueprint(blueprint)

    assert restored.strategy_id == original.strategy_id
    assert restored.version == original.version
    assert restored.label == original.label
    assert restored.description == original.description
    assert restored.investment_universe == original.investment_universe
    assert restored.execution_plan == original.execution_plan
    assert restored.risk_controls == original.risk_controls
    assert restored.extensions == original.extensions
    assert restored.portfolio_model == original.portfolio_model
    assert len(restored.signals) == len(original.signals)

    for restored_signal, original_signal in zip(restored.signals, original.signals):
        assert restored_signal.key == original_signal.key
        assert restored_signal.label == original_signal.label
        assert restored_signal.description == original_signal.description
        assert restored_signal.source_kind == original_signal.source_kind
        assert restored_signal.weight == original_signal.weight
        assert restored_signal.predictor_key == original_signal.predictor_key
        assert restored_signal.observation_spec.tickers == original_signal.observation_spec.tickers
        assert restored_signal.observation_spec.fields == original_signal.observation_spec.fields
        assert restored_signal.data_timeframe == original_signal.data_timeframe
        assert restored_signal.signal_timeframe == original_signal.signal_timeframe
        assert restored_signal.signal_parameters == original_signal.signal_parameters
        assert restored_signal.data_source_spec is not None
        assert original_signal.data_source_spec is not None
        assert restored_signal.data_source_spec.key == original_signal.data_source_spec.key
        assert restored_signal.data_source_spec.label == original_signal.data_source_spec.label
        assert restored_signal.data_source_spec.kind == original_signal.data_source_spec.kind
        assert restored_signal.data_source_spec.observation_spec.tickers == original_signal.data_source_spec.observation_spec.tickers
        assert restored_signal.data_source_spec.observation_spec.fields == original_signal.data_source_spec.observation_spec.fields
        assert restored_signal.feature_definition_spec == original_signal.feature_definition_spec
        assert restored_signal.alignment_policy == original_signal.alignment_policy


def test_run_strategy_backtest_matches_legacy_strategy_definition_run() -> None:
    strategy_definition = build_runtime_strategy_definition()
    market_bundle = build_market_bundle()
    execution_assumptions = make_execution_assumptions()

    legacy_run = evaluate_strategy_definition_run(
        closes=market_bundle["closes"],
        volumes=market_bundle["volumes"],
        strategy_definition=strategy_definition,
        initial_capital=100_000,
        split_ratio=0.6,
        bars_per_year=252.0,
        execution_assumptions=execution_assumptions,
    )
    artifact = run_strategy_backtest(
        build_strategy_blueprint_from_definition(strategy_definition),
        market_bundle,
        RunContext(
            initial_capital=100_000,
            split_ratio=0.6,
            bars_per_year=252.0,
            execution_assumptions=execution_assumptions,
            availability_policy={"minHistoryBars": 0, "allowPartialUniverse": True, "maxStaleBars": 5},
        ),
    )

    assert isinstance(artifact, BacktestArtifact)
    assert artifact.run_result["summary"] == legacy_run["summary"]
    assert artifact.run_result["selectedAssets"] == legacy_run["selectedAssets"]
    assert artifact.run_result["weights"] == legacy_run["weights"]
    assert artifact.run_result["decisionSummary"] == legacy_run["decisionSummary"]
    assert len(artifact.decision_events) == len(legacy_run["decisionEvents"])
    assert len(artifact.execution_events) == len(legacy_run["executionTrace"])
    assert artifact.performance_summary is not None
    assert artifact.performance_summary.total_return_pct == legacy_run["summary"]["totalReturnPct"]
