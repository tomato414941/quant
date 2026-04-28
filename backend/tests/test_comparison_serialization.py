from __future__ import annotations

import pytest

from app.comparison_models import (
    ComparisonSpec,
    ConditionVariant,
    EvaluationSettings,
    EvaluationSpec,
    MarketSliceSpec,
    RunSpec,
    SelectionPolicy,
    build_cost_model_spec,
    build_execution_assumptions_spec,
)
from app.comparison_serialization import (
    deserialize_comparison_run_spec_payload,
    serialize_comparison,
    serialize_condition_variant,
)
from app.portfolio import (
    build_investment_universe_spec,
    build_observation_spec,
    build_portfolio_model_spec,
    build_portfolio_state,
    build_risk_controls_spec,
    build_strategy_definition,
    build_strategy_execution_plan_spec,
    build_strategy_signal_spec,
)
from app.run_store import build_run_fingerprint
from app.timeframe_models import DEFAULT_DAILY_TIMEFRAME


def _strategy_definition():
    universe = build_investment_universe_spec(
        key="test_universe",
        label="Test universe",
        tickers=["SPY", "AGG"],
    )
    observation = build_observation_spec(
        key="test_observation",
        label="Test observation",
        tickers=["SPY", "AGG"],
        fields=["close", "volume"],
    )
    signal = build_strategy_signal_spec(
        key="test_selection_signal",
        label="Test selection",
        description="Test selection signal.",
        observation_spec=observation,
        data_timeframe=DEFAULT_DAILY_TIMEFRAME,
        source_kind="selection_signal",
        signal_parameters={
            "selectionKey": "full_universe",
            "strategyType": "full_universe",
            "scoreParameters": {"windowSpec": {"lookbackBars": 20}},
        },
    )
    return build_strategy_definition(
        strategy_id="test_strategy",
        version="v1",
        label="Test strategy",
        hypothesis="Serialization contract test.",
        description="A minimal strategy definition for serialization tests.",
        investment_universe=universe,
        signals=[signal],
        portfolio_model=build_portfolio_model_spec("equal_weight"),
        execution_plan=build_strategy_execution_plan_spec(
            key="monthly_plan",
            label="Monthly plan",
            decision_schedule="month_end",
            rebalance_schedule="month_end",
        ),
        risk_controls=build_risk_controls_spec(max_investment_ratio=0.95, max_weight=0.6),
        extensions={"decision_policy": "cost_aware_no_trade"},
    )


def _comparison_spec() -> ComparisonSpec:
    return ComparisonSpec(
        comparison_id="serializer_contract",
        title="Serializer contract",
        question="Does serializer output remain stable after separation?",
        run_spec=RunSpec(
            market_slice=MarketSliceSpec(
                period="1y",
                sanity_periods=["6mo"],
                start_date="2025-01-01",
                end_date="2025-12-31",
            ),
            portfolio_state=build_portfolio_state(
                current_weights={"SPY": 0.4, "AGG": 0.3},
                cash_weight=0.1,
            ),
            capital_base=100000.0,
            execution_assumptions=build_execution_assumptions_spec(
                label="Close execution",
                parameters={"fillPrice": "close", "costProfileKey": "test_profile"},
                cost_model=build_cost_model_spec(
                    kind="flat_cost",
                    commission_pct=0.12345,
                    slippage_pct=0.01234,
                    per_asset_overrides={"SPY": {"commissionPct": 0.2}},
                ),
            ),
            evaluation=EvaluationSpec(evaluation_settings=EvaluationSettings(split_ratio=0.7)),
        ),
        selection_policy=SelectionPolicy(
            primary_metric="sharpe_ratio",
            secondary_metric="max_drawdown_pct",
            tertiary_metric="cagr_pct",
        ),
        candidate_strategies=[_strategy_definition()],
        condition_variants=[
            ConditionVariant(
                key="high_cost",
                label="High cost",
                cost_multiplier=1.2345,
                max_investment_ratio=0.875,
                max_weight=0.3333,
            )
        ],
        result_store_dir="backend/data/run_results",
    )


def _metadata_by_timeframe() -> dict[str, dict[str, object]]:
    return {
        DEFAULT_DAILY_TIMEFRAME.key: {
            "source": "test",
            "aligned_start_date": "2025-01-02",
            "aligned_end_date": "2025-12-30",
            "row_count": 250,
            "tickers": ["SPY", "AGG"],
            "requested_tickers": ["SPY", "AGG"],
            "assetAvailability": {
                "SPY": {
                    "available": True,
                    "firstValidDate": "2025-01-02",
                    "lastValidDate": "2025-12-30",
                },
                "AGG": {
                    "available": True,
                    "firstValidDate": "2025-01-02",
                    "lastValidDate": "2025-12-30",
                },
            },
            "datasetSnapshot": {
                "source": "test",
                "timeframe": DEFAULT_DAILY_TIMEFRAME.key,
                "rowCount": 250,
            },
        }
    }


def test_serialize_comparison_keeps_run_spec_and_strategy_payload_contract() -> None:
    comparison = _comparison_spec()

    payload = serialize_comparison(
        comparison,
        _metadata_by_timeframe(),
        [DEFAULT_DAILY_TIMEFRAME],
    )

    assert payload["kind"] == "strategy_comparison"
    assert payload["schemaVersion"] == "v1"
    assert payload["runSpec"]["kind"] == "comparison_run_spec"
    assert payload["runSpec"]["marketSlice"]["startDate"] == "2025-01-01"
    assert payload["runSpec"]["marketSlice"]["endDate"] == "2025-12-31"
    assert payload["runSpec"]["portfolioState"]["weights"] == [
        {"asset": "SPY", "weightPct": 40.0},
        {"asset": "AGG", "weightPct": 30.0},
        {"asset": "CASH", "weightPct": 10.0},
    ]
    assert payload["runSpec"]["executionAssumptions"]["costModel"] == {
        "kind": "flat_cost",
        "parameters": {"commissionPct": 0.123, "slippagePct": 0.012},
        "perAssetOverrides": {"SPY": {"commissionPct": 0.2}},
    }
    assert payload["runSpec"]["evaluation"]["marketDataContexts"][0]["datasetSnapshot"] == {
        "source": "test",
        "timeframe": DEFAULT_DAILY_TIMEFRAME.key,
        "rowCount": 250,
    }
    assert payload["candidateStrategies"][0]["kind"] == "strategy_definition"
    assert payload["candidateStrategies"][0]["components"]["optional"]["signals"][0]["dataTimeframe"][
        "key"
    ] == DEFAULT_DAILY_TIMEFRAME.key
    assert payload["conditionVariants"] == [
        {
            "key": "high_cost",
            "label": "High cost",
            "costMultiplier": 1.234,
            "maxInvestmentPct": 87.5,
            "maxWeightPct": 33.3,
        }
    ]


def test_comparison_run_spec_payload_deserializes_serialized_strategy_definitions() -> None:
    comparison = _comparison_spec()
    serialized = serialize_comparison(
        comparison,
        _metadata_by_timeframe(),
        [DEFAULT_DAILY_TIMEFRAME],
    )
    payload = {
        **serialized,
        "kind": "comparison_run_spec_payload",
        "candidateStrategyCount": len(serialized["candidateStrategies"]),
        "referenceStrategyCount": len(serialized["referenceStrategies"]),
        "runSpecFingerprint": build_run_fingerprint(serialized["runSpec"]),
    }
    payload["comparisonFingerprint"] = build_run_fingerprint(
        {
            "comparisonId": payload["comparisonId"],
            "selectionPolicy": payload["selectionPolicy"],
            "candidateStrategies": payload["candidateStrategies"],
            "referenceStrategies": payload["referenceStrategies"],
            "conditionVariants": payload["conditionVariants"],
            "runSpec": payload["runSpec"],
        }
    )

    deserialized = deserialize_comparison_run_spec_payload(payload)

    assert deserialized.comparison_id == comparison.comparison_id
    assert deserialized.run_spec.market_slice.start_date == "2025-01-01"
    assert deserialized.run_spec.execution_assumptions.cost_model.parameters == {
        "commissionPct": 0.123,
        "slippagePct": 0.012,
    }
    assert deserialized.candidate_strategies[0].strategy_id == "test_strategy"
    assert deserialized.candidate_strategies[0].signals[0].data_timeframe.key == DEFAULT_DAILY_TIMEFRAME.key
    assert deserialized.condition_variants[0].max_weight == pytest.approx(0.333)


def test_serialize_condition_variant_preserves_absent_max_weight_as_null() -> None:
    payload = serialize_condition_variant(
        ConditionVariant(
            key="base",
            label="Base",
            cost_multiplier=1.0,
            max_investment_ratio=1.0,
            max_weight=None,
        )
    )

    assert payload["maxWeightPct"] is None
