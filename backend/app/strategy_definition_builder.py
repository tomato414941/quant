from __future__ import annotations

from dataclasses import dataclass

from app.portfolio import (
    AlignmentPolicySpec,
    ExecutionPolicySpec,
    InvestmentUniverseSpec,
    PortfolioModelSpec,
    RiskControlsSpec,
    SelectionSpec,
    StrategyDataSourceSpec,
    StrategyDefinition,
    StrategyFeatureDefinitionSpec,
    build_observation_spec,
    build_strategy_definition,
    build_strategy_execution_plan_spec,
    build_strategy_signal_spec,
    build_executable_strategy_spec_from_definition,
)
from app.timeframe_models import TimeframeSpec


@dataclass(frozen=True)
class SelectionDefinitionDefinition:
    strategy_id: str
    selection: SelectionSpec
    portfolio_model: PortfolioModelSpec
    timeframe: TimeframeSpec
    execution_policy: ExecutionPolicySpec
    investment_universe: InvestmentUniverseSpec
    risk_controls: RiskControlsSpec
    signal_timeframe: TimeframeSpec | None = None
    decision_schedule: str | None = None
    data_source_spec: StrategyDataSourceSpec | None = None
    feature_definition_spec: StrategyFeatureDefinitionSpec | None = None
    alignment_policy: AlignmentPolicySpec | None = None
    label: str | None = None
    hypothesis: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class PredictorDefinitionDefinition:
    strategy_id: str
    selection: SelectionSpec
    portfolio_model: PortfolioModelSpec
    predictor_key: str
    signal_weight: float
    predictor_weight: float
    timeframe: TimeframeSpec
    execution_policy: ExecutionPolicySpec
    investment_universe: InvestmentUniverseSpec
    risk_controls: RiskControlsSpec
    label: str
    hypothesis: str
    description: str
    selection_signal_timeframe: TimeframeSpec | None = None
    predictor_data_timeframe: TimeframeSpec | None = None
    predictor_signal_timeframe: TimeframeSpec | None = None
    decision_schedule: str | None = None
    selection_data_source_spec: StrategyDataSourceSpec | None = None
    selection_feature_definition_spec: StrategyFeatureDefinitionSpec | None = None
    selection_alignment_policy: AlignmentPolicySpec | None = None
    predictor_data_source_spec: StrategyDataSourceSpec | None = None
    predictor_feature_definition_spec: StrategyFeatureDefinitionSpec | None = None
    predictor_alignment_policy: AlignmentPolicySpec | None = None


@dataclass(frozen=True)
class SelectionVariantDefinition:
    key: str
    selection: SelectionSpec
    label: str | None = None
    hypothesis: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class PortfolioModelVariantDefinition:
    key: str
    portfolio_model: PortfolioModelSpec
    label: str | None = None


@dataclass(frozen=True)
class PredictorVariantDefinition:
    key: str
    predictor_key: str
    signal_weight: float
    predictor_weight: float
    label: str
    hypothesis: str
    description: str


@dataclass(frozen=True)
class ExecutionVariantDefinition:
    key: str
    timeframe: TimeframeSpec
    execution_policy: ExecutionPolicySpec
    signal_timeframe: TimeframeSpec | None = None
    decision_schedule: str | None = None
    label: str | None = None
    hypothesis: str | None = None
    description: str | None = None


def build_selection_signal_parameters(selection: SelectionSpec) -> dict[str, object]:
    return {
        "selectionKey": selection.key,
        "strategyType": selection.strategy_type,
        "scoreModelKind": selection.ranking_signal.score_model.kind,
        "scoreParameters": {
            key: value for key, value in selection.ranking_signal.score_parameters
        },
        "featureInputs": list(selection.ranking_signal.feature_inputs),
        "universePolicyKey": selection.universe_policy.key,
        "filterRuleKeys": [filter_rule.key for filter_rule in selection.filter_rules],
        "fallbackRuleKey": selection.fallback_rule.key,
    }


def build_selection_strategy_definition(
    definition: SelectionDefinitionDefinition,
) -> StrategyDefinition:
    selection = definition.selection
    selection_signal = build_strategy_signal_spec(
        key=f"signal__{definition.strategy_id}__selection",
        label=selection.label,
        description=selection.description,
        observation_spec=build_observation_spec(
            key=f"observation__{definition.strategy_id}__selection",
            label=f"{selection.label} observation",
            tickers=definition.investment_universe.tickers,
            fields=selection.ranking_signal.feature_inputs,
        ),
        data_timeframe=definition.timeframe,
        signal_timeframe=definition.signal_timeframe or definition.timeframe,
        source_kind="selection_signal",
        data_source_spec=definition.data_source_spec,
        feature_definition_spec=definition.feature_definition_spec,
        alignment_policy=definition.alignment_policy,
        signal_parameters=build_selection_signal_parameters(selection),
    )
    return build_strategy_definition(
        strategy_id=definition.strategy_id,
        label=definition.label,
        hypothesis=definition.hypothesis,
        description=definition.description,
        investment_universe=definition.investment_universe,
        signals=[selection_signal],
        portfolio_model=definition.portfolio_model,
        execution_plan=build_strategy_execution_plan_spec(
            key=f"execution_plan__{definition.execution_policy.key}",
            label=definition.execution_policy.label,
            decision_schedule=(
                definition.decision_schedule or definition.execution_policy.rebalance_schedule
            ),
            rebalance_schedule=definition.execution_policy.rebalance_schedule,
        ),
        risk_controls=definition.risk_controls,
    )


def build_predictor_strategy_definition(
    definition: PredictorDefinitionDefinition,
) -> StrategyDefinition:
    selection = definition.selection
    selection_signal = build_strategy_signal_spec(
        key=f"signal__{definition.strategy_id}__selection",
        label=selection.label,
        description=selection.description,
        observation_spec=build_observation_spec(
            key=f"observation__{definition.strategy_id}__selection",
            label=f"{selection.label} observation",
            tickers=definition.investment_universe.tickers,
            fields=selection.ranking_signal.feature_inputs,
        ),
        data_timeframe=definition.timeframe,
        signal_timeframe=(
            definition.selection_signal_timeframe or definition.timeframe
        ),
        source_kind="selection_signal",
        data_source_spec=definition.selection_data_source_spec,
        feature_definition_spec=definition.selection_feature_definition_spec,
        alignment_policy=definition.selection_alignment_policy,
        weight=definition.signal_weight,
        signal_parameters=build_selection_signal_parameters(selection),
    )
    predictor_signal = build_strategy_signal_spec(
        key=f"signal__{definition.strategy_id}__predictor",
        label=f"{definition.label} predictor overlay",
        description="Predictor overlay for supplemental momentum blending.",
        observation_spec=build_observation_spec(
            key=f"observation__{definition.strategy_id}__predictor",
            label=f"{definition.label} predictor observation",
            tickers=definition.investment_universe.tickers,
            fields=selection.ranking_signal.feature_inputs,
        ),
        data_timeframe=(
            definition.predictor_data_timeframe or definition.timeframe
        ),
        signal_timeframe=(
            definition.predictor_signal_timeframe
            or definition.predictor_data_timeframe
            or definition.timeframe
        ),
        source_kind="predictor_overlay",
        data_source_spec=definition.predictor_data_source_spec,
        feature_definition_spec=definition.predictor_feature_definition_spec,
        alignment_policy=definition.predictor_alignment_policy,
        weight=definition.predictor_weight,
        signal_parameters={
            "predictorKey": definition.predictor_key,
            "signalWeight": definition.signal_weight,
            "predictorWeight": definition.predictor_weight,
        },
        predictor_key=definition.predictor_key,
    )
    return build_strategy_definition(
        strategy_id=definition.strategy_id,
        label=definition.label,
        hypothesis=definition.hypothesis,
        description=definition.description,
        investment_universe=definition.investment_universe,
        signals=[selection_signal, predictor_signal],
        portfolio_model=definition.portfolio_model,
        execution_plan=build_strategy_execution_plan_spec(
            key=f"execution_plan__{definition.execution_policy.key}",
            label=definition.execution_policy.label,
            decision_schedule=(
                definition.decision_schedule or definition.execution_policy.rebalance_schedule
            ),
            rebalance_schedule=definition.execution_policy.rebalance_schedule,
        ),
        risk_controls=definition.risk_controls,
    )


def build_selection_strategy_definitions(
    definitions: list[SelectionDefinitionDefinition] | tuple[SelectionDefinitionDefinition, ...],
) -> list[StrategyDefinition]:
    return [build_selection_strategy_definition(definition) for definition in definitions]


def build_predictor_strategy_definitions(
    definitions: list[PredictorDefinitionDefinition] | tuple[PredictorDefinitionDefinition, ...],
) -> list[StrategyDefinition]:
    return [build_predictor_strategy_definition(definition) for definition in definitions]


def build_evaluator_strategy_specs_from_definitions(
    definitions: list[StrategyDefinition] | tuple[StrategyDefinition, ...],
):
    return [
        build_executable_strategy_spec_from_definition(strategy_definition)
        for strategy_definition in definitions
    ]


def build_selection_strategy_definition_product(
    *,
    strategy_id_pattern: str,
    selection_variants: list[SelectionVariantDefinition] | tuple[SelectionVariantDefinition, ...],
    portfolio_model_variants: list[PortfolioModelVariantDefinition] | tuple[PortfolioModelVariantDefinition, ...],
    execution_variants: list[ExecutionVariantDefinition] | tuple[ExecutionVariantDefinition, ...],
    investment_universe: InvestmentUniverseSpec,
    risk_controls: RiskControlsSpec,
) -> list[StrategyDefinition]:
    definitions: list[StrategyDefinition] = []
    for selection_variant in selection_variants:
        for portfolio_model_variant in portfolio_model_variants:
            for execution_variant in execution_variants:
                definitions.append(
                    build_selection_strategy_definition(
                        SelectionDefinitionDefinition(
                            strategy_id=strategy_id_pattern.format(
                                selection=selection_variant.key,
                                portfolio_model=portfolio_model_variant.key,
                                execution=execution_variant.key,
                            ),
                            selection=selection_variant.selection,
                            portfolio_model=portfolio_model_variant.portfolio_model,
                            timeframe=execution_variant.timeframe,
                            execution_policy=execution_variant.execution_policy,
                            signal_timeframe=execution_variant.signal_timeframe,
                            decision_schedule=execution_variant.decision_schedule,
                            investment_universe=investment_universe,
                            risk_controls=risk_controls,
                            label=(
                                execution_variant.label
                                or selection_variant.label
                                or portfolio_model_variant.label
                            ),
                            hypothesis=(
                                execution_variant.hypothesis
                                or selection_variant.hypothesis
                            ),
                            description=(
                                execution_variant.description
                                or selection_variant.description
                            ),
                        )
                    )
                )
    return definitions



def build_predictor_strategy_definition_product(
    *,
    strategy_id_pattern: str,
    selection_variants: list[SelectionVariantDefinition] | tuple[SelectionVariantDefinition, ...],
    predictor_variants: list[PredictorVariantDefinition] | tuple[PredictorVariantDefinition, ...],
    portfolio_model_variants: list[PortfolioModelVariantDefinition] | tuple[PortfolioModelVariantDefinition, ...],
    execution_variants: list[ExecutionVariantDefinition] | tuple[ExecutionVariantDefinition, ...],
    investment_universe: InvestmentUniverseSpec,
    risk_controls: RiskControlsSpec,
) -> list[StrategyDefinition]:
    definitions: list[StrategyDefinition] = []
    for selection_variant in selection_variants:
        for predictor_variant in predictor_variants:
            for portfolio_model_variant in portfolio_model_variants:
                for execution_variant in execution_variants:
                    definitions.append(
                        build_predictor_strategy_definition(
                            PredictorDefinitionDefinition(
                                strategy_id=strategy_id_pattern.format(
                                    selection=selection_variant.key,
                                    predictor=predictor_variant.key,
                                    portfolio_model=portfolio_model_variant.key,
                                    execution=execution_variant.key,
                                ),
                                selection=selection_variant.selection,
                                portfolio_model=portfolio_model_variant.portfolio_model,
                                predictor_key=predictor_variant.predictor_key,
                                signal_weight=predictor_variant.signal_weight,
                                predictor_weight=predictor_variant.predictor_weight,
                                timeframe=execution_variant.timeframe,
                                execution_policy=execution_variant.execution_policy,
                                selection_signal_timeframe=(
                                    execution_variant.signal_timeframe
                                ),
                                predictor_signal_timeframe=(
                                    execution_variant.signal_timeframe
                                ),
                                decision_schedule=execution_variant.decision_schedule,
                                investment_universe=investment_universe,
                                risk_controls=risk_controls,
                                label=(
                                    predictor_variant.label
                                    if execution_variant.label is None
                                    else f"{predictor_variant.label} × {execution_variant.label}"
                                ),
                                hypothesis=(
                                    execution_variant.hypothesis
                                    or predictor_variant.hypothesis
                                ),
                                description=(
                                    execution_variant.description
                                    or predictor_variant.description
                                ),
                            )
                        )
                    )
    return definitions
