from __future__ import annotations

from app.domain import (
    DecisionPolicyBlueprint,
    ExecutionScheduleBlueprint,
    ForecastSourceBlueprint,
    PortfolioConstructionBlueprint,
    StrategyBlueprint,
    StrategySignalBlueprint,
)
from app.portfolio_domain import (
    DEFAULT_DAILY_TIMEFRAME,
    StrategyDefinition,
    build_strategy_definition,
    build_strategy_definition_from_evaluator_strategy_spec,
)


def build_strategy_blueprint_from_definition(strategy_definition: StrategyDefinition) -> StrategyBlueprint:
    signal_blueprints = tuple(
        StrategySignalBlueprint(
            key=signal.key,
            label=signal.label,
            description=signal.description,
            source_kind=signal.source_kind,
            data_timeframe_key=signal.data_timeframe.key,
            signal_timeframe_key=signal.signal_timeframe.key,
            observation_tickers=signal.observation_spec.tickers,
            observation_fields=signal.observation_spec.fields,
            signal_parameters=signal.signal_parameters,
            weight=signal.weight,
            predictor_key=signal.predictor_key,
            data_source_key=None if signal.data_source_spec is None else signal.data_source_spec.key,
            data_source_label=None if signal.data_source_spec is None else signal.data_source_spec.label,
            data_source_kind=None if signal.data_source_spec is None else signal.data_source_spec.kind,
            feature_definition_key=(
                None if signal.feature_definition_spec is None else signal.feature_definition_spec.key
            ),
            feature_definition_label=(
                None if signal.feature_definition_spec is None else signal.feature_definition_spec.label
            ),
            feature_definition_source_fields=(
                ()
                if signal.feature_definition_spec is None
                else signal.feature_definition_spec.source_field_keys
            ),
            feature_definition_derived_fields=(
                ()
                if signal.feature_definition_spec is None
                else signal.feature_definition_spec.derived_feature_keys
            ),
            alignment_policy_key=None if signal.alignment_policy is None else signal.alignment_policy.key,
            alignment_policy_label=None if signal.alignment_policy is None else signal.alignment_policy.label,
            alignment_policy_method=None if signal.alignment_policy is None else signal.alignment_policy.method,
            alignment_policy_parameters=(
                () if signal.alignment_policy is None else signal.alignment_policy.parameters
            ),
        )
        for signal in strategy_definition.signals
    )
    predictor_signal = next((signal for signal in signal_blueprints if signal.source_kind == 'predictor_overlay'), None)
    return StrategyBlueprint(
        key=strategy_definition.strategy_id,
        version=strategy_definition.version,
        label=strategy_definition.label,
        hypothesis=strategy_definition.hypothesis,
        description=strategy_definition.description,
        investment_universe_key=strategy_definition.investment_universe.key,
        investment_universe_label=strategy_definition.investment_universe.label,
        investment_universe_tickers=strategy_definition.investment_universe.tickers,
        signals=signal_blueprints,
        portfolio_construction=PortfolioConstructionBlueprint(
            model_key=strategy_definition.portfolio_model.key,
            model_type=strategy_definition.portfolio_model.model_type,
            max_investment_ratio=strategy_definition.risk_controls.max_investment_ratio,
            max_weight=strategy_definition.risk_controls.max_weight,
        ),
        decision_policy=DecisionPolicyBlueprint(
            key=dict(strategy_definition.extensions).get('decision_policy', 'cost_aware_no_trade'),
            execution_plan_key=strategy_definition.execution_plan.key,
            execution_plan_label=strategy_definition.execution_plan.label,
            decision_schedule=strategy_definition.execution_plan.decision_schedule,
            rebalance_schedule=strategy_definition.execution_plan.rebalance_schedule,
            execution_mode=dict(strategy_definition.extensions).get('execution_mode'),
        ),
        execution_schedule=ExecutionScheduleBlueprint(
            market_data_timeframe_key=signal_blueprints[0].data_timeframe_key,
            signal_timeframe_keys=tuple(dict.fromkeys(signal.signal_timeframe_key for signal in signal_blueprints)),
        ),
        forecast_source=(
            None
            if predictor_signal is None
            else ForecastSourceBlueprint(
                kind=predictor_signal.source_kind,
                predictor_key=predictor_signal.predictor_key,
                signal_weight=_read_numeric_signal_parameter(predictor_signal.signal_parameters, 'signalWeight'),
                predictor_weight=_read_numeric_signal_parameter(predictor_signal.signal_parameters, 'predictorWeight'),
            )
        ),
        extensions=strategy_definition.extensions,
    )


def build_strategy_blueprint_from_evaluator_strategy_spec(strategy) -> StrategyBlueprint:
    return build_strategy_blueprint_from_definition(
        build_strategy_definition_from_evaluator_strategy_spec(strategy)
    )


def build_strategy_definition_from_blueprint(blueprint: StrategyBlueprint) -> StrategyDefinition:
    from app.portfolio_domain import (
        build_alignment_policy_spec,
        build_investment_universe_spec,
        build_observation_spec,
        build_portfolio_model_spec,
        build_risk_controls_spec,
        build_strategy_data_source_spec,
        build_strategy_execution_plan_spec,
        build_strategy_feature_definition_spec,
        build_strategy_signal_spec,
        resolve_timeframe_spec_from_key,
    )

    primary_timeframe = resolve_timeframe_spec_from_key(
        blueprint.execution_schedule.market_data_timeframe_key,
        fallback=DEFAULT_DAILY_TIMEFRAME,
    )
    signals = []
    for signal in blueprint.signals:
        data_source_spec = None
        if signal.data_source_key is not None:
            data_source_spec = build_strategy_data_source_spec(
                key=signal.data_source_key,
                label=signal.data_source_label or signal.label,
                kind=signal.data_source_kind or 'market_observation',
                observation_spec=build_observation_spec(
                    key=f'observation__{blueprint.key}__{signal.key}__source',
                    label=f'{signal.label} source observation',
                    tickers=signal.observation_tickers,
                    fields=signal.observation_fields,
                ),
            )
        feature_definition_spec = None
        if signal.feature_definition_key is not None:
            feature_definition_spec = build_strategy_feature_definition_spec(
                key=signal.feature_definition_key,
                label=signal.feature_definition_label or signal.label,
                source_field_keys=signal.feature_definition_source_fields,
                derived_feature_keys=signal.feature_definition_derived_fields,
            )
        alignment_policy = None
        if signal.alignment_policy_key is not None and signal.alignment_policy_method is not None:
            alignment_policy = build_alignment_policy_spec(
                key=signal.alignment_policy_key,
                label=signal.alignment_policy_label or signal.alignment_policy_key,
                method=signal.alignment_policy_method,
                parameters={key: value for key, value in signal.alignment_policy_parameters},
            )
        signals.append(
            build_strategy_signal_spec(
                key=signal.key,
                label=signal.label,
                description=signal.description,
                observation_spec=build_observation_spec(
                    key=f'observation__{blueprint.key}__{signal.key}',
                    label=f'{signal.label} observation',
                    tickers=signal.observation_tickers,
                    fields=signal.observation_fields,
                ),
                data_timeframe=resolve_timeframe_spec_from_key(signal.data_timeframe_key, fallback=primary_timeframe),
                signal_timeframe=resolve_timeframe_spec_from_key(signal.signal_timeframe_key, fallback=primary_timeframe),
                source_kind=signal.source_kind,
                data_source_spec=data_source_spec,
                feature_definition_spec=feature_definition_spec,
                signal_parameters={key: value for key, value in signal.signal_parameters},
                weight=signal.weight,
                predictor_key=signal.predictor_key,
                alignment_policy=alignment_policy,
            )
        )

    extensions = dict(blueprint.extensions)
    if blueprint.decision_policy.execution_mode is not None:
        extensions.setdefault('execution_mode', blueprint.decision_policy.execution_mode)
    extensions.setdefault('decision_policy', blueprint.decision_policy.key)
    return build_strategy_definition(
        strategy_id=blueprint.key,
        version=blueprint.version,
        hypothesis=blueprint.hypothesis,
        label=blueprint.label,
        description=blueprint.description,
        investment_universe=build_investment_universe_spec(
            tickers=blueprint.investment_universe_tickers,
            key=blueprint.investment_universe_key,
            label=blueprint.investment_universe_label,
        ),
        signals=signals,
        portfolio_model=build_portfolio_model_spec(
            blueprint.portfolio_construction.model_type,
            key=blueprint.portfolio_construction.model_key,
        ),
        execution_plan=build_strategy_execution_plan_spec(
            key=blueprint.decision_policy.execution_plan_key,
            label=blueprint.decision_policy.execution_plan_label,
            decision_schedule=blueprint.decision_policy.decision_schedule,
            rebalance_schedule=blueprint.decision_policy.rebalance_schedule,
        ),
        risk_controls=build_risk_controls_spec(
            max_investment_ratio=blueprint.portfolio_construction.max_investment_ratio,
            max_weight=blueprint.portfolio_construction.max_weight,
        ),
        extensions=extensions,
    )


def _read_numeric_signal_parameter(
    signal_parameters: tuple[tuple[str, object], ...],
    parameter_key: str,
) -> float | None:
    for key, value in signal_parameters:
        if key == parameter_key:
            return float(value)
    return None
