from __future__ import annotations

from app.portfolio_domain import (
    AlignmentPolicySpec,
    AssetRankingSpec,
    CandidateSetSpec,
    DecisionUseSpec,
    EvaluatorStrategySpec,
    ExecutionPolicySpec,
    FeatureSpec,
    ObservationSpec,
    PortfolioModelSpec,
    PredictionCombinerSpec,
    PredictionEngineSpec,
    PredictionLearnerSpec,
    PredictionOutputSpec,
    PredictionSignalSourceSpec,
    PredictionTargetSpec,
    PredictedQuantitySpec,
    PredictorUseSpec,
    RankingFeatureRecipeSpec,
    RankingSourceSpec,
    RiskControlsSpec,
    SignalSpec,
    StrategyDefinition,
    StrategyExecutionPlanSpec,
    StrategySignalSpec,
    TrainingSpec,
    get_direct_execution_strategy_definition_compatibility_issues,
    extract_ranking_score_parameters,
    thaw_strategy_parameter_value,
)


def serialize_portfolio_model_spec(model: PortfolioModelSpec) -> dict:
    return {
        "key": model.key,
        "modelType": model.model_type,
        "label": model.label,
        "description": model.description,
    }


def serialize_execution_policy_spec(execution_policy: ExecutionPolicySpec) -> dict:
    return {
        "key": execution_policy.key,
        "label": execution_policy.label,
        "entry": execution_policy.entry,
        "rebalanceSchedule": execution_policy.rebalance_schedule,
    }


def serialize_strategy_execution_plan_spec(execution_plan: StrategyExecutionPlanSpec) -> dict:
    return {
        "key": execution_plan.key,
        "label": execution_plan.label,
        "decisionSchedule": execution_plan.decision_schedule,
        "rebalanceSchedule": execution_plan.rebalance_schedule,
    }


def serialize_strategy_signal_spec(signal: StrategySignalSpec) -> dict:
    return {
        "key": signal.key,
        "label": signal.label,
        "description": signal.description,
        "sourceKind": signal.source_kind,
        "weight": signal.weight,
        "predictorKey": signal.predictor_key,
        "signalParameters": {
            key: thaw_strategy_parameter_value(value)
            for key, value in signal.signal_parameters
        },
        "observationSpec": serialize_observation_spec(signal.observation_spec),
        "dataSource": None if signal.data_source_spec is None else {
            "key": signal.data_source_spec.key,
            "label": signal.data_source_spec.label,
            "kind": signal.data_source_spec.kind,
            "observationSpec": serialize_observation_spec(signal.data_source_spec.observation_spec),
        },
        "featureDefinition": None if signal.feature_definition_spec is None else {
            "key": signal.feature_definition_spec.key,
            "label": signal.feature_definition_spec.label,
            "sourceFieldKeys": list(signal.feature_definition_spec.source_field_keys),
            "derivedFeatureKeys": list(signal.feature_definition_spec.derived_feature_keys),
        },
        "alignmentPolicy": None if signal.alignment_policy is None else {
            "key": signal.alignment_policy.key,
            "label": signal.alignment_policy.label,
            "method": signal.alignment_policy.method,
            "parameters": {key: value for key, value in signal.alignment_policy.parameters},
        },
        "dataTimeframe": {
            "key": signal.data_timeframe.key,
            "label": signal.data_timeframe.label,
            "yfinanceInterval": signal.data_timeframe.yfinance_interval,
            "barsPerYear": signal.data_timeframe.bars_per_year,
            "barSeconds": signal.data_timeframe.bar_seconds,
        },
        "signalTimeframe": {
            "key": signal.signal_timeframe.key,
            "label": signal.signal_timeframe.label,
            "yfinanceInterval": signal.signal_timeframe.yfinance_interval,
            "barsPerYear": signal.signal_timeframe.bars_per_year,
            "barSeconds": signal.signal_timeframe.bar_seconds,
        },
    }


def serialize_strategy_definition(strategy: StrategyDefinition) -> dict:
    direct_execution_issues = get_direct_execution_strategy_definition_compatibility_issues(strategy)
    return {
        "kind": "strategy_definition",
        "schemaVersion": "v1",
        "strategyId": strategy.strategy_id,
        "version": strategy.version,
        "label": strategy.label,
        "hypothesis": strategy.hypothesis,
        "description": strategy.description,
        "components": {
            "core": {
                "investmentUniverse": {
                    "key": strategy.investment_universe.key,
                    "label": strategy.investment_universe.label,
                    "assetCount": len(strategy.investment_universe.tickers),
                    "tickers": list(strategy.investment_universe.tickers),
                },
                "portfolioModel": serialize_portfolio_model_spec(strategy.portfolio_model),
                "executionPlan": serialize_strategy_execution_plan_spec(strategy.execution_plan),
            },
            "optional": {
                "signals": [serialize_strategy_signal_spec(signal) for signal in strategy.signals],
                "riskControls": serialize_risk_controls_spec(strategy.risk_controls),
            },
        },
        "executionSupport": {
            "directExecutionCompatible": not direct_execution_issues,
            "directExecutionIssues": direct_execution_issues,
        },
        "extensions": {key: value for key, value in strategy.extensions},
    }


def serialize_alignment_policy_spec(
    alignment_policy: AlignmentPolicySpec | None,
) -> dict[str, object] | None:
    if alignment_policy is None:
        return None
    return {
        "key": alignment_policy.key,
        "label": alignment_policy.label,
        "method": alignment_policy.method,
        "parameters": {key: value for key, value in alignment_policy.parameters},
    }


def serialize_ranking_source_spec(ranking_source: RankingSourceSpec) -> dict:
    return {
        "key": ranking_source.key,
        "strategyType": ranking_source.strategy_type,
        "label": ranking_source.label,
        "description": ranking_source.description,
        "featureInputs": list(ranking_source.ranking_signal.feature_inputs),
        "universePolicy": {
            "key": ranking_source.universe_policy.key,
            "label": ranking_source.universe_policy.label,
        },
        "scoreModel": {
            "kind": ranking_source.ranking_signal.score_model.kind,
            "label": ranking_source.ranking_signal.score_model.label,
        },
        "scoreParameters": {
            key: value for key, value in ranking_source.ranking_signal.score_parameters
        },
        "filterRules": [
            {
                "key": filter_rule.key,
                "label": filter_rule.label,
            }
            for filter_rule in ranking_source.filter_rules
        ],
        "fallbackRule": {
            "key": ranking_source.fallback_rule.key,
            "label": ranking_source.fallback_rule.label,
        },
    }


def serialize_candidate_set_spec(candidate_set_spec: CandidateSetSpec) -> dict:
    return {
        "kind": "candidate_set_spec",
        "schemaVersion": "v1",
        "key": candidate_set_spec.key,
        "label": candidate_set_spec.label,
        "description": candidate_set_spec.description,
        "universePolicy": {
            "key": candidate_set_spec.universe_policy.key,
            "label": candidate_set_spec.universe_policy.label,
        },
        "filterRules": [
            {
                "key": filter_rule.key,
                "label": filter_rule.label,
            }
            for filter_rule in candidate_set_spec.filter_rules
        ],
        "fallbackRule": {
            "key": candidate_set_spec.fallback_rule.key,
            "label": candidate_set_spec.fallback_rule.label,
        },
    }


def serialize_observation_spec(observation_spec: ObservationSpec) -> dict:
    return {
        "kind": "observation_spec",
        "schemaVersion": "v1",
        "key": observation_spec.key,
        "label": observation_spec.label,
        "assetCount": len(observation_spec.tickers),
        "tickers": list(observation_spec.tickers),
        "fields": list(observation_spec.fields),
    }


def serialize_signal_spec(signal_spec: SignalSpec) -> dict:
    return {
        "kind": "signal_spec",
        "schemaVersion": "v1",
        "key": signal_spec.key,
        "label": signal_spec.label,
        "observationSpec": serialize_observation_spec(signal_spec.observation_spec),
        "entityKind": signal_spec.entity_kind,
        "entityIdentifiers": list(signal_spec.entity_identifiers),
        "outputSpec": serialize_prediction_output_spec(signal_spec.output_spec),
        "decisionUseSpec": serialize_decision_use_spec(signal_spec.decision_use_spec),
    }


def serialize_decision_use_spec(
    decision_use_spec: DecisionUseSpec,
) -> dict:
    return {
        "kind": "decision_use_spec",
        "schemaVersion": "v1",
        "key": decision_use_spec.key,
        "label": decision_use_spec.label,
        "description": decision_use_spec.description,
        "useKind": decision_use_spec.kind,
        "candidateSet": serialize_candidate_set_spec(decision_use_spec.candidate_set_spec),
    }


def serialize_ranking_feature_recipe(recipe_spec: RankingFeatureRecipeSpec) -> dict:
    recipe = serialize_ranking_source_spec(recipe_spec)
    return {
        "kind": "ranking_feature_recipe",
        "schemaVersion": "v1",
        **recipe,
    }


def serialize_risk_controls_spec(risk_controls: RiskControlsSpec) -> dict:
    return {
        "maxInvestmentPct": round(risk_controls.max_investment_ratio * 100, 1),
        "maxWeightPct": round(risk_controls.max_weight * 100, 1)
        if risk_controls.max_weight is not None
        else None,
    }


def serialize_asset_ranking_model_parameters(strategy: EvaluatorStrategySpec) -> dict[str, object]:
    score_parameters = dict(strategy.selection.ranking_signal.score_parameters)
    serialized: dict[str, object] = {}

    if "windowSpec" in score_parameters:
        serialized["windowSpec"] = dict(score_parameters["windowSpec"])
    if "momentum_weight" in score_parameters:
        serialized["momentumWeight"] = float(score_parameters["momentum_weight"])
    if "low_vol_weight" in score_parameters:
        serialized["lowVolWeight"] = float(score_parameters["low_vol_weight"])
    if "macro_weight" in score_parameters:
        serialized["macroWeight"] = float(score_parameters["macro_weight"])

    return serialized


def serialize_strategy_execution_context_payload(context: dict[str, object]) -> dict[str, object]:
    payload = {
        "sourceKind": str(context.get("sourceKind", context.get("source_kind", ""))),
        "dataTimeframe": str(context.get("dataTimeframe", context.get("data_timeframe_key", ""))),
        "signalTimeframe": str(context.get("signalTimeframe", context.get("signal_timeframe_key", ""))),
        "alignmentPolicy": context.get("alignmentPolicy", context.get("alignment_policy")),
    }
    if "selectionKey" in context:
        payload["selectionKey"] = str(context["selectionKey"])
    if "strategyType" in context:
        payload["strategyType"] = str(context["strategyType"])
    if "scoreParameters" in context:
        payload["scoreParameters"] = context["scoreParameters"]
    if "weight" in context:
        payload["weight"] = float(context["weight"])
    if "predictorKey" in context:
        payload["predictorKey"] = str(context["predictorKey"])
    if "signalWeight" in context:
        payload["signalWeight"] = float(context["signalWeight"])
    if "predictorWeight" in context:
        payload["predictorWeight"] = float(context["predictorWeight"])
    return payload


def serialize_predictor_use_spec(predictor_use: PredictorUseSpec) -> dict[str, object]:
    return {
        "predictorKey": predictor_use.predictor_key,
        "signalWeight": predictor_use.signal_weight,
        "predictorWeight": predictor_use.predictor_weight,
    }


def serialize_tilt_rule(strategy: EvaluatorStrategySpec) -> dict | None:
    score_parameters = dict(strategy.selection.ranking_signal.score_parameters)
    if "tilt_strength" not in score_parameters:
        return None

    parameters = {
        "strength": float(score_parameters["tilt_strength"]),
    }
    if "tilt_shape" in score_parameters:
        parameters["shape"] = float(score_parameters["tilt_shape"])

    return {
        "kind": "ranking_weight_tilt",
        "label": "ランキング連動ティルト",
        "parameters": parameters,
    }


def serialize_evaluator_strategy_spec(strategy: EvaluatorStrategySpec) -> dict:
    ranking_model = (
        {
            "kind": strategy.selection.ranking_signal.score_model.kind,
            "label": strategy.selection.ranking_signal.score_model.label,
            "parameters": serialize_asset_ranking_model_parameters(strategy),
        }
        if strategy.selection.ranking_signal.score_model.kind != "none"
        else None
    )
    fallback_rule = (
        {
            "key": strategy.selection.fallback_rule.key,
            "label": strategy.selection.fallback_rule.label,
        }
        if strategy.selection.fallback_rule.key != "none"
        else None
    )
    return {
        "kind": "evaluator_strategy_spec",
        "schemaVersion": "v1",
        "strategyId": strategy.strategy_id,
        "version": strategy.version,
        "label": strategy.label,
        "hypothesis": strategy.hypothesis,
        "description": strategy.description,
        "components": {
            "core": {
                "dataResolution": {
                    "key": strategy.timeframe.key,
                    "label": strategy.timeframe.label,
                    "yfinanceInterval": strategy.timeframe.yfinance_interval,
                    "barsPerYear": strategy.timeframe.bars_per_year,
                    "barSeconds": strategy.timeframe.bar_seconds,
                },
                "investmentUniverse": {
                    "key": strategy.investment_universe.key,
                    "label": strategy.investment_universe.label,
                    "assetCount": len(strategy.investment_universe.tickers),
                    "tickers": list(strategy.investment_universe.tickers),
                },
                "portfolioModel": serialize_portfolio_model_spec(strategy.portfolio_model),
                "executionPolicy": serialize_execution_policy_spec(strategy.execution_policy),
                "decisionSchedule": strategy.decision_schedule or strategy.execution_policy.rebalance_schedule,
                "executionMode": strategy.execution_mode,
            },
            "optional": {
                "assetRankingModel": ranking_model,
                "predictor": None
                if strategy.predictor_use is None
                else serialize_predictor_use_spec(strategy.predictor_use),
                "featureInputs": list(strategy.selection.ranking_signal.feature_inputs),
                "filterRules": [
                    {
                        "key": filter_rule.key,
                        "label": filter_rule.label,
                    }
                    for filter_rule in strategy.selection.filter_rules
                ],
                "fallbackRule": fallback_rule,
                "riskControls": serialize_risk_controls_spec(strategy.risk_controls),
                "tiltRule": serialize_tilt_rule(strategy),
                "decisionPolicy": {
                    "key": dict(strategy.extensions).get("decision_policy", "cost_aware_no_trade"),
                },
                "signalExecutionContexts": {
                    "selectionSignals": [
                        serialize_strategy_execution_context_payload(context)
                        for context in strategy.signal_execution_contexts
                    ],
                    "predictorSignal": None
                    if strategy.predictor_signal_execution_context is None
                    else serialize_strategy_execution_context_payload(
                        strategy.predictor_signal_execution_context
                    ),
                },
            },
        },
        "extensions": {
            key: value for key, value in strategy.extensions
        },
    }


def serialize_asset_ranking_spec(
    ranking_spec: AssetRankingSpec,
) -> dict:
    selection = ranking_spec.selection
    return {
        "kind": "asset_ranking_spec",
        "schemaVersion": "v1",
        "key": ranking_spec.key,
        "label": ranking_spec.label,
        "description": ranking_spec.description,
        "timeframe": {
            "key": ranking_spec.timeframe.key,
            "label": ranking_spec.timeframe.label,
            "yfinanceInterval": ranking_spec.timeframe.yfinance_interval,
            "barsPerYear": ranking_spec.timeframe.bars_per_year,
            "barSeconds": ranking_spec.timeframe.bar_seconds,
        },
        "investmentUniverse": {
            "key": ranking_spec.investment_universe.key,
            "label": ranking_spec.investment_universe.label,
            "assetCount": len(ranking_spec.investment_universe.tickers),
            "tickers": list(ranking_spec.investment_universe.tickers),
        },
        "featureInputs": list(selection.ranking_signal.feature_inputs),
        "universePolicy": {
            "key": selection.universe_policy.key,
            "label": selection.universe_policy.label,
        },
        "rankingModel": {
            "kind": selection.ranking_signal.score_model.kind,
            "label": selection.ranking_signal.score_model.label,
        },
        "scoreParameters": extract_ranking_score_parameters(selection),
        "filterRules": [
            {
                "key": filter_rule.key,
                "label": filter_rule.label,
            }
            for filter_rule in selection.filter_rules
        ],
        "fallbackRule": {
            "key": selection.fallback_rule.key,
            "label": selection.fallback_rule.label,
        },
    }


def serialize_prediction_target_spec(
    target_spec: PredictionTargetSpec,
) -> dict:
    return {
        "kind": "prediction_target_spec",
        "schemaVersion": "v1",
        "key": target_spec.key,
        "label": target_spec.label,
        "horizonSpec": {
            key: value for key, value in target_spec.horizon_spec
        },
        "baseline": target_spec.baseline,
        "transform": target_spec.transform,
    }


def serialize_predicted_quantity_spec(
    predicted_quantity_spec: PredictedQuantitySpec,
) -> dict:
    return {
        "kind": "predicted_quantity_spec",
        "schemaVersion": "v1",
        "key": predicted_quantity_spec.key,
        "label": predicted_quantity_spec.label,
        "quantityKind": predicted_quantity_spec.kind,
    }


def serialize_feature_spec(
    feature_spec: FeatureSpec,
) -> dict:
    return {
        "kind": "feature_spec",
        "schemaVersion": "v1",
        "key": feature_spec.key,
        "label": feature_spec.label,
        "featureInputs": [
            {
                "key": feature_input.key,
                "label": feature_input.label,
            }
            for feature_input in feature_spec.feature_inputs
        ],
        "derivedFeatures": [
            {
                "key": derived_feature.key,
                "label": derived_feature.label,
            }
            for derived_feature in feature_spec.derived_features
        ],
        "rankingFeatureRecipe": serialize_ranking_feature_recipe(feature_spec.ranking_feature_recipe)
        if feature_spec.ranking_feature_recipe is not None
        else None,
    }


def serialize_prediction_signal_source_spec(
    signal_source_spec: PredictionSignalSourceSpec,
) -> dict:
    return {
        "kind": "prediction_signal_source_spec",
        "schemaVersion": "v1",
        "key": signal_source_spec.key,
        "label": signal_source_spec.label,
        "signalSourceKind": signal_source_spec.kind,
        "featureKey": signal_source_spec.feature_key,
    }


def serialize_prediction_learner_spec(
    learner_spec: PredictionLearnerSpec,
) -> dict:
    return {
        "kind": "prediction_learner_spec",
        "schemaVersion": "v1",
        "key": learner_spec.key,
        "label": learner_spec.label,
        "learnerKind": learner_spec.kind,
        "ridgeAlpha": learner_spec.ridge_alpha,
    }


def serialize_prediction_combiner_spec(
    combiner_spec: PredictionCombinerSpec,
) -> dict:
    return {
        "kind": "prediction_combiner_spec",
        "schemaVersion": "v1",
        "key": combiner_spec.key,
        "label": combiner_spec.label,
        "combinerKind": combiner_spec.kind,
        "signalSourceWeight": combiner_spec.signal_source_weight,
        "learnerWeight": combiner_spec.learner_weight,
    }


def serialize_prediction_output_spec(
    output_spec: PredictionOutputSpec,
) -> dict:
    return {
        "kind": "prediction_output_spec",
        "schemaVersion": "v1",
        "key": output_spec.key,
        "label": output_spec.label,
        "outputKind": output_spec.kind,
    }


def serialize_prediction_engine_spec(
    engine_spec: PredictionEngineSpec,
) -> dict:
    return {
        "kind": "prediction_engine_spec",
        "schemaVersion": "v1",
        "key": engine_spec.key,
        "label": engine_spec.label,
        "signalSourceSpec": (
            serialize_prediction_signal_source_spec(engine_spec.signal_source_spec)
            if engine_spec.signal_source_spec is not None
            else None
        ),
        "learnerSpec": (
            serialize_prediction_learner_spec(engine_spec.learner_spec)
            if engine_spec.learner_spec is not None
            else None
        ),
        "combinerSpec": serialize_prediction_combiner_spec(engine_spec.combiner_spec),
    }


def serialize_training_spec(
    training_spec: TrainingSpec,
) -> dict:
    return {
        "kind": "training_spec",
        "schemaVersion": "v1",
        "key": training_spec.key,
        "label": training_spec.label,
        "fitMode": training_spec.fit_mode,
        "minTrainSamples": training_spec.min_train_samples,
    }


__all__ = (
    "serialize_portfolio_model_spec",
    "serialize_execution_policy_spec",
    "serialize_strategy_execution_plan_spec",
    "serialize_strategy_signal_spec",
    "serialize_strategy_definition",
    "serialize_alignment_policy_spec",
    "serialize_ranking_source_spec",
    "serialize_candidate_set_spec",
    "serialize_observation_spec",
    "serialize_signal_spec",
    "serialize_decision_use_spec",
    "serialize_ranking_feature_recipe",
    "serialize_risk_controls_spec",
    "serialize_asset_ranking_model_parameters",
    "serialize_strategy_execution_context_payload",
    "serialize_predictor_use_spec",
    "serialize_tilt_rule",
    "serialize_evaluator_strategy_spec",
    "serialize_asset_ranking_spec",
    "serialize_prediction_target_spec",
    "serialize_predicted_quantity_spec",
    "serialize_feature_spec",
    "serialize_prediction_signal_source_spec",
    "serialize_prediction_learner_spec",
    "serialize_prediction_combiner_spec",
    "serialize_prediction_output_spec",
    "serialize_prediction_engine_spec",
    "serialize_training_spec",
)
