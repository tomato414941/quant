from __future__ import annotations

from app.strategy_presets import (
    DEFAULT_INVESTMENT_UNIVERSE,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
)
from app.portfolio import (
    PREDICTION_FEATURE_NAMES,
    PredictorSpec,
    build_observation_spec,
    build_prediction_combiner_spec,
    build_derived_feature_spec,
    build_feature_spec,
    build_feature_input_spec,
    build_prediction_output_spec,
    build_decision_use_spec,
    build_prediction_engine_spec,
    build_prediction_learner_spec,
    build_prediction_signal_source_spec,
    build_signal_spec,
    build_predicted_quantity_spec,
    build_prediction_target_spec,
    build_predictor_spec,
    build_ranking_feature_recipe_spec,
    build_training_spec,
)
from app.timeframe_models import DEFAULT_DAILY_TIMEFRAME


def _horizon_value(target_spec) -> int:
    horizon = dict(target_spec.horizon_spec)
    return int(horizon["value"])


DEFAULT_PREDICTION_TARGET_SPECS = [
    build_prediction_target_spec(
        key="next_5bar_excess_return",
        label="次の5bar超過収益",
        horizon_spec={"unit": "bars", "value": 5},
        baseline="cross_sectional_mean",
        transform="identity",
    ),
    build_prediction_target_spec(
        key="next_10bar_excess_return",
        label="次の10bar超過収益",
        horizon_spec={"unit": "bars", "value": 10},
        baseline="cross_sectional_mean",
        transform="identity",
    ),
]


def _build_registered_predictors() -> list[PredictorSpec]:
    feature_spec = build_feature_spec(
        key="features__pred-fu-momo2-supplement",
        label="2ヶ月モメンタム supplement features",
        feature_inputs=tuple(
            build_feature_input_spec(key=feature_input)
            for feature_input in FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M.ranking_signal.feature_inputs
        ),
        derived_features=tuple(
            build_derived_feature_spec(key=feature_key)
            for feature_key in PREDICTION_FEATURE_NAMES
        ),
        ranking_feature_recipe=build_ranking_feature_recipe_spec(FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M),
    )
    engine_specs = [
        build_prediction_engine_spec(
            key="engine__pred-fu-momo2-supplement-linear",
            label="2ヶ月モメンタム supplement linear",
            signal_source_spec=None,
            learner_spec=build_prediction_learner_spec(
                key="learner__pred-fu-momo2-supplement-linear",
                label="2ヶ月モメンタム supplement linear learner",
                kind="linear_regression",
            ),
            combiner_spec=build_prediction_combiner_spec(
                key="combiner__pred-fu-momo2-supplement-learner-only",
                label="Learner only",
                kind="learner_only",
            ),
        ),
        build_prediction_engine_spec(
            key="engine__pred-fu-momo2-supplement-ridge",
            label="2ヶ月モメンタム supplement ridge",
            signal_source_spec=None,
            learner_spec=build_prediction_learner_spec(
                key="learner__pred-fu-momo2-supplement-ridge",
                label="2ヶ月モメンタム supplement ridge learner",
                kind="ridge_regression",
                ridge_alpha=1.0,
            ),
            combiner_spec=build_prediction_combiner_spec(
                key="combiner__pred-fu-momo2-supplement-learner-only",
                label="Learner only",
                kind="learner_only",
            ),
        ),
    ]
    for signal_source_weight, learner_weight in ((0.9, 0.1), (0.7, 0.3), (0.5, 0.5)):
        weight_key = f"{int(signal_source_weight * 100):02d}-{int(learner_weight * 100):02d}"
        engine_specs.append(
            build_prediction_engine_spec(
                key=f"engine__pred-fu-momo2-supplement-momentum-blend-{weight_key}",
                label=(
                    "2ヶ月モメンタム supplement momentum blend "
                    f"{int(signal_source_weight * 100)}/{int(learner_weight * 100)}"
                ),
                signal_source_spec=build_prediction_signal_source_spec(
                    key="signal-source__pred-fu-momo2-supplement-momentum",
                    label="Momentum signal source",
                    kind="derived_feature",
                    feature_key="momentum",
                ),
                learner_spec=build_prediction_learner_spec(
                    key="learner__pred-fu-momo2-supplement-linear-blend",
                    label="2ヶ月モメンタム supplement linear learner",
                    kind="linear_regression",
                ),
                combiner_spec=build_prediction_combiner_spec(
                    key=f"combiner__pred-fu-momo2-supplement-weighted-blend-{weight_key}",
                    label=(
                        "Weighted blend "
                        f"{int(signal_source_weight * 100)}/{int(learner_weight * 100)}"
                    ),
                    kind="weighted_blend",
                    signal_source_weight=signal_source_weight,
                    learner_weight=learner_weight,
                ),
            )
        )
    target_specs = DEFAULT_PREDICTION_TARGET_SPECS
    predictor_specs: list[PredictorSpec] = []
    for engine_spec in engine_specs:
        for target_spec in target_specs:
            horizon_value = _horizon_value(target_spec)
            learner_suffix = engine_spec.learner_spec.kind.replace("_regression", "")
            combiner_suffix = engine_spec.combiner_spec.kind.replace("_only", "")
            signal_source_suffix = ""
            if engine_spec.signal_source_spec is not None:
                if engine_spec.signal_source_spec.kind == "ranking_signal":
                    signal_source_suffix = "-ranking-signal"
                elif engine_spec.signal_source_spec.feature_key is not None:
                    signal_source_suffix = f"-{engine_spec.signal_source_spec.feature_key}"
            combiner_weight_suffix = ""
            if engine_spec.combiner_spec.kind == "weighted_blend":
                signal_source_weight = int((engine_spec.combiner_spec.signal_source_weight or 0) * 100)
                learner_weight = int((engine_spec.combiner_spec.learner_weight or 0) * 100)
                combiner_weight_suffix = f"-{signal_source_weight:02d}{learner_weight:02d}"
            predictor_specs.append(build_predictor_spec(
                key=(
                    f"pred-fu-momo2-supplement-{horizon_value}bar-"
                    f"{learner_suffix}{signal_source_suffix}-{combiner_suffix}{combiner_weight_suffix}"
                ),
                label=f"2ヶ月モメンタム / {target_spec.label} / {engine_spec.label}",
                description=f"2ヶ月モメンタム特徴から{target_spec.label}を推定する",
                timeframe=DEFAULT_DAILY_TIMEFRAME,
                signal_spec=build_signal_spec(
                    key=f"signal__pred-fu-momo2-supplement-{horizon_value}bar",
                    label="2ヶ月モメンタム supplement signal",
                    observation_spec=build_observation_spec(
                        key=f"observation__pred-fu-momo2-supplement-{horizon_value}bar",
                        label="2ヶ月モメンタム supplement observation",
                        tickers=DEFAULT_INVESTMENT_UNIVERSE.tickers,
                        fields=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M.ranking_signal.feature_inputs,
                    ),
                    entity_kind="asset_set",
                    entity_identifiers=DEFAULT_INVESTMENT_UNIVERSE.tickers,
                    output_spec=build_prediction_output_spec(
                        key="output__cross_sectional_score",
                        label="Cross-sectional score",
                        kind="score",
                    ),
                    decision_use_spec=build_decision_use_spec(
                        FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M
                    ),
                ),
                predicted_quantity_spec=build_predicted_quantity_spec(
                    key="quantity__return",
                    label="Return",
                    kind="return",
                ),
                target_spec=target_spec,
                feature_spec=feature_spec,
                engine_spec=engine_spec,
                training_spec=build_training_spec(
                    key=(
                        f"training__pred-fu-momo2-supplement-"
                        f"{engine_spec.learner_spec.kind}-{engine_spec.combiner_spec.kind}"
                    ),
                    label="2ヶ月モメンタム supplement training",
                    fit_mode="expanding",
                    min_train_samples=50,
                ),
            ))
    return predictor_specs


REGISTERED_PREDICTOR_SPECS = _build_registered_predictors()
REGISTERED_PREDICTOR_SPECS_BY_KEY = {
    predictor_spec.key: predictor_spec for predictor_spec in REGISTERED_PREDICTOR_SPECS
}
