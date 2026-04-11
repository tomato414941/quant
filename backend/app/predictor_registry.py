from __future__ import annotations

from app.strategy_registry import (
    DEFAULT_INVESTMENT_UNIVERSE,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
)
from app.portfolio import (
    PREDICTION_FEATURE_NAMES,
    PredictorSpec,
    build_derived_feature_spec,
    build_feature_spec,
    build_feature_input_spec,
    build_prediction_calibration_spec,
    build_prediction_model_spec,
    build_prediction_objective_spec,
    build_prediction_target_spec,
    build_predictor_spec,
    build_ranking_feature_recipe_spec,
    build_training_spec,
    extract_ranking_score_parameters,
)
from app.timeframe_models import DEFAULT_DAILY_TIMEFRAME


def _horizon_value(target_spec) -> int:
    horizon = dict(target_spec.horizon_spec)
    return int(horizon["value"])


DEFAULT_PREDICTION_TARGET_SPECS = [
    build_prediction_target_spec(
        key="next_5bar_excess_return",
        label="次の5bar超過収益",
        kind="forward_excess_return",
        horizon_spec={"unit": "bars", "value": 5},
    ),
    build_prediction_target_spec(
        key="next_10bar_excess_return",
        label="次の10bar超過収益",
        kind="forward_excess_return",
        horizon_spec={"unit": "bars", "value": 10},
    ),
]


def _build_registered_predictors() -> list[PredictorSpec]:
    ranking_parameters = extract_ranking_score_parameters(FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M)
    feature_spec = build_feature_spec(
        key="features__pred-fu-momo2-supplement",
        label="2ヶ月モメンタム supplement features",
        feature_inputs=tuple(
            build_feature_input_spec(key=feature_input)
            for feature_input in FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M.feature_inputs
        ),
        derived_features=tuple(
            build_derived_feature_spec(key=feature_key)
            for feature_key in PREDICTION_FEATURE_NAMES
        ),
        ranking_feature_recipe=build_ranking_feature_recipe_spec(FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M),
    )
    model_specs = [
        build_prediction_model_spec(
            key="model__pred-fu-momo2-supplement-linear",
            kind="linear_regression",
            label="2ヶ月モメンタム supplement linear",
        ),
        build_prediction_model_spec(
            key="model__pred-fu-momo2-supplement-ridge",
            kind="ridge_regression",
            label="2ヶ月モメンタム supplement ridge",
            ridge_alpha=1.0,
        ),
    ]
    target_specs = DEFAULT_PREDICTION_TARGET_SPECS
    predictor_specs: list[PredictorSpec] = []
    for model_spec in model_specs:
        for target_spec in target_specs:
            horizon_value = _horizon_value(target_spec)
            model_suffix = model_spec.kind.replace("_regression", "")
            predictor_specs.append(build_predictor_spec(
                key=f"pred-fu-momo2-supplement-{horizon_value}bar-{model_suffix}",
                label=f"2ヶ月モメンタム / {target_spec.label} / {model_spec.label}",
                description=f"2ヶ月モメンタム特徴から{target_spec.label}を推定する",
                timeframe=DEFAULT_DAILY_TIMEFRAME,
                investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
                objective_spec=build_prediction_objective_spec(
                    key="objective__cross_sectional_alpha",
                    label="Cross-sectional alpha",
                    kind="cross_sectional_alpha_forecast",
                ),
                target_spec=target_spec,
                feature_spec=feature_spec,
                model_spec=model_spec,
                training_spec=build_training_spec(
                    key=f"training__pred-fu-momo2-supplement-{model_spec.kind}",
                    label="2ヶ月モメンタム supplement training",
                    fit_mode="expanding",
                    min_train_samples=50,
                ),
                calibration_spec=build_prediction_calibration_spec(
                    key="calibration__cross_sectional_standard_score",
                    label="Cross-sectional standardized score",
                    kind="standardized_score",
                    parameters={"scope": "cross_sectional"},
                ),
                source_strategy_keys=("stg-fu-momo2-top035-pred5blend-hrp-month",),
                source_strategy_labels=("全資産モメンタム傾斜 上位優遇 2ヶ月 + 5bar予測補助 × HRP × 月次",),
            ))
    return predictor_specs


REGISTERED_PREDICTOR_SPECS = _build_registered_predictors()
REGISTERED_PREDICTOR_SPECS_BY_KEY = {
    predictor_spec.key: predictor_spec for predictor_spec in REGISTERED_PREDICTOR_SPECS
}
