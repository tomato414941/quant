from __future__ import annotations

from app.spec_registry import (
    DEFAULT_INVESTMENT_UNIVERSE,
    FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
)
from app.portfolio import (
    PREDICTION_FEATURE_NAMES,
    PredictorSpec,
    build_feature_spec,
    build_prediction_model_spec,
    build_prediction_target_spec,
    build_predictor_spec,
    extract_ranking_score_parameters,
)
from app.timeframe_models import DEFAULT_DAILY_TIMEFRAME


def _build_registered_predictors() -> list[PredictorSpec]:
    ranking_parameters = extract_ranking_score_parameters(FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M)
    return [
        build_predictor_spec(
            key="pred-fu-momo2-supplement-5bar-linear",
            label="2ヶ月モメンタム / 5bar予測線形",
            description="2ヶ月モメンタム特徴から次の5bar超過収益を推定する",
            timeframe=DEFAULT_DAILY_TIMEFRAME,
            investment_universe=DEFAULT_INVESTMENT_UNIVERSE,
            feature_spec=build_feature_spec(
                key="features__pred-fu-momo2-supplement-5bar-linear",
                label="2ヶ月モメンタム supplement features",
                inputs=tuple(FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M.feature_inputs),
                parameters={
                    **ranking_parameters,
                    "featureKeys": PREDICTION_FEATURE_NAMES,
                },
            ),
            model_spec=build_prediction_model_spec(
                key="model__pred-fu-momo2-supplement-5bar-linear",
                kind="linear_regression",
                label="2ヶ月モメンタム supplement linear",
                parameters={
                    "scoreModelKind": FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M.score_model.kind,
                    "fitMode": "expanding",
                    "minTrainSamples": 50,
                    **ranking_parameters,
                },
            ),
            target_spec=build_prediction_target_spec(
                key="next_5bar_excess_return",
                label="次の5bar超過収益",
                kind="forward_excess_return",
                horizon_spec={"unit": "bars", "value": 5},
            ),
            selection=FULL_UNIVERSE_MOMENTUM_TILT_WEAK_TOP_2M,
            source_strategy_keys=("stg-fu-momo2-top035-pred5blend-hrp-month",),
            source_strategy_labels=("全資産モメンタム傾斜 上位優遇 2ヶ月 + 5bar予測補助 × HRP × 月次",),
        )
    ]


REGISTERED_PREDICTOR_SPECS = _build_registered_predictors()
REGISTERED_PREDICTOR_SPECS_BY_KEY = {
    predictor_spec.key: predictor_spec for predictor_spec in REGISTERED_PREDICTOR_SPECS
}
