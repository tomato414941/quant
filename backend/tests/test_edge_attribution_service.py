from app.edge_attribution_service import build_edge_attribution_strategy_variants
from app.main import DEFAULT_COMPARISON_SPEC
from app.portfolio import get_strategy_signal_execution_contexts
from app.portfolio_domain import (
    build_executable_evaluator_strategy_spec_from_definition,
    thaw_strategy_parameter_value,
)


def find_tilted_hrp_strategy():
    for strategy in DEFAULT_COMPARISON_SPEC.candidate_strategies:
        for signal in strategy.signals:
            score_parameters = thaw_strategy_parameter_value(
                dict(signal.signal_parameters).get("scoreParameters")
            )
            if isinstance(score_parameters, dict) and "tilt_strength" in score_parameters:
                return strategy
    raise AssertionError("No tilted strategy found.")


def get_score_parameters(strategy) -> dict:
    selection_signal = next(
        signal for signal in strategy.signals
        if signal.source_kind == "selection_signal"
    )
    score_parameters = thaw_strategy_parameter_value(
        dict(selection_signal.signal_parameters).get("scoreParameters")
    )
    return dict(score_parameters)


def get_runtime_score_parameters(strategy) -> dict:
    executable_strategy = build_executable_evaluator_strategy_spec_from_definition(strategy)
    selection_contexts, _ = get_strategy_signal_execution_contexts(executable_strategy)
    assert len(selection_contexts) == 1
    selection = selection_contexts[0]["selection"]
    return dict(selection.ranking_signal.score_parameters)


def test_edge_attribution_variants_split_selection_tilt_and_model() -> None:
    base_strategy = find_tilted_hrp_strategy()

    variants = build_edge_attribution_strategy_variants(base_strategy)
    variants_by_key = {variant.strategy_id.split("__")[-1]: variant for variant in variants}

    assert list(variants_by_key) == [
        "universe_equal_weight",
        "selection_pure_equal_weight",
        "selection_tilt_equal_weight",
        "selection_model_no_tilt",
        "strategy_full",
    ]
    assert variants_by_key["selection_pure_equal_weight"].portfolio_model.model_type == "equal_weight"
    assert variants_by_key["selection_tilt_equal_weight"].portfolio_model.model_type == "equal_weight"
    assert variants_by_key["selection_model_no_tilt"].portfolio_model == base_strategy.portfolio_model
    assert variants_by_key["strategy_full"].portfolio_model == base_strategy.portfolio_model

    pure_params = get_score_parameters(variants_by_key["selection_pure_equal_weight"])
    tilt_params = get_score_parameters(variants_by_key["selection_tilt_equal_weight"])
    model_params = get_score_parameters(variants_by_key["selection_model_no_tilt"])
    full_params = get_score_parameters(variants_by_key["strategy_full"])

    assert pure_params["tilt_strength"] == 0.0
    assert pure_params["tilt_shape"] == 0.0
    assert tilt_params["tilt_strength"] > 0.0
    assert "tilt_shape" in tilt_params
    assert model_params["tilt_strength"] == 0.0
    assert model_params["tilt_shape"] == 0.0
    assert full_params == get_score_parameters(base_strategy)


def test_edge_attribution_no_tilt_variants_keep_zero_tilt_at_runtime() -> None:
    base_strategy = find_tilted_hrp_strategy()

    variants = build_edge_attribution_strategy_variants(base_strategy)
    variants_by_key = {variant.strategy_id.split("__")[-1]: variant for variant in variants}

    pure_params = get_runtime_score_parameters(variants_by_key["selection_pure_equal_weight"])
    tilt_params = get_runtime_score_parameters(variants_by_key["selection_tilt_equal_weight"])
    model_params = get_runtime_score_parameters(variants_by_key["selection_model_no_tilt"])
    full_params = get_runtime_score_parameters(variants_by_key["strategy_full"])

    assert pure_params["tilt_strength"] == 0.0
    assert pure_params["tilt_shape"] == 0.0
    assert tilt_params["tilt_strength"] > 0.0
    assert model_params["tilt_strength"] == 0.0
    assert model_params["tilt_shape"] == 0.0
    assert full_params == get_runtime_score_parameters(base_strategy)
