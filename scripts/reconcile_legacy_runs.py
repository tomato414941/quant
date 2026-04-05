from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.dashboard_config import DEFAULT_DASHBOARD_CONFIG
from app.dashboard_service import (  # noqa: E402
    compact_condition_sweep_run,
    serialize_condition_variant,
    serialize_evaluation_context,
)
from app.market_data import fetch_market_universe_bundle  # noqa: E402
from app.portfolio import (  # noqa: E402
    AssetRankingDefinition,
    build_execution_policy_definition,
    build_investment_universe_definition,
    build_portfolio_model_definition,
    build_portfolio_state,
    build_portfolio_strategy_definition,
    build_risk_controls_definition,
    build_strategy_definition,
    evaluate_asset_ranking_definition,
    evaluate_strategy_run,
    serialize_asset_ranking_definition,
    serialize_portfolio_state,
    serialize_strategy_definition,
)
from app.run_store import FileRunResultStore, build_run_definition  # noqa: E402
from app.study_models import CostAssumptions, EvaluationContext, EvaluationSettings  # noqa: E402


RUN_RESULTS_DIR = ROOT / "backend" / "data" / "run_results"

DEFAULT_UNIVERSE_BY_STRATEGY = {
    "full_universe": "all_assets",
    "momentum_top3": "all_assets",
    "dual_momentum_top3": "positive_assets_only",
    "positive_momentum_low_vol_universe": "positive_assets_only",
    "positive_momentum_high_volume_universe": "positive_assets_only",
    "trailing_momentum_low_vol_universe": "all_assets",
    "full_universe_momentum_tilt": "all_assets",
    "full_universe_momentum_low_vol_tilt": "all_assets",
    "full_universe_momentum_macro_tilt": "all_assets",
}

DEFAULT_SCORE_MODEL_BY_STRATEGY = {
    "momentum_top3": "momentum",
    "dual_momentum_top3": "momentum",
    "positive_momentum_low_vol_universe": "momentum",
    "trailing_momentum_low_vol_universe": "trailing_momentum_12m",
    "positive_momentum_high_volume_universe": "volume_strength",
    "full_universe_momentum_tilt": "trailing_momentum_12m",
    "full_universe_momentum_low_vol_tilt": "momentum_low_vol",
    "full_universe_momentum_macro_tilt": "momentum_macro",
}

DEFAULT_FILTER_RULES_BY_STRATEGY = {
    "full_universe": (),
    "full_universe_momentum_tilt": (),
    "full_universe_momentum_low_vol_tilt": (),
    "full_universe_momentum_macro_tilt": (),
    "momentum_top3": ("top_3",),
    "dual_momentum_top3": ("positive_return", "top_3"),
    "trailing_momentum_low_vol_universe": ("positive_return", "low_volatility_half"),
    "positive_momentum_universe": ("positive_return",),
    "positive_momentum_low_vol_universe": ("positive_return", "low_volatility_half"),
    "positive_momentum_high_volume_universe": ("positive_return", "high_volume_half"),
}

REBALANCE_LABELS = {
    "hold": "保有",
    "monthly": "月次",
    "quarterly": "四半期",
    "annual": "年次",
}


def load_payload(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def pick_key(value):
    if isinstance(value, dict):
        return value.get("key") or value.get("label") or value.get("modelType") or value.get("strategyType")
    return value


def normalize_score_model(value, *, strategy_type: str | None = None) -> str | None:
    key = pick_key(value)
    if key in (None, "none"):
        key = DEFAULT_SCORE_MODEL_BY_STRATEGY.get(strategy_type)
    return None if key in (None, "none") else key


def normalize_universe_policy(value, *, strategy_type: str | None = None) -> str | None:
    key = pick_key(value)
    if key is None:
        key = DEFAULT_UNIVERSE_BY_STRATEGY.get(strategy_type)
    return key


def clean_none_dict(items: dict) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((key, value) for key, value in items.items() if value is not None))


def normalize_filter_rules(filters, *, strategy_type: str | None = None) -> tuple[str, ...]:
    explicit = tuple(sorted(key for item in (filters or []) if (key := pick_key(item))))
    if explicit:
        return explicit
    return tuple(sorted(DEFAULT_FILTER_RULES_BY_STRATEGY.get(strategy_type, ())))


def infer_strategy_type_from_ranking_payload(ranking: dict) -> str | None:
    for value in ranking.get("sourceStrategyKeys") or ():
        if "momentum_low_vol" in value:
            return "full_universe_momentum_low_vol_tilt"
        if "momentum_macro" in value:
            return "full_universe_momentum_macro_tilt"
        if "momentum_tilt" in value or "momo12" in value:
            return "full_universe_momentum_tilt"
    key = ranking.get("key") or ""
    if "momentum_low_vol" in key:
        return "full_universe_momentum_low_vol_tilt"
    if "momentum_macro" in key:
        return "full_universe_momentum_macro_tilt"
    if "momentum_tilt" in key or "momo12" in key:
        return "full_universe_momentum_tilt"
    return None


def normalize_features(features, *, ranking_key: str | None = None) -> tuple[str, ...]:
    if ranking_key is None:
        return ()
    values = tuple(sorted(features or []))
    return values or ("close",)


def normalize_ranking_params(score_model_key: str | None, params: dict | None) -> tuple[tuple[str, float], ...]:
    normalized = dict(params or {})
    if score_model_key in {
        "momentum",
        "volume_strength",
        "trailing_momentum_12m",
        "momentum_low_vol",
        "momentum_macro",
    }:
        normalized.setdefault("window_days", 252.0)
    if score_model_key == "momentum_low_vol":
        normalized.setdefault("momentum_weight", 0.7)
        normalized.setdefault("low_vol_weight", 0.3)
    if score_model_key == "momentum_macro":
        normalized.setdefault("momentum_weight", 0.85)
        normalized.setdefault("macro_weight", 0.15)
    keep = {}
    for key in ["window_days", "momentum_weight", "low_vol_weight", "macro_weight"]:
        if key in normalized:
            keep[key] = normalized[key]
    return clean_none_dict(keep)


def normalize_tilt_params(params: dict | None) -> tuple[tuple[str, float], ...]:
    normalized = dict(params or {})
    keep = {}
    for key in ["tilt_shape", "tilt_strength"]:
        if key in normalized:
            keep[key] = normalized[key]
    return clean_none_dict(keep)


def current_family(strategy: dict) -> tuple[str | None, str | None]:
    extensions = strategy.get("extensions") or {}
    legacy_strategy_type = extensions.get("legacyStrategyType")
    if legacy_strategy_type:
        return legacy_strategy_type, DEFAULT_UNIVERSE_BY_STRATEGY.get(legacy_strategy_type)

    strategy_id = strategy.get("strategyId", "")
    if strategy_id.startswith("stg-top3-"):
        return "momentum_top3", "all_assets"
    if strategy_id.startswith("stg-dualtop3-"):
        return "dual_momentum_top3", "positive_assets_only"
    if strategy_id.startswith("stg-poslowvol-"):
        return "positive_momentum_low_vol_universe", "positive_assets_only"
    if strategy_id.startswith("stg-posvol-"):
        return "positive_momentum_high_volume_universe", "positive_assets_only"
    if strategy_id.startswith("stg-trailmomlowvol-"):
        return "trailing_momentum_low_vol_universe", "all_assets"
    if strategy_id.startswith("stg-fu-momolv"):
        return "full_universe_momentum_low_vol_tilt", "all_assets"
    if strategy_id.startswith("stg-fu-momomac"):
        return "full_universe_momentum_macro_tilt", "all_assets"
    if strategy_id.startswith("stg-fu-momo") or strategy_id.startswith("stg-etf-momo") or strategy_id.startswith("momentum_top__"):
        return "full_universe_momentum_tilt", "all_assets"
    if strategy_id.startswith("stg-fu-"):
        return "full_universe", "all_assets"
    return strategy_id or None, None


def old_signature(run_definition: dict) -> tuple:
    raw_run_kind = run_definition.get("runKind") or "portfolio_comparison"
    run_kind = "portfolio_comparison" if raw_run_kind == "parameter_sweep" else raw_run_kind
    candidate = run_definition.get("candidate") or {}

    if run_kind == "ranking_evaluation":
        ranking = candidate.get("ranking") or {}
        strategy_type = infer_strategy_type_from_ranking_payload(ranking)
        score_model = normalize_score_model(ranking.get("rankingModel"), strategy_type=strategy_type)
        score_parameters = ranking.get("scoreParameters") or {}
        tilt_parameters = {}
        filter_rules = ranking.get("filterRules")
        feature_inputs = ranking.get("featureInputs")
        universe_policy_value = ranking.get("universePolicy")
    else:
        strategy = candidate.get("strategy") or {}
        strategy_type = pick_key(strategy.get("strategyType"))
        score_model = normalize_score_model(strategy.get("scoreModel"), strategy_type=strategy_type)
        score_parameters = strategy.get("scoreParameters") or {}
        tilt_parameters = strategy.get("scoreParameters") or {}
        filter_rules = strategy.get("filterRules")
        feature_inputs = strategy.get("featureInputs")
        universe_policy_value = strategy.get("universePolicy")

    execution_model = run_definition.get("executionModel") or {}
    dataset_spec = run_definition.get("datasetSpec") or {}
    backtest_config = run_definition.get("backtestConfig") or {}
    evaluation_context = run_definition.get("evaluationContext") or {}
    evaluation_settings = evaluation_context.get("evaluationSettings") or {}
    cost_assumptions = evaluation_context.get("costAssumptions") or {}
    portfolio_model = None if run_kind == "ranking_evaluation" else pick_key(candidate.get("portfolioModel"))
    execution_frequency = None if run_kind == "ranking_evaluation" else pick_key(execution_model.get("rebalanceFrequency"))
    max_investment_pct = None if run_kind == "ranking_evaluation" else backtest_config.get("maxInvestmentPct")
    max_weight_pct = None if run_kind == "ranking_evaluation" else backtest_config.get("maxWeightPct")
    return (
        run_kind,
        strategy_type,
        normalize_universe_policy(universe_policy_value, strategy_type=strategy_type),
        score_model,
        normalize_ranking_params(score_model, score_parameters),
        normalize_tilt_params(tilt_parameters),
        normalize_features(feature_inputs, ranking_key=score_model),
        normalize_filter_rules(filter_rules, strategy_type=strategy_type),
        portfolio_model,
        execution_frequency,
        pick_key(dataset_spec.get("period")),
        max_investment_pct,
        max_weight_pct,
        backtest_config.get("splitRatioPct"),
        pick_key(backtest_config.get("benchmark") or evaluation_settings.get("benchmark")),
        execution_model.get("commissionPct") if "commissionPct" in execution_model else cost_assumptions.get("commissionPct"),
        execution_model.get("slippagePct") if "slippagePct" in execution_model else cost_assumptions.get("slippagePct"),
    )


def current_signature(run_definition: dict) -> tuple:
    run_kind = run_definition.get("runKind") or "portfolio_comparison"
    strategy = run_definition.get("strategy") or {}

    if run_kind == "ranking_evaluation":
        ranking = strategy.get("ranking") or {}
        family = infer_strategy_type_from_ranking_payload(ranking)
        universe_policy = normalize_universe_policy(ranking.get("universePolicy"), strategy_type=family)
        ranking_model = ranking.get("rankingModel") or {}
        ranking_score_parameters = ranking.get("scoreParameters") or {}
        tilt_rule = {}
        risk_controls = {}
        feature_inputs = ranking.get("featureInputs")
        filter_rules = ranking.get("filterRules")
        portfolio_model = None
        execution_frequency = None
    else:
        family, universe_policy = current_family(strategy)
        components = strategy.get("components") or {}
        core = components.get("core") or {}
        optional = components.get("optional") or {}
        ranking_model = optional.get("assetRankingModel") or {}
        tilt_rule = optional.get("tiltRule") or {}
        risk_controls = optional.get("riskControls") or {}
        feature_inputs = optional.get("featureInputs")
        filter_rules = optional.get("filterRules")
        portfolio_model = pick_key(core.get("portfolioModel"))
        execution_frequency = pick_key(core.get("executionPolicy", {}).get("rebalanceFrequency"))

    evaluation_context = run_definition.get("evaluationContext") or {}
    dataset_context = evaluation_context.get("datasetContext") or {}
    evaluation_settings = evaluation_context.get("evaluationSettings") or {}
    cost_assumptions = evaluation_context.get("costAssumptions") or {}
    ranking_key = normalize_score_model(ranking_model, strategy_type=family)
    if run_kind == "ranking_evaluation":
        ranking_params = ranking_score_parameters
    else:
        ranking_params = ranking_model.get("parameters") or {}
    tilt_params = tilt_rule.get("parameters") or {}
    return (
        run_kind,
        family,
        universe_policy,
        ranking_key,
        normalize_ranking_params(
            ranking_key,
            {
                "window_days": ranking_params.get("windowDays"),
                "momentum_weight": ranking_params.get("momentumWeight"),
                "low_vol_weight": ranking_params.get("lowVolWeight"),
                "macro_weight": ranking_params.get("macroWeight"),
            },
        ),
        clean_none_dict(
            {
                "tilt_shape": tilt_params.get("shape"),
                "tilt_strength": tilt_params.get("strength"),
            }
        ),
        normalize_features(feature_inputs, ranking_key=ranking_key),
        normalize_filter_rules(filter_rules, strategy_type=family),
        portfolio_model,
        execution_frequency,
        pick_key(dataset_context.get("period")),
        risk_controls.get("maxInvestmentPct"),
        risk_controls.get("maxWeightPct"),
        evaluation_settings.get("splitRatioPct"),
        pick_key(evaluation_settings.get("benchmark")),
        cost_assumptions.get("commissionPct"),
        cost_assumptions.get("slippagePct"),
    )


def signature_representable(signature: tuple) -> bool:
    run_kind, strategy_type, _, _, _, _, _, _, portfolio_model, rebalance, period, *_ = signature
    if run_kind == "ranking_evaluation":
        return strategy_type in {
            "full_universe_momentum_tilt",
            "full_universe_momentum_low_vol_tilt",
            "full_universe_momentum_macro_tilt",
            "momentum_top3",
            "dual_momentum_top3",
            "positive_momentum_low_vol_universe",
            "positive_momentum_high_volume_universe",
            "trailing_momentum_low_vol_universe",
        } and period in {"3y", "10y"}
    return (
        run_kind in {"dashboard", "condition_sweep", "portfolio_comparison"}
        and strategy_type in {
            "full_universe",
            "momentum_top3",
            "dual_momentum_top3",
            "positive_momentum_low_vol_universe",
            "positive_momentum_high_volume_universe",
            "trailing_momentum_low_vol_universe",
            "full_universe_momentum_tilt",
            "full_universe_momentum_low_vol_tilt",
            "full_universe_momentum_macro_tilt",
        }
        and portfolio_model in {
            "equal_weight",
            "risk_budgeting",
            "minimum_variance",
            "hierarchical_risk_parity",
            "mean_risk_utility",
            "mean_risk_utility_conservative",
        }
        and rebalance in {"annual", "quarterly", "monthly", "hold"}
        and period in {"3y", "10y"}
    )


def build_strategy_id(signature: tuple) -> str:
    digest = hashlib.sha1(repr(signature[:14]).encode("utf-8")).hexdigest()[:12]
    return f"stg-lgcy-{digest}"


def build_strategy_from_old_run(run_definition: dict):
    candidate = run_definition.get("candidate") or {}
    strategy = candidate.get("strategy") or {}
    strategy_type = pick_key(strategy.get("strategyType"))
    score_model = normalize_score_model(strategy.get("scoreModel"), strategy_type=strategy_type)
    score_parameters = dict(strategy.get("scoreParameters") or {})
    if score_model in {
        "momentum",
        "volume_strength",
        "trailing_momentum_12m",
        "momentum_low_vol",
        "momentum_macro",
    }:
        score_parameters.setdefault("window_days", 252.0)
    if score_model == "momentum_low_vol":
        score_parameters.setdefault("momentum_weight", 0.7)
        score_parameters.setdefault("low_vol_weight", 0.3)
    if score_model == "momentum_macro":
        score_parameters.setdefault("momentum_weight", 0.85)
        score_parameters.setdefault("macro_weight", 0.15)

    dataset_spec = run_definition.get("datasetSpec") or {}
    tickers = dataset_spec.get("tickers") or [
        row["asset"]
        for row in (run_definition.get("portfolioState") or {}).get("weights", [])
        if row.get("asset") != "CASH"
    ]
    investment_universe = build_investment_universe_definition(
        tickers=tickers,
        key=f"legacy_universe_{hashlib.sha1(','.join(tickers).encode('utf-8')).hexdigest()[:10]}",
        label=f"Legacy universe ({len(tickers)} assets)",
    )

    portfolio_strategy = build_portfolio_strategy_definition(
        strategy_type=strategy_type,
        key=strategy.get("key") or strategy_type,
        label=strategy.get("label") or None,
        description=strategy.get("description") or None,
        score_parameters=score_parameters or None,
    )
    execution_model = run_definition.get("executionModel") or {}
    execution_policy = build_execution_policy_definition(
        key=pick_key(execution_model.get("key")) or pick_key(execution_model.get("rebalanceFrequency")) or "annual",
        label=pick_key(execution_model.get("label")) or REBALANCE_LABELS.get(pick_key(execution_model.get("rebalanceFrequency")), "保有"),
        entry=execution_model.get("entry") or "train_once_then_periodic_rebalance",
        rebalance_frequency=pick_key(execution_model.get("rebalanceFrequency")) or "annual",
    )
    backtest_config = run_definition.get("backtestConfig") or {}
    risk_controls = build_risk_controls_definition(
        max_investment_ratio=(backtest_config.get("maxInvestmentPct") or 100.0) / 100.0,
        max_weight=(backtest_config.get("maxWeightPct") / 100.0) if backtest_config.get("maxWeightPct") is not None else None,
    )
    strategy_definition = build_strategy_definition(
        investment_universe_definition=investment_universe,
        data_resolution="daily",
        selection_definition=portfolio_strategy,
        portfolio_model_definition=build_portfolio_model_definition(
            pick_key(candidate.get("portfolioModel"))
        ),
        execution_policy_definition=execution_policy,
        risk_controls_definition=risk_controls,
        strategy_id=build_strategy_id(old_signature(run_definition)),
        label=(strategy.get("label") or portfolio_strategy.label),
        description=strategy.get("description") or portfolio_strategy.description,
        hypothesis=None,
        extensions={"legacyStrategyType": strategy_type or "unknown"},
    )
    return strategy_definition


def build_ranking_definition_from_old_run(run_definition: dict) -> AssetRankingDefinition:
    candidate = run_definition.get("candidate") or {}
    ranking = candidate.get("ranking") or {}
    strategy_type = infer_strategy_type_from_ranking_payload(ranking)
    if strategy_type is None:
        raise ValueError("Unable to infer legacy ranking family.")

    score_parameters = dict(ranking.get("scoreParameters") or {})
    normalized_score_parameters: dict[str, float] = {}
    if "windowDays" in score_parameters:
        normalized_score_parameters["window_days"] = float(score_parameters["windowDays"])
    if "momentumWeight" in score_parameters:
        normalized_score_parameters["momentum_weight"] = float(score_parameters["momentumWeight"])
    if "lowVolWeight" in score_parameters:
        normalized_score_parameters["low_vol_weight"] = float(score_parameters["lowVolWeight"])
    if "macroWeight" in score_parameters:
        normalized_score_parameters["macro_weight"] = float(score_parameters["macroWeight"])

    dataset_spec = run_definition.get("datasetSpec") or {}
    tickers = dataset_spec.get("tickers") or []
    investment_universe = build_investment_universe_definition(
        tickers=tickers,
        key=f"legacy_universe_{hashlib.sha1(','.join(tickers).encode('utf-8')).hexdigest()[:10]}",
        label=(
            ranking.get("investmentUniverse", {}).get("label")
            or f"Legacy universe ({len(tickers)} assets)"
        ),
    )
    strategy_definition = build_portfolio_strategy_definition(
        strategy_type=strategy_type,
        key=ranking.get("key") or strategy_type,
        label=ranking.get("label") or None,
        description=ranking.get("description") or None,
        score_parameters=normalized_score_parameters or None,
    )
    return AssetRankingDefinition(
        key=ranking.get("key") or f"ranking__legacy__{strategy_type}",
        label=ranking.get("label") or strategy_definition.label,
        description=ranking.get("description") or strategy_definition.description,
        investment_universe_definition=investment_universe,
        strategy_definition=strategy_definition,
        source_strategy_keys=tuple(ranking.get("sourceStrategyKeys") or ()),
        source_strategy_labels=tuple(ranking.get("sourceStrategyLabels") or ()),
    )


def build_evaluation_context_from_old_run(run_definition: dict) -> EvaluationContext:
    backtest_config = run_definition.get("backtestConfig") or {}
    execution_model = run_definition.get("executionModel") or {}
    evaluation_context = run_definition.get("evaluationContext") or {}
    cost_assumptions = evaluation_context.get("costAssumptions") or {}
    return EvaluationContext(
        evaluation_settings=EvaluationSettings(
            split_ratio=(backtest_config.get("splitRatioPct") or 70.0) / 100.0,
            initial_capital=float(backtest_config.get("initialCapital") or 10_000),
            benchmark=backtest_config.get("benchmark") or "equal_weight_buy_and_hold_with_cash",
        ),
        cost_assumptions=CostAssumptions(
            commission_pct=float(
                execution_model.get("commissionPct")
                if execution_model.get("commissionPct") is not None
                else cost_assumptions.get("commissionPct", 0.0)
            ),
            slippage_pct=float(
                execution_model.get("slippagePct")
                if execution_model.get("slippagePct") is not None
                else cost_assumptions.get("slippagePct", 0.0)
            ),
        ),
    )


def build_portfolio_state_from_old_run(run_definition: dict):
    rows = (run_definition.get("portfolioState") or {}).get("weights", [])
    cash_weight = 0.0
    current_weights: dict[str, float] = {}
    for row in rows:
        asset = str(row.get("asset"))
        weight_pct = float(row.get("weightPct", 0.0))
        if asset == "CASH":
            cash_weight = weight_pct / 100.0
            continue
        current_weights[asset] = weight_pct / 100.0
    return build_portfolio_state(current_weights=current_weights, cash_weight=cash_weight)


def condition_variant_from_signature(signature: tuple):
    _, _, _, _, _, _, _, _, _, _, _, max_investment_pct, max_weight_pct, _, _, commission_pct, _ = signature
    max_investment_ratio = (max_investment_pct or 100.0) / 100.0
    max_weight = None if max_weight_pct is None else max_weight_pct / 100.0
    cap_label = "上限なし" if max_weight_pct is None else f"{max_weight_pct:.1f}%上限"
    key = (
        f"legacy_fee_{str(commission_pct).replace('.', '_')}"
        f"__invest_{int(round(max_investment_ratio * 100))}"
        f"__cap_{'none' if max_weight is None else str(max_weight_pct).replace('.', '_')}"
    )
    label = (
        f"手数料 {commission_pct:.2f}% / 投資 {int(round(max_investment_ratio * 100))}%"
        f" / {cap_label}"
    )
    from app.study_models import ConditionVariant  # imported late to avoid circular style noise

    return ConditionVariant(
        key=key,
        label=label,
        commission_pct=float(commission_pct),
        max_investment_ratio=max_investment_ratio,
        max_weight=max_weight,
    )


def replay_signature(
    *,
    representative_run_definition: dict,
    run_store: FileRunResultStore,
    market_cache: dict[tuple[tuple[str, ...], str], tuple[dict, dict]],
) -> bool:
    signature = old_signature(representative_run_definition)
    if not signature_representable(signature):
        return False

    dataset_spec = representative_run_definition.get("datasetSpec") or {}
    period = pick_key(dataset_spec.get("period")) or "10y"
    tickers = tuple(dataset_spec.get("tickers") or [
        row["asset"]
        for row in (representative_run_definition.get("portfolioState") or {}).get("weights", [])
        if row.get("asset") != "CASH"
    ])
    cache_key = (tickers, period)
    if cache_key not in market_cache:
        market_cache[cache_key] = fetch_market_universe_bundle(tickers=list(tickers), period=period)
    market_bundle, dataset_metadata = market_cache[cache_key]

    evaluation_context = build_evaluation_context_from_old_run(representative_run_definition)
    portfolio_state = build_portfolio_state_from_old_run(representative_run_definition)
    run_kind = signature[0]
    strategy_definition = None
    ranking_definition = None
    if run_kind == "ranking_evaluation":
        ranking_definition = build_ranking_definition_from_old_run(representative_run_definition)
    else:
        strategy_definition = build_strategy_from_old_run(representative_run_definition)

    fake_study = replace(
        DEFAULT_DASHBOARD_CONFIG,
        evaluation_context=evaluation_context,
        initial_portfolio_state=portfolio_state,
    )

    serialized_evaluation_context = serialize_evaluation_context(
        fake_study,
        dataset_metadata,
        period_override=period,
    )
    generation = {
        "method": "legacy_replay",
        "batchKey": "manual_v1_v5_reconcile_v1",
        "spec": {"sourceLogicVersions": ["v1", "v2", "v3", "v4", "v5"]},
    }

    if run_kind == "ranking_evaluation":
        serialized_strategy = {"ranking": serialize_asset_ranking_definition(ranking_definition)}
    else:
        serialized_strategy = serialize_strategy_definition(strategy_definition)

    run_definition = build_run_definition(
        run_kind=run_kind,
        strategy=serialized_strategy,
        dataset_spec={"period": period},
        evaluation_context=serialized_evaluation_context,
        portfolio_state=serialize_portfolio_state(portfolio_state),
        dataset_metadata=dataset_metadata,
        generation=generation,
    )
    if run_store.load(run_definition) is not None:
        return True

    if run_kind == "ranking_evaluation":
        strategy_universe = [
            asset
            for asset in ranking_definition.investment_universe_definition.tickers
            if asset in market_bundle["closes"].columns
        ]
        closes = market_bundle["closes"][strategy_universe]
        returns = closes.pct_change().dropna()
        aligned_volumes = None
        if market_bundle["volumes"] is not None:
            aligned_volumes = market_bundle["volumes"][strategy_universe].reindex(returns.index)
        result = evaluate_asset_ranking_definition(
            returns=returns,
            volumes=aligned_volumes,
            split_ratio=evaluation_context.evaluation_settings.split_ratio,
            ranking_definition=ranking_definition,
        )
    else:
        run = evaluate_strategy_run(
            closes=market_bundle["closes"],
            volumes=market_bundle["volumes"],
            strategy_definition=strategy_definition,
            initial_capital=evaluation_context.evaluation_settings.initial_capital,
            split_ratio=evaluation_context.evaluation_settings.split_ratio,
            transaction_cost=evaluation_context.cost_assumptions.commission_pct / 100.0,
            portfolio_state=portfolio_state,
        )
        if run_kind == "condition_sweep":
            condition_variant = condition_variant_from_signature(signature)
            result = compact_condition_sweep_run(
                run=run,
                key=f"{strategy_definition.key}__{condition_variant.key}",
                condition_variant=serialize_condition_variant(condition_variant),
            )
        else:
            result = run

    if run_kind == "condition_sweep":
        condition_variant = condition_variant_from_signature(signature)
    run_store.save(run_definition, result)
    return True


def collect_rows():
    old_rows: list[tuple[Path, tuple, dict]] = []
    current_signatures: set[tuple] = set()
    for path in RUN_RESULTS_DIR.glob("*.json"):
        payload = load_payload(path)
        if not payload:
            continue
        run_definition = payload.get("runDefinition") or {}
        logic_version = run_definition.get("logicVersion")
        if logic_version == "v9":
            current_signatures.add(current_signature(run_definition))
        elif logic_version in {"v1", "v2", "v3", "v4", "v5"}:
            old_rows.append((path, old_signature(run_definition), run_definition))
    return old_rows, current_signatures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    old_rows, current_signatures = collect_rows()
    grouped_old: dict[tuple, list[tuple[Path, dict]]] = {}
    for path, signature, run_definition in old_rows:
        grouped_old.setdefault(signature, []).append((path, run_definition))

    matched_before = {sig for sig in grouped_old if sig in current_signatures}
    unmatched_before = [sig for sig in grouped_old if sig not in current_signatures]
    representable = [sig for sig in unmatched_before if signature_representable(sig)]
    irreconcilable = [sig for sig in unmatched_before if not signature_representable(sig)]

    print(f"old_rows={len(old_rows)}")
    print(f"old_unique={len(grouped_old)}")
    print(f"matched_before={len(matched_before)}")
    print(f"unmatched_before={len(unmatched_before)}")
    print(f"representable_unmatched={len(representable)}")
    print(f"irreconcilable_unmatched={len(irreconcilable)}")

    if not args.apply:
        print("dry_run_only=true")
        return 0

    run_store = FileRunResultStore(RUN_RESULTS_DIR)
    market_cache: dict[tuple[tuple[str, ...], str], tuple[dict, dict]] = {}
    replayed = 0
    replay_failed = 0
    for signature in representable:
        representative_run_definition = grouped_old[signature][0][1]
        try:
            if replay_signature(
                representative_run_definition=representative_run_definition,
                run_store=run_store,
                market_cache=market_cache,
            ):
                replayed += 1
            else:
                replay_failed += 1
        except Exception:
            replay_failed += 1

    _, current_signatures_after = collect_rows()
    deletable_paths: list[Path] = []
    still_unmatched = 0
    for signature, rows in grouped_old.items():
        if signature in current_signatures_after:
            deletable_paths.extend(path for path, _ in rows)
        else:
            still_unmatched += 1

    for path in deletable_paths:
        path.unlink(missing_ok=True)

    print(f"replayed={replayed}")
    print(f"replay_failed={replay_failed}")
    print(f"deleted_old_rows={len(deletable_paths)}")
    print(f"still_unmatched_unique={still_unmatched}")

    family_counter = Counter(sig[1] for sig in grouped_old if sig not in current_signatures_after)
    print("remaining_families=")
    for family, count in family_counter.most_common(20):
        print(f"  {family}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
