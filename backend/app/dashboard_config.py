from __future__ import annotations

from app.portfolio import (
    build_portfolio_model_definition,
    build_portfolio_strategy_definition,
)


DEFAULT_DASHBOARD_CONFIG = {
    "study_id": "etf_portfolio_models_5y",
    "title": "ETF戦略 x ポートフォリオ構築の比較",
    "question": "同じETFユニバースに対して、どの戦略と配分法の組み合わせが最も安定するか",
    "dataset_spec": {
        "tickers": ["SPY", "QQQ", "IWM", "TLT", "GLD"],
        "period": "5y",
        "frequency": "daily",
    },
    "execution_model": {
        "entry": "train_once_hold_static_weights",
        "commission_pct": 0.1,
        "slippage_pct": 0.0,
    },
    "backtest_config": {
        "split_ratio": 0.7,
        "initial_capital": 10_000,
        "max_investment_ratio": 0.85,
        "benchmark": "equal_weight_buy_and_hold_with_cash",
    },
    "strategy_definitions": [
        build_portfolio_strategy_definition(
            strategy_type="full_universe",
            key="full_universe",
            label="全資産",
            description="全ETFを候補にして配分する",
        ),
        build_portfolio_strategy_definition(
            strategy_type="momentum_top3",
            key="momentum_top3",
            label="モメンタム上位3",
            description="学習期間で強かった上位3ETFに絞って配分する",
        ),
    ],
    "portfolio_models": [
        build_portfolio_model_definition(
            model_type="equal_weight",
            key="equal_weight",
            label="等金額配分",
            description="全ETFを同じ比率で持つ",
        ),
        build_portfolio_model_definition(
            model_type="risk_budgeting",
            key="risk_budgeting",
            label="リスク予算配分",
            description="各ETFのリスク寄与が近づくように配分する",
        ),
        build_portfolio_model_definition(
            model_type="minimum_variance",
            key="minimum_variance",
            label="最小分散",
            description="全体の分散が最小になるように配分する",
        ),
    ],
}
