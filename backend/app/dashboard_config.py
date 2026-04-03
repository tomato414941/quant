from __future__ import annotations

from app.strategy import build_strategy_definition


DEFAULT_DASHBOARD_CONFIG = {
    "study_id": "short_term_reaction_spy_2y",
    "title": "SPY短期反応の戦略比較",
    "question": "同じSPY日足データに対して、短期逆張りと短期順張りのどちらが安定するか",
    "dataset_spec": {
        "ticker": "SPY",
        "period": "2y",
        "frequency": "daily",
    },
    "execution_model": {
        "entry": "close_to_close",
        "commission_pct": 0.1,
        "slippage_pct": 0.0,
    },
    "backtest_config": {
        "split_ratio": 0.7,
        "initial_capital": 10_000,
        "benchmark": "buy_and_hold",
    },
    "strategy_definitions": [
        build_strategy_definition(
            engine="mean_reversion",
            threshold=0.03,
            holding_days=1,
            key="mean_reversion_3_1",
            label="逆張り 3.0% / 1日",
            hypothesis="大きく下げた直後の短期反発を拾う",
        ),
        build_strategy_definition(
            engine="mean_reversion",
            threshold=0.02,
            holding_days=2,
            key="mean_reversion_2_2",
            label="逆張り 2.0% / 2日",
            hypothesis="浅めの下落を少し長く持つと反発を取りやすい",
        ),
        build_strategy_definition(
            engine="momentum",
            threshold=0.03,
            holding_days=1,
            key="momentum_3_1",
            label="上昇継続 3.0% / 1日",
            hypothesis="強い上昇の翌日は短期で勢いが続く",
        ),
    ],
}
