# Prediction Surface

この文書は、予測機能を実装詳細に引きずられずに整理するための公開用メモである。

公開版では、予測対象を整理するための一般的な軸だけを記載する。具体的な探索アイデア、未検証のモデル構想、研究途中の優先順位は載せない。

## Purpose

局所的な予測問題に閉じず、予測対象を分解して考える。

この文書でやりたいこと:

- 予測対象を分類するための軸を明示する
- 新しい予測問題を追加するときに、何が増えたのかを切り分ける
- model名から入らず、意思決定上の使い道から整理する

## Core View

予測は単一の数値だけではなく、意思決定に必要な不確実性を扱うための入力である。

少なくとも次の観点を分けて考える。

- 何を予測するか
- どのhorizonで予測するか
- どの資産集合に適用するか
- どの意思決定に使うか
- どの形式で出力するか

## Predictive Axes

### Predicted Quantity

何を予測するか。

例:

- return
- volatility
- drawdown risk
- market regime
- liquidity condition

### Horizon

どれくらい先を予測するか。

例:

- short horizon
- medium horizon
- long horizon

### Decision Use

予測を何に使うか。

例:

- ranking
- weight tilt
- exposure gate
- risk budget adjustment
- strategy comparison

### Candidate Set

どの集合の中で比較・適用するか。

例:

- broad asset set
- filtered asset set
- defensive subset

### Output Form

どの形で出すか。

例:

- score
- probability
- estimate
- state label
- confidence

## Practical Rule

新しいpredictorを設計するときは、最低限次を先に書く。

1. 何を予測するのか
2. どのhorizonで予測するのか
3. どの集合の中で使うのか
4. どういう意思決定に使うのか
5. 出力の形は何か

この5つが曖昧なままmodelから入らない。
