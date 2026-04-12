# Prediction Surface

この文書は、「何を予測すべきか」を現在の実装に引きずられずに考えるための設計メモである。

## Purpose

局所的な予測問題に閉じず、あるべき予測対象の全体を見渡す。

この文書でやりたいこと:
- 予測対象の全体像を明示する
- 予測問題が増えたときに、何の軸が増えたのかを切り分ける
- 現在の実装が狭い問いに落ちていても、思考まで狭くしない

## Core View

最終的に予測したいものは、単一の数値ではなく、将来の市場世界の条件付き分布全体である。

言い換えると、理想的には次を知りたい。
- 将来どの資産がどう動くか
- それらが一緒にどう動くか
- どれくらい確からしいか
- その予測を実際に執行したとき、どんなコストと制約が乗るか

## Prediction Surface

計算量が無限にあると仮定したとき、少なくとも次は予測対象候補に入る。

### Future Paths

- 将来の価格パス
- 将来の出来高パス
- 将来のスプレッド
- 将来の板厚
- 将来の約定可能性

### Asset-Level Returns

- 資産ごとの将来超過収益
- 資産ごとの将来絶対収益
- 資産ごとの将来リターン分布
- 上側テール
- 下側テール
- 閾値超過確率

### Cross-Sectional Outcomes

- 資産間の将来相対順位
- 上位集合に入る確率
- 下位集合に落ちる確率
- pairwise relative strength
- top-minus-bottom spread

### Risk Outcomes

- 将来ボラティリティ
- 将来ドローダウン
- 将来最大ドローダウン確率
- ジャンプ確率
- tail risk

### Market State

- market regime
- risk-on / risk-off
- inflationary / disinflationary
- growth / slowdown
- trending / mean-reverting
- dispersion regime
- liquidity stress regime

### Dependency Structure

- 将来相関
- 将来共分散
- conditional correlation
- tail dependence
- co-jump probability

### Execution and Capacity

- 将来 slippage
- 将来 market impact
- fill probability
- 実効コスト
- capacity
- funding cost
- borrow cost

### Predictive Reliability

- 予測の不確実性
- calibration error
- model drift
- distribution shift
- edge decay
- その予測が今効く確率

## Predictive Axes

予測問題を増やすときは、1つの `kind` に押し込まず、何の軸が増えたのかを切り分けて考える。

### Predicted Quantity

何を予測するか。

例:
- 超過収益
- absolute return
- regime
- volatility

### Horizon

どれくらい先を予測するか。

例:
- 1bar
- 5bar
- 20bar
- 3ヶ月

### Decision Use

予測を何に使うか。

例:
- 候補集合内の順位付け
- ウェイト傾斜の補助
- exposure gate
- strategy switching
- risk budget adjustment

### Candidate Set

どの集合の中で比較・適用するか。

例:
- 全資産
- 上昇資産のみ
- defensive subset

### Output Form

どの形で出すか。

例:
- score
- probability
- return estimate
- state label
- state probability vector

## Current Bias To Avoid

現在の実装にある predictor だけを見て、「予測すべきもの」を定義しない。

避けたい局所化:
- 今ある target だけを本質とみなす
- 今ある model だけで task を定義する
- 今ある strategy の使い方だけで predictor の責務を決める

## Current Working Question

現時点の実装で主に解いている問いは次。

- 候補資産集合の中で、次の数 bar で相対的に強い資産はどれか

これは `Prediction Surface` 全体の一部にすぎない。

この問いの位置づけ:
- `Predicted Quantity`: 次の5bar / 10bar 超過収益
- `Decision Use`: 候補集合内の順位付けや weighting 補助
- `Candidate Set`: ranking によって絞られた集合

## Near-Term Expansion

近いうちに扱いたい別系統の問いとして、少なくとも次がある。

### Regime Prediction

例:
- risk-on / risk-off
- inflation-sensitive / growth-sensitive
- trend-following が効きやすい局面か

主な使い道:
- exposure 制御
- risk budget 調整
- strategy 切り替え

補足:
- これは現在の cross-sectional な順位予測とは別問題である
- よって predictor の拡張を考えるとき、単一の task 名だけで押し込めない

## Practical Rule

新しい predictor を設計するときは、最低限次を先に書く。

1. 何を予測するのか
2. どの horizon で予測するのか
3. どの集合の中で使うのか
4. どういう意思決定に使うのか
5. 出力の形は何か

この5つが曖昧なまま model から入らない。
