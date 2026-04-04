# Quant Glossary

このプロジェクトで使う主要な用語を整理する。

## 用語の流れ

`Hypothesis -> Raw Data -> Features -> Asset Ranking Model -> Strategy -> Portfolio Model -> Execution Model -> Run -> Study`

## Core Terms

### Hypothesis

何が効くと思っているか、という仮説。

例:
- 強い資産を少し厚くすると Sharpe が改善する
- 上昇していて出来高も強い資産は、その後も相対的に強い

### Raw Data

観測した元データ。

例:
- close
- volume

### Feature

Raw Data を加工して作る説明変数。

例:
- 12ヶ月モメンタム
- 実現ボラティリティ
- 出来高強度

### Asset Ranking Model

Feature から各資産の相対順位や相対的な持ちたさを作る層。  
このプロジェクトでは、正式な用語として `Asset Ranking Model` を使う。

理由:
- まだ超過収益や絶対的な期待リターンを直接推定しているわけではない
- 実際には資産の順位付けや重みの傾きに使う相対評価に近い
- `alpha` より意味が狭く、今の実装に合っている

例:
- モメンタム順位
- 低ボラ補正付きモメンタム順位

別名:
- `Score Model`
  実装や画面で暫定的に残ることがある互換名

### Strategy

Asset Ranking Model をどう使うかを決めるルール。  
候補集合、スコアの使い方、フィルタ、フォールバックを含む。

例:
- 全資産を残して、上位だけ少し厚くする
- 上昇資産だけを候補にする
- 候補が空なら CASH に逃がす

### Universe Policy

どの資産を候補集合として扱うかを決めるルール。

例:
- 全資産
- 上昇資産のみ

### Filter Rule

候補資産を除外・選別するルール。

例:
- 上位3
- 低ボラ半分
- 出来高上位半分

### Fallback Rule

候補資産が空、または少なすぎるときにどうするかを決めるルール。

例:
- 候補ゼロなら CASH
- フォールバックなし

### Portfolio Model

Strategy が作った候補やスコアを受けて、最終ウェイトを決めるモデル。

例:
- Equal Weight
- Risk Budgeting
- Minimum Variance
- HRP

### Portfolio State

現在の保有状態。  
今のプロジェクトでは最小構成として、現在ウェイトと cash 比率を持つ。

### Execution Model

目標ウェイトへの変更をどう実行したとみなすかを決めるモデル。

例:
- 年次リバランス
- 手数料 0.05%
- slippage 0.0%

### Candidate

比較対象の単位。  
このプロジェクトでは基本的に `Strategy x Portfolio Model` の組み合わせを指す。

例:
- 全資産 x HRP
- 全資産モメンタム傾斜 弱 上位優遇 x HRP

### Condition Variant

Candidate に対して追加で振る実験条件。

例:
- 手数料
- 最大投資比率
- weight cap

### Run

1つの条件セットで実行した1回の結果。

構成:
- Candidate
- Dataset
- Execution Model
- Backtest Config
- Portfolio State
- Condition Variant

出力:
- Sharpe
- total return
- max drawdown
- turnover
- weights

### Study

複数の Run をまとめて比較する実験全体。

例:
- 20資産ユニバースで、複数 candidate を同条件比較する

## Inputs to Portfolio Construction

ポートフォリオの入力として自然なのは次の5つ。

- expected return
- risk
- constraints
- costs
- objective

このプロジェクトでの対応はおおむね次の通り。

- expected return: Asset Ranking Model の出力、またはその proxy
- risk: HRP / minimum variance / risk budgeting
- constraints: max investment ratio / weight cap
- costs: commission / turnover / slippage
- objective: 各 Portfolio Model が暗黙に持つ最適化目的

## Preferred Naming

このプロジェクトでは、次の用語を優先する。

- `Asset Ranking Model`
  - `Alpha Model` より優先
  - `Score Model` は互換名としてのみ使う
- `Portfolio Model`
  - `Allocator` より優先
- `Execution Model`
  - `Trading Rule` より優先
- `Run`
  - `Result` 単体より優先
- `Study`
  - 実験全体の単位として使う

## Current Mental Model

いまの勝ち筋は、強い選別戦略というより

- 広い universe
- 穏やかな ranking tilt
- 強い portfolio construction
- 妥当な constraints

の組み合わせでできている。

つまり、現在の主語は
`sharp prediction` ではなく
`ranking-guided portfolio construction`
である。
