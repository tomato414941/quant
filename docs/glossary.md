# Quant Glossary

このプロジェクトで使う主要な用語を整理する。

## 用語の流れ

`Hypothesis -> Raw Data -> Features -> Asset Ranking Model -> Strategy -> Run -> Study`

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

## Prediction Terms

### Predictor

将来の量を予測し、Strategy の意思決定に使うためのモデル。

補足:
- `Predictor` は必ずしも `alpha` そのものを直接予測するとは限らない
- 大事なのは、何を予測するかと、その予測をどう使うか
- 予測対象が違えば、必要な入力、評価指標、下流の使い方も変わる

### Predicted Quantity

Predictor が予測したい対象そのもの。

例:
- 次の5bar超過収益
- 次の10bar超過収益
- 市場レジーム
- 将来ボラティリティ

補足:
- `Predicted Quantity` は「何を予測するか」を表す
- horizon はここに含めてもよいが、実装上は別の `Prediction Target` として分けてもよい

### Prediction Target

予測対象を、実際の学習・推論に使う形に落とした定義。

現在の感覚では、少なくとも次を含む。
- 何を予測するか
- どの horizon で予測するか
- どの baseline からの差として扱うか

例:
- 次の5bar超過収益
- 次の10bar超過収益

### Decision Use

予測結果を Strategy 側でどう使うかを表す。

例:
- 候補集合内の順位付け
- ウェイト傾斜の補助
- exposure の gate
- regime に応じた Strategy 切り替え

補足:
- `Predictor` の価値は、予測精度だけでなく `Decision Use` まで含めて評価すべき
- 同じ予測対象でも、使い方が違えば別物として扱う方が自然なことがある

### Candidate Set

予測や順位付けの対象として実際に比較する資産集合。

例:
- 全資産
- 上昇資産のみ
- 上昇資産のうち低ボラ群

補足:
- `Investment Universe` は Strategy 全体の投資対象
- `Candidate Set` はその時点の予測や選別の入力となる比較集合
- `Predictor` は Strategy ID ではなく、このような集合定義に依存する方が自然

### Current Prediction Question

現時点で主に解いている予測問題は次。

- 候補資産集合の中で、次の数 bar で相対的に強い資産はどれか

このため、今の Predictor は主に
- `Predicted Quantity`: 次の5bar / 10bar 超過収益
- `Decision Use`: 候補集合内の順位付けや weighting 補助

を担っている。

### Regime Prediction

近いうちに扱いたい別系統の予測問題。

例:
- risk-on / risk-off
- inflation-sensitive / growth-sensitive
- trend-following が効きやすい局面かどうか

補足:
- `Regime Prediction` は、現在の cross-sectional な順位予測とは別問題
- 主な使い道は、銘柄順位そのものではなく、exposure 制御、risk budget 調整、Strategy 切り替え
- したがって、将来的には `Predicted Quantity` や `Decision Use` の違いとして表現するのが自然

参照:
- 詳しい設計原則は [prediction-surface.md](/home/dev/projects/quant/docs/prediction-surface.md)

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

このプロジェクトで最終的に採用候補として選ぶ、意思決定ルールの完全な仕様。  
`Strategy` は上位概念であり、内部に複数の signal / feature / predictor / execution rule を持つ。

コード上の正本は `StrategyDefinition`。
`EvaluatorStrategySpec` は公開正本ではなく、低レベル evaluator に渡すための内部 DTO として残っている。

最低限持つもの:
- `strategy_id`
- `version`
- `label`
- `hypothesis`
- Investment Universe
- Signals
- Portfolio Model
- Execution Plan
- Risk Controls
- `extensions`

考え方:
- `Strategy` は「我々が選ぶ対象」
- `Run` は「その Strategy を特定の前提で評価した結果」
- `Study` は「複数 Strategy を共通の問いで比べる枠」

含まれるもの:
- Investment Universe
- Signal definitions
- Data Source / Feature Definition
- Alignment Policy
- Asset Ranking Model
- Filter Rule
- Fallback Rule
- Predictor overlay
- Portfolio Model
- Execution Plan
- Risk Controls

含まれないもの:
- 評価期間
- benchmark
- train/test split
- generation method
- fee のような市場前提そのもの

補足:
- `data_timeframe` / `signal_timeframe` / `decision_schedule` / `rebalance_schedule` は分けて扱う
- runSpec では由来となる戦略を `strategyDefinition`、実評価対象を `evaluationSubject` として分離する
- `executionSupport.evaluatorAdapter*` は内部 DTO へ変換できるかを表す
- `executionSupport.directExecution*` は direct execution 経路で評価できるかを表す

要するに、`Strategy` は「何をどう持つか」の仕様であり、  
`Portfolio Model` や `Execution Plan` はその構成要素である。

## Strategy Components

`Strategy` を構成する要素の総称。

このプロジェクトでは、Strategy Components を次の2種類に分けて考える。

- `Core Components`
  多くの Strategy に存在する中核要素
- `Optional Components`
  Strategy によって存在したりしなかったりする追加要素

### Core Components

現在の設計では、少なくとも次を `Core Components` とみなす。

- Investment Universe
- Portfolio Model
- Execution Plan

### Optional Components

現在の設計では、次は `Optional Components` とみなす。

- Asset Ranking Model
- Feature Inputs
- Filter Rules
- Fallback Rule
- Risk Controls
- 拡張的な Data Sources / custom logic
- Predictor overlay

例:
- 全資産を候補にし、12ヶ月モメンタムで上位優遇 tilt をかけ、HRP で配分し、年次で更新する
- 上昇資産のみを候補にし、モメンタム上位3へ絞り、候補ゼロなら CASH に逃がす

### Investment Universe

Strategy が実際に投資対象として扱う資産集合そのもの。

例:
- 20資産マルチアセット
- crypto only

補足:
- `Investment Universe` は「何を対象にするか」
- `Universe Policy` は「その対象集合をどう扱うか」

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

Strategy の構成要素の1つ。  
候補資産やランキングを受けて、最終ウェイトを決めるモデル。

例:
- Equal Weight
- Risk Budgeting
- Minimum Variance
- HRP

### Portfolio State

現在の保有状態。  
今のプロジェクトでは最小構成として、現在ウェイトと cash 比率を持つ。

### Execution Plan

Strategy の構成要素の1つ。  
いつ判断し、いつ実際にリバランスするかを表す。

例:
- 判断は毎日、売買は月末
- 判断も売買も週次
- 判断は毎 bar、売買は月末

最低限分けて扱うもの:
- `decision_schedule`
- `rebalance_schedule`

補足:
- `Execution Policy` は、低レベル evaluator DTO や builder 内で使う再利用可能な実行方針として残る
- StrategyDefinition の公開正本では、strategy-level の実行条件は `Execution Plan` として扱う
- 手数料や slippage の実数値そのものは、普通は Strategy ではなく評価前提やコスト前提として扱う
- ただし「どのコストモデルを使うか」は比較対象になりうる

### Evaluation Context

Strategy そのものではなく、Strategy を評価するために外側から与える文脈全体。  
以前は `Assumptions` と呼んでいたが、内容が「仮定」だけではないため、現在は `Evaluation Context` を優先する。

含まれるもの:
- Dataset Context
- Evaluation Settings
- Cost Assumptions
- Initial State
- Generation Metadata

整理:
- `Strategy` は選ぶ対象
- `Evaluation Context` は評価の土俵
- `Run` は `Strategy + Evaluation Context + Result`

### Dataset Context

Strategy の外側で、どの期間・頻度・整列済みデータ範囲で評価したかを表す文脈。

例:
- `10y`
- `daily`
- `alignedStartDate / alignedEndDate`
- `rowCount`

補足:
- `investment universe` そのものは Strategy に含める
- `その universe をどの期間で評価したか` は Dataset Context に含める
- 期間途中で資産が現れる/消える場合の扱いは [Universe Availability](./universe-availability.md) を正本にする

### Evaluation Settings

評価のやり方に関する設定。

例:
- split ratio
- initial capital
- benchmark

### Cost Assumptions

コスト前提。

例:
- fee
- slippage

補足:
- fee の実数値そのものは通常 Strategy ではなく Evaluation Context
- ただし、どの cost model を採用するかは Strategy や Study の論点になりうる

### Initial State

評価開始時の状態。

例:
- portfolio state
- cash 比率

### Generation Metadata

その Run がどのように作られたかを表すメタ情報。

例:
- manual
- condition grid
- parameter sweep

### Candidate

比較対象として並べる Strategy の候補。

補足:
- 以前は実装上 `Candidate` を強く使っていたが、現在の主語は `Strategy`
- 今後は UI や保存設計でも、`Candidate` より `Strategy` を優先する

例:
- 全資産モメンタム傾斜 最良 上位優遇 × HRP × 年次
- 上昇資産のみ × モメンタム上位3 × HRP × 年次

### Condition Variant

Run を生成するために追加で振る評価条件。

例:
- 手数料
- 最大投資比率
- weight cap

### Run

1つの Strategy を、特定の前提条件・制約条件・評価条件のもとで実行した1回の結果。

式で書くと:

`Run = Strategy + Evaluation Context + Result`

ここでいう `Evaluation Context` には、たとえば次が入る。
- dataset context
- evaluation settings
- cost assumptions
- initial state
- generation metadata

構成:
- Strategy
- Evaluation Context
- Result

出力:
- Sharpe
- total return
- max drawdown
- turnover
- weights

### Study

複数の Strategy を、共通の問いのもとで比較する実験全体。

Study は「何を比較したいか」を表し、Run は「その比較の中で1回どうだったか」を表す。

例:
- 20資産ユニバースで、複数 Strategy を同条件比較する
- ある Strategy 群を、10y と 3y で比較する

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
  - Strategy の構成要素として扱う
- `Execution Plan`
  - strategy-level の実行条件名として使う
  - `decision_schedule` と `rebalance_schedule` を分けて扱う
- `Execution Policy`
  - 低レベル evaluator DTO や builder の再利用可能な実行方針名として使う
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
