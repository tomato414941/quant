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
`Strategy` は上位概念であり、内部に複数の構成要素を持つ。  
コード上では、`Strategy` は固定的な単一手法名ではなく、`StrategySpec` に近い拡張可能な仕様として扱う。

最低限持つもの:
- `strategy_id`
- `version`
- `label`
- `hypothesis`
- `components`
- `extensions`

考え方:
- `Strategy` は「我々が選ぶ対象」
- `Run` は「その Strategy を特定の前提で評価した結果」
- `Study` は「複数 Strategy を共通の問いで比べる枠」

含まれるもの:
- Universe Policy
- Asset Ranking Model
- Filter Rule
- Fallback Rule
- Portfolio Model
- Execution Policy
- Risk Controls

含まれないもの:
- 評価期間
- benchmark
- train/test split
- generation method
- fee のような市場前提そのもの

要するに、`Strategy` は「何をどう持つか」の仕様であり、  
`Portfolio Model` や `Execution Policy` はその構成要素である。

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
- Execution Policy

### Optional Components

現在の設計では、次は `Optional Components` とみなす。

- Asset Ranking Model
- Feature Inputs
- Filter Rules
- Fallback Rule
- Risk Controls
- 拡張的な Data Sources / custom logic

例:
- 全資産を候補にし、12ヶ月モメンタムで上位優遇 tilt をかけ、HRP で配分し、年次で更新する
- 上昇資産のみを候補にし、モメンタム上位3へ絞り、候補ゼロなら CASH に逃がす

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

### Execution Policy

Strategy の構成要素の1つ。  
目標ウェイトへの変更をどう実行したとみなすかを決めるモデル。

例:
- 年次リバランス
- 月次リバランス

補足:
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
- `Execution Policy`
  - `Trading Rule` より優先
  - Strategy の構成要素として扱う
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
