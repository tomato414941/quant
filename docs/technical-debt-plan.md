# Technical Debt Plan

この文書は、既存実装を壊さないために横へ増築してきた結果として生まれた歪みを、意図的に返済するための計画である。

## Goal

予測、選択、配分、執行、評価を明確に分離し、診断機能が本体の歪みを外側から補う構造をやめる。

この計画では互換維持を最優先にしない。使われていない経路、正本が二重になっている定義、名前だけ残った抽象は削除または統合する。

## Principles

- 正本は1つにする。
- 診断は本体概念を検証するものであり、本体の代替実装にしない。
- 予測の評価と売買判断を分ける。
- Portfolio Model は予測の代用品にしない。
- HRP は比較対象の1つであり、必ず通る主経路にしない。
- 互換層は移行期間だけ許容し、期限なく残さない。
- 実装後は関連テストを実行し、コミットして push する。

## Current Distortions

### 1. `portfolio.py` が巨大な中心になっている

`portfolio.py` は戦略変換、予測、スコア計算、バックテスト、配分 orchestration を抱えすぎている。

返済方針:
- 予測、選択、配分、執行、評価を別モジュールへ切り出す。
- `portfolio.py` は backtest orchestration の薄い層に寄せる。
- 新機能は `portfolio.py` へ追加しない。

### 2. 期待リターン経路が存在するが主経路で使われていない

以前の `fit_portfolio_model()` は `expected_return_proxy` を受け取っていたが、主経路では実質使っていなかった。現在は未使用引数を削除し、risk allocator 入力と forecast-aware allocator 入力を分離している。

返済方針:
- まず `expected_return_proxy` を未使用引数として削除するか、使うモデルだけに閉じ込める。
- 期待リターンを使う場合は、HRP へ曖昧に渡すのではなく、expected-return-aware allocator として明示する。
- 予測値、信頼度、リスク見積もり、コストを allocator input として定義する。

### 3. HRP が主経路のように見える

HRP は risk structure allocator であり、期待リターンを主目的にしない。今の構造では、予測の良し悪しより配分モデルの癖が結果を支配しやすい。

返済方針:
- HRP を特別扱いせず、equal weight、risk budgeting、expected-return-aware allocator と同列の portfolio model にする。
- 戦略評価では allocator 別の比較を標準化する。
- forecast/ranking の単体評価を allocator 評価より前段に置く。

### 4. `signal_diagnostics_service.py` が独自の評価世界を持っている

Signal diagnostics は有用だが、日次で全観測し、年次、資産クラス、診断フラグを独自に組み立てている。戦略の意思決定頻度と観測頻度が揃っていない。

返済方針:
- 観測単位を strategy decision schedule に合わせる。
- 日次の重複観測は明示オプションに落とす。
- signal diagnostics は「予測の standalone 評価」として位置づける。
- 年次、資産クラス、horizon 別の評価は共通 evaluation utility へ切り出す。

### 5. 資産分類の正本が二重になっている

`instrument_registry.py` に asset class がある一方、`signal_diagnostics_service.py` に別の `ASSET_CLASS_BY_TICKER` がある。

返済方針:
- asset class の正本は `instrument_registry.py` に統一する。
- diagnostics 側の ticker mapping は削除する。
- 表示上の大分類が必要なら、registry 側に normalized asset class helper を置く。

### 6. 診断系が本体の外側で無理に分解している

`edge_attribution_service.py` は strategy variant を派生コピーして selection、tilt、model、full を比較している。これは便利だが、本体側に明確な component boundary がないため、診断側が構造を推測している。

返済方針:
- Strategy の component boundary を本体概念として明示する。
- selection、tilt、portfolio model、execution を本体の trace として記録する。
- edge attribution は variant を大量生成する方式から、実行 trace を読む方式へ寄せる。

### 7. `StrategyDefinition` と `EvaluatorStrategySpec` の二重構造が重い

`StrategyDefinition` が正本で、`EvaluatorStrategySpec` は evaluator DTO として残っている。ただし変換、互換性チェック、テストが広く残り、長期化すると再び正本がぼやける。

返済方針:
- `StrategyDefinition` を公開正本として固定する。
- `EvaluatorStrategySpec` は evaluator 内部 DTO として名前と配置を明確化する。
- 変換関数は1箇所へ集約する。
- 互換性チェックは移行目的のものを削除し、実行前 validation に置き換える。

### 8. CLI が肥大化している

`cli.py` が command 定義、payload 作成、rendering、実行分岐を抱えている。

返済方針:
- CLI は command wiring だけに寄せる。
- rendering は command ごとの renderer に分離する。
- service payload schema と text rendering を近づけすぎない。

## Execution Order

### Phase 1: 正本の二重化を消す

- `signal_diagnostics_service.py` の asset class mapping を削除し、`instrument_registry.py` に統一する。
- `expected_return_proxy` の未使用経路を整理し、使わないなら削除する。
- `StrategyDefinition` から evaluator DTO への変換入口を1つに集約する。

完了条件:
- asset class mapping が1箇所だけになる。
- 未使用の expected return 引数が残らない、または使う allocator にだけ渡る。
- strategy conversion の入口が明確になる。

### Phase 2: 予測評価を意思決定頻度へ揃える

- `signal-diagnostics` に observation schedule を導入する。
- default は strategy decision schedule にする。
- daily overlapping evaluation は明示オプションにする。
- 結果 payload に observation schedule と observation count semantics を出す。

完了条件:
- 月次戦略は月次観測で signal diagnostics できる。
- 日次重複観測と意思決定観測を比較できる。
- standalone signal 評価の結果が売買ロジックと混ざらない。

### Phase 3: Portfolio Model を予測から分離する

- HRP、equal weight、risk budgeting、expected-return-aware allocator を同列に扱う。
- HRP を通らない評価経路を自然な本線として用意する。
- 予測、confidence、risk、cost を allocator input として明示する。

進捗:
- risk structure allocator の入力は `PortfolioAllocationInput` に分離済み。
- forecast-aware allocator 用の入力は `ForecastAllocationInput` として分けたが、未校正 forecast を risk allocator へ混ぜる経路は作らない。

完了条件:
- 「予測が良いか」と「配分モデルが良いか」を別々に評価できる。
- HRP なしでも strategy run が自然に成立する。
- expected-return-aware allocator を追加しても HRP の意味を歪めない。

### Phase 4: 実行 trace を本体に入れる

- backtest runtime で selection、score、tilt、allocation、execution decision を trace として保存する。
- edge attribution は variant 生成より trace 分析を優先する。
- 診断は本体 trace を読むだけに寄せる。

進捗:
- runtime は `executionTrace` として decision、rebalance、forced universe change を記録する。
- edge attribution の trace 読み取り化は次段階に残っている。

完了条件:
- 診断が strategy をコピーして推測しなくても、どこで成績が変わったか見える。
- selection effect、tilt effect、allocator effect、execution cost を同じ trace から説明できる。

### Phase 5: CLI と service 境界を整理する

- CLI の renderer を分離する。
- 診断 service は payload 作成に集中する。
- text output は renderer の責務にする。

完了条件:
- `cli.py` に診断表示ロジックが増え続けない。
- 新しい診断を追加しても CLI 本体の変更が小さい。

## Near-Term Next Action

最初にやるべき作業は、`signal-diagnostics` の観測単位を strategy decision schedule に合わせることである。

理由:
- 現在の signal diagnostics は、月次戦略でも日次に重複 horizon を評価している。
- これでは sample count が大きく見え、信号の安定性を誤認しやすい。
- HRP や allocator を責める前に、予測自体が意思決定日に効いているかを測る必要がある。

その次に、asset class mapping の正本を `instrument_registry.py` に統一する。

## Push Policy

各 phase または小さな完了単位ごとに次を行う。

1. 関連テストを実行する。
2. `git diff --check` を実行する。
3. 小さく commit する。
4. `main` に push する。
