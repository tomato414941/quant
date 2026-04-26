# Technical Debt Plan

この文書は、公開リポジトリで共有する技術的負債の整理方針である。

公開版では、具体的な内部探索ロードマップではなく、保守性を上げるための一般的な改善方針だけを記載する。

## Goal

予測、選択、配分、執行、評価を明確に分離し、診断機能が本体の歪みを外側から補う構造を減らす。

## Principles

- 正本は1つにする。
- 診断は本体概念を検証するものであり、本体の代替実装にしない。
- 予測の評価と売買判断を分ける。
- Portfolio Model は予測の代用品にしない。
- 互換層は移行期間だけ許容し、期限なく残さない。
- 実装後は関連テストを実行し、コミットしてpushする。

## Improvement Areas

### Strategy Definition

Strategy定義、評価用DTO、実行用payloadの責務を分ける。Strategyの正本が複数に見える状態を避ける。

### Evaluation Contexts

評価条件はStrategy定義を上書きしない。期間、コスト、執行前提などの測定条件は、Strategy本体とは別に管理する。

### Prediction Evaluation

予測そのものの評価と、予測を使ったPortfolio成績を分ける。予測が有効かどうかを、配分モデルの癖だけで判断しない。

### Portfolio Models

Equal Weight、Equal Risk Contribution、Minimum Variance、HRPなどの配分モデルを同列に扱う。特定モデルを暗黙の主経路にしない。

### Execution Trace

Strategy runの中で、selection、signal、allocation、execution decisionを後から説明できる形にする。

### CLI And Services

CLIはcommand wiringに寄せ、payload生成やrenderingの責務を分ける。

## Near-Term Direction

短期的には、Strategy Catalog、Evaluation Contexts、Evaluation Plan、Leaderboardの責務を明確に保つ。

コード側の整理は、docs上の概念が安定してから小さく進める。
