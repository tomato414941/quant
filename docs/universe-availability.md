# Universe Availability

この文書は、評価期間の途中で資産が現れたり消えたりする場合に、Strategy評価をどう扱うべきかを整理する公開用メモである。

公開版では、設計原則と用語だけを記載する。具体的な実装進捗、cache version、内部payload詳細、未公開の検証ケースは載せない。

## Purpose

固定ticker universeを全期間で評価する前提を避け、日付ごとの投資可能性を明示的に扱う。

この文書で決めること:

- 資産の存在開始・終了を評価にどう反映するか
- `requested universe` と `eligible universe` をどう分けるか
- market dataの欠損をどこまで補完してよいか
- backtestやwalk-forwardで何を記録すべきか

この文書で決めないこと:

- survivorship biasを完全に解決する銘柄データベースの選定
- delisting returnを正確に復元する外部データソースの導入
- 個別Strategyの採用判断

## Problem

固定universeを全期間に強制すると、途中からデータが始まる資産に評価期間全体が引っ張られることがある。

これは次の意味で問題である。

- 早い期間に存在していた資産の情報が捨てられる
- Strategyが対象にしたい集合と、実際にその日に投資可能だった集合が混ざる
- universeの違いが、資産構成だけでなく評価期間まで変えてしまう
- walk-forwardのwindowごとの比較可能性が不透明になる

本来は、ある資産が存在しない期間はその資産なしで評価し、存在し、十分な履歴がたまったあとにだけ候補へ入るべきである。

## Core Terms

### Requested Universe

Strategyが対象にしたい全候補の集合。

これはStrategyの意図を表す。全期間で常に投資可能だったことは意味しない。

### Available Universe

ある日付で、データ上その資産の価格が観測可能な集合。

上場前、データ提供前、長期欠損期間はavailableではない。

### Eligible Universe

ある日付で、Strategyがselection、ranking、allocationに使ってよい集合。

`available universe` のうち、最低履歴本数や必要フィールドなどの条件を満たしたものだけがeligibleになる。

### Tradable Universe

ある日付で、売買・保有を継続してよい集合。

初期設計では `eligible universe` と近いが、将来的には「保有はできるが新規買いはできない」「売却だけできる」などを分ける余地を残す。

### Availability Policy

`requested universe` から `available / eligible / tradable universe` を作るための評価前提。

## Target Behavior

### Asset Appears Mid-Period

途中から現れる資産は、その日以前には存在しなかったものとして扱う。

期待する挙動:

- 上場前やデータ開始前はreturns、ranking、allocationに入らない
- 初回価格が観測されても、すぐにはeligibleにしない
- 必要な履歴がたまった日からeligibleにする
- eligibleになった日以降は通常の候補資産として扱う

### Asset Disappears Mid-Period

途中で価格が取れなくなった資産は、以後そのまま保有し続けた扱いにしない。

期待する挙動:

- tradableから外れたことを明示する
- 既存保有がある場合の扱いをpolicyで決める
- delisting returnが不明な場合はwarningとして残す

### Short Missing Data

短い欠損は、存在しない資産とは別に扱う。

期待する挙動:

- 短期のprovider欠損や休日は限定的に補完してよい
- 補完できる最大期間を制限する
- 制限を超えたらavailableから外す

## Backtest Semantics

各barでは次の順序で処理する。

1. その日までのmarket dataからasset availabilityを解決する
2. 必要な履歴とフィールドを満たすassetだけをeligibleにする
3. 既存保有がtradableから外れていたらpolicyに従って処理する
4. selection、ranking、predictorはeligible assetsだけを見る
5. allocationはeligible assetsとcashの範囲で行う
6. 結果にeligible asset countとwarningsを記録する

重要な制約:

- 上場前の資産を過去に補完しない
- futureのfirst valid dateを過去時点の判断に使わない
- window全体を見てからuniverseを決めない
- benchmarkやreference strategyも同じavailability policyで評価する

## Interaction With Walk-Forward

walk-forwardでは、windowごとにeligible universeの推移を記録する。

見るべき指標:

- window開始時のeligible asset count
- window終了時のeligible asset count
- window中にeligibleになったasset
- window中にremovedされたasset
- availability warning count

これにより、あるwindowの成績が「Strategyが良かった」のか、「そのwindowで候補資産が増えた/減った」のかを切り分けやすくする。

## Known Limitations

この設計はsurvivorship biasを完全には解決しない。

理由:

- 利用できるデータソースは現在生きている資産に偏りやすい
- delisted assetsの最終returnを正しく取得できるとは限らない
- ETFの統合、償還、ticker変更を完全には追跡しない

ただし、この設計により「途中からデータが始まる資産が全体評価期間を短くしてしまう」問題は軽減できる。

## Decision

このプロジェクトでは、Strategyの `investment universe` は requested universe として扱う。

実際に評価で使える集合は、Evaluation Contextのavailability policyとmarket data metadataから日付ごとに決める。

つまり:

- Strategyは「何を対象にしたいか」を持つ
- Evaluation Contextは「その期間で何が実際に使えたか」を持つ
- Runは「その条件でどう動いたか」を記録する

Market data metadata includes `datasetSnapshot` so a run can identify the requested universe, available universe, date range, timeframe, row count, adjustment policy, and metadata fingerprint used for availability decisions.

This is intentionally metadata-only. Do not commit raw provider downloads, adjusted price panels, volume panels, or other market data extracts; commit only reproducibility metadata and documentation.
