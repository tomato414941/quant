# Universe Availability

この文書は、評価期間の途中で資産が現れたり消えたりする場合に、Strategy 評価をどう扱うべきかを定める設計メモである。

## Purpose

固定 ticker universe を全期間で評価する前提をやめ、日付ごとの投資可能性を明示的に扱えるようにする。

この文書で決めること:
- 資産の存在開始・終了を評価にどう反映するか
- `investment universe` と `eligible universe` をどう分けるか
- market data の欠損をどこまで補完してよいか
- backtest / walk-forward / run cache にどの情報を含めるべきか

この文書で決めないこと:
- survivorship bias を完全に解決する銘柄データベースの選定
- delisting return を正確に復元する外部データソースの導入
- 個別戦略の採用判断

## Current Problem

現在の market data pipeline は、取得できた ticker を横持ち DataFrame に揃え、最後に全体を `dropna()` する固定 universe 寄りの設計である。

このため、途中からデータが始まる資産があると、評価開始日全体が後ろにずれる。例として ETH を含めると、2015-2025 の評価でも ETH のデータ開始に引っ張られて 2017 以降の評価になりやすい。

これは次の意味で問題である。
- 2015-2017 に存在していた ETF の情報が捨てられる
- `requested universe` と `実際にその日に投資可能だった universe` が混ざる
- crypto を含めるかどうかで、資産構成だけでなく評価期間まで変わる
- walk-forward の window ごとの比較可能性が不透明になる

本来は、ETH が存在しない期間は ETH なしで評価し、ETH が存在し、十分な履歴がたまったあとにだけ候補へ入るべきである。

## Core Terms

### Requested Universe

Strategy が対象にしたい全候補の集合。

例:
- `SPY`
- `QQQ`
- `BTC-USD`
- `ETH-USD`

これは Strategy の意図を表す。全期間で常に投資可能だったことは意味しない。

### Available Universe

ある日付で、データ上その資産の価格が観測可能な集合。

上場前、データ提供前、長期欠損期間は available ではない。

### Eligible Universe

ある日付で、Strategy が selection / ranking / allocation に使ってよい集合。

`available universe` のうち、最低履歴本数や必要フィールドなどの条件を満たしたものだけが eligible になる。

### Tradable Universe

ある日付で、売買・保有を継続してよい集合。

初期実装では `eligible universe` とほぼ同じでよい。ただし将来的には、保有はできるが新規買いはできない、売却だけできる、などを分ける余地を残す。

### Availability Policy

`requested universe` から `available / eligible / tradable universe` を作るための評価前提。

初期方針:
- 上場前・データ開始前は `not_listed` として扱う
- 価格を過去に遡って補完しない
- 初登場後も `min_history_bars` を満たすまで selection 対象にしない
- 途中で使えなくなった資産は cash 化する
- delisting return が取れない場合は warning を出す

## Target Behavior

### Asset Appears Mid-Period

途中から現れる資産は、その日以前には存在しなかったものとして扱う。

期待する挙動:
- 上場前は returns / ranking / allocation に入らない
- 初回価格が観測されても、すぐには eligible にしない
- `min_history_bars` を満たした日から eligible にする
- eligible になった日以降は通常の候補資産として扱う

例:
- ETH の first valid date が 2017-11-09
- `min_history_bars = 252`
- ETH は 2017-11-09 から available
- ETH は十分な履歴がたまるまで eligible ではない

### Asset Disappears Mid-Period

途中で価格が取れなくなった資産は、以後そのまま保有し続けた扱いにしない。

初期実装の方針:
- `lastValidDate` 以降は `not_tradable` とする
- 既存保有がある場合、最後に有効だった価格で cash 化する
- delisting return が不明な場合は `availabilityWarnings` に出す

これは完全な delisting return 処理ではない。だが、消えた資産を永久に last price で保有し続けるよりは明示的である。

### Short Missing Data

短い欠損は、存在しない資産とは別に扱う。

初期方針:
- 日次データの短い欠損は前方補完を許容する
- ただし補完できる最大期間を `max_stale_bars` として制限する
- `max_stale_bars` を超えたら available から外す

これにより、休日や一時的な provider 欠損と、上場前・消滅後を混同しない。

## Data Model

market data metadata には、ticker ごとの availability summary を追加する。

最小 payload:

```json
{
  "assetAvailability": {
    "SPY": {
      "firstValidDate": "2015-01-02",
      "lastValidDate": "2025-12-31",
      "validRowCount": 2767
    },
    "ETH-USD": {
      "firstValidDate": "2017-11-09",
      "lastValidDate": "2025-12-31",
      "validRowCount": 2975
    }
  }
}
```

Evaluation payload には、availability policy と集計結果を含める。

最小 payload:

```json
{
  "availabilityPolicy": {
    "kind": "asset_availability_policy",
    "minHistoryBars": 252,
    "maxStaleBars": 5,
    "delistedAssetPolicy": "liquidate_to_cash"
  },
  "availabilitySummary": {
    "requestedAssetCount": 20,
    "minEligibleAssetCount": 18,
    "maxEligibleAssetCount": 20
  }
}
```

Backtest series には、必要に応じて日付ごとの universe 状態を出す。

例:

```json
{
  "date": "2020-01-03",
  "eligibleAssetCount": 18,
  "newlyEligibleAssets": [],
  "removedAssets": []
}
```

## Backtest Semantics

各 bar では次の順序で処理する。

1. その日までの market data から asset availability を解決する
2. `min_history_bars` と必要フィールドを満たす asset だけを eligible にする
3. 既存保有が tradable から外れていたら cash 化する
4. selection / ranking / predictor は eligible assets だけを見る
5. allocation は eligible assets と cash の範囲で行う
6. 結果 series に eligible asset count と warnings を記録する

重要な制約:
- 上場前の資産を過去に補完しない
- future の first valid date を過去時点の判断に使わない
- window 全体を見てから universe を決めない
- benchmark / reference strategy も同じ availability policy で評価する

## Interaction With Walk-Forward

walk-forward では、window ごとに eligible universe の推移を記録する。

見るべき指標:
- window 開始時の eligible asset count
- window 終了時の eligible asset count
- window 中に newly eligible になった assets
- window 中に removed された assets
- availability warning count

これにより、ある window の成績が「戦略が良かった」のか、「その window で候補資産が増えた / 減った」のかを切り分けやすくする。

## Implementation Status

### Phase 1: Market Data Metadata

Status: implemented.

- ticker ごとの `requested`, `available`, `failedReason`, `firstValidDate`, `lastValidDate`, `validRowCount` を `assetAvailability` に保存する
- market data は union calendar に揃え、全体 `dropna()` ではなく `dropna(how="all")` で評価開始日を維持する
- 短期欠損は `maxStaleBars = 5` まで close の前方補完を許容し、volume は未知なら 0 とする

### Phase 2: Availability Policy

Status: implemented for evaluation payload and backtest runtime.

- デフォルト policy は `minHistoryBars = 252`, `maxStaleBars = 5`, `delistedAssetPolicy = liquidate_to_cash`
- policy は evaluation payload に入り、run spec fingerprint と cache version の対象になる
- unit test の短い toy data では、履歴本数不足で全テストが cash だけにならないよう、実効 `minHistoryBars` は観測済み履歴長を上限にする

### Phase 3: Backtest Integration

Status: implemented.

- `compare_portfolio_runs` / `run_portfolio_backtest` は日付ごとの eligible assets を解決する
- selection / predictor / allocation は eligible assets に絞った履歴だけを見る
- eligible assets が2未満の場合は cash fallback する
- eligible から外れた既存 weight はその bar で cash 化する

### Phase 4: Reporting

Status: implemented.

- run payload と evaluation payload に `availabilityPolicy` / `availabilitySummary` / `availabilityDiagnostics` を出す
- backtest series に `availableAssetCount`, `eligibleAssetCount`, `newlyEligibleAssets`, `removedAssets` を出す
- Dataset Context に `assetAvailability` と asset-level availability warnings を出す
- CLI は `availabilityDiagnostics` を表示し、raw warnings の分類判断を持たない

### Phase 5: Cache Versioning

Status: implemented.

- run store logic version は `v63`
- availability policy と asset availability summary を evaluation payload に含め、旧結果と新結果を混ぜない

## Test Scenarios

必須テスト:
- 途中から現れる資産があっても評価開始日全体が後ろにずれない
- 上場前の資産は selection / allocation に入らない
- `min_history_bars` 未満の資産は eligible にならない
- 途中で消える資産は cash 化される
- short missing data は `max_stale_bars` 以内だけ前方補完される
- `crypto_included` で ETH が 2017 以降だけ available になり、十分な履歴後に eligible になる
- walk-forward の各 window に eligible asset count が出る

回帰テスト:
- 既存の固定 universe で全資産が全期間 available な場合、従来結果と大きく変わらない
- `no_crypto` の 2015-2025 評価開始日が維持される
- run store の cache が旧 logic version と混ざらない

## Known Limitations

この設計は survivorship bias を完全には解決しない。

理由:
- Yahoo Finance で取得できる ticker は現在生きている資産に偏りやすい
- delisted assets の最終 return を正しく取得できるとは限らない
- ETF の統合・償還・ticker 変更を完全には追跡しない

ただし、この設計により「途中からデータが始まる資産が全体評価期間を短くしてしまう」問題は解決できる。

## Decision

このプロジェクトでは、Strategy の `investment universe` は引き続き requested universe として扱う。

実際に評価で使える集合は、Evaluation Context の availability policy と market data metadata から日付ごとに決める。

つまり:
- Strategy は「何を対象にしたいか」を持つ
- Evaluation Context は「その期間で何が実際に使えたか」を持つ
- Run は「その条件でどう動いたか」を記録する
