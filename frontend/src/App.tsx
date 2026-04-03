import { startTransition, useDeferredValue, useEffect, useEffectEvent, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'

type SummaryMetrics = {
  totalReturnPct: number
  cagrPct: number
  sharpeRatio: number
  maxDrawdownPct: number
  tradeCount?: number
  winRatePct?: number
}

type DatasetInfo = {
  ticker: string
  period: string
  source: string
}

type BacktestResult = {
  dataset: DatasetInfo
  summary: {
    strategy: SummaryMetrics
    benchmark: SummaryMetrics
    config: {
      thresholdPct: number
      initialCapital: number
      holdingDays: number
      transactionCostPct: number
    }
  }
  series: Array<{
    date: string
    close: number
    previousDayReturnPct: number
    position: number
    signal: boolean
    strategyEquity: number
    benchmarkEquity: number
    strategyReturnPct: number
  }>
}

type GridSearchRow = {
  rank: number
  thresholdPct: number
  holdingDays: number
  totalReturnPct: number
  cagrPct: number
  sharpeRatio: number
  maxDrawdownPct: number
  tradeCount: number
  winRatePct: number
}

type GridSearchResult = {
  dataset: DatasetInfo
  config: {
    thresholdValuesPct: number[]
    holdingDaysValues: number[]
    initialCapital: number
    transactionCostPct: number
  }
  results: GridSearchRow[]
}

type TickerCompareResult = {
  config: {
    thresholdPct: number
    holdingDays: number
    initialCapital: number
    transactionCostPct: number
  }
  results: Array<{
    ticker: string
    period: string
    strategy: SummaryMetrics
    benchmark: SummaryMetrics
  }>
}

type StatusState = {
  tone: 'success' | 'error'
  text: string
}

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  `${window.location.protocol}//${window.location.hostname}:8000`

function formatPercent(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat('ja-JP', {
    maximumFractionDigits: 0,
  }).format(value)
}

function isValidPositiveInteger(value: number): boolean {
  return Number.isInteger(value) && value > 0
}

function isValidTicker(value: string): boolean {
  return value.trim().length > 0
}

function isValidTransactionCost(value: number): boolean {
  return value >= 0 && value < 100
}

function App() {
  const [ticker, setTicker] = useState('SPY')
  const [comparisonTickers, setComparisonTickers] = useState('SPY,QQQ,IWM,TLT,GLD,BTC-USD')
  const [period, setPeriod] = useState('2y')
  const [threshold, setThreshold] = useState(3)
  const [holdingDays, setHoldingDays] = useState(1)
  const [initialCapital, setInitialCapital] = useState(10000)
  const [transactionCost, setTransactionCost] = useState(0.1)
  const [thresholdGrid, setThresholdGrid] = useState('1.5,2,3,4,5')
  const [holdingDaysGrid, setHoldingDaysGrid] = useState('1,2,3')
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [gridSearch, setGridSearch] = useState<GridSearchResult | null>(null)
  const [tickerCompare, setTickerCompare] = useState<TickerCompareResult | null>(null)
  const [singleLoading, setSingleLoading] = useState(false)
  const [gridLoading, setGridLoading] = useState(false)
  const [compareLoading, setCompareLoading] = useState(false)
  const [status, setStatus] = useState<StatusState>({
    tone: 'success',
    text: '実データの結果を自動で読み込みます。',
  })

  const deferredTicker = useDeferredValue(ticker)
  const deferredComparisonTickers = useDeferredValue(comparisonTickers)
  const deferredPeriod = useDeferredValue(period)
  const deferredThreshold = useDeferredValue(threshold)
  const deferredHoldingDays = useDeferredValue(holdingDays)
  const deferredInitialCapital = useDeferredValue(initialCapital)
  const deferredTransactionCost = useDeferredValue(transactionCost)
  const deferredThresholdGrid = useDeferredValue(thresholdGrid)
  const deferredHoldingDaysGrid = useDeferredValue(holdingDaysGrid)

  const refreshSingleBacktest = useEffectEvent(async () => {
    if (!isValidTicker(deferredTicker)) {
      setStatus({ tone: 'error', text: '銘柄コードを入力してください。' })
      return
    }
    if (!isValidPositiveInteger(deferredHoldingDays)) {
      setStatus({ tone: 'error', text: '保有日数は 1 以上の整数で入力してください。' })
      return
    }
    if (!isValidPositiveInteger(deferredInitialCapital)) {
      setStatus({ tone: 'error', text: '初期資金は 1 以上の整数で入力してください。' })
      return
    }
    if (!isValidTransactionCost(deferredTransactionCost)) {
      setStatus({ tone: 'error', text: '片道コストは 0 以上 100 未満で入力してください。' })
      return
    }

    setSingleLoading(true)
    try {
      const params = new URLSearchParams({
        ticker: deferredTicker.trim().toUpperCase(),
        period: deferredPeriod,
        threshold: (deferredThreshold / 100).toString(),
        initial_capital: deferredInitialCapital.toString(),
        holding_days: deferredHoldingDays.toString(),
        transaction_cost: (deferredTransactionCost / 100).toString(),
      })
      const response = await fetch(`${API_BASE_URL}/api/backtest?${params.toString()}`)
      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail ?? 'バックテスト結果の取得に失敗しました。')
      }

      startTransition(() => setResult(payload))
      setStatus({
        tone: 'success',
        text: `${payload.dataset.ticker} の単発結果を表示しています。`,
      })
    } catch (caughtError) {
      setStatus({
        tone: 'error',
        text: caughtError instanceof Error ? caughtError.message : '不明なエラーが発生しました。',
      })
    } finally {
      setSingleLoading(false)
    }
  })

  const refreshGridSearch = useEffectEvent(async () => {
    if (!isValidTicker(deferredTicker)) {
      setStatus({ tone: 'error', text: '銘柄コードを入力してください。' })
      return
    }
    if (!isValidPositiveInteger(deferredInitialCapital)) {
      setStatus({ tone: 'error', text: '初期資金は 1 以上の整数で入力してください。' })
      return
    }
    if (!isValidTransactionCost(deferredTransactionCost)) {
      setStatus({ tone: 'error', text: '片道コストは 0 以上 100 未満で入力してください。' })
      return
    }

    setGridLoading(true)
    try {
      const params = new URLSearchParams({
        ticker: deferredTicker.trim().toUpperCase(),
        period: deferredPeriod,
        threshold_values: deferredThresholdGrid,
        holding_days_values: deferredHoldingDaysGrid,
        initial_capital: deferredInitialCapital.toString(),
        transaction_cost: (deferredTransactionCost / 100).toString(),
      })
      const response = await fetch(`${API_BASE_URL}/api/grid-search?${params.toString()}`)
      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail ?? '条件比較の取得に失敗しました。')
      }

      startTransition(() => setGridSearch(payload))
    } catch (caughtError) {
      setStatus({
        tone: 'error',
        text: caughtError instanceof Error ? caughtError.message : '不明なエラーが発生しました。',
      })
    } finally {
      setGridLoading(false)
    }
  })

  const refreshTickerCompare = useEffectEvent(async () => {
    if (!isValidTicker(deferredComparisonTickers)) {
      setStatus({ tone: 'error', text: '比較する銘柄を 1 つ以上入力してください。' })
      return
    }
    if (!isValidPositiveInteger(deferredInitialCapital)) {
      setStatus({ tone: 'error', text: '初期資金は 1 以上の整数で入力してください。' })
      return
    }
    if (!isValidTransactionCost(deferredTransactionCost)) {
      setStatus({ tone: 'error', text: '片道コストは 0 以上 100 未満で入力してください。' })
      return
    }

    setCompareLoading(true)
    try {
      const params = new URLSearchParams({
        tickers: deferredComparisonTickers,
        period: deferredPeriod,
        threshold: (deferredThreshold / 100).toString(),
        initial_capital: deferredInitialCapital.toString(),
        holding_days: deferredHoldingDays.toString(),
        transaction_cost: (deferredTransactionCost / 100).toString(),
      })
      const response = await fetch(`${API_BASE_URL}/api/ticker-compare?${params.toString()}`)
      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail ?? '複数銘柄比較の取得に失敗しました。')
      }

      startTransition(() => setTickerCompare(payload))
    } catch (caughtError) {
      setStatus({
        tone: 'error',
        text: caughtError instanceof Error ? caughtError.message : '不明なエラーが発生しました。',
      })
    } finally {
      setCompareLoading(false)
    }
  })

  useEffect(() => {
    void refreshSingleBacktest()
  }, [
    deferredTicker,
    deferredPeriod,
    deferredThreshold,
    deferredHoldingDays,
    deferredInitialCapital,
    deferredTransactionCost,
  ])

  useEffect(() => {
    void refreshGridSearch()
  }, [deferredTicker, deferredPeriod, deferredThresholdGrid, deferredHoldingDaysGrid, deferredInitialCapital, deferredTransactionCost])

  useEffect(() => {
    void refreshTickerCompare()
  }, [deferredComparisonTickers, deferredPeriod, deferredThreshold, deferredHoldingDays, deferredInitialCapital, deferredTransactionCost])

  return (
    <main className="app-shell">
      <div className="page">
        <section className="hero panel">
          <p className="eyebrow">ミニマルクオンツ</p>
          <h1>1つのルールから始める、最小のクオンツ検証。</h1>
          <p className="hero-copy">
            実データだけを使う最小構成のクオンツ検証です。銘柄コードを入れると価格系列を取得し、
            明示的なルールでバックテストして結果をそのまま確認できます。
          </p>
        </section>

        <div className="layout">
          <section className="panel results">
            <div className="panel-title">
              <div>
                <h2>結果</h2>
                <p>
                  {result
                    ? `${result.dataset.ticker} / ${result.dataset.period} / 前日下落が ${result.summary.config.thresholdPct.toFixed(2)}% 以上なら ${result.summary.config.holdingDays} 日保有。`
                    : 'まだ結果がありません。'}
                </p>
              </div>
            </div>

            {result ? (
              <>
                <div className="cards">
                  <article className="metric-card">
                    <h3>戦略</h3>
                    <div className="metric-grid">
                      <div className="metric">
                        <span className="metric-label">総リターン</span>
                        <strong className="metric-value">
                          {formatPercent(result.summary.strategy.totalReturnPct)}
                        </strong>
                      </div>
                      <div className="metric">
                        <span className="metric-label">シャープレシオ</span>
                        <strong className="metric-value">
                          {result.summary.strategy.sharpeRatio.toFixed(2)}
                        </strong>
                      </div>
                      <div className="metric">
                        <span className="metric-label">最大ドローダウン</span>
                        <strong className="metric-value">
                          {formatPercent(-result.summary.strategy.maxDrawdownPct)}
                        </strong>
                      </div>
                      <div className="metric">
                        <span className="metric-label">勝率</span>
                        <strong className="metric-value">
                          {formatPercent(result.summary.strategy.winRatePct ?? 0)}
                        </strong>
                      </div>
                    </div>
                    <p className="chart-note">
                      初期資金 {formatNumber(result.summary.config.initialCapital)} 円相当で
                      {result.summary.strategy.tradeCount} 回トレード、保有日数は {result.summary.config.holdingDays} 日です。
                    </p>
                    <p className="footnote">
                      {singleLoading
                        ? '単発結果を更新中…'
                        : `${result.dataset.source} の実データを表示しています。片道コスト ${result.summary.config.transactionCostPct.toFixed(3)}%。`}
                    </p>
                  </article>

                  <article className="metric-card">
                    <h3>ベンチマーク</h3>
                    <div className="metric-grid">
                      <div className="metric">
                        <span className="metric-label">総リターン</span>
                        <strong className="metric-value">
                          {formatPercent(result.summary.benchmark.totalReturnPct)}
                        </strong>
                      </div>
                      <div className="metric">
                        <span className="metric-label">シャープレシオ</span>
                        <strong className="metric-value">
                          {result.summary.benchmark.sharpeRatio.toFixed(2)}
                        </strong>
                      </div>
                      <div className="metric">
                        <span className="metric-label">最大ドローダウン</span>
                        <strong className="metric-value">
                          {formatPercent(-result.summary.benchmark.maxDrawdownPct)}
                        </strong>
                      </div>
                      <div className="metric">
                        <span className="metric-label">CAGR</span>
                        <strong className="metric-value">
                          {formatPercent(result.summary.benchmark.cagrPct)}
                        </strong>
                      </div>
                    </div>
                    <p className="chart-note">同じ価格系列での単純な買い持ちです。</p>
                  </article>
                </div>

                <div className="content-block">
                  <ResponsiveContainer width="100%" height={320}>
                    <LineChart data={result.series}>
                      <CartesianGrid stroke="rgba(88, 67, 51, 0.08)" vertical={false} />
                      <XAxis
                        dataKey="date"
                        tick={{ fill: '#6c5a49', fontSize: 12 }}
                        minTickGap={40}
                      />
                      <YAxis tick={{ fill: '#6c5a49', fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{
                          borderRadius: 16,
                          border: '1px solid rgba(90, 63, 40, 0.14)',
                          backgroundColor: 'rgba(255, 251, 245, 0.96)',
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="strategyEquity"
                        stroke="#b45b2a"
                        strokeWidth={2.5}
                        dot={false}
                        name="戦略"
                      />
                      <Line
                        type="monotone"
                        dataKey="benchmarkEquity"
                        stroke="#667d5d"
                        strokeWidth={2.1}
                        dot={false}
                        name="買い持ち"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="chart-note">
                    戦略の資産推移を買い持ちと直接比較して、このルールに意味があるかを見ます。
                  </p>
                </div>

                <div className="content-block">
                  <div className="table-header">
                    <div>
                      <h3>条件比較</h3>
                      <p>
                        {gridSearch
                          ? `${gridSearch.dataset.ticker} / ${gridSearch.dataset.period} の実データで、閾値 ${gridSearch.config.thresholdValuesPct.join(', ')}% と保有日数 ${gridSearch.config.holdingDaysValues.join(', ')} 日を比較しています。`
                          : '条件比較を読み込んでいます。'}
                      </p>
                      <p className="footnote">
                        {gridLoading ? '条件比較を更新中…' : '比較条件を変えると自動で更新されます。'}
                      </p>
                    </div>
                  </div>

                  {gridSearch && gridSearch.results.length > 0 ? (
                    <div className="table-scroll">
                      <table className="results-table">
                        <thead>
                          <tr>
                            <th>順位</th>
                            <th>閾値</th>
                            <th>保有</th>
                            <th>総リターン</th>
                            <th>Sharpe</th>
                            <th>最大DD</th>
                            <th>勝率</th>
                            <th>回数</th>
                          </tr>
                        </thead>
                        <tbody>
                          {gridSearch.results.slice(0, 8).map((row) => (
                            <tr key={`${row.thresholdPct}-${row.holdingDays}`}>
                              <td>{row.rank}</td>
                              <td>{row.thresholdPct.toFixed(2)}%</td>
                              <td>{row.holdingDays}日</td>
                              <td>{formatPercent(row.totalReturnPct)}</td>
                              <td>{row.sharpeRatio.toFixed(2)}</td>
                              <td>{formatPercent(-row.maxDrawdownPct)}</td>
                              <td>{formatPercent(row.winRatePct)}</td>
                              <td>{row.tradeCount}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="empty-state">条件比較を読み込めませんでした。</div>
                  )}
                </div>

                <div className="content-block">
                  <div className="table-header">
                    <div>
                      <h3>複数銘柄比較</h3>
                      <p>
                        {tickerCompare
                          ? `同じルールを ${tickerCompare.results.length} 銘柄に当てています。片道コスト ${tickerCompare.config.transactionCostPct.toFixed(3)}%。`
                          : '複数銘柄比較を読み込んでいます。'}
                      </p>
                      <p className="footnote">
                        {compareLoading ? '複数銘柄比較を更新中…' : '銘柄一覧を変えると自動で更新されます。'}
                      </p>
                    </div>
                  </div>

                  {tickerCompare && tickerCompare.results.length > 0 ? (
                    <div className="table-scroll">
                      <table className="results-table">
                        <thead>
                          <tr>
                            <th>銘柄</th>
                            <th>戦略</th>
                            <th>Sharpe</th>
                            <th>最大DD</th>
                            <th>勝率</th>
                            <th>回数</th>
                            <th>買い持ち</th>
                          </tr>
                        </thead>
                        <tbody>
                          {tickerCompare.results.map((row) => (
                            <tr key={row.ticker}>
                              <td>{row.ticker}</td>
                              <td>{formatPercent(row.strategy.totalReturnPct)}</td>
                              <td>{row.strategy.sharpeRatio.toFixed(2)}</td>
                              <td>{formatPercent(-row.strategy.maxDrawdownPct)}</td>
                              <td>{formatPercent(row.strategy.winRatePct ?? 0)}</td>
                              <td>{row.strategy.tradeCount ?? 0}</td>
                              <td>{formatPercent(row.benchmark.totalReturnPct)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="empty-state">複数銘柄比較を読み込めませんでした。</div>
                  )}
                </div>
              </>
            ) : (
              <div className="empty-state">実データの結果を読み込んでいます。</div>
            )}
          </section>

          <section className="panel controls">
            <div className="panel-title">
              <div>
                <h2>条件</h2>
                <p>ここを変えると結果が自動更新されます。</p>
              </div>
            </div>

            <div className="control-grid">
              <div className="field">
                <label htmlFor="ticker">銘柄コード</label>
                <input
                  id="ticker"
                  type="text"
                  value={ticker}
                  onChange={(event) => setTicker(event.target.value)}
                />
                <p className="footnote">例: SPY, QQQ, BTC-USD, AAPL</p>
              </div>

              <div className="field">
                <label htmlFor="comparison-tickers">比較する銘柄一覧</label>
                <input
                  id="comparison-tickers"
                  type="text"
                  value={comparisonTickers}
                  onChange={(event) => setComparisonTickers(event.target.value)}
                />
                <p className="footnote">例: SPY,QQQ,IWM,TLT,GLD,BTC-USD</p>
              </div>

              <div className="field">
                <label htmlFor="period">取得期間</label>
                <select id="period" value={period} onChange={(event) => setPeriod(event.target.value)}>
                  <option value="6mo">6か月</option>
                  <option value="1y">1年</option>
                  <option value="2y">2年</option>
                  <option value="5y">5年</option>
                  <option value="10y">10年</option>
                  <option value="max">全期間</option>
                </select>
              </div>

              <div className="field">
                <div className="range-header">
                  <label htmlFor="threshold">下落閾値</label>
                  <span className="range-value">{threshold.toFixed(1)}%</span>
                </div>
                <input
                  id="threshold"
                  type="range"
                  min="1"
                  max="8"
                  step="0.5"
                  value={threshold}
                  onChange={(event) => setThreshold(Number(event.target.value))}
                />
              </div>

              <div className="field">
                <label htmlFor="holding-days">保有日数</label>
                <input
                  id="holding-days"
                  type="number"
                  min="1"
                  max="30"
                  step="1"
                  value={holdingDays}
                  onChange={(event) => setHoldingDays(Number(event.target.value))}
                />
              </div>

              <div className="field">
                <label htmlFor="capital">初期資金</label>
                <input
                  id="capital"
                  type="number"
                  min="1000"
                  step="1000"
                  value={initialCapital}
                  onChange={(event) => setInitialCapital(Number(event.target.value))}
                />
              </div>

              <div className="field">
                <label htmlFor="transaction-cost">片道コスト (%)</label>
                <input
                  id="transaction-cost"
                  type="number"
                  min="0"
                  step="0.01"
                  value={transactionCost}
                  onChange={(event) => setTransactionCost(Number(event.target.value))}
                />
                <p className="footnote">例: 0.10 は 0.10%</p>
              </div>

              <div className="field">
                <label htmlFor="threshold-grid">比較する閾値一覧</label>
                <input
                  id="threshold-grid"
                  type="text"
                  value={thresholdGrid}
                  onChange={(event) => setThresholdGrid(event.target.value)}
                />
                <p className="footnote">例: 1.5,2,3,4,5</p>
              </div>

              <div className="field">
                <label htmlFor="holding-days-grid">比較する保有日数</label>
                <input
                  id="holding-days-grid"
                  type="text"
                  value={holdingDaysGrid}
                  onChange={(event) => setHoldingDaysGrid(event.target.value)}
                />
                <p className="footnote">例: 1,2,3</p>
              </div>
            </div>

            <p className={`status ${status.tone}`}>{status.text}</p>
          </section>
        </div>
      </div>
    </main>
  )
}

export default App
