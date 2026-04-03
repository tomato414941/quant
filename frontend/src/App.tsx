import { startTransition, useEffect, useEffectEvent, useState } from 'react'
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
  period?: string
  source: string
}

type StrategyConfig = {
  strategyId: string
  strategyLabel: string
  thresholdPct: number
  initialCapital: number
  holdingDays: number
  transactionCostPct: number
}

type SplitSegment = {
  startDate: string
  endDate: string
  dayCount: number
  strategy: SummaryMetrics
  benchmark: SummaryMetrics
}

type BacktestResult = {
  dataset: DatasetInfo
  summary: {
    strategy: SummaryMetrics
    benchmark: SummaryMetrics
    config: StrategyConfig
  }
  splitAnalysis: {
    config: {
      splitRatioPct: number
    }
    train: SplitSegment
    test: SplitSegment
  }
  series: Array<{
    date: string
    strategyEquity: number
    benchmarkEquity: number
  }>
}

type GridSearchResult = {
  dataset: DatasetInfo
  config: {
    strategyLabel: string
    thresholdValuesPct: number[]
    holdingDaysValues: number[]
  }
  results: Array<{
    rank: number
    thresholdPct: number
    holdingDays: number
    totalReturnPct: number
    sharpeRatio: number
    maxDrawdownPct: number
    winRatePct: number
    tradeCount: number
  }>
}

type TickerCompareResult = {
  config: {
    strategyLabel: string
    transactionCostPct: number
  }
  results: Array<{
    ticker: string
    period: string
    strategy: SummaryMetrics
    benchmark: SummaryMetrics
  }>
}

type PeriodCompareResult = {
  dataset: DatasetInfo
  config: {
    strategyLabel: string
  }
  results: Array<{
    period: string
    strategy: SummaryMetrics
    benchmark: SummaryMetrics
  }>
}

type DashboardResult = {
  config: {
    ticker: string
    period: string
    periodComparePeriods: string[]
    comparisonTickers: string[]
    strategyId: string
    strategyLabel: string
    thresholdPct: number
    holdingDays: number
    splitRatioPct: number
    initialCapital: number
    transactionCostPct: number
    thresholdGridPct: number[]
    holdingDaysGrid: number[]
  }
  single: BacktestResult
  gridSearch: GridSearchResult
  tickerCompare: TickerCompareResult
  periodCompare: PeriodCompareResult
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

function describeRule(config: DashboardResult['config']): string {
  if (config.strategyId === 'mean_reversion') {
    return `前日が ${config.thresholdPct.toFixed(2)}% 以上下落したら ${config.holdingDays} 日保有`
  }
  return `前日が ${config.thresholdPct.toFixed(2)}% 以上上昇したら ${config.holdingDays} 日保有`
}

function App() {
  const [dashboard, setDashboard] = useState<DashboardResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<StatusState>({
    tone: 'success',
    text: '固定プリセットの結果を読み込んでいます。',
  })

  const refreshDashboard = useEffectEvent(async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/dashboard`)
      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail ?? 'ダッシュボードの取得に失敗しました。')
      }

      startTransition(() => setDashboard(payload))
      setStatus({
        tone: 'success',
        text: `${payload.config.ticker} / ${payload.config.period} の固定ダッシュボードを表示しています。`,
      })
    } catch (caughtError) {
      setStatus({
        tone: 'error',
        text: caughtError instanceof Error ? caughtError.message : '不明なエラーが発生しました。',
      })
    } finally {
      setLoading(false)
    }
  })

  useEffect(() => {
    void refreshDashboard()
  }, [])

  return (
    <main className="app-shell">
      <div className="page">
        <section className="hero panel">
          <p className="eyebrow">ミニマルクオンツ</p>
          <h1>1つのルールから始める、最小のクオンツ検証。</h1>
          <p className="hero-copy">
            UI は固定プリセットの結果を見るためだけにしています。調整用の入力は置かず、
            採用中の条件で単発・学習/検証・複数期間・複数銘柄をそのまま比較します。
          </p>
        </section>

        <section className="panel results">
          <div className="panel-title">
            <div>
              <h2>結果</h2>
              <p>
                {dashboard
                  ? `${dashboard.config.ticker} / ${dashboard.config.period} / ${dashboard.config.strategyLabel} / ${describeRule(dashboard.config)}`
                  : 'まだ結果がありません。'}
              </p>
            </div>
          </div>

          <p className={`status ${status.tone}`}>{status.text}</p>

          {dashboard ? (
            <>
              <div className="cards cards-compact">
                <article className="metric-card">
                  <h3>採用条件</h3>
                  <div className="metric-grid">
                    <div className="metric">
                      <span className="metric-label">主銘柄</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.config.ticker}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">主期間</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.config.period}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">ルール</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.config.strategyLabel}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">片道コスト</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.config.transactionCostPct.toFixed(3)}%
                      </strong>
                    </div>
                  </div>
                  <p className="chart-note">
                    初期資金 {formatNumber(dashboard.config.initialCapital)} 円、学習期間比率{' '}
                    {dashboard.config.splitRatioPct.toFixed(1)}% です。
                  </p>
                </article>

                <article className="metric-card">
                  <h3>戦略</h3>
                  <div className="metric-grid">
                    <div className="metric">
                      <span className="metric-label">総リターン</span>
                      <strong className="metric-value">
                        {formatPercent(dashboard.single.summary.strategy.totalReturnPct)}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">シャープレシオ</span>
                      <strong className="metric-value">
                        {dashboard.single.summary.strategy.sharpeRatio.toFixed(2)}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">最大ドローダウン</span>
                      <strong className="metric-value">
                        {formatPercent(-dashboard.single.summary.strategy.maxDrawdownPct)}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">勝率</span>
                      <strong className="metric-value">
                        {formatPercent(dashboard.single.summary.strategy.winRatePct ?? 0)}
                      </strong>
                    </div>
                  </div>
                </article>

                <article className="metric-card">
                  <h3>ベンチマーク</h3>
                  <div className="metric-grid">
                    <div className="metric">
                      <span className="metric-label">総リターン</span>
                      <strong className="metric-value">
                        {formatPercent(dashboard.single.summary.benchmark.totalReturnPct)}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">シャープレシオ</span>
                      <strong className="metric-value">
                        {dashboard.single.summary.benchmark.sharpeRatio.toFixed(2)}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">最大ドローダウン</span>
                      <strong className="metric-value">
                        {formatPercent(-dashboard.single.summary.benchmark.maxDrawdownPct)}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">CAGR</span>
                      <strong className="metric-value">
                        {formatPercent(dashboard.single.summary.benchmark.cagrPct)}
                      </strong>
                    </div>
                  </div>
                </article>
              </div>

              <div className="content-block">
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={dashboard.single.series}>
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
                      name={dashboard.config.strategyLabel}
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
                  {loading
                    ? 'ダッシュボードを更新中…'
                    : `${dashboard.single.dataset.source} の実データです。UI では条件変更を受け付けません。`}
                </p>
              </div>

              <div className="content-block">
                <div className="table-header">
                  <div>
                    <h3>学習 / 検証分割</h3>
                    <p>
                      前半 {dashboard.single.splitAnalysis.config.splitRatioPct.toFixed(1)}% を学習、
                      後半を検証として固定表示しています。
                    </p>
                  </div>
                </div>
                <div className="table-scroll">
                  <table className="results-table">
                    <thead>
                      <tr>
                        <th>区間</th>
                        <th>期間</th>
                        <th>戦略</th>
                        <th>Sharpe</th>
                        <th>最大DD</th>
                        <th>勝率</th>
                        <th>回数</th>
                        <th>買い持ち</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        { label: '学習', value: dashboard.single.splitAnalysis.train },
                        { label: '検証', value: dashboard.single.splitAnalysis.test },
                      ].map((row) => (
                        <tr key={row.label}>
                          <td>{row.label}</td>
                          <td>{`${row.value.startDate} - ${row.value.endDate}`}</td>
                          <td>{formatPercent(row.value.strategy.totalReturnPct)}</td>
                          <td>{row.value.strategy.sharpeRatio.toFixed(2)}</td>
                          <td>{formatPercent(-row.value.strategy.maxDrawdownPct)}</td>
                          <td>{formatPercent(row.value.strategy.winRatePct ?? 0)}</td>
                          <td>{row.value.strategy.tradeCount ?? 0}</td>
                          <td>{formatPercent(row.value.benchmark.totalReturnPct)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="content-block">
                <div className="table-header">
                  <div>
                    <h3>条件比較</h3>
                    <p>
                      閾値 {dashboard.config.thresholdGridPct.join(', ')}% と保有日数{' '}
                      {dashboard.config.holdingDaysGrid.join(', ')} 日を固定で比較しています。
                    </p>
                  </div>
                </div>
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
                      {dashboard.gridSearch.results.slice(0, 8).map((row) => (
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
              </div>

              <div className="content-block">
                <div className="table-header">
                  <div>
                    <h3>複数期間比較</h3>
                    <p>
                      {dashboard.periodCompare.dataset.ticker} を{' '}
                      {dashboard.config.periodComparePeriods.join(', ')} で固定比較しています。
                    </p>
                  </div>
                </div>
                <div className="table-scroll">
                  <table className="results-table">
                    <thead>
                      <tr>
                        <th>期間</th>
                        <th>戦略</th>
                        <th>Sharpe</th>
                        <th>最大DD</th>
                        <th>勝率</th>
                        <th>回数</th>
                        <th>買い持ち</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dashboard.periodCompare.results.map((row) => (
                        <tr key={row.period}>
                          <td>{row.period}</td>
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
              </div>

              <div className="content-block">
                <div className="table-header">
                  <div>
                    <h3>複数銘柄比較</h3>
                    <p>
                      {dashboard.config.comparisonTickers.join(', ')} を同じルールで固定比較しています。
                    </p>
                  </div>
                </div>
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
                      {dashboard.tickerCompare.results.map((row) => (
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
              </div>
            </>
          ) : (
            <div className="empty-state">固定ダッシュボードを読み込んでいます。</div>
          )}
        </section>
      </div>
    </main>
  )
}

export default App
