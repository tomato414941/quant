import { startTransition, useEffect, useEffectEvent, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
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
}

type PortfolioStrategyDefinition = {
  key: string
  strategyType: string
  label: string
  description: string
}

type PortfolioModelDefinition = {
  key: string
  modelType: string
  label: string
  description: string
}

type SplitSegment = {
  startDate: string
  endDate: string
  dayCount: number
  portfolio: SummaryMetrics
  benchmark: SummaryMetrics
}

type PortfolioRun = {
  key: string
  strategy: PortfolioStrategyDefinition
  portfolioModel: PortfolioModelDefinition
  weights: Array<{
    asset: string
    weightPct: number
  }>
  selectedAssets: string[]
  summary: SummaryMetrics
  benchmark: SummaryMetrics
  splitAnalysis: {
    config: {
      splitRatioPct: number
    }
    train: SplitSegment
    test: SplitSegment
  }
  series: Array<{
    date: string
    portfolioEquity: number
    benchmarkEquity: number
  }>
}

type ComparisonRow = {
  date: string
  benchmarkEquity: number
} & Record<string, number | string>

type StudyResult = {
  id: string
  title: string
  question: string
  datasetSpec: {
    tickers: string[]
    period: string
    frequency: string
    source: string
  }
  executionModel: {
    entry: string
    commissionPct: number
    slippagePct: number
  }
  backtestConfig: {
    splitRatioPct: number
    initialCapital: number
    benchmark: string
  }
  strategyDefinitions: PortfolioStrategyDefinition[]
  portfolioModels: PortfolioModelDefinition[]
}

type DashboardResult = {
  study: StudyResult
  runs: PortfolioRun[]
  comparisonSeries: ComparisonRow[]
}

type StatusState = {
  tone: 'success' | 'error'
  text: string
}

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  `${window.location.protocol}//${window.location.hostname}:8000`

const RUN_COLORS = ['#b45b2a', '#748a6c', '#2f6c74', '#9f734f', '#7a4d72', '#5d698f', '#8b5f3d']

function formatPercent(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat('ja-JP', {
    maximumFractionDigits: 0,
  }).format(value)
}

function formatWeights(weights: PortfolioRun['weights']): string {
  return weights
    .filter((row) => row.weightPct > 0)
    .slice(0, 3)
    .map((row) => `${row.asset} ${row.weightPct.toFixed(1)}%`)
    .join(' / ')
}

function formatRunLabel(run: PortfolioRun): string {
  return `${run.strategy.label} × ${run.portfolioModel.label}`
}

function App() {
  const [dashboard, setDashboard] = useState<DashboardResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<StatusState>({
    tone: 'success',
    text: '戦略とポートフォリオ比較実験を読み込んでいます。',
  })

  const refreshDashboard = useEffectEvent(async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/dashboard`)
      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail ?? '比較実験の取得に失敗しました。')
      }

      startTransition(() => setDashboard(payload))
      setStatus({
        tone: 'success',
        text: `${payload.study.strategyDefinitions.length} 戦略 x ${payload.study.portfolioModels.length} ポートフォリオ構築法を比較しています。`,
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

  const splitDate = dashboard?.runs[0]?.splitAnalysis.test.startDate
  const benchmarkSummary = dashboard?.runs[0]?.benchmark

  return (
    <main className="app-shell">
      <div className="page">
        <section className="hero panel">
          <p className="eyebrow">ミニマルクオンツ</p>
          <h1>戦略と配分法を、同じ時系列で比べる。</h1>
          <p className="hero-copy">
            この画面は、同じETFユニバースに対して `戦略 x ポートフォリオ構築法` の組み合わせを比較する実験です。
            戦略は候補資産を選び、`skfolio` の配分法がその中で重みを決めます。
          </p>
        </section>

        <section className="panel results">
          <div className="panel-title">
            <div>
              <h2>{dashboard?.study.title ?? '比較実験'}</h2>
              <p>{dashboard?.study.question ?? 'まだ結果がありません。'}</p>
            </div>
          </div>

          <p className={`status ${status.tone}`}>{status.text}</p>

          {dashboard ? (
            <>
              <div className="cards cards-compact">
                <article className="metric-card">
                  <h3>ユニバース</h3>
                  <div className="metric-grid">
                    <div className="metric">
                      <span className="metric-label">銘柄群</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.datasetSpec.tickers.join(', ')}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">期間</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.datasetSpec.period}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">戦略数</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.strategyDefinitions.length}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">配分法数</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.portfolioModels.length}
                      </strong>
                    </div>
                  </div>
                </article>

                <article className="metric-card">
                  <h3>執行モデル</h3>
                  <div className="metric-grid">
                    <div className="metric">
                      <span className="metric-label">約定前提</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.executionModel.entry}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">手数料</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.executionModel.commissionPct.toFixed(3)}%
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">スリッページ</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.executionModel.slippagePct.toFixed(3)}%
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">ベンチマーク</span>
                      <strong className="metric-value metric-value-text">等金額買い持ち</strong>
                    </div>
                  </div>
                </article>

                <article className="metric-card">
                  <h3>検証設定</h3>
                  <div className="metric-grid">
                    <div className="metric">
                      <span className="metric-label">組み合わせ数</span>
                      <strong className="metric-value metric-value-text">{dashboard.runs.length}</strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">初期資金</span>
                      <strong className="metric-value metric-value-text">
                        {formatNumber(dashboard.study.backtestConfig.initialCapital)}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">学習比率</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.backtestConfig.splitRatioPct.toFixed(1)}%
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">等金額買い持ち</span>
                      <strong className="metric-value">
                        {formatPercent(benchmarkSummary?.totalReturnPct ?? 0)}
                      </strong>
                    </div>
                  </div>
                </article>
              </div>

              <div className="content-block">
                <ResponsiveContainer width="100%" height={380}>
                  <LineChart data={dashboard.comparisonSeries}>
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
                    {splitDate ? (
                      <ReferenceLine
                        x={splitDate}
                        stroke="#8c745f"
                        strokeDasharray="5 5"
                        label={{
                          value: '検証開始',
                          position: 'insideTopRight',
                          fill: '#8c745f',
                          fontSize: 12,
                        }}
                      />
                    ) : null}
                    <Line
                      type="monotone"
                      dataKey="benchmarkEquity"
                      stroke="#7d8f6f"
                      strokeWidth={2.2}
                      dot={false}
                      name="等金額買い持ち"
                    />
                    {dashboard.runs.map((run, index) => (
                      <Line
                        key={run.key}
                        type="monotone"
                        dataKey={run.key}
                        stroke={RUN_COLORS[index % RUN_COLORS.length]}
                        strokeWidth={2.3}
                        dot={false}
                        name={formatRunLabel(run)}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
                <p className="chart-note">
                  {loading
                    ? '比較実験を更新中…'
                    : '各線は 戦略 x 配分法 の組み合わせです。縦線より後ろが検証期間です。'}
                </p>
              </div>

              <div className="content-block">
                <div className="table-header">
                  <div>
                    <h3>組み合わせ結果</h3>
                    <p>戦略が候補資産を選び、配分法が重みを決めた結果を並べています。</p>
                  </div>
                </div>
                <div className="table-scroll">
                  <table className="results-table">
                    <thead>
                      <tr>
                        <th>戦略</th>
                        <th>配分法</th>
                        <th>候補資産</th>
                        <th>代表ウェイト</th>
                        <th>総リターン</th>
                        <th>Sharpe</th>
                        <th>最大DD</th>
                        <th>検証</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dashboard.runs.map((run) => (
                        <tr key={run.key}>
                          <td>{run.strategy.label}</td>
                          <td>{run.portfolioModel.label}</td>
                          <td>{run.selectedAssets.join(', ')}</td>
                          <td>{formatWeights(run.weights)}</td>
                          <td>{formatPercent(run.summary.totalReturnPct)}</td>
                          <td>{run.summary.sharpeRatio.toFixed(2)}</td>
                          <td>{formatPercent(-run.summary.maxDrawdownPct)}</td>
                          <td>{formatPercent(run.splitAnalysis.test.portfolio.totalReturnPct)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          ) : (
            <div className="empty-state">戦略とポートフォリオ比較実験を読み込んでいます。</div>
          )}
        </section>
      </div>
    </main>
  )
}

export default App
