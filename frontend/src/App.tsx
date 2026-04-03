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
  tradeCount?: number
  winRatePct?: number
}

type StrategyDefinition = {
  key: string
  engine: string
  label: string
  hypothesis: string
  thresholdPct: number
  holdingDays: number
}

type SplitSegment = {
  startDate: string
  endDate: string
  dayCount: number
  strategy: SummaryMetrics
  benchmark: SummaryMetrics
}

type StrategyRun = {
  definition: StrategyDefinition
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
    strategyEquity: number
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
    ticker: string
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
  strategyDefinitions: StrategyDefinition[]
}

type DashboardResult = {
  study: StudyResult
  runs: StrategyRun[]
  comparisonSeries: ComparisonRow[]
}

type StatusState = {
  tone: 'success' | 'error'
  text: string
}

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  `${window.location.protocol}//${window.location.hostname}:8000`

const STRATEGY_COLORS = ['#b45b2a', '#748a6c', '#2f6c74', '#9f734f', '#7a4d72']

function formatPercent(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat('ja-JP', {
    maximumFractionDigits: 0,
  }).format(value)
}

function describeRule(definition: StrategyDefinition): string {
  if (definition.engine === 'mean_reversion') {
    return `前日が ${definition.thresholdPct.toFixed(2)}% 以上下落したら ${definition.holdingDays} 日保有`
  }
  return `前日が ${definition.thresholdPct.toFixed(2)}% 以上上昇したら ${definition.holdingDays} 日保有`
}

function App() {
  const [dashboard, setDashboard] = useState<DashboardResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<StatusState>({
    tone: 'success',
    text: '比較実験を読み込んでいます。',
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
        text: `${payload.study.datasetSpec.ticker} / ${payload.study.datasetSpec.period} で ${payload.runs.length} 本の戦略を同条件比較しています。`,
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
          <h1>比較実験を、時系列で見る。</h1>
          <p className="hero-copy">
            この画面の主語は戦略そのものではなく、同一条件で複数戦略を比較する 1 つの比較実験です。
            何を検証しているか、どの条件で比べているか、結果がどうだったかを同じ構造で見せています。
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
                  <h3>データセット</h3>
                  <div className="metric-grid">
                    <div className="metric">
                      <span className="metric-label">銘柄</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.datasetSpec.ticker}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">期間</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.datasetSpec.period}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">頻度</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.datasetSpec.frequency}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">データ元</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.datasetSpec.source}
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
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.backtestConfig.benchmark}
                      </strong>
                    </div>
                  </div>
                </article>

                <article className="metric-card">
                  <h3>検証設定</h3>
                  <div className="metric-grid">
                    <div className="metric">
                      <span className="metric-label">戦略数</span>
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
                      <span className="metric-label">買い持ち</span>
                      <strong className="metric-value">
                        {formatPercent(benchmarkSummary?.totalReturnPct ?? 0)}
                      </strong>
                    </div>
                  </div>
                </article>
              </div>

              <div className="content-block">
                <ResponsiveContainer width="100%" height={360}>
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
                      name="買い持ち"
                    />
                    {dashboard.runs.map((run, index) => (
                      <Line
                        key={run.definition.key}
                        type="monotone"
                        dataKey={run.definition.key}
                        stroke={STRATEGY_COLORS[index % STRATEGY_COLORS.length]}
                        strokeWidth={2.5}
                        dot={false}
                        name={run.definition.label}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
                <p className="chart-note">
                  {loading
                    ? '比較実験を更新中…'
                    : '同じ比較実験の中で、買い持ちと各戦略の資産曲線を重ねています。縦線より後ろが検証期間です。'}
                </p>
              </div>

              <div className="content-block">
                <div className="table-header">
                  <div>
                    <h3>戦略ラン</h3>
                    <p>この比較実験の中で実行した各戦略ランのルール、仮説、結果です。</p>
                  </div>
                </div>
                <div className="table-scroll">
                  <table className="results-table">
                    <thead>
                      <tr>
                        <th>戦略</th>
                        <th>ルール</th>
                        <th>仮説</th>
                        <th>総リターン</th>
                        <th>Sharpe</th>
                        <th>最大DD</th>
                        <th>回数</th>
                        <th>検証</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dashboard.runs.map((run) => (
                        <tr key={run.definition.key}>
                          <td>{run.definition.label}</td>
                          <td>{describeRule(run.definition)}</td>
                          <td>{run.definition.hypothesis}</td>
                          <td>{formatPercent(run.summary.totalReturnPct)}</td>
                          <td>{run.summary.sharpeRatio.toFixed(2)}</td>
                          <td>{formatPercent(-run.summary.maxDrawdownPct)}</td>
                          <td>{run.summary.tradeCount ?? 0}</td>
                          <td>{formatPercent(run.splitAnalysis.test.strategy.totalReturnPct)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          ) : (
            <div className="empty-state">比較実験を読み込んでいます。</div>
          )}
        </section>
      </div>
    </main>
  )
}

export default App
