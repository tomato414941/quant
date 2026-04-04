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
  turnoverPct: number
}

type StrategyComponent = {
  key: string
  label: string
}

type PortfolioStrategyDefinition = {
  key: string
  strategyType: string
  label: string
  description: string
  featureInputs: string[]
  universePolicy: StrategyComponent
  scoreModel: StrategyComponent
  scoreParameters: Record<string, number>
  filterRules: StrategyComponent[]
  fallbackRule: StrategyComponent
}

type PortfolioModelDefinition = {
  key: string
  modelType: string
  label: string
  description: string
}

type ExecutionModel = {
  key: string
  label: string
  entry: string
  commissionPct: number
  slippagePct: number
  rebalanceFrequency: string
}

type ConditionVariant = {
  key: string
  label: string
  commissionPct: number
  maxInvestmentPct: number
  maxWeightPct: number | null
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
  executionModel: ExecutionModel
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
    sanityPeriods: string[]
    frequency: string
    source: string
    alignedStartDate: string
    alignedEndDate: string
    rowCount: number
  }
  executionVariants: ExecutionModel[]
  backtestConfig: {
    splitRatioPct: number
    initialCapital: number
    maxInvestmentPct: number
    maxWeightPct: number | null
    benchmark: string
  }
  portfolioState: {
    weights: Array<{
      asset: string
      weightPct: number
    }>
  }
  strategyDefinitions: PortfolioStrategyDefinition[]
  portfolioModels: PortfolioModelDefinition[]
}

type RunStoreSummary = {
  cachedRunCount: number
  computedRunCount: number
}

type DashboardResult = {
  study: StudyResult
  runs: PortfolioRun[]
  comparisonSeries: ComparisonRow[]
  runStoreSummary: RunStoreSummary
  sanityChecks: Array<{
    period: string
    datasetSpec: {
      tickers: string[]
      period: string
      frequency: string
      source: string
      alignedStartDate: string
      alignedEndDate: string
      rowCount: number
    }
    runStoreSummary: RunStoreSummary
    runs: PortfolioRun[]
  }>
}

type ConditionSweepRun = {
  key: string
  strategy: PortfolioStrategyDefinition
  portfolioModel: PortfolioModelDefinition
  executionModel: ExecutionModel
  conditionVariant: ConditionVariant
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
}

type ConditionSweepResult = {
  study: StudyResult
  conditionVariants: ConditionVariant[]
  resultCount: number
  runStoreSummary: RunStoreSummary
  results: ConditionSweepRun[]
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
  const positiveWeights = weights.filter((row) => row.weightPct > 0)
  const cashRow = positiveWeights.find((row) => row.asset === 'CASH')
  const assetRows = positiveWeights.filter((row) => row.asset !== 'CASH').slice(0, 3)
  const rows = cashRow ? [...assetRows, cashRow] : assetRows
  return rows.map((row) => `${row.asset} ${row.weightPct.toFixed(1)}%`).join(' / ')
}

function formatPortfolioStateWeights(weights: StudyResult['portfolioState']['weights']): string {
  const topRows = weights.filter((row) => row.weightPct > 0).slice(0, 5)
  return topRows.map((row) => `${row.asset} ${row.weightPct.toFixed(1)}%`).join(' / ')
}

function formatRunLabel(run: PortfolioRun): string {
  return `${run.strategy.label} × ${run.portfolioModel.label} × ${run.executionModel.label}`
}

function formatWeightCap(value: number | null): string {
  return value === null ? '上限なし' : `${value.toFixed(1)}%`
}

function formatFeatureInputs(inputs: string[]): string {
  return inputs.join(' + ')
}

function formatFilterRules(filters: StrategyComponent[]): string {
  if (filters.length === 0) {
    return 'なし'
  }
  return filters.map((filter) => filter.label).join(' / ')
}

function formatScoreParameters(parameters: Record<string, number>): string {
  const entries = Object.entries(parameters)
  if (entries.length === 0) {
    return 'なし'
  }
  return entries
    .map(([key, value]) => `${key}=${value.toFixed(2)}`)
    .join(' / ')
}

function App() {
  const [dashboard, setDashboard] = useState<DashboardResult | null>(null)
  const [conditionSweep, setConditionSweep] = useState<ConditionSweepResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<StatusState>({
    tone: 'success',
    text: '戦略とポートフォリオ比較実験を読み込んでいます。',
  })

  const refreshDashboard = useEffectEvent(async () => {
    setLoading(true)
    try {
      const [dashboardResponse, sweepResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/api/dashboard`),
        fetch(`${API_BASE_URL}/api/condition-sweep`),
      ])
      const dashboardPayload = await dashboardResponse.json()
      const sweepPayload = await sweepResponse.json()
      if (!dashboardResponse.ok) {
        throw new Error(dashboardPayload.detail ?? '比較実験の取得に失敗しました。')
      }
      if (!sweepResponse.ok) {
        throw new Error(sweepPayload.detail ?? '条件感度の取得に失敗しました。')
      }

      startTransition(() => {
        setDashboard(dashboardPayload)
        setConditionSweep(sweepPayload)
      })
      setStatus({
        tone: 'success',
        text: `${dashboardPayload.study.datasetSpec.period} を主期間に、最良条件へ固定した ${dashboardPayload.runs.length} 候補を比較しています。条件感度は ${sweepPayload.resultCount} run です。比較実験は再利用 ${dashboardPayload.runStoreSummary.cachedRunCount} 件、条件感度は再利用 ${sweepPayload.runStoreSummary.cachedRunCount} 件です。`,
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
            この画面は、条件スイープで最良だった `年次 / 100%投資 / 45%上限 / 0.05%手数料` を固定し、
            10y を主期間にした `戦略 x ポートフォリオ構築法` の比較実験です。直近 3y は sanity check として
            別に並べ、長期で強いかと最近も壊れていないかを分けて見ます。
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
                      <span className="metric-label">補助確認</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.datasetSpec.sanityPeriods.join(', ')}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">戦略数</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.strategyDefinitions.length}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">共通期間</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.datasetSpec.alignedStartDate} - {dashboard.study.datasetSpec.alignedEndDate}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">共通行数</span>
                      <strong className="metric-value metric-value-text">
                        {formatNumber(dashboard.study.datasetSpec.rowCount)}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">配分法数</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.portfolioModels.length}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">執行頻度数</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.executionVariants.length}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">主条件</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.executionVariants[0]?.label ?? '-'} / {dashboard.study.backtestConfig.maxInvestmentPct.toFixed(0)}% 投資
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">現在ポートフォリオ</span>
                      <strong className="metric-value metric-value-text">
                        {formatPortfolioStateWeights(dashboard.study.portfolioState.weights)}
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
                        {dashboard.study.executionVariants[0]?.entry ?? '-'}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">手数料</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.executionVariants[0]?.commissionPct.toFixed(3) ?? '0.000'}%
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">スリッページ</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.executionVariants[0]?.slippagePct.toFixed(3) ?? '0.000'}%
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">執行頻度</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.executionVariants.map((variant) => variant.label).join(', ')}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">ベンチマーク</span>
                      <strong className="metric-value metric-value-text">等金額買い持ち + CASH</strong>
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
                      <span className="metric-label">最大投資比率</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.study.backtestConfig.maxInvestmentPct.toFixed(1)}%
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">weight cap</span>
                      <strong className="metric-value metric-value-text">
                        {formatWeightCap(dashboard.study.backtestConfig.maxWeightPct)}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">条件感度パターン</span>
                      <strong className="metric-value metric-value-text">
                        {conditionSweep?.conditionVariants.length ?? 0}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">等金額買い持ち + CASH</span>
                      <strong className="metric-value">
                        {formatPercent(benchmarkSummary?.totalReturnPct ?? 0)}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">再利用候補</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.runStoreSummary.cachedRunCount}
                      </strong>
                    </div>
                    <div className="metric">
                      <span className="metric-label">今回計算</span>
                      <strong className="metric-value metric-value-text">
                        {dashboard.runStoreSummary.computedRunCount}
                      </strong>
                    </div>
                  </div>
                </article>
              </div>

              <div className="content-block">
                <div className="table-header">
                  <div>
                    <h3>戦略定義</h3>
                    <p>戦略を `候補集合 / スコア / フィルタ / フォールバック` に分けて見せています。全資産とモメンタムは背反ではなく、別レイヤーです。</p>
                  </div>
                </div>
                <div className="cards cards-compact">
                  {dashboard.study.strategyDefinitions.map((strategy) => (
                    <article key={strategy.key} className="metric-card">
                      <h3>{strategy.label}</h3>
                      <div className="metric-grid">
                        <div className="metric">
                          <span className="metric-label">候補集合</span>
                          <strong className="metric-value metric-value-text">
                            {strategy.universePolicy.label}
                          </strong>
                        </div>
                        <div className="metric">
                          <span className="metric-label">スコア</span>
                          <strong className="metric-value metric-value-text">
                            {strategy.scoreModel.label}
                          </strong>
                        </div>
                        <div className="metric">
                          <span className="metric-label">フィルタ</span>
                          <strong className="metric-value metric-value-text">
                            {formatFilterRules(strategy.filterRules)}
                          </strong>
                        </div>
                        <div className="metric">
                          <span className="metric-label">スコア係数</span>
                          <strong className="metric-value metric-value-text">
                            {formatScoreParameters(strategy.scoreParameters)}
                          </strong>
                        </div>
                        <div className="metric">
                          <span className="metric-label">フォールバック</span>
                          <strong className="metric-value metric-value-text">
                            {strategy.fallbackRule.label}
                          </strong>
                        </div>
                        <div className="metric">
                          <span className="metric-label">入力データ</span>
                          <strong className="metric-value metric-value-text">
                            {formatFeatureInputs(strategy.featureInputs)}
                          </strong>
                        </div>
                      </div>
                      <p className="card-copy">{strategy.description}</p>
                    </article>
                  ))}
                </div>
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
                      name="等金額買い持ち + CASH"
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
                    : `${dashboard.study.datasetSpec.period} の各線は、最良条件に固定した 戦略 x 配分法 の組み合わせです。縦線より後ろが検証期間で、比較はすべて年次更新で揃えています。`}
                </p>
              </div>

              <div className="content-block">
                <div className="table-header">
                    <div>
                      <h3>組み合わせ結果</h3>
                    <p>戦略が候補資産を選び、配分法が重みを決めた結果を並べています。執行条件は最良だった年次 / 100%投資 / 45%上限 / 0.05%手数料で固定です。</p>
                  </div>
                </div>
                <div className="table-scroll">
                  <table className="results-table">
                    <thead>
                        <tr>
                          <th>戦略</th>
                          <th>配分法</th>
                          <th>執行頻度</th>
                          <th>候補資産</th>
                          <th>直近ウェイト</th>
                          <th>総リターン</th>
                        <th>Sharpe</th>
                        <th>最大DD</th>
                        <th>Turnover</th>
                        <th>検証</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dashboard.runs.map((run) => (
                        <tr key={run.key}>
                          <td>{run.strategy.label}</td>
                          <td>{run.portfolioModel.label}</td>
                          <td>{run.executionModel.label}</td>
                          <td>{run.selectedAssets.join(', ')}</td>
                          <td>{formatWeights(run.weights)}</td>
                          <td>{formatPercent(run.summary.totalReturnPct)}</td>
                          <td>{run.summary.sharpeRatio.toFixed(2)}</td>
                          <td>{formatPercent(-run.summary.maxDrawdownPct)}</td>
                          <td>{formatPercent(run.summary.turnoverPct)}</td>
                          <td>{formatPercent(run.splitAnalysis.test.portfolio.totalReturnPct)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {conditionSweep ? (
                <div className="content-block">
                  <div className="table-header">
                    <div>
                      <h3>条件感度トップ10</h3>
                      <p>
                        手数料・最大投資比率・weight cap の 27 条件を、現在の 27 候補すべてに当てた結果です。
                        合計 {conditionSweep.resultCount} run を Sharpe 順に並べています。再利用{' '}
                        {conditionSweep.runStoreSummary.cachedRunCount} 件、再計算{' '}
                        {conditionSweep.runStoreSummary.computedRunCount} 件です。
                      </p>
                    </div>
                  </div>
                  <div className="table-scroll">
                    <table className="results-table">
                      <thead>
                        <tr>
                          <th>戦略</th>
                          <th>配分法</th>
                          <th>執行頻度</th>
                          <th>手数料</th>
                          <th>最大投資</th>
                          <th>weight cap</th>
                          <th>総リターン</th>
                          <th>Sharpe</th>
                          <th>最大DD</th>
                          <th>Turnover</th>
                          <th>検証</th>
                        </tr>
                      </thead>
                      <tbody>
                        {conditionSweep.results.slice(0, 10).map((run) => (
                          <tr key={run.key}>
                            <td>{run.strategy.label}</td>
                            <td>{run.portfolioModel.label}</td>
                            <td>{run.executionModel.label}</td>
                            <td>{run.conditionVariant.commissionPct.toFixed(2)}%</td>
                            <td>{run.conditionVariant.maxInvestmentPct.toFixed(1)}%</td>
                            <td>{formatWeightCap(run.conditionVariant.maxWeightPct)}</td>
                            <td>{formatPercent(run.summary.totalReturnPct)}</td>
                            <td>{run.summary.sharpeRatio.toFixed(2)}</td>
                            <td>{formatPercent(-run.summary.maxDrawdownPct)}</td>
                            <td>{formatPercent(run.summary.turnoverPct)}</td>
                            <td>{formatPercent(run.splitAnalysis.test.portfolio.totalReturnPct)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : null}

              {dashboard.sanityChecks.map((sanityCheck) => (
                <div className="content-block" key={sanityCheck.period}>
                  <div className="table-header">
                    <div>
                      <h3>直近 {sanityCheck.period} の sanity check</h3>
                      <p>
                        主期間の結論が最近の相場でも大きく崩れていないかを見るための補助比較です。
                        共通期間は {sanityCheck.datasetSpec.alignedStartDate} - {sanityCheck.datasetSpec.alignedEndDate}
                        です。再利用 {sanityCheck.runStoreSummary.cachedRunCount} 件、再計算{' '}
                        {sanityCheck.runStoreSummary.computedRunCount} 件です。
                      </p>
                    </div>
                  </div>
                  <div className="table-scroll">
                    <table className="results-table">
                      <thead>
                        <tr>
                          <th>戦略</th>
                          <th>配分法</th>
                          <th>執行頻度</th>
                          <th>候補資産</th>
                          <th>直近ウェイト</th>
                          <th>総リターン</th>
                          <th>Sharpe</th>
                          <th>最大DD</th>
                          <th>Turnover</th>
                          <th>検証</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sanityCheck.runs.map((run) => (
                          <tr key={`${sanityCheck.period}-${run.key}`}>
                            <td>{run.strategy.label}</td>
                            <td>{run.portfolioModel.label}</td>
                            <td>{run.executionModel.label}</td>
                            <td>{run.selectedAssets.join(', ')}</td>
                            <td>{formatWeights(run.weights)}</td>
                            <td>{formatPercent(run.summary.totalReturnPct)}</td>
                            <td>{run.summary.sharpeRatio.toFixed(2)}</td>
                            <td>{formatPercent(-run.summary.maxDrawdownPct)}</td>
                            <td>{formatPercent(run.summary.turnoverPct)}</td>
                            <td>{formatPercent(run.splitAnalysis.test.portfolio.totalReturnPct)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
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
