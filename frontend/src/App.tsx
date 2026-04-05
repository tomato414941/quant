import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

type SummaryMetrics = {
  totalReturnPct: number
  sharpeRatio: number
  maxDrawdownPct: number
  turnoverPct: number
}

type StrategyComponent = {
  key: string
  label: string
}

type InvestmentUniverse = {
  key: string
  label: string
  assetCount: number
  tickers: string[]
}

type StrategyDefinition = {
  strategyId: string
  version: string
  label: string
  hypothesis: string
  components: {
    core: {
      investmentUniverse: InvestmentUniverse
      portfolioModel: {
        key: string
        label: string
      }
      executionPolicy: {
        key: string
        label: string
      }
    }
    optional: {
      assetRankingModel: StrategyComponent | null
      featureInputs: string[]
      filterRules: StrategyComponent[]
      fallbackRule: StrategyComponent | null
      riskControls: {
        maxInvestmentPct: number
        maxWeightPct: number | null
      }
    }
  }
}

type PortfolioRun = {
  key: string
  strategy: StrategyDefinition
  summary: SummaryMetrics
  splitAnalysis: {
    test: {
      portfolio: SummaryMetrics
    }
  }
}

type RunCatalogRecord = {
  runKey: string
  savedAtUtc: string | null
  strategyLabel: string
  strategyVersion: string
  strategyHypothesis: string
  investmentUniverseLabel: string
  investmentUniverseAssetCount: number
  portfolioModelLabel: string
  executionLabel: string
  period: string
  commissionPct: number
  maxInvestmentPct: number
  maxWeightPct: number | null
  sharpeRatio: number
  totalReturnPct: number
  maxDrawdownPct: number
}

type RunCatalogResult = {
  records: RunCatalogRecord[]
}

type DashboardResult = {
  study: {
    id: string
    title: string
    question: string
    selectionPolicy: {
      primaryMetric: string
      secondaryMetric: string
      tertiaryMetric: string
    }
    marketUniverse: {
      assetCount: number
      tickers: string[]
    }
    evaluationContext: {
      datasetContext: {
        period: string
        frequency: string
        sanityPeriods: string[]
      }
      evaluationSettings: {
        splitRatioPct: number
        benchmark: string
      }
      costAssumptions: {
        commissionPct: number
      }
    }
  }
  runs: PortfolioRun[]
}

type ViewKey = 'decision' | 'strategies' | 'runs'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  `${window.location.protocol}//${window.location.hostname}:8000`

function formatPercent(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
}

function formatWeightCap(value: number | null): string {
  return value === null ? '上限なし' : `${value.toFixed(1)}%`
}

function formatBenchmarkLabel(value: string): string {
  if (value === 'equal_weight_buy_and_hold_with_cash') {
    return '等金額買い持ち + CASH'
  }
  return value
}

function formatMetricLabel(value: string): string {
  if (value === 'sharpe_ratio') {
    return 'シャープレシオ'
  }
  if (value === 'total_return') {
    return '総リターン'
  }
  if (value === 'max_drawdown') {
    return '最大ドローダウン'
  }
  return value
}

function formatRiskControls(maxInvestmentPct: number, maxWeightPct: number | null): string {
  return `${maxInvestmentPct.toFixed(0)}%投資 / ${formatWeightCap(maxWeightPct)}`
}

function formatFilters(filters: StrategyComponent[]): string {
  if (filters.length === 0) {
    return 'なし'
  }
  return filters.map((filter) => filter.label).join(' / ')
}

function formatSavedAt(value: string | null): string {
  if (!value) {
    return '-'
  }
  return new Date(value).toLocaleString('ja-JP', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function App() {
  const [dashboard, setDashboard] = useState<DashboardResult | null>(null)
  const [runCatalog, setRunCatalog] = useState<RunCatalogResult | null>(null)
  const [view, setView] = useState<ViewKey>('decision')
  const [loading, setLoading] = useState(true)
  const [catalogLoading, setCatalogLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [catalogError, setCatalogError] = useState<string | null>(null)
  const hasLoadedRef = useRef(false)

  useEffect(() => {
    if (hasLoadedRef.current) {
      return
    }
    hasLoadedRef.current = true

    async function loadDashboard() {
      setLoading(true)
      setError(null)

      try {
        const response = await fetch(`${API_BASE_URL}/api/dashboard`)
        const payload = await response.json()

        if (!response.ok) {
          throw new Error(payload.detail ?? '比較実験の取得に失敗しました。')
        }

        setDashboard(payload)
      } catch (caughtError) {
        setError(
          caughtError instanceof Error ? caughtError.message : '不明なエラーが発生しました。',
        )
      } finally {
        setLoading(false)
      }
    }

    void loadDashboard()
  }, [])

  useEffect(() => {
    if (view !== 'runs' || runCatalog !== null || catalogLoading) {
      return
    }

    async function loadCatalog() {
      setCatalogLoading(true)
      setCatalogError(null)

      try {
        const response = await fetch(`${API_BASE_URL}/api/run-catalog?run_kind=dashboard&limit=20`)
        const payload = await response.json()

        if (!response.ok) {
          throw new Error(payload.detail ?? 'Run Catalog の取得に失敗しました。')
        }

        setRunCatalog(payload)
      } catch (caughtError) {
        setCatalogError(
          caughtError instanceof Error ? caughtError.message : '不明なエラーが発生しました。',
        )
      } finally {
        setCatalogLoading(false)
      }
    }

    void loadCatalog()
  }, [view, runCatalog, catalogLoading])

  const sortedRuns = useMemo(() => {
    if (!dashboard) {
      return []
    }

    return dashboard.runs.slice().sort((left, right) => {
      if (right.summary.sharpeRatio !== left.summary.sharpeRatio) {
        return right.summary.sharpeRatio - left.summary.sharpeRatio
      }
      if (right.summary.totalReturnPct !== left.summary.totalReturnPct) {
        return right.summary.totalReturnPct - left.summary.totalReturnPct
      }
      return left.summary.maxDrawdownPct - right.summary.maxDrawdownPct
    })
  }, [dashboard])

  const bestRun = sortedRuns[0] ?? null

  if (loading && !dashboard) {
    return (
      <main className="workspace-shell">
        <section className="panel">
          <p className="panel-kicker">Decision</p>
          <h1>Strategy Workbench</h1>
          <p className="panel-summary">比較実験を読み込んでいます。</p>
        </section>
      </main>
    )
  }

  if (error || !dashboard || !bestRun) {
    return (
      <main className="workspace-shell">
        <section className="panel">
          <p className="panel-kicker">Decision</p>
          <h1>Strategy Workbench</h1>
          <p className="panel-summary error-text">{error ?? '結果を取得できませんでした。'}</p>
        </section>
      </main>
    )
  }

  return (
    <main className="workspace-shell">
      <section className="workspace-header">
        <p className="panel-kicker">Study</p>
        <h1>{dashboard.study.title}</h1>
        <p className="panel-summary">{dashboard.study.question}</p>
        <nav className="workspace-nav" aria-label="Primary">
          <button
            className={view === 'decision' ? 'is-active' : ''}
            onClick={() => setView('decision')}
            type="button"
          >
            Decision
          </button>
          <button
            className={view === 'strategies' ? 'is-active' : ''}
            onClick={() => setView('strategies')}
            type="button"
          >
            Strategies
          </button>
          <button
            className={view === 'runs' ? 'is-active' : ''}
            onClick={() => setView('runs')}
            type="button"
          >
            Run Catalog
          </button>
        </nav>
      </section>

      {view === 'decision' ? (
        <section className="panel">
          <p className="panel-kicker">Strategy</p>
          <h2 className="panel-title">{bestRun.strategy.label}</h2>

          <div className="metric-grid">
            <article>
              <span>シャープレシオ</span>
              <strong>{bestRun.summary.sharpeRatio.toFixed(2)}</strong>
            </article>
            <article>
              <span>総リターン</span>
              <strong>{formatPercent(bestRun.summary.totalReturnPct)}</strong>
            </article>
            <article>
              <span>最大ドローダウン</span>
              <strong>{formatPercent(-bestRun.summary.maxDrawdownPct)}</strong>
            </article>
            <article>
              <span>検証リターン</span>
              <strong>{formatPercent(bestRun.splitAnalysis.test.portfolio.totalReturnPct)}</strong>
            </article>
          </div>

          <div className="section-grid">
            <section className="subpanel">
              <h3>Strategy</h3>
              <dl className="detail-list">
                <div>
                  <dt>投資対象</dt>
                  <dd>{bestRun.strategy.components.core.investmentUniverse.label}</dd>
                </div>
                <div>
                  <dt>資産数</dt>
                  <dd>{bestRun.strategy.components.core.investmentUniverse.assetCount}資産</dd>
                </div>
                <div>
                  <dt>ランキングモデル</dt>
                  <dd>{bestRun.strategy.components.optional.assetRankingModel?.label ?? 'なし'}</dd>
                </div>
                <div>
                  <dt>特徴量</dt>
                  <dd>{bestRun.strategy.components.optional.featureInputs.join(' + ')}</dd>
                </div>
                <div>
                  <dt>フィルタ</dt>
                  <dd>{formatFilters(bestRun.strategy.components.optional.filterRules)}</dd>
                </div>
                <div>
                  <dt>ポートフォリオモデル</dt>
                  <dd>{bestRun.strategy.components.core.portfolioModel.label}</dd>
                </div>
                <div>
                  <dt>執行方針</dt>
                  <dd>{bestRun.strategy.components.core.executionPolicy.label}</dd>
                </div>
                <div>
                  <dt>リスク制御</dt>
                  <dd>
                    {formatRiskControls(
                      bestRun.strategy.components.optional.riskControls.maxInvestmentPct,
                      bestRun.strategy.components.optional.riskControls.maxWeightPct,
                    )}
                  </dd>
                </div>
              </dl>
            </section>

            <section className="subpanel">
              <h3>Evaluation Context</h3>
              <dl className="detail-list">
                <div>
                  <dt>評価期間</dt>
                  <dd>{dashboard.study.evaluationContext.datasetContext.period}</dd>
                </div>
                <div>
                  <dt>補助確認</dt>
                  <dd>{dashboard.study.evaluationContext.datasetContext.sanityPeriods.join(' / ') || 'なし'}</dd>
                </div>
                <div>
                  <dt>頻度</dt>
                  <dd>{dashboard.study.evaluationContext.datasetContext.frequency}</dd>
                </div>
                <div>
                  <dt>手数料</dt>
                  <dd>{dashboard.study.evaluationContext.costAssumptions.commissionPct.toFixed(2)}%</dd>
                </div>
                <div>
                  <dt>分割</dt>
                  <dd>
                    学習 {dashboard.study.evaluationContext.evaluationSettings.splitRatioPct.toFixed(1)}% / 検証{' '}
                    {(100 - dashboard.study.evaluationContext.evaluationSettings.splitRatioPct).toFixed(1)}%
                  </dd>
                </div>
                <div>
                  <dt>ベンチマーク</dt>
                  <dd>{formatBenchmarkLabel(dashboard.study.evaluationContext.evaluationSettings.benchmark)}</dd>
                </div>
                <div>
                  <dt>選定基準</dt>
                  <dd>
                    {[
                      formatMetricLabel(dashboard.study.selectionPolicy.primaryMetric),
                      formatMetricLabel(dashboard.study.selectionPolicy.secondaryMetric),
                      formatMetricLabel(dashboard.study.selectionPolicy.tertiaryMetric),
                    ].join(' → ')}
                  </dd>
                </div>
                <div>
                  <dt>Study Universe</dt>
                  <dd>{dashboard.study.marketUniverse.assetCount}資産</dd>
                </div>
              </dl>
            </section>
          </div>
        </section>
      ) : null}

      {view === 'strategies' ? (
        <section className="panel">
          <p className="panel-kicker">Strategies</p>
          <h2 className="panel-title">Strategy 比較</h2>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>投資対象</th>
                  <th>Portfolio</th>
                  <th>Execution</th>
                  <th>Sharpe</th>
                  <th>Return</th>
                  <th>MaxDD</th>
                  <th>Test</th>
                </tr>
              </thead>
              <tbody>
                {sortedRuns.map((run) => (
                  <tr key={run.key}>
                    <td>
                      <strong>{run.strategy.label}</strong>
                      <span>{run.strategy.hypothesis}</span>
                    </td>
                    <td>{run.strategy.components.core.investmentUniverse.label}</td>
                    <td>{run.strategy.components.core.portfolioModel.label}</td>
                    <td>{run.strategy.components.core.executionPolicy.label}</td>
                    <td>{run.summary.sharpeRatio.toFixed(2)}</td>
                    <td>{formatPercent(run.summary.totalReturnPct)}</td>
                    <td>{formatPercent(-run.summary.maxDrawdownPct)}</td>
                    <td>{formatPercent(run.splitAnalysis.test.portfolio.totalReturnPct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {view === 'runs' ? (
        <section className="panel">
          <p className="panel-kicker">Run Catalog</p>
          <h2 className="panel-title">保存済み Run</h2>
          {catalogLoading && !runCatalog ? (
            <p className="panel-summary">Run Catalog を読み込んでいます。</p>
          ) : null}
          {catalogError ? <p className="panel-summary error-text">{catalogError}</p> : null}
          {runCatalog ? (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Saved</th>
                    <th>Strategy</th>
                    <th>投資対象</th>
                    <th>期間</th>
                    <th>Portfolio</th>
                    <th>Execution</th>
                    <th>Sharpe</th>
                    <th>Return</th>
                    <th>MaxDD</th>
                  </tr>
                </thead>
                <tbody>
                  {runCatalog.records.map((record) => (
                    <tr key={record.runKey}>
                      <td>{formatSavedAt(record.savedAtUtc)}</td>
                      <td>
                        <strong>{record.strategyLabel}</strong>
                        <span>{record.strategyHypothesis}</span>
                      </td>
                      <td>{record.investmentUniverseLabel}</td>
                      <td>{record.period}</td>
                      <td>{record.portfolioModelLabel}</td>
                      <td>{record.executionLabel}</td>
                      <td>{record.sharpeRatio.toFixed(2)}</td>
                      <td>{formatPercent(record.totalReturnPct)}</td>
                      <td>{formatPercent(-record.maxDrawdownPct)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </section>
      ) : null}
    </main>
  )
}

export default App
