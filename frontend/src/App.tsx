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

type StrategyComponentWithParameters = StrategyComponent & {
  parameters: Record<string, number>
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
      dataResolution: {
        key: string
        label: string
      }
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
      assetRankingModel: StrategyComponentWithParameters | null
      featureInputs: string[]
      filterRules: StrategyComponent[]
      fallbackRule: StrategyComponent | null
      tiltRule: StrategyComponentWithParameters | null
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
    evaluationContext: {
      datasetContext: {
        period: string
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

function formatStrategyParameters(parameters: Record<string, number>): string {
  const entries = Object.entries(parameters)
  if (entries.length === 0) {
    return 'なし'
  }
  return entries
    .map(([key, value]) => `${key}=${Number.isInteger(value) ? value.toFixed(0) : value.toFixed(2)}`)
    .join(' / ')
}

function App() {
  const [dashboard, setDashboard] = useState<DashboardResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
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
          <p className="panel-summary">比較実験を読み込んでいます。</p>
        </section>
      </main>
    )
  }

  if (error || !dashboard || !bestRun) {
    return (
      <main className="workspace-shell">
        <section className="panel">
          <p className="panel-summary error-text">{error ?? '結果を取得できませんでした。'}</p>
        </section>
      </main>
    )
  }

  return (
    <main className="workspace-shell">
      <section className="panel">
        <h2 className="panel-title">{bestRun.strategy.label}</h2>
        {bestRun.strategy.hypothesis ? (
          <p className="panel-summary">{bestRun.strategy.hypothesis}</p>
        ) : null}

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
                <dd>
                  {bestRun.strategy.components.optional.assetRankingModel
                    ? `${bestRun.strategy.components.optional.assetRankingModel.label} (${formatStrategyParameters(
                        bestRun.strategy.components.optional.assetRankingModel.parameters,
                      )})`
                    : 'なし'}
                </dd>
              </div>
              <div>
                <dt>データ粒度</dt>
                <dd>{bestRun.strategy.components.core.dataResolution.label}</dd>
              </div>
              <div>
                <dt>特徴量</dt>
                <dd>{bestRun.strategy.components.optional.featureInputs.join(' + ')}</dd>
              </div>
              <div>
                <dt>ティルト</dt>
                <dd>
                  {bestRun.strategy.components.optional.tiltRule
                    ? `${bestRun.strategy.components.optional.tiltRule.label} (${formatStrategyParameters(
                        bestRun.strategy.components.optional.tiltRule.parameters,
                      )})`
                    : 'なし'}
                </dd>
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
            </dl>
          </section>
        </div>
      </section>
    </main>
  )
}

export default App
