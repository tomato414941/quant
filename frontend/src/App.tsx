import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

type SummaryMetrics = {
  totalReturnPct: number
  sharpeRatio: number
  maxDrawdownPct: number
  turnoverPct: number
}

type StrategyComponent = {
  label: string
}

type SelectionDefinition = {
  label: string
  description: string
  featureInputs: string[]
  universePolicy: StrategyComponent
  scoreModel: StrategyComponent
  scoreParameters: Record<string, number>
  filterRules: StrategyComponent[]
}

type PortfolioModelDefinition = {
  label: string
}

type ExecutionPolicy = {
  label: string
}

type RiskControls = {
  maxInvestmentPct: number
  maxWeightPct: number | null
}

type StrategyDefinition = {
  label: string
  selectionDefinition: SelectionDefinition
  portfolioModel: PortfolioModelDefinition
  executionPolicy: ExecutionPolicy
  riskControls: RiskControls
}

type SplitSegment = {
  portfolio: SummaryMetrics
}

type PortfolioRun = {
  key: string
  strategy: StrategyDefinition
  selectedAssets: string[]
  weights: Array<{
    asset: string
    weightPct: number
  }>
  summary: SummaryMetrics
  splitAnalysis: {
    test: SplitSegment
  }
}

type DashboardResult = {
  study: {
    marketUniverse: {
      assetCount: number
      tickers: string[]
    }
    evaluationContext: {
      datasetContext: {
        period: string
      }
      evaluationSettings: {
        splitRatioPct: number
        benchmark: string
      }
      costAssumptions: {
        commissionPct: number
      }
    }
    datasetSpec?: {
      period: string
      tickers: string[]
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

function formatRiskControls(maxInvestmentPct: number, maxWeightPct: number | null): string {
  return `${maxInvestmentPct.toFixed(0)}%投資 / ${formatWeightCap(maxWeightPct)}`
}

function formatScoreParameters(parameters: Record<string, number>): string {
  const entries = Object.entries(parameters)
  if (entries.length === 0) {
    return 'なし'
  }

  return entries.map(([key, value]) => `${key}=${value.toFixed(2)}`).join(' / ')
}

function formatFilters(filters: StrategyComponent[]): string {
  if (filters.length === 0) {
    return 'なし'
  }

  return filters.map((filter) => filter.label).join(' / ')
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

  const bestRun = useMemo(() => {
    if (!dashboard) {
      return null
    }

    return dashboard.runs
      .slice()
      .sort((left, right) => right.summary.sharpeRatio - left.summary.sharpeRatio)[0] ?? null
  }, [dashboard])

  if (loading && !dashboard) {
    return (
      <main className="decision-shell">
        <section className="decision-card">
          <p className="decision-kicker">Strategy</p>
          <h1>現在の最有力 Strategy</h1>
          <p className="decision-summary">比較実験を読み込んでいます。</p>
        </section>
      </main>
    )
  }

  if (error || !dashboard || !bestRun) {
    return (
      <main className="decision-shell">
        <section className="decision-card">
          <p className="decision-kicker">Strategy</p>
          <h1>現在の最有力 Strategy</h1>
          <p className="decision-summary error-text">{error ?? '結果を取得できませんでした。'}</p>
        </section>
      </main>
    )
  }

  return (
    <main className="decision-shell">
      <section className="decision-card">
        <p className="decision-kicker">Strategy</p>
        <h1>現在の最有力 Strategy</h1>

        <div className="decision-highlight">
          <h2>
            {bestRun.strategy.label}
          </h2>
        </div>

        <div className="decision-metrics">
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

        <section className="decision-section">
          <h3>Strategyの構成</h3>
          <dl className="decision-details">
            <div>
              <dt>投資対象</dt>
              <dd>{bestRun.strategy.selectionDefinition.universePolicy.label}</dd>
            </div>
            <div>
              <dt>ランキングモデル</dt>
              <dd>{bestRun.strategy.selectionDefinition.scoreModel.label}</dd>
            </div>
            <div>
              <dt>特徴量</dt>
              <dd>{bestRun.strategy.selectionDefinition.featureInputs.join(' + ')}</dd>
            </div>
            <div>
              <dt>ランキング係数</dt>
              <dd>{formatScoreParameters(bestRun.strategy.selectionDefinition.scoreParameters)}</dd>
            </div>
            <div>
              <dt>フィルタ</dt>
              <dd>{formatFilters(bestRun.strategy.selectionDefinition.filterRules)}</dd>
            </div>
            <div>
              <dt>ポートフォリオモデル</dt>
              <dd>{bestRun.strategy.portfolioModel.label}</dd>
            </div>
            <div>
              <dt>執行方針</dt>
              <dd>{bestRun.strategy.executionPolicy.label}</dd>
            </div>
            <div>
              <dt>リスク制御</dt>
              <dd>
                {formatRiskControls(
                  bestRun.strategy.riskControls.maxInvestmentPct,
                  bestRun.strategy.riskControls.maxWeightPct,
                )}
              </dd>
            </div>
          </dl>
        </section>

        <section className="decision-section">
          <h3>この評価の前提</h3>
          <dl className="decision-details">
            <div>
              <dt>評価期間</dt>
              <dd>{dashboard.study.evaluationContext.datasetContext.period}</dd>
            </div>
            <div>
              <dt>資産数</dt>
              <dd>{dashboard.study.marketUniverse.assetCount}資産</dd>
            </div>
            <div>
              <dt>手数料前提</dt>
              <dd>{dashboard.study.evaluationContext.costAssumptions.commissionPct.toFixed(2)}%</dd>
            </div>
            <div>
              <dt>分割</dt>
              <dd>学習 {dashboard.study.evaluationContext.evaluationSettings.splitRatioPct.toFixed(1)}% / 検証 {(100 - dashboard.study.evaluationContext.evaluationSettings.splitRatioPct).toFixed(1)}%</dd>
            </div>
            <div>
              <dt>比較基準</dt>
              <dd>{formatBenchmarkLabel(dashboard.study.evaluationContext.evaluationSettings.benchmark)}</dd>
            </div>
            <div>
              <dt>直近Turnover</dt>
              <dd>{formatPercent(bestRun.summary.turnoverPct)}</dd>
            </div>
          </dl>
        </section>
      </section>
    </main>
  )
}

export default App
