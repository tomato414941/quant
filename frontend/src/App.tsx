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

type PortfolioStrategyDefinition = {
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

type ExecutionModel = {
  label: string
  commissionPct: number
}

type SplitSegment = {
  portfolio: SummaryMetrics
}

type PortfolioRun = {
  key: string
  strategy: PortfolioStrategyDefinition
  portfolioModel: PortfolioModelDefinition
  executionModel: ExecutionModel
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
    datasetSpec: {
      period: string
      tickers: string[]
    }
    backtestConfig: {
      maxInvestmentPct: number
      maxWeightPct: number | null
      benchmark: string
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

function formatWeights(weights: PortfolioRun['weights']): string {
  return weights
    .filter((row) => row.weightPct > 0)
    .slice(0, 4)
    .map((row) => `${row.asset} ${row.weightPct.toFixed(1)}%`)
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
          <p className="decision-kicker">Decision</p>
          <h1>現在の暫定結論</h1>
          <p className="decision-summary">比較実験を読み込んでいます。</p>
        </section>
      </main>
    )
  }

  if (error || !dashboard || !bestRun) {
    return (
      <main className="decision-shell">
        <section className="decision-card">
          <p className="decision-kicker">Decision</p>
          <h1>現在の暫定結論</h1>
          <p className="decision-summary error-text">{error ?? '結果を取得できませんでした。'}</p>
        </section>
      </main>
    )
  }

  return (
    <main className="decision-shell">
      <section className="decision-card">
        <p className="decision-kicker">Decision</p>
        <h1>現在の暫定結論</h1>

        <div className="decision-highlight">
          <h2>{bestRun.strategy.label} × {bestRun.portfolioModel.label}</h2>
        </div>

        <div className="decision-metrics">
          <article>
            <span>Sharpe</span>
            <strong>{bestRun.summary.sharpeRatio.toFixed(2)}</strong>
          </article>
          <article>
            <span>Total Return</span>
            <strong>{formatPercent(bestRun.summary.totalReturnPct)}</strong>
          </article>
          <article>
            <span>Max Drawdown</span>
            <strong>{formatPercent(-bestRun.summary.maxDrawdownPct)}</strong>
          </article>
          <article>
            <span>Test Return</span>
            <strong>{formatPercent(bestRun.splitAnalysis.test.portfolio.totalReturnPct)}</strong>
          </article>
        </div>

        <dl className="decision-details">
          <div>
            <dt>投資対象</dt>
            <dd>{dashboard.study.datasetSpec.tickers.length}資産のマルチアセット</dd>
          </div>
          <div>
            <dt>評価期間</dt>
            <dd>{dashboard.study.datasetSpec.period}</dd>
          </div>
          <div>
            <dt>執行条件</dt>
            <dd>
              {bestRun.executionModel.label} / {dashboard.study.backtestConfig.maxInvestmentPct.toFixed(0)}%投資 / {formatWeightCap(dashboard.study.backtestConfig.maxWeightPct)} / {bestRun.executionModel.commissionPct.toFixed(2)}%手数料
            </dd>
          </div>
          <div>
            <dt>ランキング</dt>
            <dd>{bestRun.strategy.scoreModel.label}</dd>
          </div>
          <div>
            <dt>特徴量</dt>
            <dd>{bestRun.strategy.featureInputs.join(' + ')}</dd>
          </div>
          <div>
            <dt>スコア係数</dt>
            <dd>{formatScoreParameters(bestRun.strategy.scoreParameters)}</dd>
          </div>
          <div>
            <dt>フィルタ</dt>
            <dd>{formatFilters(bestRun.strategy.filterRules)}</dd>
          </div>
          <div>
            <dt>候補資産</dt>
            <dd>{bestRun.selectedAssets.join(', ')}</dd>
          </div>
          <div>
            <dt>直近ウェイト</dt>
            <dd>{formatWeights(bestRun.weights)}</dd>
          </div>
          <div>
            <dt>Turnover</dt>
            <dd>{formatPercent(bestRun.summary.turnoverPct)}</dd>
          </div>
          <div>
            <dt>比較基準</dt>
            <dd>{formatBenchmarkLabel(dashboard.study.backtestConfig.benchmark)}</dd>
          </div>
        </dl>
      </section>
    </main>
  )
}

export default App
