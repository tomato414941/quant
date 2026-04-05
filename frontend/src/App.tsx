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

function formatDataResolutionLabel(value: string): string {
  if (value === 'daily') {
    return '日次'
  }
  if (value === 'weekly') {
    return '週次'
  }
  if (value === 'monthly') {
    return '月次'
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

function formatFeatureInputs(values: string[]): string {
  const labels = values.map((value) => {
    if (value === 'close') {
      return '価格'
    }
    if (value === 'volume') {
      return '出来高'
    }
    return value
  })
  return labels.join(' + ')
}

function formatWindowDays(value: number): string {
  const knownWindows: Record<number, string> = {
    21: '1ヶ月',
    63: '3ヶ月',
    126: '6ヶ月',
    252: '12ヶ月',
  }
  return knownWindows[value] ?? `${value.toFixed(0)}日`
}

function formatRankingModelSummary(model: StrategyComponentWithParameters | null): string {
  if (!model) {
    return '使わない'
  }

  const parts = [model.label]
  const parameterLabels: string[] = []

  if (typeof model.parameters.windowDays === 'number') {
    parameterLabels.push(`窓 ${formatWindowDays(model.parameters.windowDays)}`)
  }
  if (typeof model.parameters.momentumWeight === 'number') {
    parameterLabels.push(`モメンタム ${model.parameters.momentumWeight.toFixed(2)}`)
  }
  if (typeof model.parameters.lowVolWeight === 'number') {
    parameterLabels.push(`低ボラ ${model.parameters.lowVolWeight.toFixed(2)}`)
  }
  if (typeof model.parameters.macroWeight === 'number') {
    parameterLabels.push(`マクロ ${model.parameters.macroWeight.toFixed(2)}`)
  }

  if (parameterLabels.length > 0) {
    parts.push(`(${parameterLabels.join(' / ')})`)
  }
  return parts.join(' ')
}

function formatTiltShape(value: number | undefined): string {
  if (value === 1) {
    return '上位優遇'
  }
  if (value === 2) {
    return 'softmax'
  }
  return '線形'
}

function formatTiltRuleSummary(rule: StrategyComponentWithParameters | null): string {
  if (!rule) {
    return '使わない'
  }

  const strength = typeof rule.parameters.strength === 'number' ? rule.parameters.strength.toFixed(2) : '-'
  const shape = formatTiltShape(rule.parameters.shape)
  return `${shape} / 強度 ${strength}`
}

function buildStrategyDetails(strategy: StrategyDefinition): Array<{ label: string; value: string }> {
  const details = [
    {
      label: '投資対象',
      value: `${strategy.components.core.investmentUniverse.label} (${strategy.components.core.investmentUniverse.assetCount}資産)`,
    },
    {
      label: 'データ粒度',
      value: formatDataResolutionLabel(strategy.components.core.dataResolution.label),
    },
    {
      label: '配分',
      value: strategy.components.core.portfolioModel.label,
    },
    {
      label: '執行',
      value: strategy.components.core.executionPolicy.label,
    },
    {
      label: 'リスク制御',
      value: formatRiskControls(
        strategy.components.optional.riskControls.maxInvestmentPct,
        strategy.components.optional.riskControls.maxWeightPct,
      ),
    },
  ]

  if (strategy.components.optional.assetRankingModel) {
    details.splice(2, 0, {
      label: '資産評価',
      value: formatRankingModelSummary(strategy.components.optional.assetRankingModel),
    })
  }

  if (strategy.components.optional.tiltRule) {
    details.splice(3, 0, {
      label: '重み付け',
      value: formatTiltRuleSummary(strategy.components.optional.tiltRule),
    })
  }

  if (strategy.components.optional.featureInputs.length > 0) {
    details.splice(2, 0, {
      label: '入力データ',
      value: formatFeatureInputs(strategy.components.optional.featureInputs),
    })
  }

  if (strategy.components.optional.filterRules.length > 0) {
    details.splice(details.length - 2, 0, {
      label: 'フィルタ',
      value: formatFilters(strategy.components.optional.filterRules),
    })
  }

  if (strategy.components.optional.fallbackRule) {
    details.push({
      label: 'フォールバック',
      value: strategy.components.optional.fallbackRule.label,
    })
  }

  return details
}

function buildEvaluationDetails(dashboard: DashboardResult): Array<{ label: string; value: string }> {
  return [
    {
      label: '評価期間',
      value: dashboard.study.evaluationContext.datasetContext.period,
    },
    {
      label: '補助確認',
      value: dashboard.study.evaluationContext.datasetContext.sanityPeriods.join(' / ') || 'なし',
    },
    {
      label: '手数料',
      value: `${dashboard.study.evaluationContext.costAssumptions.commissionPct.toFixed(2)}%`,
    },
    {
      label: '分割',
      value: `学習 ${dashboard.study.evaluationContext.evaluationSettings.splitRatioPct.toFixed(1)}% / 検証 ${(
        100 - dashboard.study.evaluationContext.evaluationSettings.splitRatioPct
      ).toFixed(1)}%`,
    },
    {
      label: 'ベンチマーク',
      value: formatBenchmarkLabel(dashboard.study.evaluationContext.evaluationSettings.benchmark),
    },
    {
      label: '選定基準',
      value: [
        formatMetricLabel(dashboard.study.selectionPolicy.primaryMetric),
        formatMetricLabel(dashboard.study.selectionPolicy.secondaryMetric),
        formatMetricLabel(dashboard.study.selectionPolicy.tertiaryMetric),
      ].join(' → '),
    },
  ]
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
  const strategyDetails = bestRun ? buildStrategyDetails(bestRun.strategy) : []
  const evaluationDetails = dashboard ? buildEvaluationDetails(dashboard) : []

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
              {strategyDetails.map((detail) => (
                <div key={detail.label}>
                  <dt>{detail.label}</dt>
                  <dd>{detail.value}</dd>
                </div>
              ))}
            </dl>
          </section>

          <section className="subpanel">
            <h3>Evaluation Context</h3>
            <dl className="detail-list">
              {evaluationDetails.map((detail) => (
                <div key={detail.label}>
                  <dt>{detail.label}</dt>
                  <dd>{detail.value}</dd>
                </div>
              ))}
            </dl>
          </section>
        </div>
      </section>
    </main>
  )
}

export default App
