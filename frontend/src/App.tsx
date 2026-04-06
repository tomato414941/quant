import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import './App.css'

type SummaryMetrics = {
  cagrPct?: number
  totalReturnPct: number
  sharpeRatio: number
  maxDrawdownPct: number
  turnoverPct: number
}

type StrategyComponent = {
  key: string
  label: string
}

type StrategyModelWithParameters = {
  kind: string
  label: string
  parameters: Record<string, number>
}

type InvestmentUniverse = {
  key: string
  label: string
  assetCount: number
  tickers: string[]
}

type StrategySpec = {
  kind?: string
  schemaVersion?: string
  strategyId: string
  version: string
  label: string
  hypothesis: string | null
  description?: string
  extensions?: Record<string, unknown>
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
        entry: string
        rebalanceFrequency: string
      }
    }
    optional: {
      assetRankingModel: StrategyModelWithParameters | null
      featureInputs: string[]
      filterRules: StrategyComponent[]
      fallbackRule: StrategyComponent | null
      tiltRule: StrategyModelWithParameters | null
      riskControls: {
        maxInvestmentPct: number
        maxWeightPct: number | null
      }
    }
  }
}

type PortfolioRun = {
  kind?: string
  schemaVersion?: string
  key: string
  strategy: StrategySpec
  summary: SummaryMetrics
  splitAnalysis: {
    config: {
      splitRatioPct: number
    }
    train: {
      dayCount: number
      endDate: string
      portfolio: SummaryMetrics
      startDate: string
    }
    test: {
      dayCount: number
      endDate: string
      portfolio: SummaryMetrics
      startDate: string
    }
  }
}

type DashboardResult = {
  comparison: {
    kind?: string
    schemaVersion?: string
    comparisonId: string
    title: string
    question: string
    selectionPolicy: {
      primaryMetric: string
      secondaryMetric: string
      tertiaryMetric: string
    }
    marketData: {
      period: string
      sanityPeriods: string[]
      timeframe: {
        key: string
        label: string
        barsPerYear: number
        barSeconds: number
      }
      fields: string[]
    }
    runInput: {
      portfolioState: {
        weights: Array<{ asset: string; weightPct: number }>
      }
      capitalBase: number
    }
    executionAssumptions: {
      kind?: string
      label: string
      parameters: Record<string, string | number | boolean>
      costModel: {
        kind: string
        parameters: {
          commissionPct?: number
          slippagePct?: number
          [key: string]: number | undefined
        }
        perAssetOverrides: Record<string, Record<string, number>>
      }
    }
    evaluation: {
      kind?: string
      schemaVersion?: string
      marketDataContext: {
        period: string
        sanityPeriods: string[]
        timeframe: {
          key: string
          label: string
          barsPerYear: number
          barSeconds: number
        }
        fields: string[]
        source: string
        alignedStartDate: string
        alignedEndDate: string
        rowCount: number
      }
      evaluationSettings: {
        splitRatioPct: number
      }
    }
    candidateStrategies: StrategySpec[]
    referenceStrategies: StrategySpec[]
  }
  candidateRuns: PortfolioRun[]
  referenceRuns: PortfolioRun[]
}

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  `${window.location.protocol}//${window.location.hostname}:8000`

function formatPercent(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
}

function formatMetricValue(value: string | number | null | undefined): string {
  if (value === null) {
    return 'null'
  }
  if (value === undefined) {
    return '-'
  }
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toString() : value.toFixed(2)
  }
  return value
}

function formatJson(value: unknown): string {
  return JSON.stringify(value, null, 2)
}

function DataListRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="data-list-row">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}

function ValueList({ values }: { values: string[] }) {
  return (
    <ul className="value-list">
      {values.map((value) => (
        <li key={value}>{value}</li>
      ))}
    </ul>
  )
}

function FlipCard({
  title,
  face,
  onToggle,
  front,
  back,
  className,
}: {
  title: string
  face: 'front' | 'back'
  onToggle: () => void
  front: ReactNode
  back: unknown
  className?: string
}) {
  return (
    <section className={`subpanel ${className ?? ''}`.trim()}>
      <div className="subpanel-header">
        <h3>{title}</h3>
        <button type="button" className="flip-button" onClick={onToggle}>
          {face === 'front' ? 'JSON' : '項目表示'}
        </button>
      </div>
      {face === 'front' ? front : <pre className="raw-block">{formatJson(back)}</pre>}
    </section>
  )
}

function App() {
  const [dashboard, setDashboard] = useState<DashboardResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [cardFaces, setCardFaces] = useState<{
    strategy: 'front' | 'back'
    reference: 'front' | 'back'
    portfolioState: 'front' | 'back'
    executionAssumptions: 'front' | 'back'
    evaluation: 'front' | 'back'
    result: 'front' | 'back'
  }>({
    strategy: 'front',
    reference: 'front',
    portfolioState: 'front',
    executionAssumptions: 'front',
    evaluation: 'front',
    result: 'front',
  })
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

    return dashboard.candidateRuns.slice().sort((left, right) => {
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
  const referenceRun = dashboard?.referenceRuns[0] ?? null
  const rawRunResult = bestRun
    ? {
        kind: bestRun.kind,
        schemaVersion: bestRun.schemaVersion,
        key: bestRun.key,
        summary: bestRun.summary,
        splitAnalysis: bestRun.splitAnalysis,
      }
    : null

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

        <div className="section-grid">
          <FlipCard
            title="Strategy"
            face={cardFaces.strategy}
            onToggle={() =>
              setCardFaces((current) => ({
                ...current,
                strategy: current.strategy === 'front' ? 'back' : 'front',
              }))
            }
            back={bestRun.strategy}
            front={
              <div className="card-body">
                <dl className="data-list">
                  <DataListRow label="strategyId" value={bestRun.strategy.strategyId} />
                  <DataListRow label="version" value={bestRun.strategy.version} />
                  <DataListRow label="label" value={bestRun.strategy.label} />
                  <DataListRow label="hypothesis" value={bestRun.strategy.hypothesis ?? 'null'} />
                  <DataListRow
                    label="investmentUniverse.key"
                    value={bestRun.strategy.components.core.investmentUniverse.key}
                  />
                  <DataListRow
                    label="investmentUniverse.label"
                    value={bestRun.strategy.components.core.investmentUniverse.label}
                  />
                  <DataListRow
                    label="investmentUniverse.assetCount"
                    value={bestRun.strategy.components.core.investmentUniverse.assetCount}
                  />
                  <DataListRow
                    label="investmentUniverse.tickers"
                    value={<ValueList values={bestRun.strategy.components.core.investmentUniverse.tickers} />}
                  />
                  <DataListRow
                    label="portfolioModel"
                    value={`${bestRun.strategy.components.core.portfolioModel.label} (${bestRun.strategy.components.core.portfolioModel.key})`}
                  />
                  <DataListRow
                    label="executionPolicy"
                    value={`${bestRun.strategy.components.core.executionPolicy.label} (${bestRun.strategy.components.core.executionPolicy.key})`}
                  />
                  <DataListRow
                    label="executionPolicy.entry"
                    value={bestRun.strategy.components.core.executionPolicy.entry}
                  />
                  <DataListRow
                    label="executionPolicy.rebalanceFrequency"
                    value={bestRun.strategy.components.core.executionPolicy.rebalanceFrequency}
                  />
                  <DataListRow
                    label="featureInputs"
                    value={<ValueList values={bestRun.strategy.components.optional.featureInputs} />}
                  />
                  <DataListRow
                    label="assetRankingModel"
                    value={
                      bestRun.strategy.components.optional.assetRankingModel
                        ? `${bestRun.strategy.components.optional.assetRankingModel.label} (${bestRun.strategy.components.optional.assetRankingModel.kind})`
                        : 'null'
                    }
                  />
                  {bestRun.strategy.components.optional.assetRankingModel ? (
                    <DataListRow
                      label="assetRankingModel.parameters"
                      value={
                        <pre className="inline-json">
                          {formatJson(bestRun.strategy.components.optional.assetRankingModel.parameters)}
                        </pre>
                      }
                    />
                  ) : null}
                  <DataListRow
                    label="filterRules"
                    value={
                      bestRun.strategy.components.optional.filterRules.length > 0 ? (
                        <ValueList
                          values={bestRun.strategy.components.optional.filterRules.map(
                            (rule) => `${rule.label} (${rule.key})`,
                          )}
                        />
                      ) : (
                        '[]'
                      )
                    }
                  />
                  <DataListRow
                    label="fallbackRule"
                    value={
                      bestRun.strategy.components.optional.fallbackRule
                        ? `${bestRun.strategy.components.optional.fallbackRule.label} (${bestRun.strategy.components.optional.fallbackRule.key})`
                        : 'null'
                    }
                  />
                  <DataListRow
                    label="tiltRule"
                    value={
                      bestRun.strategy.components.optional.tiltRule
                        ? `${bestRun.strategy.components.optional.tiltRule.label} (${bestRun.strategy.components.optional.tiltRule.kind})`
                        : 'null'
                    }
                  />
                  {bestRun.strategy.components.optional.tiltRule ? (
                    <DataListRow
                      label="tiltRule.parameters"
                      value={
                        <pre className="inline-json">
                          {formatJson(bestRun.strategy.components.optional.tiltRule.parameters)}
                        </pre>
                      }
                    />
                  ) : null}
                  <DataListRow
                    label="riskControls.maxInvestmentPct"
                    value={formatMetricValue(
                      bestRun.strategy.components.optional.riskControls.maxInvestmentPct,
                    )}
                  />
                  <DataListRow
                    label="riskControls.maxWeightPct"
                    value={formatMetricValue(bestRun.strategy.components.optional.riskControls.maxWeightPct)}
                  />
                </dl>
              </div>
            }
          />

          {referenceRun ? (
            <FlipCard
              title="Reference Strategy"
              face={cardFaces.reference}
              onToggle={() =>
                setCardFaces((current) => ({
                  ...current,
                  reference: current.reference === 'front' ? 'back' : 'front',
                }))
              }
              back={referenceRun.strategy}
              front={
                <div className="card-body">
                  <dl className="data-list">
                    <DataListRow label="strategyId" value={referenceRun.strategy.strategyId} />
                    <DataListRow label="version" value={referenceRun.strategy.version} />
                    <DataListRow label="label" value={referenceRun.strategy.label} />
                    <DataListRow label="hypothesis" value={referenceRun.strategy.hypothesis ?? 'null'} />
                    <DataListRow
                      label="investmentUniverse.key"
                      value={referenceRun.strategy.components.core.investmentUniverse.key}
                    />
                    <DataListRow
                      label="investmentUniverse.label"
                      value={referenceRun.strategy.components.core.investmentUniverse.label}
                    />
                    <DataListRow
                      label="investmentUniverse.assetCount"
                      value={referenceRun.strategy.components.core.investmentUniverse.assetCount}
                    />
                    <DataListRow
                      label="investmentUniverse.tickers"
                      value={<ValueList values={referenceRun.strategy.components.core.investmentUniverse.tickers} />}
                    />
                    <DataListRow
                      label="portfolioModel"
                      value={`${referenceRun.strategy.components.core.portfolioModel.label} (${referenceRun.strategy.components.core.portfolioModel.key})`}
                    />
                    <DataListRow
                      label="executionPolicy"
                      value={`${referenceRun.strategy.components.core.executionPolicy.label} (${referenceRun.strategy.components.core.executionPolicy.key})`}
                    />
                    <DataListRow
                      label="executionPolicy.entry"
                      value={referenceRun.strategy.components.core.executionPolicy.entry}
                    />
                    <DataListRow
                      label="executionPolicy.rebalanceFrequency"
                      value={referenceRun.strategy.components.core.executionPolicy.rebalanceFrequency}
                    />
                    <DataListRow
                      label="featureInputs"
                      value={<ValueList values={referenceRun.strategy.components.optional.featureInputs} />}
                    />
                    <DataListRow
                      label="assetRankingModel"
                      value={
                        referenceRun.strategy.components.optional.assetRankingModel
                          ? `${referenceRun.strategy.components.optional.assetRankingModel.label} (${referenceRun.strategy.components.optional.assetRankingModel.kind})`
                          : 'null'
                      }
                    />
                    <DataListRow
                      label="filterRules"
                      value={
                        referenceRun.strategy.components.optional.filterRules.length > 0 ? (
                          <ValueList
                            values={referenceRun.strategy.components.optional.filterRules.map(
                              (rule) => `${rule.label} (${rule.key})`,
                            )}
                          />
                        ) : (
                          '[]'
                        )
                      }
                    />
                    <DataListRow
                      label="fallbackRule"
                      value={
                        referenceRun.strategy.components.optional.fallbackRule
                          ? `${referenceRun.strategy.components.optional.fallbackRule.label} (${referenceRun.strategy.components.optional.fallbackRule.key})`
                          : 'null'
                      }
                    />
                    <DataListRow
                      label="tiltRule"
                      value={
                        referenceRun.strategy.components.optional.tiltRule
                          ? `${referenceRun.strategy.components.optional.tiltRule.label} (${referenceRun.strategy.components.optional.tiltRule.kind})`
                          : 'null'
                      }
                    />
                    <DataListRow
                      label="riskControls.maxInvestmentPct"
                      value={formatMetricValue(
                        referenceRun.strategy.components.optional.riskControls.maxInvestmentPct,
                      )}
                    />
                    <DataListRow
                      label="riskControls.maxWeightPct"
                      value={formatMetricValue(
                        referenceRun.strategy.components.optional.riskControls.maxWeightPct,
                      )}
                    />
                  </dl>
                </div>
              }
            />
          ) : null}

          <FlipCard
            title="Portfolio State"
            face={cardFaces.portfolioState}
            onToggle={() =>
              setCardFaces((current) => ({
                ...current,
                portfolioState: current.portfolioState === 'front' ? 'back' : 'front',
              }))
            }
            back={dashboard.comparison.runInput.portfolioState}
            front={
              <div className="card-body">
                <dl className="data-list">
                  <DataListRow
                    label="weights"
                    value={
                      <pre className="inline-json">
                        {formatJson(dashboard.comparison.runInput.portfolioState.weights)}
                      </pre>
                    }
                  />
                  <DataListRow
                    label="capitalBase"
                    value={formatMetricValue(dashboard.comparison.runInput.capitalBase)}
                  />
                </dl>
              </div>
            }
          />

          <FlipCard
            title="Execution Assumptions"
            face={cardFaces.executionAssumptions}
            onToggle={() =>
              setCardFaces((current) => ({
                ...current,
                executionAssumptions:
                  current.executionAssumptions === 'front' ? 'back' : 'front',
              }))
            }
            back={dashboard.comparison.executionAssumptions}
            front={
              <div className="card-body">
                <dl className="data-list">
                  <DataListRow
                    label="kind"
                    value={dashboard.comparison.executionAssumptions.kind ?? '-'}
                  />
                  <DataListRow
                    label="label"
                    value={dashboard.comparison.executionAssumptions.label}
                  />
                  <DataListRow
                    label="parameters"
                    value={
                      <pre className="inline-json">
                        {formatJson(dashboard.comparison.executionAssumptions.parameters)}
                      </pre>
                    }
                  />
                  <DataListRow
                    label="costModel.kind"
                    value={dashboard.comparison.executionAssumptions.costModel.kind}
                  />
                  <DataListRow
                    label="costModel.parameters"
                    value={
                      <pre className="inline-json">
                        {formatJson(dashboard.comparison.executionAssumptions.costModel.parameters)}
                      </pre>
                    }
                  />
                  <DataListRow
                    label="costModel.perAssetOverrides"
                    value={
                      <pre className="inline-json">
                        {formatJson(
                          dashboard.comparison.executionAssumptions.costModel.perAssetOverrides,
                        )}
                      </pre>
                    }
                  />
                </dl>
              </div>
            }
          />

          <FlipCard
            title="Evaluation"
            face={cardFaces.evaluation}
            onToggle={() =>
              setCardFaces((current) => ({
                ...current,
                evaluation: current.evaluation === 'front' ? 'back' : 'front',
              }))
            }
            back={dashboard.comparison.evaluation}
            front={
              <div className="card-body">
                <dl className="data-list">
                  <DataListRow
                    label="marketDataContext.period"
                    value={dashboard.comparison.evaluation.marketDataContext.period}
                  />
                  <DataListRow
                    label="marketDataContext.sanityPeriods"
                    value={
                      <ValueList values={dashboard.comparison.evaluation.marketDataContext.sanityPeriods} />
                    }
                  />
                  <DataListRow
                    label="marketDataContext.timeframe"
                    value={dashboard.comparison.evaluation.marketDataContext.timeframe.label}
                  />
                  <DataListRow
                    label="marketDataContext.fields"
                    value={<ValueList values={dashboard.comparison.evaluation.marketDataContext.fields} />}
                  />
                  <DataListRow
                    label="marketDataContext.source"
                    value={dashboard.comparison.evaluation.marketDataContext.source}
                  />
                  <DataListRow
                    label="marketDataContext.alignedStartDate"
                    value={dashboard.comparison.evaluation.marketDataContext.alignedStartDate}
                  />
                  <DataListRow
                    label="marketDataContext.alignedEndDate"
                    value={dashboard.comparison.evaluation.marketDataContext.alignedEndDate}
                  />
                  <DataListRow
                    label="marketDataContext.rowCount"
                    value={dashboard.comparison.evaluation.marketDataContext.rowCount}
                  />
                  <DataListRow
                    label="evaluationSettings.splitRatioPct"
                    value={formatMetricValue(
                      dashboard.comparison.evaluation.evaluationSettings.splitRatioPct,
                    )}
                  />
                </dl>
              </div>
            }
          />

          <FlipCard
            title="Run Result"
            className="section-span-full"
            face={cardFaces.result}
            onToggle={() =>
              setCardFaces((current) => ({
                ...current,
                result: current.result === 'front' ? 'back' : 'front',
              }))
            }
            back={rawRunResult}
            front={
              <div className="card-body">
                <dl className="data-list">
                  <DataListRow label="key" value={bestRun.key} />
                  <DataListRow label="summary.totalReturnPct" value={formatPercent(bestRun.summary.totalReturnPct)} />
                  <DataListRow label="summary.cagrPct" value={formatPercent(bestRun.summary.cagrPct ?? 0)} />
                  <DataListRow label="summary.sharpeRatio" value={formatMetricValue(bestRun.summary.sharpeRatio)} />
                  <DataListRow
                    label="summary.maxDrawdownPct"
                    value={formatPercent(-bestRun.summary.maxDrawdownPct)}
                  />
                  <DataListRow label="summary.turnoverPct" value={formatPercent(bestRun.summary.turnoverPct)} />
                  <DataListRow
                    label="splitAnalysis.config.splitRatioPct"
                    value={formatMetricValue(bestRun.splitAnalysis.config.splitRatioPct)}
                  />
                  <DataListRow label="train.startDate" value={bestRun.splitAnalysis.train.startDate} />
                  <DataListRow label="train.endDate" value={bestRun.splitAnalysis.train.endDate} />
                  <DataListRow label="train.dayCount" value={bestRun.splitAnalysis.train.dayCount} />
                  <DataListRow
                    label="train.portfolio"
                    value={<pre className="inline-json">{formatJson(bestRun.splitAnalysis.train.portfolio)}</pre>}
                  />
                  <DataListRow label="test.startDate" value={bestRun.splitAnalysis.test.startDate} />
                  <DataListRow label="test.endDate" value={bestRun.splitAnalysis.test.endDate} />
                  <DataListRow label="test.dayCount" value={bestRun.splitAnalysis.test.dayCount} />
                  <DataListRow
                    label="test.portfolio"
                    value={<pre className="inline-json">{formatJson(bestRun.splitAnalysis.test.portfolio)}</pre>}
                  />
                </dl>
              </div>
            }
          />
        </div>
      </section>
    </main>
  )
}

export default App
