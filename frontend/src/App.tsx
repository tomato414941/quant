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

type BacktestResult = {
  summary: {
    strategy: SummaryMetrics
    benchmark: SummaryMetrics
    config: {
      thresholdPct: number
      initialCapital: number
      holdingRule: string
    }
  }
  series: Array<{
    date: string
    close: number
    previousDayReturnPct: number
    position: number
    signal: boolean
    strategyEquity: number
    benchmarkEquity: number
    strategyReturnPct: number
  }>
}

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  `${window.location.protocol}//${window.location.hostname}:8000`

async function fetchDemoBacktest(threshold: number, initialCapital: number): Promise<BacktestResult> {
  const params = new URLSearchParams({
    threshold: (threshold / 100).toString(),
    initial_capital: initialCapital.toString(),
  })
  const response = await fetch(`${API_BASE_URL}/api/backtest/demo?${params.toString()}`)
  if (!response.ok) {
    const payload = await response.json()
    throw new Error(payload.detail ?? 'Failed to fetch demo backtest.')
  }
  return response.json()
}

async function uploadBacktest(
  file: File,
  threshold: number,
  initialCapital: number,
): Promise<BacktestResult> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('threshold', (threshold / 100).toString())
  formData.append('initial_capital', initialCapital.toString())

  const response = await fetch(`${API_BASE_URL}/api/backtest/upload`, {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    const payload = await response.json()
    throw new Error(payload.detail ?? 'Failed to upload CSV.')
  }
  return response.json()
}

function formatPercent(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 0,
  }).format(value)
}

function App() {
  const [threshold, setThreshold] = useState(3)
  const [initialCapital, setInitialCapital] = useState(10000)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('Synthetic market demo loaded automatically.')

  const loadDemo = async (nextThreshold: number, nextCapital: number) => {
    setLoading(true)
    setError('')

    try {
      const payload = await fetchDemoBacktest(nextThreshold, nextCapital)
      startTransition(() => setResult(payload))
      setMessage('Demo backtest updated.')
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Unknown error.')
    } finally {
      setLoading(false)
    }
  }

  const bootDemo = useEffectEvent(async () => {
    await loadDemo(threshold, initialCapital)
  })

  useEffect(() => {
    void bootDemo()
  }, [])

  const handleRunDemo = async () => {
    setSelectedFile(null)
    await loadDemo(threshold, initialCapital)
  }

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Upload a CSV file with date and close columns first.')
      return
    }

    setLoading(true)
    setError('')
    try {
      const payload = await uploadBacktest(selectedFile, threshold, initialCapital)
      startTransition(() => setResult(payload))
      setMessage(`CSV backtest updated from ${selectedFile.name}.`)
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Unknown error.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <div className="page">
        <section className="hero panel">
          <p className="eyebrow">Minimal Quant</p>
          <h1>One idea. One rule. One chart.</h1>
          <p className="hero-copy">
            This is the smallest useful quant workflow: price series in, explicit rule,
            backtest out. The rule is fixed to a one-day mean reversion trade after a large drop.
          </p>
          <div className="hero-grid">
            <div className="hero-chip">
              <div>
                <strong>Rule</strong>
                <span>Buy after a drop larger than threshold</span>
              </div>
            </div>
            <div className="hero-chip">
              <div>
                <strong>Input</strong>
                <span>Built-in synthetic data or your own CSV</span>
              </div>
            </div>
            <div className="hero-chip">
              <div>
                <strong>Output</strong>
                <span>Return, Sharpe, drawdown, win rate</span>
              </div>
            </div>
          </div>
        </section>

        <div className="layout">
          <section className="panel controls">
            <div className="panel-title">
              <div>
                <h2>Controls</h2>
                <p>Frontend and backend are connected only through the API.</p>
              </div>
            </div>

            <div className="control-grid">
              <div className="field">
                <div className="range-header">
                  <label htmlFor="threshold">Drop Threshold</label>
                  <span className="range-value">{threshold.toFixed(1)}%</span>
                </div>
                <input
                  id="threshold"
                  type="range"
                  min="1"
                  max="8"
                  step="0.5"
                  value={threshold}
                  onChange={(event) => setThreshold(Number(event.target.value))}
                />
              </div>

              <div className="field">
                <label htmlFor="capital">Initial Capital</label>
                <input
                  id="capital"
                  type="number"
                  min="1000"
                  step="1000"
                  value={initialCapital}
                  onChange={(event) => setInitialCapital(Number(event.target.value))}
                />
              </div>

              <div className="field">
                <label htmlFor="csv">Optional CSV Upload</label>
                <input
                  id="csv"
                  type="file"
                  accept=".csv,text/csv"
                  onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
                />
                <p className="footnote">Expected columns: date, close</p>
              </div>

              <div className="actions">
                <button className="button button-primary" disabled={loading} onClick={() => void handleRunDemo()}>
                  {loading ? 'Running…' : 'Run Demo Backtest'}
                </button>
                <button className="button button-secondary" disabled={loading} onClick={() => void handleUpload()}>
                  {loading ? 'Uploading…' : 'Run Uploaded CSV'}
                </button>
              </div>
            </div>

            <p className={`status ${error ? 'error' : 'success'}`}>{error || message}</p>
          </section>

          <section className="panel results">
            <div className="panel-title">
              <div>
                <h2>Results</h2>
                <p>
                  {result
                    ? `${result.summary.config.holdingRule} Threshold ${result.summary.config.thresholdPct.toFixed(2)}%.`
                    : 'No results yet.'}
                </p>
              </div>
            </div>

            {result ? (
              <>
                <div className="cards">
                  <article className="metric-card">
                    <h3>Strategy</h3>
                    <div className="metric-grid">
                      <div className="metric">
                        <span className="metric-label">Total Return</span>
                        <strong className="metric-value">
                          {formatPercent(result.summary.strategy.totalReturnPct)}
                        </strong>
                      </div>
                      <div className="metric">
                        <span className="metric-label">Sharpe</span>
                        <strong className="metric-value">
                          {result.summary.strategy.sharpeRatio.toFixed(2)}
                        </strong>
                      </div>
                      <div className="metric">
                        <span className="metric-label">Max Drawdown</span>
                        <strong className="metric-value">
                          {formatPercent(-result.summary.strategy.maxDrawdownPct)}
                        </strong>
                      </div>
                      <div className="metric">
                        <span className="metric-label">Win Rate</span>
                        <strong className="metric-value">
                          {formatPercent(result.summary.strategy.winRatePct ?? 0)}
                        </strong>
                      </div>
                    </div>
                    <p className="chart-note">
                      {result.summary.strategy.tradeCount} trades from starting capital $
                      {formatNumber(result.summary.config.initialCapital)}.
                    </p>
                  </article>

                  <article className="metric-card">
                    <h3>Benchmark</h3>
                    <div className="metric-grid">
                      <div className="metric">
                        <span className="metric-label">Total Return</span>
                        <strong className="metric-value">
                          {formatPercent(result.summary.benchmark.totalReturnPct)}
                        </strong>
                      </div>
                      <div className="metric">
                        <span className="metric-label">Sharpe</span>
                        <strong className="metric-value">
                          {result.summary.benchmark.sharpeRatio.toFixed(2)}
                        </strong>
                      </div>
                      <div className="metric">
                        <span className="metric-label">Max Drawdown</span>
                        <strong className="metric-value">
                          {formatPercent(-result.summary.benchmark.maxDrawdownPct)}
                        </strong>
                      </div>
                      <div className="metric">
                        <span className="metric-label">CAGR</span>
                        <strong className="metric-value">
                          {formatPercent(result.summary.benchmark.cagrPct)}
                        </strong>
                      </div>
                    </div>
                    <p className="chart-note">Buy and hold on the same price path.</p>
                  </article>
                </div>

                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height={320}>
                    <LineChart data={result.series}>
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
                        name="Strategy"
                      />
                      <Line
                        type="monotone"
                        dataKey="benchmarkEquity"
                        stroke="#667d5d"
                        strokeWidth={2.1}
                        dot={false}
                        name="Buy & Hold"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="chart-note">
                    Strategy equity is compared directly against buy and hold so the rule either
                    earns its keep or it does not.
                  </p>
                </div>
              </>
            ) : (
              <div className="empty-state">Run the demo backtest to see metrics and the equity curve.</div>
            )}
          </section>
        </div>
      </div>
    </main>
  )
}

export default App
