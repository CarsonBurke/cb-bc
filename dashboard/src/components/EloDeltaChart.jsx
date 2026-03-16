import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from 'chart.js'
import { botColor, getAllBots, CHART_THEME } from '../data'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend)

export default function EloDeltaChart({ rounds }) {
  const allBots = getAllBots()
  if (rounds.length < 2) return <p className="text-overlay0 text-sm">Need at least 2 rounds for deltas.</p>

  const labels = rounds.slice(1).map(r => `Round ${r.round_id}`)

  const datasets = allBots.map((bot, i) => ({
    label: bot,
    data: rounds.slice(1).map((r, idx) => {
      const prev = rounds[idx]
      const cur = r.bots[bot]?.elo
      const pre = prev.bots[bot]?.elo
      if (cur == null || pre == null) return null
      return cur - pre
    }),
    borderColor: botColor(i),
    backgroundColor: botColor(i) + '22',
    tension: 0.3,
    spanGaps: true,
    pointRadius: 5,
    pointHoverRadius: 7,
    borderWidth: 2.5,
  }))

  return (
    <div className="min-h-[240px]">
      <Line
        data={{ labels, datasets }}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          plugins: {
            legend: {
              labels: {
                color: CHART_THEME.text,
                font: { size: 11 },
                usePointStyle: true,
                pointStyle: 'circle',
                padding: 16,
              },
            },
            tooltip: {
              backgroundColor: CHART_THEME.surface,
              borderColor: CHART_THEME.border,
              borderWidth: 1,
              titleColor: CHART_THEME.text,
              bodyColor: CHART_THEME.subtext,
              padding: 12,
              cornerRadius: 4,
              callbacks: {
                label(ctx) {
                  const v = ctx.parsed.y
                  if (v == null) return null
                  return `${ctx.dataset.label}: ${v > 0 ? '+' : ''}${v}`
                },
              },
            },
          },
          scales: {
            x: {
              ticks: { color: CHART_THEME.subtext, font: { size: 11 } },
              grid: { color: CHART_THEME.grid },
            },
            y: {
              ticks: {
                color: CHART_THEME.subtext,
                font: { size: 11 },
                callback: v => (v > 0 ? '+' : '') + v,
              },
              grid: { color: CHART_THEME.grid },
            },
          },
        }}
      />
    </div>
  )
}
