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

export default function EloProgressionChart({ rounds }) {
  const allBots = getAllBots()
  const labels = rounds.map(r => `Round ${r.round_id}`)

  const datasets = allBots.map((bot, i) => {
    const data = rounds.map(r => r.bots[bot]?.elo ?? null)
    return {
      label: bot,
      data,
      borderColor: botColor(i),
      backgroundColor: botColor(i) + '22',
      tension: 0.3,
      spanGaps: true,
      pointRadius: 5,
      pointHoverRadius: 7,
      borderWidth: 2.5,
    }
  })

  // Compute deltas for the tooltip
  const botDataMap = {}
  allBots.forEach((bot, i) => {
    botDataMap[bot] = rounds.map(r => r.bots[bot]?.elo ?? null)
  })

  return (
    <div className="min-h-[280px]">
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
                  const bot = ctx.dataset.label
                  const elo = ctx.parsed.y
                  if (elo == null) return null
                  const idx = ctx.dataIndex
                  const data = botDataMap[bot]
                  const prev = idx > 0 ? data[idx - 1] : null
                  const delta = prev != null ? elo - prev : null
                  const deltaStr = delta != null ? ` (${delta > 0 ? '+' : ''}${delta})` : ''
                  return `${bot}: ${elo}${deltaStr}`
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
              ticks: { color: CHART_THEME.subtext, font: { size: 11 } },
              grid: { color: CHART_THEME.grid },
            },
          },
        }}
      />
    </div>
  )
}
