import { useMemo } from 'react'
import { Line } from 'react-chartjs-2'
import { botColor, CHART_THEME } from '../data'
import '../chartjs'

export default function EloProgressionChart({ timeline, allBots }) {
  const { labels, datasets, botDataMap } = useMemo(() => {
    const labels = timeline.map((round) => `Round ${round.round_id}`)
    const botDataMap = Object.fromEntries(
      allBots.map((bot) => [bot, timeline.map((round) => round.bots[bot] ?? null)]),
    )
    const datasets = allBots.map((bot, index) => ({
      label: bot,
      data: botDataMap[bot],
      borderColor: botColor(index),
      backgroundColor: `${botColor(index)}22`,
      tension: 0.3,
      spanGaps: true,
      pointRadius: 5,
      pointHoverRadius: 7,
      borderWidth: 2.5,
    }))
    return { labels, datasets, botDataMap }
  }, [allBots, timeline])

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
