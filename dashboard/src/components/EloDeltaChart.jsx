import { useMemo } from 'react'
import { Line } from 'react-chartjs-2'
import { botColor, CHART_THEME } from '../data'
import '../chartjs'

export default function EloDeltaChart({ timeline, allBots }) {
  const { labels, datasets } = useMemo(() => {
    const labels = timeline.slice(1).map((round) => `Round ${round.round_id}`)
    const datasets = allBots.map((bot, index) => ({
      label: bot,
      data: timeline.slice(1).map((round, offset) => {
        const previous = timeline[offset]
        const currentElo = round.bots[bot]
        const previousElo = previous.bots[bot]
        if (currentElo == null || previousElo == null) return null
        return currentElo - previousElo
      }),
      borderColor: botColor(index),
      backgroundColor: `${botColor(index)}22`,
      tension: 0.3,
      spanGaps: true,
      pointRadius: 5,
      pointHoverRadius: 7,
      borderWidth: 2.5,
    }))
    return { labels, datasets }
  }, [allBots, timeline])

  if (timeline.length < 2) return <p className="text-overlay0 text-sm">Need at least 2 rounds for deltas.</p>

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
