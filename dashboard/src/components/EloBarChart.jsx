import { useMemo } from 'react'
import { Bar } from 'react-chartjs-2'
import { botColor, CHART_THEME } from '../data'
import '../chartjs'

export default function EloBarChart({ bots, allBots }) {
  const { names, elos, borderColors, backgroundColors } = useMemo(() => {
    const indexByBot = new Map(allBots.map((name, index) => [name, index]))
    const sorted = Object.entries(bots).sort((a, b) => b[1].elo - a[1].elo)
    const names = sorted.map(([name]) => name)
    const elos = sorted.map(([, data]) => data.elo)
    const borderColors = names.map((name) => botColor(indexByBot.get(name) || 0))
    const backgroundColors = borderColors.map((color) => `${color}80`)
    return { names, elos, borderColors, backgroundColors }
  }, [allBots, bots])

  return (
    <div className="min-h-[260px]">
      <Bar
        data={{
          labels: names,
          datasets: [{
            data: elos,
            backgroundColor: backgroundColors,
            borderColor: borderColors,
            borderWidth: 1.5,
            borderRadius: 3,
          }],
        }}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: CHART_THEME.surface,
              borderColor: CHART_THEME.border,
              borderWidth: 1,
              titleColor: CHART_THEME.text,
              bodyColor: CHART_THEME.subtext,
              padding: 12,
              cornerRadius: 4,
            },
          },
          scales: {
            x: {
              ticks: { color: CHART_THEME.subtext, font: { size: 11 } },
              grid: { display: false },
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
