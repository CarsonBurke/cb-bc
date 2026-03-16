import { Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
} from 'chart.js'
import { botColor, getAllBots, CHART_THEME } from '../data'

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip)

export default function EloBarChart({ bots }) {
  const allBotsList = getAllBots()
  const sorted = Object.entries(bots).sort((a, b) => b[1].elo - a[1].elo)
  const names = sorted.map(([n]) => n)
  const elos = sorted.map(([, d]) => d.elo)

  return (
    <div className="min-h-[260px]">
      <Bar
        data={{
          labels: names,
          datasets: [{
            data: elos,
            backgroundColor: names.map(n => {
              const ci = allBotsList.indexOf(n)
              return botColor(ci >= 0 ? ci : 0) + '80'
            }),
            borderColor: names.map(n => {
              const ci = allBotsList.indexOf(n)
              return botColor(ci >= 0 ? ci : 0)
            }),
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
