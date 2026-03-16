import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js'
import { CHART_THEME } from '../data'
import { Table, Th, Td, Thead, Tr } from './Table'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip, Legend)

const CHART_OPTS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: { labels: { color: CHART_THEME.text, font: { size: 11 }, usePointStyle: true, pointStyle: 'circle', padding: 12 } },
    tooltip: {
      backgroundColor: CHART_THEME.surface, borderColor: CHART_THEME.border, borderWidth: 1,
      titleColor: CHART_THEME.text, bodyColor: CHART_THEME.subtext,
      padding: 10, cornerRadius: 4,
    },
  },
  scales: {
    x: { ticks: { color: CHART_THEME.subtext, font: { size: 10 } }, grid: { color: CHART_THEME.grid } },
    y: { ticks: { color: CHART_THEME.subtext, font: { size: 10 } }, grid: { color: CHART_THEME.grid }, beginAtZero: true },
  },
}

function mergeRounds(snapsA, snapsB) {
  return [...new Set([...(snapsA || []).map(s => s.round), ...(snapsB || []).map(s => s.round)])].sort((a, b) => a - b)
}

function lookupField(snaps, field, rounds) {
  const map = Object.fromEntries((snaps || []).map(s => [s.round, s[field]]))
  return rounds.map(r => map[r] ?? null)
}

function TimelineChart({ teamA, teamB, botA, botB, datasets: datasetsFn, title }) {
  const snapsA = teamA?.snapshots || []
  const snapsB = teamB?.snapshots || []
  const rounds = mergeRounds(snapsA, snapsB)
  const labels = rounds.map(r => `${r}`)
  const datasets = datasetsFn(snapsA, snapsB, rounds)

  return (
    <div>
      <h5 className="text-[12px] text-overlay0 font-medium mb-2">{title}</h5>
      <div className="min-h-[220px]">
        <Line data={{ labels, datasets }} options={CHART_OPTS} />
      </div>
    </div>
  )
}

function avgField(team, field) {
  const snaps = team?.snapshots
  if (!snaps?.length) return null
  const relevant = snaps.filter(s => s.round > 0 && s[field] != null)
  if (!relevant.length) return null
  return relevant.reduce((sum, s) => sum + s[field], 0) / relevant.length
}

function avgOreUtil(team) {
  const snaps = team?.snapshots
  if (!snaps?.length) return null
  const relevant = snaps.filter(s => s.round > 0 && s.visible_ore > 0)
  if (!relevant.length) return null
  const avg = relevant.reduce((sum, s) => sum + (s.mined_ore / s.visible_ore) * 100, 0) / relevant.length
  return Math.round(avg)
}

function StatSummary({ teamA, teamB, botA, botB }) {
  if (!teamA && !teamB) return null

  function row(label, aVal, bVal, fmt = v => v) {
    return (
      <Tr>
        <Td className="text-overlay0 text-[12px]">{label}</Td>
        <Td className="text-[12px] text-blue">{aVal != null ? fmt(aVal) : '—'}</Td>
        <Td className="text-[12px] text-red">{bVal != null ? fmt(bVal) : '—'}</Td>
      </Tr>
    )
  }

  return (
    <Table className="mb-4">
      <Thead>
        <Th className="w-48">Metric</Th>
        <Th className="text-blue">{botA}</Th>
        <Th className="text-red">{botB}</Th>
      </Thead>
      <tbody>
        {row('Final Core HP', teamA?.final_core_hp, teamB?.final_core_hp)}
        {row('Peak Titanium', teamA?.peak_titanium, teamB?.peak_titanium)}
        {row('Peak Axionite', teamA?.peak_axionite, teamB?.peak_axionite)}
        {row('Total Ti Delivered', teamA?.total_income_ti, teamB?.total_income_ti)}
        {row('Total Ax Delivered', teamA?.total_income_ax, teamB?.total_income_ax)}
        {row('Total Ti Spent', teamA?.total_spent_ti, teamB?.total_spent_ti)}
        {row('Total Ax Spent', teamA?.total_spent_ax, teamB?.total_spent_ax)}
        {row('Final Ti', teamA?.final?.titanium, teamB?.final?.titanium)}
        {row('Final Ax', teamA?.final?.axionite, teamB?.final?.axionite)}
        {row('Avg Unmined Ore (visible)', avgField(teamA, 'unmined_ore'), avgField(teamB, 'unmined_ore'), v => v.toFixed(1))}
        {row('Avg Ore Utilization', avgOreUtil(teamA), avgOreUtil(teamB), v => v + '%')}
      </tbody>
    </Table>
  )
}

// Color constants for chart datasets
const A1 = '#89b4fa'
const A2 = '#a6e3a1'
const B1 = '#f38ba8'
const B2 = '#f9e2af'

export default function MatchStats({ stats, botA, botB }) {
  if (!stats) return null
  const { team_a, team_b } = stats

  return (
    <div className="space-y-5 mt-3">
      <StatSummary teamA={team_a} teamB={team_b} botA={botA} botB={botB} />

      <TimelineChart teamA={team_a} teamB={team_b} botA={botA} botB={botB}
        title="Resources at Core"
        datasets={(a, b, rounds) => [
          { label: `${botA} Ti`, data: lookupField(a, 'titanium', rounds), borderColor: A1, backgroundColor: A1 + '22', tension: 0.3, borderWidth: 2, pointRadius: 2, fill: true },
          { label: `${botB} Ti`, data: lookupField(b, 'titanium', rounds), borderColor: B1, backgroundColor: B1 + '22', tension: 0.3, borderWidth: 2, pointRadius: 2, fill: true },
          { label: `${botA} Ax`, data: lookupField(a, 'axionite', rounds), borderColor: A2, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
          { label: `${botB} Ax`, data: lookupField(b, 'axionite', rounds), borderColor: B2, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
        ]}
      />

      <TimelineChart teamA={team_a} teamB={team_b} botA={botA} botB={botB}
        title="Cumulative Income (Delivered to Core)"
        datasets={(a, b, rounds) => [
          { label: `${botA} Ti`, data: lookupField(a, 'cumulative_income_ti', rounds), borderColor: A1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botB} Ti`, data: lookupField(b, 'cumulative_income_ti', rounds), borderColor: B1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botA} Ax`, data: lookupField(a, 'cumulative_income_ax', rounds), borderColor: A2, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
          { label: `${botB} Ax`, data: lookupField(b, 'cumulative_income_ax', rounds), borderColor: B2, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
        ]}
      />

      <TimelineChart teamA={team_a} teamB={team_b} botA={botA} botB={botB}
        title="Cumulative Spending"
        datasets={(a, b, rounds) => [
          { label: `${botA} Ti`, data: lookupField(a, 'cumulative_spent_ti', rounds), borderColor: A1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botB} Ti`, data: lookupField(b, 'cumulative_spent_ti', rounds), borderColor: B1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
        ]}
      />

      <TimelineChart teamA={team_a} teamB={team_b} botA={botA} botB={botB}
        title="Resources In Transit"
        datasets={(a, b, rounds) => [
          { label: `${botA} Ti`, data: lookupField(a, 'in_transit_ti', rounds), borderColor: A1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botB} Ti`, data: lookupField(b, 'in_transit_ti', rounds), borderColor: B1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botA} Ax`, data: lookupField(a, 'in_transit_ax', rounds), borderColor: A2, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
          { label: `${botB} Ax`, data: lookupField(b, 'in_transit_ax', rounds), borderColor: B2, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
        ]}
      />

      <TimelineChart teamA={team_a} teamB={team_b} botA={botA} botB={botB}
        title="Ore Utilization (Visible)"
        datasets={(a, b, rounds) => [
          { label: `${botA} mined`, data: lookupField(a, 'mined_ore', rounds), borderColor: A1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botA} unmined`, data: lookupField(a, 'unmined_ore', rounds), borderColor: A1, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
          { label: `${botB} mined`, data: lookupField(b, 'mined_ore', rounds), borderColor: B1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botB} unmined`, data: lookupField(b, 'unmined_ore', rounds), borderColor: B1, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
        ]}
      />

      <TimelineChart teamA={team_a} teamB={team_b} botA={botA} botB={botB}
        title="Total Units (Visible)"
        datasets={(a, b, rounds) => {
          function totalUnits(snaps) {
            const map = Object.fromEntries(snaps.map(s => [s.round, Object.values(s.own_units || {}).reduce((sum, v) => sum + v, 0)]))
            return rounds.map(r => map[r] ?? null)
          }
          return [
            { label: `${botA}`, data: totalUnits(a), borderColor: A1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
            { label: `${botB}`, data: totalUnits(b), borderColor: B1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          ]
        }}
      />

      <TimelineChart teamA={team_a} teamB={team_b} botA={botA} botB={botB}
        title="Core HP"
        datasets={(a, b, rounds) => [
          { label: `${botA}`, data: lookupField(a, 'core_hp', rounds), borderColor: A1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botB}`, data: lookupField(b, 'core_hp', rounds), borderColor: B1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
        ]}
      />

      <TimelineChart teamA={team_a} teamB={team_b} botA={botA} botB={botB}
        title="Cost Scale %"
        datasets={(a, b, rounds) => [
          { label: `${botA}`, data: lookupField(a, 'scale_percent', rounds), borderColor: A1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botB}`, data: lookupField(b, 'scale_percent', rounds), borderColor: B1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
        ]}
      />
    </div>
  )
}
