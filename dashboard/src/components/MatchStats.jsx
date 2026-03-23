import { useMemo } from 'react'
import { Line } from 'react-chartjs-2'
import { CHART_THEME } from '../data'
import { Table, Th, Td, Thead, Tr } from './Table'
import '../chartjs'

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

function TimelineChart({ title, labels, datasets }) {
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
  const team_a = stats?.team_a || null
  const team_b = stats?.team_b || null
  const snapsA = useMemo(() => team_a?.snapshots || [], [team_a])
  const snapsB = useMemo(() => team_b?.snapshots || [], [team_b])

  const { labels, rounds, totalUnitsA, totalUnitsB } = useMemo(() => {
    const rounds = mergeRounds(snapsA, snapsB)
    const labels = rounds.map((round) => `${round}`)
    const totalUnitsA = lookupField(
      snapsA.map((snap) => ({
        round: snap.round,
        total: Object.values(snap.own_units || {}).reduce((sum, value) => sum + value, 0),
      })),
      'total',
      rounds,
    )
    const totalUnitsB = lookupField(
      snapsB.map((snap) => ({
        round: snap.round,
        total: Object.values(snap.own_units || {}).reduce((sum, value) => sum + value, 0),
      })),
      'total',
      rounds,
    )
    return { labels, rounds, totalUnitsA, totalUnitsB }
  }, [snapsA, snapsB])

  if (!stats) return null

  return (
    <div className="space-y-5 mt-3">
      <StatSummary teamA={team_a} teamB={team_b} botA={botA} botB={botB} />

      <TimelineChart
        title="Resources at Core"
        labels={labels}
        datasets={[
          { label: `${botA} Ti`, data: lookupField(snapsA, 'titanium', rounds), borderColor: A1, backgroundColor: A1 + '22', tension: 0.3, borderWidth: 2, pointRadius: 2, fill: true },
          { label: `${botB} Ti`, data: lookupField(snapsB, 'titanium', rounds), borderColor: B1, backgroundColor: B1 + '22', tension: 0.3, borderWidth: 2, pointRadius: 2, fill: true },
          { label: `${botA} Ax`, data: lookupField(snapsA, 'axionite', rounds), borderColor: A2, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
          { label: `${botB} Ax`, data: lookupField(snapsB, 'axionite', rounds), borderColor: B2, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
        ]}
      />

      <TimelineChart
        title="Cumulative Income (Delivered to Core)"
        labels={labels}
        datasets={[
          { label: `${botA} Ti`, data: lookupField(snapsA, 'cumulative_income_ti', rounds), borderColor: A1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botB} Ti`, data: lookupField(snapsB, 'cumulative_income_ti', rounds), borderColor: B1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botA} Ax`, data: lookupField(snapsA, 'cumulative_income_ax', rounds), borderColor: A2, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
          { label: `${botB} Ax`, data: lookupField(snapsB, 'cumulative_income_ax', rounds), borderColor: B2, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
        ]}
      />

      <TimelineChart
        title="Cumulative Spending"
        labels={labels}
        datasets={[
          { label: `${botA} Ti`, data: lookupField(snapsA, 'cumulative_spent_ti', rounds), borderColor: A1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botB} Ti`, data: lookupField(snapsB, 'cumulative_spent_ti', rounds), borderColor: B1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
        ]}
      />

      <TimelineChart
        title="Resources In Transit"
        labels={labels}
        datasets={[
          { label: `${botA} Ti`, data: lookupField(snapsA, 'in_transit_ti', rounds), borderColor: A1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botB} Ti`, data: lookupField(snapsB, 'in_transit_ti', rounds), borderColor: B1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botA} Ax`, data: lookupField(snapsA, 'in_transit_ax', rounds), borderColor: A2, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
          { label: `${botB} Ax`, data: lookupField(snapsB, 'in_transit_ax', rounds), borderColor: B2, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
        ]}
      />

      <TimelineChart
        title="Ore Utilization (Visible)"
        labels={labels}
        datasets={[
          { label: `${botA} mined`, data: lookupField(snapsA, 'mined_ore', rounds), borderColor: A1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botA} unmined`, data: lookupField(snapsA, 'unmined_ore', rounds), borderColor: A1, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
          { label: `${botB} mined`, data: lookupField(snapsB, 'mined_ore', rounds), borderColor: B1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botB} unmined`, data: lookupField(snapsB, 'unmined_ore', rounds), borderColor: B1, borderDash: [4, 2], tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
        ]}
      />

      <TimelineChart
        title="Total Units (Visible)"
        labels={labels}
        datasets={[
          { label: `${botA}`, data: totalUnitsA, borderColor: A1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botB}`, data: totalUnitsB, borderColor: B1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
        ]}
      />

      <TimelineChart
        title="Core HP"
        labels={labels}
        datasets={[
          { label: `${botA}`, data: lookupField(snapsA, 'core_hp', rounds), borderColor: A1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botB}`, data: lookupField(snapsB, 'core_hp', rounds), borderColor: B1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
        ]}
      />

      <TimelineChart
        title="Cost Scale %"
        labels={labels}
        datasets={[
          { label: `${botA}`, data: lookupField(snapsA, 'scale_percent', rounds), borderColor: A1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: `${botB}`, data: lookupField(snapsB, 'scale_percent', rounds), borderColor: B1, tension: 0.3, borderWidth: 2, pointRadius: 2 },
        ]}
      />
    </div>
  )
}
