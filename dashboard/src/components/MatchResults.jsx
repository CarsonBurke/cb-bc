import { useState } from 'react'
import MatchStats from './MatchStats'
import { Table, Th, Td, Thead, Tr } from './Table'

function WatchButton({ matchId }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(false)

  async function handleWatch() {
    setLoading(true)
    setError(false)
    try {
      const res = await fetch(`/api/watch?match_id=${matchId}`)
      const data = await res.json()
      if (data.url) {
        window.open(data.url, '_blank')
      } else {
        setError(true)
      }
    } catch {
      setError(true)
    } finally {
      setLoading(false)
    }
  }

  return (
    <button
      onClick={handleWatch}
      disabled={loading}
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-blue/10 text-blue hover:bg-blue/20 transition-colors disabled:opacity-50"
      title="Watch replay"
    >
      <svg className="w-3 h-3" viewBox="0 0 16 16" fill="currentColor">
        <path d="M4 2l10 6-10 6V2z" />
      </svg>
      {loading ? '...' : error ? 'N/A' : 'Watch'}
    </button>
  )
}

function MatchRow({ match }) {
  const [expanded, setExpanded] = useState(false)
  const m = match
  const draw = m.winner === 'draw' || !m.winner
  const hasStats = !!m.stats

  return (
    <>
      <Tr
        className={hasStats ? 'cursor-pointer' : ''}
        onClick={() => hasStats && setExpanded(!expanded)}
      >
        <Td className={m.winner === m.bot_a ? 'text-green font-semibold' : ''}>{m.bot_a}</Td>
        <Td className={m.winner === m.bot_b ? 'text-green font-semibold' : ''}>{m.bot_b}</Td>
        <Td>
          {draw
            ? <span className="text-yellow">Draw</span>
            : <span className="text-green">{m.winner}</span>
          }
        </Td>
        <Td>
          <div className="flex items-center gap-2">
            <WatchButton matchId={m.match_id ?? m._idx} />
            {hasStats && (
              <button
                onClick={(e) => { e.stopPropagation(); setExpanded(!expanded) }}
                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                  expanded ? 'bg-mauve/15 text-mauve' : 'bg-surface0/60 text-overlay1 hover:bg-surface0'
                }`}
              >
                <svg className={`w-2.5 h-2.5 transition-transform ${expanded ? 'rotate-90' : ''}`} fill="currentColor" viewBox="0 0 8 10">
                  <path d="M0 0 L8 5 L0 10Z" />
                </svg>
                Stats
              </button>
            )}
          </div>
        </Td>
      </Tr>
      {expanded && hasStats && (
        <tr>
          <td colSpan={4} className="p-4 bg-crust/50">
            <MatchStats stats={m.stats} botA={m.bot_a} botB={m.bot_b} />
          </td>
        </tr>
      )}
    </>
  )
}

export default function MatchResults({ matches }) {
  const byMap = {}
  ;(matches || []).forEach((m, idx) => {
    const map = m.map || 'unknown'
    if (!byMap[map]) byMap[map] = []
    byMap[map].push({ ...m, _idx: idx })
  })

  return (
    <div className="space-y-5">
      {Object.keys(byMap).sort().map(map => (
        <div key={map}>
          <h4 className="text-[12px] text-overlay0 font-medium mb-2">{map}</h4>
          <Table>
            <Thead>
              <Th>Bot A</Th>
              <Th>Bot B</Th>
              <Th>Winner</Th>
              <Th className="w-32">Actions</Th>
            </Thead>
            <tbody>
              {byMap[map].map((m, i) => (
                <MatchRow key={i} match={m} />
              ))}
            </tbody>
          </Table>
        </div>
      ))}
    </div>
  )
}
