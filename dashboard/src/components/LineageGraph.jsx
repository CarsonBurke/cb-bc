import { useState } from 'react'
import { DATA, botColor, getAllBots, getLatestRound } from '../data'

export default function LineageGraph() {
  const lineage = DATA.lineage || {}
  const nodes = Object.keys(lineage)
  const [hovered, setHovered] = useState(null)

  if (!nodes.length) return <p className="text-overlay0 text-sm">No lineage data.</p>

  const children = {}
  const roots = []
  nodes.forEach(n => {
    const p = lineage[n]?.parent
    if (!p || !lineage[p]) roots.push(n)
    else { children[p] = children[p] || []; children[p].push(n) }
  })

  const positions = {}
  let xCounter = 0
  const xSpacing = 140
  const ySpacing = 90

  function layout(node, depth) {
    const kids = (children[node] || []).sort()
    if (!kids.length) {
      positions[node] = { x: xCounter * xSpacing, y: depth * ySpacing }
      xCounter++
    } else {
      kids.forEach(k => layout(k, depth + 1))
      const fx = positions[kids[0]].x
      const lx = positions[kids[kids.length - 1]].x
      positions[node] = { x: (fx + lx) / 2, y: depth * ySpacing }
    }
  }
  roots.sort().forEach(r => layout(r, 0))

  const allBots = getAllBots()
  const latest = getLatestRound()

  const elos = latest ? Object.values(latest.bots).map(b => b.elo) : []
  const minElo = Math.min(...elos, 1000)
  const maxElo = Math.max(...elos, 1000)

  function eloColor(elo) {
    if (elo == null || maxElo === minElo) return '#f9e2af'
    const t = (elo - minElo) / (maxElo - minElo)
    // red(f38ba8) → yellow(f9e2af) → green(a6e3a1)
    if (t < 0.5) {
      const s = t * 2
      const r = Math.round(0xf3 + (0xf9 - 0xf3) * s)
      const g = Math.round(0x8b + (0xe2 - 0x8b) * s)
      const b = Math.round(0xa8 + (0xaf - 0xa8) * s)
      return `rgb(${r},${g},${b})`
    } else {
      const s = (t - 0.5) * 2
      const r = Math.round(0xf9 + (0xa6 - 0xf9) * s)
      const g = Math.round(0xe2 + (0xe3 - 0xe2) * s)
      const b = Math.round(0xaf + (0xa1 - 0xaf) * s)
      return `rgb(${r},${g},${b})`
    }
  }

  const allX = Object.values(positions).map(p => p.x)
  const allY = Object.values(positions).map(p => p.y)
  const margin = 70
  const minX = Math.min(...allX) - margin
  const width = Math.max(...allX) - minX + margin + 160
  const height = Math.max(...allY) + margin * 2

  return (
    <div className="overflow-auto">
      <svg
        viewBox={`${minX} ${-margin} ${width} ${height}`}
        style={{ height: Math.max(height, 200), width: '100%' }}
        className="overflow-visible"
      >
        {nodes.map(n => {
          const p = lineage[n]?.parent
          if (!p || !positions[p] || !positions[n]) return null
          const px = positions[p]
          const cx = positions[n]
          return (
            <path
              key={`link-${n}`}
              d={`M${px.x},${px.y} C${px.x},${(px.y + cx.y) / 2} ${cx.x},${(px.y + cx.y) / 2} ${cx.x},${cx.y}`}
              fill="none"
              stroke="#45475a"
              strokeWidth={2}
            />
          )
        })}

        {nodes.map(n => {
          if (!positions[n]) return null
          const { x, y } = positions[n]
          const cl = lineage[n]?.changelog || ''
          const elo = latest?.bots[n]?.elo
          const ci = allBots.indexOf(n)
          const color = ci >= 0 ? botColor(ci) : '#a6e3a1'
          const isHovered = hovered === n

          return (
            <g
              key={n}
              transform={`translate(${x},${y})`}
              onMouseEnter={() => setHovered(n)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: 'default' }}
            >
              <rect x={-14} y={-14} width={28} height={28} fill="transparent" />
              <circle
                r={isHovered ? 12 : 10}
                fill={color}
                stroke={color}
                strokeOpacity={isHovered ? 0.5 : 0.3}
                strokeWidth={isHovered ? 8 : 6}
                style={{ transition: 'all 0.15s ease' }}
              />
              <text x={18} y={4} fill="#cdd6f4" fontSize={13} fontWeight={600}>{n}</text>
              {elo != null && (
                <text x={18} y={18} fill={eloColor(elo)} fontSize={11} fontWeight={600}>
                  ELO {elo}
                </text>
              )}
              {isHovered && cl && (
                <foreignObject x={16} y={elo != null ? 24 : 10} width={280} height={200}>
                  <div
                    xmlns="http://www.w3.org/1999/xhtml"
                    style={{
                      background: '#181825',
                      border: '1px solid #45475a',
                      borderRadius: 4,
                      padding: '8px 10px',
                      color: '#a6adc8',
                      fontSize: 11,
                      lineHeight: 1.5,
                      width: 'fit-content',
                      maxWidth: 280,
                    }}
                  >
                    {cl}
                  </div>
                </foreignObject>
              )}
            </g>
          )
        })}
      </svg>
    </div>
  )
}
