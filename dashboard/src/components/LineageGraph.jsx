import { useMemo, useState } from 'react'
import { botColor } from '../data'

export default function LineageGraph({ lineage, allBots, latestBots }) {
  const nodes = useMemo(() => Object.keys(lineage || {}), [lineage])
  const [hovered, setHovered] = useState(null)

  const { positions, minX, width, height } = useMemo(() => {
    if (!nodes.length) {
      return { positions: {}, minX: 0, width: 320, height: 200 }
    }

    const children = {}
    const roots = []
    nodes.forEach((node) => {
      const parent = lineage[node]?.parent
      if (!parent || !lineage[parent]) roots.push(node)
      else {
        children[parent] = children[parent] || []
        children[parent].push(node)
      }
    })

    const positions = {}
    let xCounter = 0
    const xSpacing = 140
    const ySpacing = 90

    function layout(node, depth) {
      const kids = (children[node] || []).sort()
      if (!kids.length) {
        positions[node] = { x: xCounter * xSpacing, y: depth * ySpacing }
        xCounter += 1
      } else {
        kids.forEach((child) => layout(child, depth + 1))
        positions[node] = {
          x: (positions[kids[0]].x + positions[kids[kids.length - 1]].x) / 2,
          y: depth * ySpacing,
        }
      }
    }

    roots.sort().forEach((root) => layout(root, 0))

    const allX = Object.values(positions).map((point) => point.x)
    const allY = Object.values(positions).map((point) => point.y)
    const margin = 70
    const minX = Math.min(...allX) - margin
    const width = Math.max(...allX) - minX + margin + 160
    const height = Math.max(...allY) + margin * 2

    return { positions, minX, width, height }
  }, [lineage, nodes])

  const latestElos = useMemo(
    () => Object.values(latestBots || {}).filter((elo) => elo != null),
    [latestBots],
  )
  const minElo = Math.min(...latestElos, 1000)
  const maxElo = Math.max(...latestElos, 1000)
  const botIndex = useMemo(
    () => new Map(allBots.map((name, index) => [name, index])),
    [allBots],
  )

  if (!nodes.length) return <p className="text-overlay0 text-sm">No lineage data.</p>

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

  return (
    <div className="overflow-auto">
      <svg
        viewBox={`${minX} -70 ${width} ${height}`}
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
          const elo = latestBots?.[n]
          const ci = botIndex.get(n)
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
