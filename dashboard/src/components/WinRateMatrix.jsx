import { useMemo } from 'react'

export default function WinRateMatrix({ bots, matches }) {
  const { sortedBots, cells } = useMemo(() => {
    const sortedBots = Object.keys(bots).sort()
    const idx = new Map(sortedBots.map((b, i) => [b, i]))
    const n = sortedBots.length
    const wins = new Int32Array(n * n)
    const games = new Int32Array(n * n)

    for (const m of matches || []) {
      const a = idx.get(m.bot_a)
      const b = idx.get(m.bot_b)
      if (a === undefined || b === undefined) continue
      games[a * n + b]++
      games[b * n + a]++
      if (m.winner === m.bot_a) wins[a * n + b]++
      else if (m.winner === m.bot_b) wins[b * n + a]++
    }

    const cells = new Array(n)
    for (let i = 0; i < n; i++) {
      const row = new Array(n)
      for (let j = 0; j < n; j++) {
        if (i === j) {
          row[j] = { style: 'text-overlay0', text: '—' }
        } else {
          const g = games[i * n + j]
          if (g === 0) {
            row[j] = { style: 'text-overlay0', text: '—' }
          } else {
            const rate = Math.round(100 * wins[i * n + j] / g)
            let style
            if (rate >= 60) style = 'bg-green/12 text-green font-semibold'
            else if (rate <= 40) style = 'bg-red/12 text-red font-semibold'
            else style = 'bg-yellow/10 text-yellow font-semibold'
            row[j] = { style, text: rate + '%' }
          }
        }
      }
      cells[i] = row
    }
    return { sortedBots, cells }
  }, [bots, matches])

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-[12px]">
        <thead>
          <tr>
            <th className="p-2 text-left text-[12px] text-subtext0 font-medium" />
            {sortedBots.map(b => (
              <th key={b} className="p-2 text-center text-[12px] text-subtext0 font-medium whitespace-nowrap">
                {b}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedBots.map((b, i) => (
            <tr key={b} className="border-b border-surface0/50">
              <td className="p-2 font-medium text-text whitespace-nowrap">{b}</td>
              {sortedBots.map((o, j) => (
                <td key={o} className={`p-2 text-center ${cells[i][j].style}`}>
                  {cells[i][j].text}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
