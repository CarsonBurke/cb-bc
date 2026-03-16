export default function WinRateMatrix({ bots, matches }) {
  const sortedBots = Object.keys(bots).sort()
  const wins = {}
  const games = {}

  sortedBots.forEach(b => {
    wins[b] = {}
    games[b] = {}
    sortedBots.forEach(o => { wins[b][o] = 0; games[b][o] = 0 })
  })

  ;(matches || []).forEach(m => {
    if (!(m.bot_a in bots) || !(m.bot_b in bots)) return
    games[m.bot_a][m.bot_b]++
    games[m.bot_b][m.bot_a]++
    if (m.winner === m.bot_a) wins[m.bot_a][m.bot_b]++
    else if (m.winner === m.bot_b) wins[m.bot_b][m.bot_a]++
  })

  function cellStyle(bot, opp) {
    if (bot === opp) return 'text-overlay0'
    if (games[bot][opp] === 0) return 'text-overlay0'
    const rate = Math.round(100 * wins[bot][opp] / games[bot][opp])
    if (rate >= 60) return 'bg-green/12 text-green font-semibold'
    if (rate <= 40) return 'bg-red/12 text-red font-semibold'
    return 'bg-yellow/10 text-yellow font-semibold'
  }

  function cellText(bot, opp) {
    if (bot === opp) return '—'
    if (games[bot][opp] === 0) return '—'
    return Math.round(100 * wins[bot][opp] / games[bot][opp]) + '%'
  }

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
          {sortedBots.map(b => (
            <tr key={b} className="border-b border-surface0/50">
              <td className="p-2 font-medium text-text whitespace-nowrap">{b}</td>
              {sortedBots.map(o => (
                <td key={o} className={`p-2 text-center ${cellStyle(b, o)}`}>
                  {cellText(b, o)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
