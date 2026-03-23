import { Table, Th, Td, Thead, Tr } from './Table'

export default function RankingsTable({ bots, prevBots, lineage }) {
  const sorted = Object.entries(bots).sort((a, b) => b[1].elo - a[1].elo)

  return (
    <Table>
      <Thead>
        <Th className="w-8">#</Th>
        <Th>Bot</Th>
        <Th>ELO</Th>
        <Th>Delta</Th>
        <Th>W</Th>
        <Th>L</Th>
        <Th>D</Th>
        <Th>Notes</Th>
      </Thead>
      <tbody>
        {sorted.map(([name, d], i) => {
          const prevElo = prevBots?.[name]?.elo
          const delta = prevElo != null ? d.elo - prevElo : null
          const changelog = lineage[name]?.changelog || d.changelog || ''

          return (
            <Tr key={name}>
              <Td className="text-overlay0">{i + 1}</Td>
              <Td className="font-semibold text-text">{name}</Td>
              <Td className="font-semibold">{d.elo}</Td>
              <Td><DeltaBadge delta={delta} /></Td>
              <Td className="text-green">{d.wins}</Td>
              <Td className="text-red">{d.losses}</Td>
              <Td className="text-overlay0">{d.draws}</Td>
              <Td className="text-[12px] text-overlay0 max-w-[200px] truncate" title={changelog}>
                {changelog}
              </Td>
            </Tr>
          )
        })}
      </tbody>
    </Table>
  )
}

function DeltaBadge({ delta }) {
  if (delta == null) {
    return <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue/15 text-blue font-medium">new</span>
  }
  if (delta > 0) {
    return <span className="text-green font-semibold">+{delta}</span>
  }
  if (delta < 0) {
    return <span className="text-red font-semibold">{delta}</span>
  }
  return <span className="text-overlay0">0</span>
}
