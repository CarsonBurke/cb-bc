import { DATA, formatDate } from '../data'
import Card from './Card'
import RankingsTable from './RankingsTable'
import WinRateMatrix from './WinRateMatrix'
import EloBarChart from './EloBarChart'
import MatchResults from './MatchResults'

export default function RoundView({ roundId }) {
  const round = DATA.rounds.find(r => r.round_id === roundId)
  if (!round) return <p className="text-overlay0">Round not found.</p>

  const prevRound = DATA.rounds.find(r => r.round_id === roundId - 1) || null

  return (
    <>
      <Card title="Rankings">
        <RankingsTable bots={round.bots} prevBots={prevRound?.bots} />
      </Card>

      <Card title="Win Rate Matrix">
        <WinRateMatrix bots={round.bots} matches={round.matches} />
      </Card>

      <Card title="ELO Distribution" defaultOpen={false}>
        <EloBarChart bots={round.bots} />
      </Card>

      <Card title={`Match Results (${(round.matches || []).length})`} defaultOpen={false}>
        <MatchResults matches={round.matches} />
      </Card>
    </>
  )
}
