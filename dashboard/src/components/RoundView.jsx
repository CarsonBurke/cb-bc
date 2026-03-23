import Card from './Card'
import RankingsTable from './RankingsTable'
import WinRateMatrix from './WinRateMatrix'
import EloBarChart from './EloBarChart'
import MatchResults from './MatchResults'
import { useDashboardRoundQuery } from '../hooks/useDashboardQueries'

export default function RoundView({ roundId, lineage, allBots }) {
  const roundQuery = useDashboardRoundQuery(roundId)
  const roundData = roundQuery.data
  if (roundQuery.isError) return <p className="text-overlay0">Round failed to load.</p>
  if (!roundData) return <p className="text-overlay0">Loading round...</p>

  const { round, prevBots } = roundData
  if (!round) return <p className="text-overlay0">Round not found.</p>

  return (
    <>
      <Card title="Rankings">
        <RankingsTable bots={round.bots} prevBots={prevBots} lineage={lineage} />
      </Card>

      <Card title="Win Rate Matrix">
        <WinRateMatrix bots={round.bots} matches={round.matches} />
      </Card>

      <Card title="ELO Distribution" defaultOpen={false}>
        <EloBarChart bots={round.bots} allBots={allBots} />
      </Card>

      <Card title={`Match Results (${(round.matches || []).length})`} defaultOpen={false}>
        <MatchResults matches={round.matches} />
      </Card>
    </>
  )
}
