import Card from './Card'
import EloProgressionChart from './EloProgressionChart'
import EloDeltaChart from './EloDeltaChart'
import LineageGraph from './LineageGraph'

export default function AllRoundsView({ overview }) {
  const latestBots = overview.timeline[overview.timeline.length - 1]?.bots || null

  return (
    <>
      <Card title="ELO Progression">
        <EloProgressionChart timeline={overview.timeline} allBots={overview.allBots} />
      </Card>

      <Card title="ELO Delta" defaultOpen={false}>
        <EloDeltaChart timeline={overview.timeline} allBots={overview.allBots} />
      </Card>

      <Card title="Bot Lineage">
        <LineageGraph lineage={overview.lineage} allBots={overview.allBots} latestBots={latestBots} />
      </Card>
    </>
  )
}
