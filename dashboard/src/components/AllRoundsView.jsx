import { DATA } from '../data'
import Card from './Card'
import EloProgressionChart from './EloProgressionChart'
import EloDeltaChart from './EloDeltaChart'
import LineageGraph from './LineageGraph'

export default function AllRoundsView() {
  const { rounds } = DATA

  return (
    <>
      <Card title="ELO Progression">
        <EloProgressionChart rounds={rounds} />
      </Card>

      <Card title="ELO Delta" defaultOpen={false}>
        <EloDeltaChart rounds={rounds} />
      </Card>

      <Card title="Bot Lineage">
        <LineageGraph />
      </Card>
    </>
  )
}
