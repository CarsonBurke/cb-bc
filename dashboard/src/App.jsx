import { useAtom } from 'jotai'
import { currentViewPersistAtom } from './store'
import { DATA } from './data'
import Sidebar from './components/Sidebar'
import AllRoundsView from './components/AllRoundsView'
import RoundView from './components/RoundView'
import ReplayViewer from './components/ReplayViewer'
import TrainingView from './components/TrainingView'
import GameViewer from './components/GameViewer'

export default function App() {
  const [currentView, setCurrentView] = useAtom(currentViewPersistAtom)

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        rounds={DATA.rounds}
        currentView={currentView}
        onSelect={setCurrentView}
      />
      <main className="flex-1 overflow-y-auto p-6 lg:p-8">
        {currentView === 'training' ? (
          <TrainingView />
        ) : currentView === 'replays' ? (
          <ReplayViewer />
        ) : currentView === 'all' ? (
          <AllRoundsView />
        ) : (
          <RoundView roundId={currentView} />
        )}
      </main>
      <GameViewer />
    </div>
  )
}
