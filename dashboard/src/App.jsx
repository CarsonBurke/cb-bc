import { useEffect } from 'react'
import { useAtom } from 'jotai'
import { useAtomValue } from 'jotai'
import { useStore } from 'jotai'
import { currentViewPersistAtom, syncFromHash, trainSessionAtom } from './store'
import { DATA } from './data'
import Sidebar from './components/Sidebar'
import AllRoundsView from './components/AllRoundsView'
import RoundView from './components/RoundView'
import ReplayViewer from './components/ReplayViewer'
import TrainingView from './components/TrainingView'
import GameViewer from './components/GameViewer'

export default function App() {
  const [currentView, setCurrentView] = useAtom(currentViewPersistAtom)
  const trainSession = useAtomValue(trainSessionAtom)
  const store = useStore()

  // Sync atoms from URL on back/forward navigation
  useEffect(() => {
    const onPop = () => {
      syncFromHash((atom, val) => store.set(atom, val))
    }
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [store])

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        rounds={DATA.rounds}
        currentView={currentView}
        onSelect={setCurrentView}
      />
      <main className={`flex-1 ${currentView === 'training' ? 'overflow-hidden p-3' : 'overflow-y-auto p-6 lg:p-8'}`}>
        {currentView === 'training' ? (
          <TrainingView key={`training:${trainSession || 'default'}`} />
        ) : currentView === 'replays' ? (
          <ReplayViewer key="replays" />
        ) : currentView === 'all' ? (
          <AllRoundsView key="all" />
        ) : (
          <RoundView key={`round:${currentView}`} roundId={currentView} />
        )}
      </main>
      <GameViewer />
    </div>
  )
}
