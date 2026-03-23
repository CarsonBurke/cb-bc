import { lazy, Suspense, useEffect } from 'react'
import { useAtom } from 'jotai'
import { useAtomValue } from 'jotai'
import { useStore } from 'jotai'
import { currentViewPersistAtom, syncFromHash, trainSessionAtom } from './store'
import Sidebar from './components/Sidebar'
import GameViewer from './components/GameViewer'
import { useDashboardOverviewQuery } from './hooks/useDashboardQueries'

const AllRoundsView = lazy(() => import('./components/AllRoundsView'))
const RoundView = lazy(() => import('./components/RoundView'))
const ReplayViewer = lazy(() => import('./components/ReplayViewer'))
const TrainingView = lazy(() => import('./components/TrainingView'))

export default function App() {
  const [currentView, setCurrentView] = useAtom(currentViewPersistAtom)
  const trainSession = useAtomValue(trainSessionAtom)
  const store = useStore()
  const overviewQuery = useDashboardOverviewQuery()
  const overview = overviewQuery.data

  // Sync atoms from URL on back/forward navigation
  useEffect(() => {
    const onPop = () => {
      syncFromHash((atom, val) => store.set(atom, val))
    }
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [store])

  let content
  if (currentView === 'training') {
    content = <TrainingView key={`training:${trainSession || 'default'}`} />
  } else if (currentView === 'replays') {
    content = <ReplayViewer key="replays" />
  } else if (overviewQuery.isPending) {
    content = <LoadingState message="Loading dashboard data..." />
  } else if (overviewQuery.isError) {
    content = <LoadingState message="Failed to load dashboard data." />
  } else if (currentView === 'all') {
    content = <AllRoundsView key="all" overview={overview} />
  } else {
    content = (
      <RoundView
        key={`round:${currentView}`}
        roundId={currentView}
        lineage={overview.lineage}
        allBots={overview.allBots}
      />
    )
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        rounds={overview?.rounds || []}
        currentView={currentView}
        onSelect={setCurrentView}
      />
      <main className={`flex-1 ${currentView === 'training' ? 'overflow-hidden p-3' : 'overflow-y-auto p-6 lg:p-8'}`}>
        <Suspense fallback={<LoadingState message="Loading view..." />}>
          {content}
        </Suspense>
      </main>
      <GameViewer />
    </div>
  )
}

function LoadingState({ message }) {
  return <p className="text-overlay0 text-sm py-6">{message}</p>
}
