import { useState } from 'react'
import { DATA } from './data'
import Sidebar from './components/Sidebar'
import AllRoundsView from './components/AllRoundsView'
import RoundView from './components/RoundView'

export default function App() {
  const [currentView, setCurrentView] = useState('all')

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        rounds={DATA.rounds}
        currentView={currentView}
        onSelect={setCurrentView}
      />
      <main className="flex-1 overflow-y-auto p-6 lg:p-8">
        {currentView === 'all' ? (
          <AllRoundsView />
        ) : (
          <RoundView roundId={currentView} />
        )}
      </main>
    </div>
  )
}
