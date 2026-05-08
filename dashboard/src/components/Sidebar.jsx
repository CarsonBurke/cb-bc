import { useEffect, useMemo } from 'react'
import { useAtom } from 'jotai'
import { modePersistAtom, trainSessionPersistAtom } from '../store'
import { relativeTime } from '../data'
import { useTrainSessionsQuery } from '../hooks/useDashboardQueries'

function formatSessionDate(ts) {
  if (!ts) return ''
  return relativeTime(ts * 1000)
}

export default function Sidebar({ rounds, currentView, onSelect }) {
  const [mode, setMode] = useAtom(modePersistAtom)
  const [activeSession, setActiveSession] = useAtom(trainSessionPersistAtom)
  const sessionsQuery = useTrainSessionsQuery()
  const sessions = useMemo(() => sessionsQuery.data || [], [sessionsQuery.data])
  const reversed = useMemo(() => [...rounds].reverse(), [rounds])

  // Auto-select most recent session if none selected
  useEffect(() => {
    if (mode === 'tensor' && !activeSession && sessions.length > 0) {
      setActiveSession(sessions[0].id)
    }
  }, [mode, sessions, activeSession, setActiveSession])

  function selectSession(id) {
    setActiveSession(id)
    onSelect('training')
  }

  return (
    <aside className="w-60 min-w-60 bg-mantle border-r border-surface0/60 flex flex-col overflow-hidden max-md:w-full max-md:min-w-full max-md:h-auto max-md:max-h-48 max-md:border-r-0 max-md:border-b">
      <div className="px-3 py-2.5 border-b border-surface0">
        <div className="flex bg-surface0/60 rounded p-0.5">
          <ModeTab active={mode === 'code'} onClick={() => setMode('code')}>Code</ModeTab>
          <ModeTab active={mode === 'tensor'} onClick={() => setMode('tensor')}>Tensor</ModeTab>
        </div>
      </div>
      <nav className="flex-1 overflow-y-auto p-1.5 max-md:flex max-md:overflow-x-auto max-md:overflow-y-hidden max-md:gap-1">
        {mode === 'code' ? (
          <>
            <SidebarItem
              active={currentView === 'all'}
              onClick={() => onSelect('all')}
              title="All Rounds"
              meta="Aggregate"
            />
            <SidebarItem
              active={currentView === 'replays'}
              onClick={() => onSelect('replays')}
              title="Replays"
              meta="Watch matches"
            />
            <div className="h-px bg-surface0 mx-2 my-1 max-md:w-px max-md:h-auto max-md:mx-0 max-md:my-0" />
            {reversed.map(r => (
              <SidebarItem
                key={r.round_id}
                active={currentView === r.round_id}
                onClick={() => onSelect(r.round_id)}
                title={`Round ${r.round_id}`}
                meta={`${relativeTime(r.timestamp)} · ${r.bot_count} bots`}
              />
            ))}
          </>
        ) : (
          <>
            {sessions.length === 0 ? (
              <SidebarItem
                active={currentView === 'training'}
                onClick={() => onSelect('training')}
                title="Training"
                meta="No sessions yet"
              />
            ) : (
              sessions.map(s => (
                <SidebarItem
                  key={s.id}
                  active={activeSession === s.id && currentView === 'training'}
                  onClick={() => selectSession(s.id)}
                  title={s.name || s.id}
                  meta={`${s.episodes || 0} eps · ${s.mode || ''}${s.created ? ' · ' + formatSessionDate(s.created) : ''}`}
                />
              ))
            )}
          </>
        )}
      </nav>
    </aside>
  )
}

function ModeTab({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 text-center px-3 py-1.5 rounded text-[12px] font-semibold transition-colors ${
        active
          ? 'bg-blue/15 text-blue'
          : 'text-overlay1 hover:text-subtext1'
      }`}
    >
      {children}
    </button>
  )
}

function SidebarItem({ active, onClick, title, meta }) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2 rounded transition-colors max-md:whitespace-nowrap max-md:shrink-0 ${
        active
          ? 'bg-blue/15 text-blue'
          : 'hover:bg-surface0/60 text-subtext1'
      }`}
    >
      <div className="text-[13px] font-medium">{title}</div>
      <div className={`text-[11px] mt-0.5 ${active ? 'text-blue/70' : 'text-overlay0'}`}>
        {meta}
      </div>
    </button>
  )
}
