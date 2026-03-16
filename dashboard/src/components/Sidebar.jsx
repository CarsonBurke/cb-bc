import { formatDate } from '../data'

export default function Sidebar({ rounds, currentView, onSelect }) {
  const reversed = [...rounds].reverse()

  return (
    <aside className="w-60 min-w-60 bg-mantle border-r border-surface0/60 flex flex-col overflow-hidden max-md:w-full max-md:min-w-full max-md:h-auto max-md:max-h-48 max-md:border-r-0 max-md:border-b">
      <div className="px-4 py-3 border-b border-surface0">
        <h1 className="text-[13px] font-semibold text-text">Ablation Rounds</h1>
      </div>
      <nav className="flex-1 overflow-y-auto p-1.5 max-md:flex max-md:overflow-x-auto max-md:overflow-y-hidden max-md:gap-1">
        <SidebarItem
          active={currentView === 'all'}
          onClick={() => onSelect('all')}
          title="All Rounds"
          meta="Aggregate"
        />
        <div className="h-px bg-surface0 mx-2 my-1 max-md:w-px max-md:h-auto max-md:mx-0 max-md:my-0" />
        {reversed.map(r => (
          <SidebarItem
            key={r.round_id}
            active={currentView === r.round_id}
            onClick={() => onSelect(r.round_id)}
            title={`Round ${r.round_id}`}
            meta={`${formatDate(r.timestamp, { month: 'short', day: 'numeric' })} · ${Object.keys(r.bots).length} bots`}
          />
        ))}
      </nav>
    </aside>
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
