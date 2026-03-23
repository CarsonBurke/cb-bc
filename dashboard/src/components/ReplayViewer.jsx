import { useSetAtom } from 'jotai'
import { viewerAtom, viewerMinimizedAtom } from '../store'
import Card from './Card'
import { Table, Th, Td, Thead, Tr } from './Table'
import { relativeTime } from '../data'
import { viewerUrl } from '../api'
import { useReplayListQuery } from '../hooks/useDashboardQueries'

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}



function parseReplayName(name) {
  const m = name.match(/^match_(\d+)\.replay26$/)
  if (m) return `Match ${m[1]}`
  return name.replace('.replay26', '')
}

export default function ReplayViewer() {
  const replayQuery = useReplayListQuery()
  const replays = replayQuery.data || []
  const loading = replayQuery.isPending || replayQuery.isFetching
  const setViewer = useSetAtom(viewerAtom)
  const setMinimized = useSetAtom(viewerMinimizedAtom)

  function refresh() {
    replayQuery.refetch()
  }

  function openReplay(name) {
    setViewer({
      url: viewerUrl(`/api/replay/${encodeURIComponent(name)}`),
      title: parseReplayName(name),
    })
    setMinimized(false)
  }

  return (
    <Card title="Replays" id="replay-list">
      <div className="flex items-center gap-3 mb-3">
        <button
          onClick={refresh}
          disabled={loading}
          className="px-3 py-1.5 rounded text-[12px] font-medium bg-surface0/60 text-overlay1 hover:bg-surface0 transition-colors disabled:opacity-50"
        >
          {loading ? 'Loading...' : 'Refresh'}
        </button>
        <span className="text-[11px] text-overlay0">
          {replays.length} replay{replays.length !== 1 ? 's' : ''} found
        </span>
      </div>

      {replays.length === 0 && !loading ? (
        <p className="text-[13px] text-overlay0 py-4">
          No replays found in <code className="text-subtext0">replays/</code>. Run a match to generate one.
        </p>
      ) : (
        <Table>
          <Thead>
            <Th>Replay</Th>
            <Th>Size</Th>
            <Th>Date</Th>
            <Th className="w-28">Actions</Th>
          </Thead>
          <tbody>
            {replays.map(r => (
              <Tr
                key={r.name}
                className="cursor-pointer"
                onClick={() => openReplay(r.name)}
              >
                <Td>{parseReplayName(r.name)}</Td>
                <Td>{formatSize(r.size)}</Td>
                <Td>{relativeTime(r.mtime)}</Td>
                <Td>
                  <button
                    onClick={(e) => { e.stopPropagation(); openReplay(r.name) }}
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-blue/10 text-blue hover:bg-blue/20 transition-colors"
                  >
                    <svg className="w-3 h-3" viewBox="0 0 16 16" fill="currentColor">
                      <path d="M4 2l10 6-10 6V2z" />
                    </svg>
                    Watch
                  </button>
                </Td>
              </Tr>
            ))}
          </tbody>
        </Table>
      )}
    </Card>
  )
}
