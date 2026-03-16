import { useState, useEffect, useRef } from 'react'
import { useAtomValue, useSetAtom } from 'jotai'
import { viewerAtom, viewerMinimizedAtom, trainSessionAtom } from '../store'
import { Line } from 'react-chartjs-2'
import Card from './Card'
import { Table, Th, Td, Thead, Tr } from './Table'
import { CHART_THEME } from '../data'

const POLL_MS = 5000

function useTrainLog(session) {
  const [data, setData] = useState({ episodes: [], config: {} })
  const [loading, setLoading] = useState(true)
  const timer = useRef(null)

  useEffect(() => {
    setLoading(true)
    function fetch_log() {
      const qs = session ? `?session=${encodeURIComponent(session)}` : ''
      fetch(`/api/train-log${qs}`)
        .then(r => r.json())
        .then(setData)
        .catch(() => {})
        .finally(() => setLoading(false))
    }
    fetch_log()
    timer.current = setInterval(fetch_log, POLL_MS)
    return () => clearInterval(timer.current)
  }, [session])

  return { data, loading }
}

function fmt(v, decimals = 2) {
  if (v == null) return '—'
  return typeof v === 'number' ? v.toFixed(decimals) : String(v)
}

function SingleChart({ episodes, label, dataKey, color, fillColor }) {
  if (!episodes.length) return null

  const opts = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    scales: {
      x: {
        ticks: { color: CHART_THEME.subtext, maxTicksLimit: 10 },
        grid: { color: CHART_THEME.grid },
      },
      y: {
        ticks: { color: CHART_THEME.subtext },
        grid: { color: CHART_THEME.grid },
      },
    },
    plugins: { legend: { display: false } },
  }

  return (
    <div className="min-h-[200px]">
      <Line
        data={{
          labels: episodes.map(e => e.episode),
          datasets: [{
            label,
            data: episodes.map(e => e[dataKey]),
            borderColor: color,
            backgroundColor: fillColor || color + '20',
            fill: !!fillColor,
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 1.5,
          }],
        }}
        options={opts}
      />
    </div>
  )
}

function LossChart({ episodes }) {
  if (!episodes.length) return null

  return (
    <div className="min-h-[200px]">
      <Line
        data={{
          labels: episodes.map(e => e.episode),
          datasets: [
            { label: 'Policy Loss', data: episodes.map(e => e.pi_loss), borderColor: '#f38ba8', tension: 0.3, pointRadius: 0, borderWidth: 1.5 },
            { label: 'Value Loss', data: episodes.map(e => e.vf_loss), borderColor: '#f9e2af', tension: 0.3, pointRadius: 0, borderWidth: 1.5 },
            { label: 'Entropy', data: episodes.map(e => e.entropy), borderColor: '#cba6f7', tension: 0.3, pointRadius: 0, borderWidth: 1.5 },
          ],
        }}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          scales: {
            x: { ticks: { color: CHART_THEME.subtext, maxTicksLimit: 10 }, grid: { color: CHART_THEME.grid } },
            y: { ticks: { color: CHART_THEME.subtext }, grid: { color: CHART_THEME.grid } },
          },
          plugins: { legend: { labels: { color: CHART_THEME.text, font: { size: 10 } } } },
        }}
      />
    </div>
  )
}

function useTrainReplays(session) {
  const [replays, setReplays] = useState([])
  const timer = useRef(null)

  useEffect(() => {
    function refresh() {
      const qs = session ? `?session=${encodeURIComponent(session)}` : ''
      fetch(`/api/train-replays${qs}`)
        .then(r => r.json())
        .then(setReplays)
        .catch(() => {})
    }
    refresh()
    timer.current = setInterval(refresh, POLL_MS)
    return () => clearInterval(timer.current)
  }, [session])

  return replays
}

function parseReplayName(name) {
  const m = name.match(/^ep(\d+)_game(\d+)\.replay26$/)
  if (m) return `Episode ${m[1]}, Game ${m[2]}`
  return name.replace('.replay26', '')
}

function formatTime(ms) {
  return new Date(ms).toLocaleString('en-US', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

function TrainingReplays({ session }) {
  const replays = useTrainReplays(session)
  const setViewer = useSetAtom(viewerAtom)
  const setMinimized = useSetAtom(viewerMinimizedAtom)

  function openReplay(name) {
    const sessionQs = session ? `?session=${encodeURIComponent(session)}` : ''
    const replayPath = `/api/train-replay/${encodeURIComponent(name)}${sessionQs}`
    setViewer({
      url: `/viewer/?replayUrl=${encodeURIComponent(replayPath)}`,
      title: parseReplayName(name),
    })
    setMinimized(false)
  }

  return (
    <Card title="Training Replays" id="train-replays">
      {replays.length === 0 ? (
        <p className="text-[13px] text-overlay0 py-2">
          No training replays yet. Run <code className="text-subtext0">train_engine.py</code> to generate.
        </p>
      ) : (
        <div className="max-h-64 overflow-y-auto">
          <Table>
            <Thead>
              <Th>Replay</Th>
              <Th>Date</Th>
              <Th className="w-24">Actions</Th>
            </Thead>
            <tbody>
              {replays.map(r => (
                <Tr key={r.name} className="cursor-pointer" onClick={() => openReplay(r.name)}>
                  <Td>{parseReplayName(r.name)}</Td>
                  <Td>{formatTime(r.mtime)}</Td>
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
        </div>
      )}
    </Card>
  )
}

export default function TrainingView() {
  const session = useAtomValue(trainSessionAtom)
  const { data, loading } = useTrainLog(session)
  const { episodes, config } = data

  const latest = episodes[episodes.length - 1]
  const progress = config.total_episodes
    ? `${episodes.length} / ${config.total_episodes}`
    : `${episodes.length}`

  return (
    <>
      {/* Summary */}
      <Card title="Training Status" id="train-status">
        {loading && !episodes.length ? (
          <p className="text-[13px] text-overlay0 py-4">Loading training data...</p>
        ) : !episodes.length ? (
          <p className="text-[13px] text-overlay0 py-4">
            No training log found. Run <code className="text-subtext0">python bots/tensor/torch/train_gpu.py</code> to start.
          </p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <Stat label="Episodes" value={progress} />
            <Stat label="Last Reward" value={fmt(latest?.reward, 3)} />
            <Stat label="Last Resources" value={fmt(latest?.resources, 1)} />
            <Stat label="Expl. Variance" value={fmt(latest?.explained_var, 3)} />
            <Stat label="Entropy" value={fmt(latest?.entropy, 3)} />
            <Stat label="Approx KL" value={fmt(latest?.approx_kl, 5)} />
            <Stat label="Elapsed" value={latest ? `${latest.elapsed}s` : '—'} />
            {config.lr && <Stat label="LR" value={config.lr} />}
          </div>
        )}
      </Card>

      {/* Charts — two per row */}
      {episodes.length > 1 && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card title="Reward / Episode" id="train-ep-reward">
              <SingleChart episodes={episodes} label="Reward" dataKey="reward" color="#a6e3a1" />
            </Card>
            <Card title="Resources / Episode" id="train-ep-res">
              <SingleChart episodes={episodes} label="Resources" dataKey="resources" color="#89b4fa" />
            </Card>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card title="Loss & Entropy" id="train-loss-chart" defaultOpen={false}>
              <LossChart episodes={episodes} />
            </Card>
            <Card title="Explained Variance" id="train-ev">
              <SingleChart episodes={episodes} label="Explained Var" dataKey="explained_var" color="#a6e3a1" fillColor="#a6e3a120" />
            </Card>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card title="Approx KL" id="train-kl" defaultOpen={false}>
              <SingleChart episodes={episodes} label="KL" dataKey="approx_kl" color="#fab387" />
            </Card>
            <Card title="Clip Fraction" id="train-clip" defaultOpen={false}>
              <SingleChart episodes={episodes} label="Clip Frac" dataKey="clip_frac" color="#f9e2af" />
            </Card>
          </div>
        </>
      )}

      {/* Episode table */}
      {episodes.length > 0 && (
        <Card title="Episode Log" id="train-log-table" defaultOpen={false}>
          <div className="max-h-96 overflow-y-auto">
            <Table>
              <Thead>
                <Th>Ep</Th>
                <Th>Reward</Th>
                <Th>Resources</Th>
                <Th>Pi Loss</Th>
                <Th>Entropy</Th>
                <Th>KL</Th>
                <Th>Clip%</Th>
                <Th>ExplVar</Th>
                <Th>SPS</Th>
              </Thead>
              <tbody>
                {[...episodes].reverse().map(e => (
                  <Tr key={e.episode}>
                    <Td>{e.episode}</Td>
                    <Td>{fmt(e.reward, 3)}</Td>
                    <Td>{fmt(e.resources, 1)}</Td>
                    <Td>{fmt(e.pi_loss, 4)}</Td>
                    <Td>{fmt(e.entropy, 3)}</Td>
                    <Td>{fmt(e.approx_kl, 5)}</Td>
                    <Td>{fmt(e.clip_frac, 3)}</Td>
                    <Td>{fmt(e.explained_var, 3)}</Td>
                    <Td>{e.sps?.toLocaleString()}</Td>
                  </Tr>
                ))}
              </tbody>
            </Table>
          </div>
        </Card>
      )}

      {/* Training replays */}
      <TrainingReplays session={session} />
    </>
  )
}

function Stat({ label, value }) {
  return (
    <div>
      <div className="text-[11px] text-overlay0">{label}</div>
      <div className="text-[15px] font-semibold text-text mt-0.5">{value}</div>
    </div>
  )
}
