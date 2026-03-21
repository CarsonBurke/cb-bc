import { useState, useEffect, useRef, memo } from 'react'
import { useAtomValue, useSetAtom } from 'jotai'
import {
  viewerAtom, viewerMinimizedAtom, trainSessionAtom,
  trainEpisodesAtom, trainConfigAtom, trainLoadingAtom,
  trainLeagueModelsAtom, trainGameLogAtom,
} from '../store'
import { Line, Bar } from 'react-chartjs-2'
import { Chart as ChartJS, BarElement } from 'chart.js'

ChartJS.register(BarElement)
import { Table, Th, Td, Thead, Tr } from './Table'
import { CHART_THEME, relativeTime } from '../data'
import TileLayout from './TileLayout'
import { useCardSettings } from './Card'

const POLL_MS = 2000

// ---- Data hooks ----

function useTrainLog(session) {
  const setEpisodes = useSetAtom(trainEpisodesAtom)
  const setConfig = useSetAtom(trainConfigAtom)
  const setLoading = useSetAtom(trainLoadingAtom)
  const timer = useRef(null)
  const accum = useRef({ episodes: [], config: {}, total: 0 })

  useEffect(() => {
    setLoading(true)
    accum.current = { episodes: [], config: {}, total: 0 }
    function fetch_log() {
      const after = accum.current.total
      const params = new URLSearchParams()
      if (session) params.set('session', session)
      if (after > 0) params.set('after', String(after))
      const qs = params.toString() ? `?${params}` : ''
      fetch(`/api/train-log${qs}`)
        .then(r => r.json())
        .then(d => {
          const newEps = d.episodes || []
          const serverTotal = d.total ?? (after + newEps.length)
          if (newEps.length === 0 && serverTotal === accum.current.total) return
          if (after > 0) {
            accum.current.episodes = accum.current.episodes.concat(newEps)
          } else {
            accum.current.episodes = newEps
            accum.current.config = d.config || {}
            setConfig(accum.current.config)
          }
          accum.current.total = serverTotal
          setEpisodes(accum.current.episodes)
        })
        .catch(() => {})
        .finally(() => setLoading(false))
    }
    fetch_log()
    timer.current = setInterval(fetch_log, POLL_MS)
    return () => clearInterval(timer.current)
  }, [session, setEpisodes, setConfig, setLoading])
}

function useLeague(session) {
  const setModels = useSetAtom(trainLeagueModelsAtom)
  const timer = useRef(null)
  const lastHash = useRef('')

  useEffect(() => {
    lastHash.current = ''
    function fetch_league() {
      const qs = session ? `?session=${encodeURIComponent(session)}` : ''
      fetch(`/api/league${qs}`)
        .then(r => r.text())
        .then(text => {
          if (text !== lastHash.current) {
            lastHash.current = text
            try { setModels(JSON.parse(text).models || []) } catch {}
          }
        })
        .catch(() => {})
    }
    fetch_league()
    timer.current = setInterval(fetch_league, POLL_MS)
    return () => clearInterval(timer.current)
  }, [session, setModels])
}

function useTrainReplays(session) {
  const [replays, setReplays] = useState([])
  const timer = useRef(null)
  const lastLen = useRef(0)

  useEffect(() => {
    lastLen.current = 0
    function refresh() {
      const qs = session ? `?session=${encodeURIComponent(session)}` : ''
      fetch(`/api/train-replays${qs}`)
        .then(r => r.json())
        .then(d => {
          if (d.length !== lastLen.current) {
            lastLen.current = d.length
            setReplays(d)
          }
        })
        .catch(() => {})
    }
    refresh()
    timer.current = setInterval(refresh, POLL_MS)
    return () => clearInterval(timer.current)
  }, [session])

  return replays
}

function useGameLog(session) {
  const setGames = useSetAtom(trainGameLogAtom)
  const timer = useRef(null)
  const accum = useRef({ map: {}, total: 0 })

  useEffect(() => {
    accum.current = { map: {}, total: 0 }
    function fetch_games() {
      const after = accum.current.total
      const params = new URLSearchParams()
      if (session) params.set('session', session)
      if (after > 0) params.set('after', String(after))
      const qs = params.toString() ? `?${params}` : ''
      fetch(`/api/game-log${qs}`)
        .then(r => r.json())
        .then(d => {
          const newGames = d.games || d
          const serverTotal = d.total ?? (after + newGames.length)
          if (newGames.length === 0 && serverTotal === accum.current.total) return
          for (const g of newGames) if (g.replay) accum.current.map[g.replay] = g
          accum.current.total = serverTotal
          setGames({ ...accum.current.map })
        })
        .catch(() => {})
    }
    fetch_games()
    timer.current = setInterval(fetch_games, POLL_MS)
    return () => clearInterval(timer.current)
  }, [session, setGames])
}

// ---- Chart helpers ----

const MAX_CHART_POINTS = 200

function downsample(arr, max = MAX_CHART_POINTS) {
  if (arr.length <= max) return arr
  // Use fixed-width buckets so existing buckets stay stable as data grows.
  // Bucket width is ceil(length/max) — only the last bucket changes size.
  const bucketSize = Math.ceil(arr.length / max)
  const result = []
  for (let i = 0; i < arr.length; i += bucketSize) {
    const end = Math.min(i + bucketSize, arr.length)
    // Average numeric fields, keep structure from the midpoint element
    const mid = arr[Math.floor((i + end - 1) / 2)]
    const bucket = arr.slice(i, end)
    const averaged = { ...mid }
    for (const key of Object.keys(mid)) {
      if (typeof mid[key] === 'number') {
        let sum = 0, count = 0
        for (let j = 0; j < bucket.length; j++) {
          const v = bucket[j][key]
          if (v != null) { sum += v; count++ }
        }
        averaged[key] = count > 0 ? sum / count : mid[key]
      }
    }
    // Preserve episode number from the last element in the bucket for x-axis
    averaged.episode = bucket[bucket.length - 1].episode
    result.push(averaged)
  }
  return result
}

function fmt(v, decimals = 2) {
  if (v == null) return '\u2014'
  return typeof v === 'number' ? v.toFixed(decimals) : String(v)
}

function ema(data, span = 10) {
  if (!data.length) return []
  const alpha = 2 / (span + 1)
  const result = [data[0]]
  for (let i = 1; i < data.length; i++) {
    const v = data[i]
    result.push(v != null ? alpha * v + (1 - alpha) * result[i - 1] : result[i - 1])
  }
  return result
}

function percentileValue(data, p) {
  const values = data.filter(v => v != null).sort((a, b) => a - b)
  if (!values.length) return null
  const pos = Math.max(0, Math.min(values.length - 1, (values.length - 1) * p))
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  const frac = pos - lo
  return values[lo] + (values[hi] - values[lo]) * frac
}

function percentileViewport(data, low = 0.05, high = 0.95) {
  const min = percentileValue(data, low)
  const max = percentileValue(data, high)
  if (min == null || max == null) return null
  if (min === max) {
    const pad = Math.abs(min || 1) * 0.05 || 1
    return { min: min - pad, max: max + pad }
  }
  const pad = (max - min) * 0.05
  return { min: min - pad, max: max + pad }
}

function chartScales() {
  return {
    x: { ticks: { color: CHART_THEME.subtext, maxTicksLimit: 10 }, grid: { color: CHART_THEME.grid } },
    y: { ticks: { color: CHART_THEME.subtext }, grid: { color: CHART_THEME.grid } },
  }
}

function chartBase(yViewport = null) {
  const scales = chartScales()
  if (yViewport) {
    scales.y = { ...scales.y, min: yViewport.min, max: yViewport.max }
  }
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    scales,
    plugins: { legend: { display: false } },
  }
}

// ---- Card content components ----

function StatusPanel() {
  const episodes = useAtomValue(trainEpisodesAtom)
  const config = useAtomValue(trainConfigAtom)
  const loading = useAtomValue(trainLoadingAtom)
  const latest = episodes[episodes.length - 1]
  const progress = config.total_episodes
    ? `${episodes.length} / ${config.total_episodes}`
    : `${episodes.length}`

  if (loading && !episodes.length) {
    return <p className="text-[13px] text-overlay0 py-4">Loading training data...</p>
  }
  if (!episodes.length) {
    return (
      <p className="text-[13px] text-overlay0 py-4">
        No training log found. Run <code className="text-subtext0">python bots/tensor/torch/train_gpu.py</code> to start.
      </p>
    )
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
      <Stat label="Episodes" value={progress} />
      <Stat label="Last Reward / Unit" value={fmt(latest?.reward, 3)} />
      <Stat label="Last Resources" value={fmt(latest?.resources, 1)} />
      <Stat label="Expl. Variance" value={fmt(latest?.explained_var, 3)} />
      <Stat label="Entropy" value={fmt(latest?.entropy, 3)} />
      <Stat label="Approx KL" value={fmt(latest?.approx_kl, 5)} />
      <Stat label="Elapsed" value={latest ? `${latest.elapsed}s` : '\u2014'} />
      {config.lr && <Stat label="LR" value={config.lr} />}
    </div>
  )
}

const RewardChart = memo(function RewardChart() {
  const episodes = useAtomValue(trainEpisodesAtom)
  const { ema: emaOn, bands: bandsOn } = useCardSettings()
  if (episodes.length <= 1) return <NoData />
  const ds = downsample(episodes)
  const raw = ds.map(e => e.reward)
  const yViewport = bandsOn ? percentileViewport(raw) : null
  const datasets = [{
    label: 'Reward', data: raw,
    borderColor: emaOn ? '#a6e3a140' : '#a6e3a1', backgroundColor: '#a6e3a120',
    tension: 0, pointRadius: 0, borderWidth: emaOn ? 1 : 1.5, spanGaps: true,
  }]
  if (emaOn) datasets.push({
    label: 'EMA', data: ema(raw),
    borderColor: '#a6e3a1', borderWidth: 2, pointRadius: 0, tension: 0.3, spanGaps: true,
  })
  return (
    <div className="min-h-[200px]">
      <Line data={{ labels: ds.map(e => e.episode), datasets }} options={chartBase(yViewport)} />
    </div>
  )
})

const RewardBreakdownChart = memo(function RewardBreakdownChart() {
  const episodes = useAtomValue(trainEpisodesAtom)
  const { ema: emaOn, bands: bandsOn } = useCardSettings()
  if (episodes.length <= 1) return <NoData />
  const ds = downsample(episodes)
  const dim = emaOn
  const series = [
    { key: 'r_harv', label: 'Harvester', color: '#a6e3a1' },
    { key: 'r_flow', altKey: 'r_conv', label: 'Flow', color: '#89b4fa' },
    { key: 'r_work', altKey: 'r_deliv', label: 'Working', color: '#f9e2af' },
    { key: 'r_dmg', label: 'Damage', color: '#f38ba8' },
    { key: 'r_own_dmg', label: 'Own Core Dmg', color: '#cba6f7' },
  ]
  const datasets = series.flatMap(({ key, altKey, label, color }) => {
    const raw = ds.map(e => e[key] ?? e[altKey] ?? 0)
    const result = [{
      label, data: raw, borderColor: dim ? color + '40' : color,
      tension: 0, pointRadius: 0, borderWidth: dim ? 1 : 1.5,
    }]
    if (emaOn) result.push({
      label: `${label} EMA`, data: ema(raw), borderColor: color,
      borderWidth: 2, pointRadius: 0, tension: 0.3,
    })
    return result
  })
  const yViewport = bandsOn
    ? percentileViewport(series.flatMap(({ key, altKey }) => ds.map(e => e[key] ?? e[altKey] ?? 0)))
    : null
  return (
    <div className="min-h-[200px]">
      <Line
        data={{ labels: ds.map(e => e.episode), datasets }}
        options={{
          ...chartBase(yViewport),
          plugins: { legend: { labels: { color: CHART_THEME.text, font: { size: 10 } } } },
        }}
      />
    </div>
  )
})

const HarvesterDebugChart = memo(function HarvesterDebugChart() {
  const episodes = useAtomValue(trainEpisodesAtom)
  const { ema: emaOn, bands: bandsOn } = useCardSettings()
  if (episodes.length <= 1) return <NoData />
  const ds = downsample(episodes)
  const dim = emaOn
  const series = [
    { key: 'harv_attempts', label: 'Attempts', color: '#89b4fa' },
    { key: 'harv_failures', label: 'Failures', color: '#f38ba8' },
    { key: 'harv_prox_gate', label: 'Gate', color: '#f9e2af' },
    { key: 'harv_prox_paid', label: 'Paid', color: '#a6e3a1' },
  ]
  const datasets = series.flatMap(({ key, label, color }) => {
    const raw = ds.map(e => e[key] ?? 0)
    const result = [{
      label, data: raw, borderColor: dim ? color + '40' : color,
      tension: 0, pointRadius: 0, borderWidth: dim ? 1 : 1.5,
    }]
    if (emaOn) result.push({
      label: `${label} EMA`, data: ema(raw), borderColor: color,
      borderWidth: 2, pointRadius: 0, tension: 0.3,
    })
    return result
  })
  const yViewport = bandsOn
    ? percentileViewport(series.flatMap(({ key }) => ds.map(e => e[key] ?? 0)))
    : null
  return (
    <div className="min-h-[200px]">
      <Line
        data={{ labels: ds.map(e => e.episode), datasets }}
        options={{
          ...chartBase(yViewport),
          plugins: { legend: { labels: { color: CHART_THEME.text, font: { size: 10 } } } },
        }}
      />
    </div>
  )
})

const ConveyorDebugChart = memo(function ConveyorDebugChart() {
  const episodes = useAtomValue(trainEpisodesAtom)
  const { ema: emaOn, bands: bandsOn } = useCardSettings()
  if (episodes.length <= 1) return <NoData />
  const ds = downsample(episodes)
  const dim = emaOn
  const series = [
    { key: 'conv_attempts', label: 'Attempts', color: '#89b4fa' },
    { key: 'conv_failures', label: 'Failures', color: '#f38ba8' },
    { key: 'conv_prox_gate', label: 'Gate', color: '#f9e2af' },
    { key: 'conv_prox_paid', label: 'Paid', color: '#a6e3a1' },
  ]
  const datasets = series.flatMap(({ key, label, color }) => {
    const raw = ds.map(e => e[key] ?? 0)
    const result = [{
      label, data: raw, borderColor: dim ? color + '40' : color,
      tension: 0, pointRadius: 0, borderWidth: dim ? 1 : 1.5,
    }]
    if (emaOn) result.push({
      label: `${label} EMA`, data: ema(raw), borderColor: color,
      borderWidth: 2, pointRadius: 0, tension: 0.3,
    })
    return result
  })
  const yViewport = bandsOn
    ? percentileViewport(series.flatMap(({ key }) => ds.map(e => e[key] ?? 0)))
    : null
  return (
    <div className="min-h-[200px]">
      <Line
        data={{ labels: ds.map(e => e.episode), datasets }}
        options={{
          ...chartBase(yViewport),
          plugins: { legend: { labels: { color: CHART_THEME.text, font: { size: 10 } } } },
        }}
      />
    </div>
  )
})

function makeSimpleChart(label, dataKey, color, opts = {}) {
  return memo(function SimpleChart() {
    "use no memo"
    const episodes = useAtomValue(trainEpisodesAtom)
    const { ema: emaOn, bands: bandsOn } = useCardSettings()
    if (episodes.length <= 1) return <NoData />
    const ds = downsample(episodes)
    const raw = ds.map(e => e[dataKey] ?? e[opts.altKey] ?? null)
    const yViewport = bandsOn ? percentileViewport(raw) : null
    const dim = emaOn
    const datasets = [{
      label, data: raw,
      borderColor: dim ? color + '40' : color,
      backgroundColor: opts.fill || color + '20',
      fill: !!opts.fill,
      tension: 0, pointRadius: 0, borderWidth: dim ? 1 : 1.5, spanGaps: true,
    }]
    if (emaOn) datasets.push({
      label: `${label} EMA`, data: ema(raw),
      borderColor: color, borderWidth: 2, pointRadius: 0, tension: 0.3, spanGaps: true,
    })
    return (
      <div className="min-h-[200px]">
        <Line data={{ labels: ds.map(e => e.episode), datasets }} options={chartBase(yViewport)} />
      </div>
    )
  })
}

const ResourcesChart = makeSimpleChart('Resources', 'resources', '#89b4fa')
const WinRateChart = makeSimpleChart('Win Rate', 'winrate', '#cba6f7')
const EntropyChart = makeSimpleChart('Entropy', 'entropy', '#cba6f7', { fill: '#cba6f720' })
const ExplVarChart = makeSimpleChart('Explained Var', 'explained_var', '#a6e3a1', { fill: '#a6e3a120' })
const KLChart = makeSimpleChart('KL', 'approx_kl', '#fab387')
const ClipChart = makeSimpleChart('Clip Frac', 'clip_frac', '#f9e2af')

const LossChart = memo(function LossChart() {
  const episodes = useAtomValue(trainEpisodesAtom)
  const { ema: emaOn, bands: bandsOn } = useCardSettings()
  if (episodes.length <= 1) return <NoData />
  const dim = emaOn
  const ds = downsample(episodes)
  const series = [
    { key: 'pi_loss', label: 'Policy Loss', color: '#f38ba8' },
    { key: 'vf_loss', label: 'Value Loss', color: '#f9e2af' },
  ]
  const datasets = series.flatMap(({ key, label, color }) => {
    const raw = ds.map(e => e[key])
    const result = [{
      label, data: raw, borderColor: dim ? color + '40' : color,
      tension: 0, pointRadius: 0, borderWidth: dim ? 1 : 1.5,
    }]
    if (emaOn) result.push({
      label: `${label} EMA`, data: ema(raw), borderColor: color,
      borderWidth: 2, pointRadius: 0, tension: 0.3,
    })
    return result
  })
  const yViewport = bandsOn
    ? percentileViewport(series.flatMap(({ key }) => ds.map(e => e[key])))
    : null
  return (
    <div className="min-h-[200px]">
      <Line
        data={{ labels: ds.map(e => e.episode), datasets }}
        options={{
          ...chartBase(yViewport),
          plugins: { legend: { labels: { color: CHART_THEME.text, font: { size: 10 } } } },
        }}
      />
    </div>
  )
})

const LeagueEloCard = memo(function LeagueEloCard() {
  const leagueModels = useAtomValue(trainLeagueModelsAtom)
  if (!leagueModels.length) return <NoData />
  const sorted = [...leagueModels].sort((a, b) => a.gen - b.gen)
  const labels = sorted.map(m => m.id)
  const elos = sorted.map(m => m.elo)
  const colors = sorted.map((_, i) => i === sorted.length - 1 ? '#a6e3a1' : '#89b4fa')

  return (
    <div className="min-h-[200px]">
      <Bar
        data={{
          labels,
          datasets: [{
            label: 'ELO', data: elos,
            backgroundColor: colors, borderColor: colors, borderWidth: 1,
          }],
        }}
        options={{
          responsive: true, maintainAspectRatio: false,
          scales: chartScales(),
          plugins: { legend: { display: false } },
        }}
      />
    </div>
  )
})

function EpisodeLogTable() {
  const episodes = useAtomValue(trainEpisodesAtom)
  if (!episodes.length) return <NoData />
  return (
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
              <Td title={`Harv: ${fmt(e.r_harv, 2)}  Work: ${fmt(e.r_work ?? e.r_deliv, 2)}  Flow: ${fmt(e.r_flow ?? e.r_conv, 2)}  Dmg: ${fmt(e.r_dmg, 2)}  OwnDmg: ${fmt(e.r_own_dmg, 2)}  |  H a/f/g/p: ${fmt(e.harv_attempts, 2)}/${fmt(e.harv_failures, 2)}/${fmt(e.harv_prox_gate, 2)}/${fmt(e.harv_prox_paid, 2)}  C a/f/g/p: ${fmt(e.conv_attempts, 2)}/${fmt(e.conv_failures, 2)}/${fmt(e.conv_prox_gate, 2)}/${fmt(e.conv_prox_paid, 2)}`}>{fmt(e.reward, 3)}</Td>
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
  )
}

function TrainingReplaysContent() {
  const session = useAtomValue(trainSessionAtom)
  const gameLog = useAtomValue(trainGameLogAtom)
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

  if (replays.length === 0) {
    return (
      <p className="text-[13px] text-overlay0 py-2">
        No training replays yet. Run <code className="text-subtext0">train_engine.py</code> to generate.
      </p>
    )
  }

  return (
    <div className="max-h-64 overflow-y-auto">
      <Table>
        <Thead>
          <Th>Replay</Th>
          <Th>Result</Th>
          <Th>Reward</Th>
          <Th>Resources</Th>
          <Th>Date</Th>
        </Thead>
        <tbody>
          {replays.map(r => (
            <Tr key={r.name} className="cursor-pointer" onClick={() => openReplay(r.name)}>
              <Td>{parseReplayName(r.name)}</Td>
              <Td>{(() => {
                const g = gameLog[r.name]
                if (!g) return '\u2014'
                return g.winner === 'a' ? <span className="text-green">W</span>
                  : g.winner === 'b' ? <span className="text-red">L</span>
                  : <span className="text-overlay0">-</span>
              })()}</Td>
              <Td>{(() => { const g = gameLog[r.name]; return g ? fmt(g.reward, 1) : '\u2014' })()}</Td>
              <Td>{(() => { const g = gameLog[r.name]; return g ? fmt(g.resources, 1) : '\u2014' })()}</Td>
              <Td>{relativeTime(r.mtime)}</Td>
            </Tr>
          ))}
        </tbody>
      </Table>
    </div>
  )
}

// ---- Small helpers ----

function parseReplayName(name) {
  const m = name.match(/^ep(\d+)_game(\d+)\.replay26$/)
  if (m) return `Episode ${m[1]}, Game ${m[2]}`
  return name.replace('.replay26', '')
}



function Stat({ label, value }) {
  return (
    <div>
      <div className="text-[11px] text-overlay0">{label}</div>
      <div className="text-[15px] font-semibold text-text mt-0.5">{value}</div>
    </div>
  )
}

function NoData() {
  return <p className="text-[12px] text-overlay0/50 py-3">Waiting for data...</p>
}

// ---- Card registry (module-level, stable reference) ----

const TRAINING_CARDS = {
  'train-status':           { title: 'Training Status',     render: () => <StatusPanel /> },
  'train-ep-reward':        { title: 'Reward / Episode',    subtitle: 'per unit', hasEma: true, render: () => <RewardChart /> },
  'train-reward-breakdown': { title: 'Reward Breakdown',    subtitle: 'per unit', hasEma: true, render: () => <RewardBreakdownChart /> },
  'train-harv-debug':       { title: 'Harvester Debug',     subtitle: 'per unit', hasEma: true, render: () => <HarvesterDebugChart /> },
  'train-conv-debug':       { title: 'Conveyor Debug',      subtitle: 'per unit', hasEma: true, render: () => <ConveyorDebugChart /> },
  'train-ep-res':           { title: 'Resources / Episode', hasEma: true, render: () => <ResourcesChart /> },
  'train-winrate':          { title: 'Win Rate',            hasEma: true, render: () => <WinRateChart /> },
  'train-elo':              { title: 'League ELO',          render: () => <LeagueEloCard /> },
  'train-loss-chart':       { title: 'Loss',                defaultOpen: false, hasEma: true, render: () => <LossChart /> },
  'train-entropy':          { title: 'Entropy',             defaultOpen: false, hasEma: true, render: () => <EntropyChart /> },
  'train-ev':               { title: 'Explained Variance',  hasEma: true, render: () => <ExplVarChart /> },
  'train-kl':               { title: 'Approx KL',           defaultOpen: false, hasEma: true, render: () => <KLChart /> },
  'train-clip':             { title: 'Clip Fraction',       defaultOpen: false, hasEma: true, render: () => <ClipChart /> },
  'train-log-table':        { title: 'Episode Log',         defaultOpen: false, render: () => <EpisodeLogTable /> },
  'train-replays':          { title: 'Training Replays',    render: () => <TrainingReplaysContent /> },
}

const DEFAULT_LAYOUT = {
  nextId: 1,
  root: { type: 'leaf', id: 'v0', cards: Object.keys(TRAINING_CARDS) },
}

// ---- Main export ----

export default function TrainingView() {
  const session = useAtomValue(trainSessionAtom)
  useTrainLog(session)
  useLeague(session)
  useGameLog(session)

  return (
    <TileLayout
      cards={TRAINING_CARDS}
      storageKey="training-tiles"
      defaultLayout={DEFAULT_LAYOUT}
    />
  )
}
