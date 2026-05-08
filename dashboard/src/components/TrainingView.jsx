import { createContext, memo, useContext, useMemo, useState } from 'react'
import { useAtomValue, useSetAtom } from 'jotai'
import { viewerAtom, viewerMinimizedAtom, trainSessionAtom } from '../store'
import { Line, Bar } from 'react-chartjs-2'
import { viewerUrl } from '../api'
import {
  useGameLogQuery,
  useGameLogRangedQuery,
  useGameLogEpisodesQuery,
  useGameLogForEpisodeQuery,
  useLeagueQuery,
  useTrainLogQuery,
  useTrainReplaysQuery,
} from '../hooks/useDashboardQueries'
import '../chartjs'

import { Table, Th, Td, Thead, Tr } from './Table'
import { CHART_THEME, relativeTime } from '../data'
import TileLayout from './TileLayout'
import { useCardSettings } from './CardSettingsContext'

const TrainEpisodesCtx = createContext([])
const EMPTY_LABELS = []
const DownsampledCtx = createContext({ ds: EMPTY_LABELS, labels: EMPTY_LABELS })
const TrainConfigCtx = createContext({})
const TrainLoadingCtx = createContext(true)
const TrainLeagueCtx = createContext([])
const TrainGameLogCtx = createContext({})
const EMPTY_ARRAY = []
const EMPTY_OBJECT = {}

function useTrainEpisodes() {
  return useContext(TrainEpisodesCtx)
}

function useTrainConfig() {
  return useContext(TrainConfigCtx)
}

function useTrainLoading() {
  return useContext(TrainLoadingCtx)
}

function useTrainLeagueModels() {
  return useContext(TrainLeagueCtx)
}

function useTrainGameLog() {
  return useContext(TrainGameLogCtx)
}

function useDownsampled() {
  return useContext(DownsampledCtx).ds
}

function useDownsampledLabels() {
  return useContext(DownsampledCtx).labels
}

function TrainingDataProvider({ session, children }) {
  const trainLogQuery = useTrainLogQuery(session)
  const leagueQuery = useLeagueQuery(session)
  const gameLogQuery = useGameLogQuery(session)

  const episodes = trainLogQuery.data?.episodes ?? EMPTY_ARRAY
  const config = trainLogQuery.data?.config ?? EMPTY_OBJECT
  const loading = trainLogQuery.isPending && !episodes.length
  const leagueModels = leagueQuery.data ?? EMPTY_ARRAY
  const gameLog = gameLogQuery.data?.map ?? EMPTY_OBJECT

  return (
    <TrainEpisodesCtx.Provider value={episodes}>
      <TrainConfigCtx.Provider value={config}>
        <TrainLoadingCtx.Provider value={loading}>
          <TrainLeagueCtx.Provider value={leagueModels}>
            <TrainGameLogCtx.Provider value={gameLog}>
              {children}
            </TrainGameLogCtx.Provider>
          </TrainLeagueCtx.Provider>
        </TrainLoadingCtx.Provider>
      </TrainConfigCtx.Provider>
    </TrainEpisodesCtx.Provider>
  )
}

// ---- Chart helpers ----

const MAX_CHART_POINTS = 200

function downsample(arr, max = MAX_CHART_POINTS) {
  if (arr.length <= max) return arr
  const bucketSize = Math.ceil(arr.length / max)
  const result = []
  for (let i = 0; i < arr.length; i += bucketSize) {
    const end = Math.min(i + bucketSize, arr.length)
    const mid = arr[Math.floor((i + end - 1) / 2)]
    const averaged = { ...mid }
    for (const key of Object.keys(mid)) {
      if (typeof mid[key] === 'number') {
        let sum = 0, count = 0
        for (let j = i; j < end; j++) {
          const v = arr[j][key]
          if (v != null) { sum += v; count++ }
        }
        averaged[key] = count > 0 ? sum / count : mid[key]
      }
    }
    averaged.episode = arr[end - 1].episode
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

function percentileValueMulti(arrays, p) {
  let total = 0
  for (const a of arrays) for (let i = 0; i < a.length; i++) if (a[i] != null) total++
  if (!total) return null
  const flat = new Float64Array(total)
  let k = 0
  for (const a of arrays) for (let i = 0; i < a.length; i++) {
    const v = a[i]
    if (v != null) flat[k++] = v
  }
  flat.sort()
  const pos = Math.max(0, Math.min(total - 1, (total - 1) * p))
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  const frac = pos - lo
  return flat[lo] + (flat[hi] - flat[lo]) * frac
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

function percentileViewportMulti(arrays, low = 0.05, high = 0.95) {
  const min = percentileValueMulti(arrays, low)
  const max = percentileValueMulti(arrays, high)
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
    animation: false,
    interaction: { mode: 'index', intersect: false },
    scales,
    plugins: { legend: { display: false } },
  }
}

function DownsampledProvider({ children }) {
  const episodes = useRangedEpisodes()
  const value = useMemo(() => {
    const ds = downsample(episodes)
    const labels = ds.map(e => e.episode)
    return { ds, labels }
  }, [episodes])
  return <DownsampledCtx.Provider value={value}>{children}</DownsampledCtx.Provider>
}

// ---- Card content components ----

function StatusPanel() {
  const episodes = useTrainEpisodes()
  const config = useTrainConfig()
  const loading = useTrainLoading()
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
  const ds = useDownsampled()
  const labels = useDownsampledLabels()
  const { ema: emaOn, bands: bandsOn } = useCardSettings()
  const raw = useMemo(() => ds.map(e => e.reward), [ds])
  const yViewport = useMemo(() => (bandsOn ? percentileViewport(raw) : null), [raw, bandsOn])
  const options = useMemo(() => chartBase(yViewport), [yViewport])
  if (ds.length <= 1) return <NoData />
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
      <Line data={{ labels, datasets }} options={options} />
    </div>
  )
})

const REWARD_BREAKDOWN_SERIES = [
  { key: 'r_flow', altKey: 'r_work', label: 'Flow', color: '#f9e2af' },
  { key: 'r_delivery', label: 'Delivery', color: '#a6e3a1' },
  { key: 'r_dmg', label: 'Damage', color: '#f38ba8' },
  { key: 'r_own_dmg', label: 'Own Core Dmg', color: '#cba6f7' },
]

const RewardBreakdownChart = memo(function RewardBreakdownChart() {
  const ds = useDownsampled()
  const labels = useDownsampledLabels()
  const { ema: emaOn, bands: bandsOn } = useCardSettings()
  const raws = useMemo(
    () => REWARD_BREAKDOWN_SERIES.map(({ key, altKey }) => ds.map(e => e[key] ?? e[altKey] ?? 0)),
    [ds],
  )
  const yViewport = useMemo(
    () => bandsOn ? percentileViewportMulti(raws) : null,
    [raws, bandsOn],
  )
  const options = useMemo(
    () => ({
      ...chartBase(yViewport),
      plugins: { legend: { labels: { color: CHART_THEME.text, font: { size: 10 } } } },
    }),
    [yViewport],
  )
  if (ds.length <= 1) return <NoData />
  const dim = emaOn
  const datasets = []
  for (let i = 0; i < REWARD_BREAKDOWN_SERIES.length; i++) {
    const { label, color } = REWARD_BREAKDOWN_SERIES[i]
    const raw = raws[i]
    datasets.push({
      label, data: raw, borderColor: dim ? color + '40' : color,
      tension: 0, pointRadius: 0, borderWidth: dim ? 1 : 1.5,
    })
    if (emaOn) datasets.push({
      label: `${label} EMA`, data: ema(raw), borderColor: color,
      borderWidth: 2, pointRadius: 0, tension: 0.3,
    })
  }
  return (
    <div className="min-h-[200px]">
      <Line data={{ labels, datasets }} options={options} />
    </div>
  )
})


const HARV_DEBUG_SERIES = [
  { key: 'harv_attempts', label: 'Attempts', color: '#89b4fa' },
  { key: 'harv_failures', label: 'Failures', color: '#f38ba8' },
  { key: 'harv_prox_gate', label: 'Gate', color: '#f9e2af' },
  { key: 'harv_prox_paid', label: 'Paid', color: '#a6e3a1' },
]

const CONV_DEBUG_SERIES = [
  { key: 'conv_attempts', label: 'Attempts', color: '#89b4fa' },
  { key: 'conv_failures', label: 'Failures', color: '#f38ba8' },
  { key: 'conv_prox_gate', label: 'Gate', color: '#f9e2af' },
  { key: 'conv_prox_paid', label: 'Paid', color: '#a6e3a1' },
]

const HarvesterDebugChart = memo(function HarvesterDebugChart() {
  const ds = useDownsampled()
  const labels = useDownsampledLabels()
  const { ema: emaOn, bands: bandsOn } = useCardSettings()
  const raws = useMemo(
    () => HARV_DEBUG_SERIES.map(({ key }) => ds.map(e => e[key] ?? 0)),
    [ds],
  )
  const yViewport = useMemo(
    () => bandsOn ? percentileViewportMulti(raws) : null,
    [raws, bandsOn],
  )
  const options = useMemo(
    () => ({
      ...chartBase(yViewport),
      plugins: { legend: { labels: { color: CHART_THEME.text, font: { size: 10 } } } },
    }),
    [yViewport],
  )
  if (ds.length <= 1) return <NoData />
  const dim = emaOn
  const datasets = []
  for (let i = 0; i < HARV_DEBUG_SERIES.length; i++) {
    const { label, color } = HARV_DEBUG_SERIES[i]
    const raw = raws[i]
    datasets.push({
      label, data: raw, borderColor: dim ? color + '40' : color,
      tension: 0, pointRadius: 0, borderWidth: dim ? 1 : 1.5,
    })
    if (emaOn) datasets.push({
      label: `${label} EMA`, data: ema(raw), borderColor: color,
      borderWidth: 2, pointRadius: 0, tension: 0.3,
    })
  }
  return (
    <div className="min-h-[200px]">
      <Line data={{ labels, datasets }} options={options} />
    </div>
  )
})

const ConveyorDebugChart = memo(function ConveyorDebugChart() {
  const ds = useDownsampled()
  const labels = useDownsampledLabels()
  const { ema: emaOn, bands: bandsOn } = useCardSettings()
  const raws = useMemo(
    () => CONV_DEBUG_SERIES.map(({ key }) => ds.map(e => e[key] ?? 0)),
    [ds],
  )
  const yViewport = useMemo(
    () => bandsOn ? percentileViewportMulti(raws) : null,
    [raws, bandsOn],
  )
  const options = useMemo(
    () => ({
      ...chartBase(yViewport),
      plugins: { legend: { labels: { color: CHART_THEME.text, font: { size: 10 } } } },
    }),
    [yViewport],
  )
  if (ds.length <= 1) return <NoData />
  const dim = emaOn
  const datasets = []
  for (let i = 0; i < CONV_DEBUG_SERIES.length; i++) {
    const { label, color } = CONV_DEBUG_SERIES[i]
    const raw = raws[i]
    datasets.push({
      label, data: raw, borderColor: dim ? color + '40' : color,
      tension: 0, pointRadius: 0, borderWidth: dim ? 1 : 1.5,
    })
    if (emaOn) datasets.push({
      label: `${label} EMA`, data: ema(raw), borderColor: color,
      borderWidth: 2, pointRadius: 0, tension: 0.3,
    })
  }
  return (
    <div className="min-h-[200px]">
      <Line data={{ labels, datasets }} options={options} />
    </div>
  )
})

function makeSimpleChart(label, dataKey, color, opts = {}) {
  const altKey = opts.altKey
  const fill = opts.fill
  return memo(function SimpleChart() {
    "use no memo"
    const ds = useDownsampled()
    const labels = useDownsampledLabels()
    const { ema: emaOn, bands: bandsOn } = useCardSettings()
    const raw = useMemo(
      () => ds.map(e => e[dataKey] ?? (altKey ? e[altKey] : null) ?? null),
      [ds],
    )
    const yViewport = useMemo(() => (bandsOn ? percentileViewport(raw) : null), [raw, bandsOn])
    const options = useMemo(() => chartBase(yViewport), [yViewport])
    if (ds.length <= 1) return <NoData />
    const dim = emaOn
    const datasets = [{
      label, data: raw,
      borderColor: dim ? color + '40' : color,
      backgroundColor: fill || color + '20',
      fill: !!fill,
      tension: 0, pointRadius: 0, borderWidth: dim ? 1 : 1.5, spanGaps: true,
    }]
    if (emaOn) datasets.push({
      label: `${label} EMA`, data: ema(raw),
      borderColor: color, borderWidth: 2, pointRadius: 0, tension: 0.3, spanGaps: true,
    })
    return (
      <div className="min-h-[200px]">
        <Line data={{ labels, datasets }} options={options} />
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
const ActorGradNormChart = makeSimpleChart('Actor Grad Norm', 'actor_grad_norm', '#94e2d5')
const CriticGradNormChart = makeSimpleChart('Critic Grad Norm', 'critic_grad_norm', '#74c7ec')
const CriticValueDeltaChart = makeSimpleChart('Critic Value Delta', 'critic_value_delta', '#f2cdcd')
const RetOobFracChart = makeSimpleChart('Return OOB Frac', 'ret_oob_frac', '#f38ba8')
const ActorLrChart = makeSimpleChart('Actor LR', 'actor_lr', '#89dceb')

const LogitScaleChart = memo(function LogitScaleChart({ series }) {
  const ds = useDownsampled()
  const labels = useDownsampledLabels()
  const { ema: emaOn, bands: bandsOn } = useCardSettings()
  const raws = useMemo(
    () => series.map(({ key }) => ds.map(e => e[key])),
    [ds, series],
  )
  const yViewport = useMemo(
    () => (bandsOn ? percentileViewportMulti(raws) : null),
    [raws, bandsOn],
  )
  const options = useMemo(
    () => ({
      ...chartBase(yViewport),
      plugins: { legend: { labels: { color: CHART_THEME.text, font: { size: 10 } } } },
    }),
    [yViewport],
  )
  if (ds.length <= 1) return <NoData />
  const dim = emaOn
  const datasets = []
  for (let i = 0; i < series.length; i++) {
    const { label, color } = series[i]
    const raw = raws[i]
    datasets.push({
      label, data: raw, borderColor: dim ? color + '40' : color,
      tension: 0, pointRadius: 0, borderWidth: dim ? 1 : 1.5, spanGaps: true,
    })
    if (emaOn) datasets.push({
      label: `${label} EMA`, data: ema(raw), borderColor: color,
      borderWidth: 2, pointRadius: 0, tension: 0.3, spanGaps: true,
    })
  }
  return (
    <div className="min-h-[200px]">
      <Line data={{ labels, datasets }} options={options} />
    </div>
  )
})

const TEMP_MEAN_SERIES = [
  { key: 'temp_bt_mean', label: 'Build Type', color: '#f38ba8' },
  { key: 'temp_bd_mean', label: 'Build Dir', color: '#89b4fa' },
  { key: 'temp_cd_mean', label: 'Conv Dir', color: '#f9e2af' },
  { key: 'temp_brt_mean', label: 'Bridge Target', color: '#cba6f7' },
  { key: 'temp_md_mean', label: 'Move Dir', color: '#a6e3a1' },
]

const TEMP_STD_SERIES = [
  { key: 'temp_bt_std', label: 'Build Type', color: '#f38ba8' },
  { key: 'temp_bd_std', label: 'Build Dir', color: '#89b4fa' },
  { key: 'temp_cd_std', label: 'Conv Dir', color: '#f9e2af' },
  { key: 'temp_brt_std', label: 'Bridge Target', color: '#cba6f7' },
  { key: 'temp_md_std', label: 'Move Dir', color: '#a6e3a1' },
]

const TempMeanChart = memo(function TempMeanChart() {
  return <LogitScaleChart series={TEMP_MEAN_SERIES} />
})

const TempStdChart = memo(function TempStdChart() {
  return <LogitScaleChart series={TEMP_STD_SERIES} />
})

const ENTROPY_HEAD_SERIES = [
  { key: 'entropy_bt', label: 'Build Type', color: '#f38ba8' },
  { key: 'entropy_bd', label: 'Build Dir', color: '#89b4fa' },
  { key: 'entropy_cd', label: 'Conv Dir', color: '#f9e2af' },
  { key: 'entropy_brt', label: 'Bridge Target', color: '#cba6f7' },
  { key: 'entropy_md', label: 'Move Dir', color: '#a6e3a1' },
]

const HeadEntropyChart = memo(function HeadEntropyChart() {
  return <LogitScaleChart series={ENTROPY_HEAD_SERIES} />
})

const LOSS_SERIES = [
  { key: 'pi_loss', label: 'Policy Loss', color: '#f38ba8' },
  { key: 'vf_loss', label: 'Value Loss', color: '#f9e2af' },
]

const LossChart = memo(function LossChart() {
  const ds = useDownsampled()
  const labels = useDownsampledLabels()
  const { ema: emaOn, bands: bandsOn } = useCardSettings()
  const raws = useMemo(
    () => LOSS_SERIES.map(({ key }) => ds.map(e => e[key])),
    [ds],
  )
  const yViewport = useMemo(
    () => (bandsOn ? percentileViewportMulti(raws) : null),
    [raws, bandsOn],
  )
  const options = useMemo(
    () => ({
      ...chartBase(yViewport),
      plugins: { legend: { labels: { color: CHART_THEME.text, font: { size: 10 } } } },
    }),
    [yViewport],
  )
  if (ds.length <= 1) return <NoData />
  const dim = emaOn
  const datasets = []
  for (let i = 0; i < LOSS_SERIES.length; i++) {
    const { label, color } = LOSS_SERIES[i]
    const raw = raws[i]
    datasets.push({
      label, data: raw, borderColor: dim ? color + '40' : color,
      tension: 0, pointRadius: 0, borderWidth: dim ? 1 : 1.5,
    })
    if (emaOn) datasets.push({
      label: `${label} EMA`, data: ema(raw), borderColor: color,
      borderWidth: 2, pointRadius: 0, tension: 0.3,
    })
  }
  return (
    <div className="min-h-[200px]">
      <Line data={{ labels, datasets }} options={options} />
    </div>
  )
})

const LeagueEloCard = memo(function LeagueEloCard() {
  const leagueModels = useTrainLeagueModels()
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
  const episodes = useRangedEpisodes()
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
              <Td title={`Flow: ${fmt(e.r_flow ?? e.r_work, 2)}  Delivery: ${fmt(e.r_delivery, 2)}  Dmg: ${fmt(e.r_dmg, 2)}  OwnDmg: ${fmt(e.r_own_dmg, 2)}  |  H a/f/g/p: ${fmt(e.harv_attempts, 2)}/${fmt(e.harv_failures, 2)}/${fmt(e.harv_prox_gate, 2)}/${fmt(e.harv_prox_paid, 2)}  C a/f/g/p: ${fmt(e.conv_attempts, 2)}/${fmt(e.conv_failures, 2)}/${fmt(e.conv_prox_gate, 2)}/${fmt(e.conv_prox_paid, 2)}`}>{fmt(e.reward, 3)}</Td>
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
  const gameLog = useTrainGameLog()
  const [rangeMin, rangeMax] = useEpisodeRange()
  const replays = useTrainReplaysQuery(session, rangeMin, rangeMax).data || []
  const setViewer = useSetAtom(viewerAtom)
  const setMinimized = useSetAtom(viewerMinimizedAtom)

  function openReplay(name) {
    const sessionQs = session ? `?session=${encodeURIComponent(session)}` : ''
    const replayPath = `/api/train-replay/${encodeURIComponent(name)}${sessionQs}`
    setViewer({
      url: viewerUrl(replayPath),
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

// ---- Episode selection for game-log charts ----

const SelectedEpisodeCtx = createContext({ episode: null, setEpisode: () => {}, range: [null, null], setRange: () => {} })

function useSelectedEpisode() {
  return useContext(SelectedEpisodeCtx)
}

function useEpisodeRange() {
  const { range } = useSelectedEpisode()
  return range
}

function useRangedEpisodes() {
  const episodes = useTrainEpisodes()
  const [rangeMin, rangeMax] = useEpisodeRange()
  return useMemo(() => {
    if (rangeMin == null && rangeMax == null) return episodes
    return episodes.filter(e =>
      (rangeMin == null || e.episode >= rangeMin) &&
      (rangeMax == null || e.episode <= rangeMax)
    )
  }, [episodes, rangeMin, rangeMax])
}

const WinConfidenceChart = memo(function WinConfidenceChart() {
  const session = useAtomValue(trainSessionAtom)
  const [rangeMin, rangeMax] = useEpisodeRange()
  const epsData = useGameLogEpisodesQuery(session, rangeMin, rangeMax).data
  const gameLogEps = epsData?.episodes ?? []
  const { episode: selectedEp } = useSelectedEpisode()
  const { legend: legendOn } = useCardSettings()

  const targetEp = selectedEp ?? gameLogEps[gameLogEps.length - 1] ?? null
  const epQuery = useGameLogForEpisodeQuery(session, targetEp)
  const epGames = (epQuery.data?.games ?? []).filter(g => g.win_confidence?.length)
  if (!epGames.length) return <NoData />

  const maxRound = Math.max(...epGames.map(g => (g.round_offset ?? 0) + g.win_confidence.length))
  const labels = Array.from({ length: maxRound }, (_, i) => i)

  const datasets = epGames.map((g) => {
    const known = g.winner === 'a' || g.winner === 'b'
    const target = g.winner === 'a' ? 100 : g.winner === 'b' ? 0 : null
    const tag = g.winner === 'a' ? 'W' : g.winner === 'b' ? 'L' : '-'
    const offset = g.round_offset ?? 0
    const raw = g.win_confidence.map(v => v * 100)
    const data = [...Array(offset).fill(null), ...raw]

    if (!known) {
      return {
        label: `${g.game}`,
        data,
        borderColor: '#a6adc8',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.2,
        spanGaps: false,
      }
    }

    const correct = target > 50 ? v => v > 50 : v => v < 50
    const segColors = data.map(v =>
      v == null ? '#a6adc8' : correct(v) ? '#a6e3a1' : '#f38ba8'
    )
    return {
      label: `${g.game}`,
      data,
      segment: { borderColor: ctx => segColors[ctx.p0DataIndex] || '#a6adc8' },
      borderColor: segColors[0] || '#a6adc8',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.2,
      spanGaps: false,
    }
  })

  return (
    <div className="min-h-[200px]">
      <Line
        data={{ labels, datasets }}
        options={{
          ...chartBase(),
          scales: {
            ...chartScales(),
            y: { ...chartScales().y, min: 0, max: 100, title: { display: true, text: '%', color: CHART_THEME.subtext } },
            x: { ...chartScales().x, title: { display: true, text: 'Round', color: CHART_THEME.subtext } },
          },
          plugins: { legend: { display: legendOn, labels: { color: CHART_THEME.text, font: { size: 10 } } } },
        }}
      />
    </div>
  )
})

const PERCENT_0_100_OPTIONS = (() => {
  const scales = chartScales()
  return {
    ...chartBase(),
    scales: {
      ...scales,
      y: { ...scales.y, min: 0, max: 100, title: { display: true, text: '%', color: CHART_THEME.subtext } },
    },
    plugins: { legend: { labels: { color: CHART_THEME.text, font: { size: 10 } } } },
  }
})()

const WinPredictionAccuracyChart = memo(function WinPredictionAccuracyChart() {
  const session = useAtomValue(trainSessionAtom)
  const { ema: emaOn } = useCardSettings()
  const [rangeMin, rangeMax] = useEpisodeRange()
  const data = useGameLogEpisodesQuery(session, rangeMin, rangeMax).data
  const summaries = data?.summaries ?? []

  const valid = summaries.filter(s => s.accuracy != null)
  if (valid.length <= 1) return <NoData />

  const labels = valid.map(s => s.episode)
  const accRaw = valid.map(s => s.accuracy * 100)
  const confRaw = valid.map(s => (s.mean_confidence ?? 0) * 100)

  const dim = emaOn
  const datasets = [
    {
      label: 'Accuracy',
      data: accRaw,
      borderColor: dim ? '#a6e3a140' : '#a6e3a1',
      borderWidth: dim ? 1 : 1.5,
      pointRadius: 0, tension: 0, spanGaps: true,
    },
    {
      label: 'Mean Confidence',
      data: confRaw,
      borderColor: dim ? '#89b4fa40' : '#89b4fa',
      borderWidth: dim ? 1 : 1.5,
      pointRadius: 0, tension: 0, spanGaps: true,
    },
  ]
  if (emaOn) {
    datasets.push(
      { label: 'Accuracy EMA', data: ema(accRaw), borderColor: '#a6e3a1', borderWidth: 2, pointRadius: 0, tension: 0.3, spanGaps: true },
      { label: 'Confidence EMA', data: ema(confRaw), borderColor: '#89b4fa', borderWidth: 2, pointRadius: 0, tension: 0.3, spanGaps: true },
    )
  }

  return (
    <div className="min-h-[200px]">
      <Line data={{ labels, datasets }} options={PERCENT_0_100_OPTIONS} />
    </div>
  )
})

const ROUND_ACCURACY_OPTIONS = (() => {
  const scales = chartScales()
  return {
    ...chartBase(),
    scales: {
      ...scales,
      y: { ...scales.y, min: 0, max: 100, title: { display: true, text: '%', color: CHART_THEME.subtext } },
      x: { ...scales.x, title: { display: true, text: 'Round', color: CHART_THEME.subtext } },
    },
    plugins: { legend: { display: false } },
  }
})()

const RoundAccuracyChart = memo(function RoundAccuracyChart() {
  const session = useAtomValue(trainSessionAtom)
  const [rangeMin, rangeMax] = useEpisodeRange()
  const queryData = useGameLogRangedQuery(session, rangeMin, rangeMax).data

  const games = useMemo(() => {
    const all = queryData?.games ?? []
    return all.filter(g => g.win_confidence?.length && g.winner != null)
  }, [queryData])

  if (!games.length) return <NoData />

  const roundCorrect = []
  const roundTotal = []
  for (const g of games) {
    const offset = g.round_offset ?? 0
    const wc = g.win_confidence
    for (let i = 0; i < wc.length; i++) {
      const r = offset + i
      while (roundCorrect.length <= r) { roundCorrect.push(0); roundTotal.push(0) }
      roundTotal[r]++
      const pred = wc[i] > 0.5 ? 'a' : 'b'
      if (pred === g.winner) roundCorrect[r]++
    }
  }

  const labels = []
  const accData = []
  for (let r = 0; r < roundTotal.length; r++) {
    if (roundTotal[r] >= 3) {
      labels.push(r)
      accData.push((roundCorrect[r] / roundTotal[r]) * 100)
    }
  }
  if (labels.length < 2) return <NoData />

  const datasets = [
    {
      label: 'Accuracy',
      data: accData,
      borderColor: '#a6e3a1',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
    },
  ]

  return (
    <div className="min-h-[200px]">
      <Line data={{ labels, datasets }} options={ROUND_ACCURACY_OPTIONS} />
    </div>
  )
})

function EpisodeSelector() {
  const session = useAtomValue(trainSessionAtom)
  const episodes = useTrainEpisodes()
  const epsData = useGameLogEpisodesQuery(session).data
  const gameLogEps = epsData?.episodes ?? []
  const { episode, setEpisode, range, setRange } = useSelectedEpisode()
  const [rangeMin, rangeMax] = range

  const hasGameLog = gameLogEps.length > 0
  const current = episode ?? (hasGameLog ? gameLogEps[gameLogEps.length - 1] : null)
  const idx = hasGameLog ? gameLogEps.indexOf(current) : -1

  const firstEp = episodes.length ? episodes[0].episode : null
  const lastEp = episodes.length ? episodes[episodes.length - 1].episode : null

  const parseRange = (v) => { const n = parseInt(v, 10); return isNaN(n) ? null : n }

  return (
    <div className="flex items-center gap-3 px-5 py-2 text-[11px] text-overlay0">
      {hasGameLog && (
        <div className="flex items-center gap-1.5">
          <span>Episode:</span>
          <select
            className="px-1 py-0.5 rounded bg-surface0 text-text text-[11px] border border-surface1"
            value={episode ?? ''}
            onChange={e => setEpisode(e.target.value === '' ? null : Number(e.target.value))}
          >
            <option value="">Latest</option>
            {[...gameLogEps].reverse().map(ep => (
              <option key={ep} value={ep}>{ep}</option>
            ))}
          </select>
        </div>
      )}
      {episodes.length > 1 && (
        <div className="flex items-center gap-1.5">
          <span>Range:</span>
          <input
            type="number"
            className="w-14 px-1 py-0.5 rounded bg-surface0 text-text text-[11px] border border-surface1 text-center"
            placeholder={firstEp ?? ''}
            value={rangeMin ?? ''}
            onChange={e => setRange([parseRange(e.target.value), rangeMax])}
          />
          <span>–</span>
          <input
            type="number"
            className="w-14 px-1 py-0.5 rounded bg-surface0 text-text text-[11px] border border-surface1 text-center"
            placeholder={lastEp ?? ''}
            value={rangeMax ?? ''}
            onChange={e => setRange([rangeMin, parseRange(e.target.value)])}
          />
          {(rangeMin != null || rangeMax != null) && (
            <button
              onClick={() => setRange([null, null])}
              className="px-1.5 py-0.5 rounded text-overlay0 hover:text-subtext0 hover:bg-surface0/50"
            >✕</button>
          )}
        </div>
      )}
    </div>
  )
}

const RETURN_RANGE_SERIES = [
  { key: 'ret_min', label: 'Min', color: '#f38ba8' },
  { key: 'ret_mean', label: 'Mean', color: '#a6e3a1' },
  { key: 'ret_max', label: 'Max', color: '#89b4fa' },
]

const ReturnRangeChart = memo(function ReturnRangeChart() {
  const ds = useDownsampled()
  const labels = useDownsampledLabels()
  const { ema: emaOn, bands: bandsOn } = useCardSettings()
  const raws = useMemo(
    () => RETURN_RANGE_SERIES.map(({ key }) => ds.map(e => e[key] ?? null)),
    [ds],
  )
  const yViewport = useMemo(
    () => (bandsOn ? percentileViewportMulti(raws) : null),
    [raws, bandsOn],
  )
  const options = useMemo(
    () => ({
      ...chartBase(yViewport),
      plugins: { legend: { labels: { color: CHART_THEME.text, font: { size: 10 } } } },
    }),
    [yViewport],
  )
  if (ds.length <= 1) return <NoData />
  const dim = emaOn
  const datasets = []
  for (let i = 0; i < RETURN_RANGE_SERIES.length; i++) {
    const { label, color } = RETURN_RANGE_SERIES[i]
    const raw = raws[i]
    datasets.push({
      label, data: raw, borderColor: dim ? color + '40' : color,
      tension: 0, pointRadius: 0, borderWidth: dim ? 1 : 1.5, spanGaps: true,
    })
    if (emaOn) datasets.push({
      label: `${label} EMA`, data: ema(raw), borderColor: color,
      borderWidth: 2, pointRadius: 0, tension: 0.3, spanGaps: true,
    })
  }
  return (
    <div className="min-h-[200px]">
      <Line data={{ labels, datasets }} options={options} />
    </div>
  )
})

// ---- Pretrain chart components ----

const BCLossChart = makeSimpleChart('Loss', 'loss', '#f38ba8')
const BCGradNormChart = makeSimpleChart('Grad Norm', 'grad_norm', '#94e2d5')

const BC_HEAD_LOSS_SERIES = [
  { key: 'loss_bt', label: 'Build Type', color: '#89b4fa' },
  { key: 'loss_bd', label: 'Build Dir', color: '#a6e3a1' },
  { key: 'loss_cd', label: 'Card', color: '#fab387' },
  { key: 'loss_brt', label: 'Bridge Target', color: '#cba6f7' },
  { key: 'loss_md', label: 'Move Dir', color: '#f9e2af' },
]

const BCHeadLossChart = memo(function BCHeadLossChart() {
  const ds = useDownsampled()
  const labels = useDownsampledLabels()
  const { ema: emaOn, bands: bandsOn } = useCardSettings()
  const raws = useMemo(
    () => BC_HEAD_LOSS_SERIES.map(({ key }) => ds.map(e => e[key] ?? null)),
    [ds],
  )
  const yViewport = useMemo(
    () => (bandsOn ? percentileViewportMulti(raws) : null),
    [raws, bandsOn],
  )
  const options = useMemo(
    () => ({
      ...chartBase(yViewport),
      plugins: { legend: { labels: { color: CHART_THEME.text, font: { size: 10 } } } },
    }),
    [yViewport],
  )
  if (ds.length <= 1) return <NoData />
  const dim = emaOn
  const datasets = []
  for (let i = 0; i < BC_HEAD_LOSS_SERIES.length; i++) {
    const { label, color } = BC_HEAD_LOSS_SERIES[i]
    const raw = raws[i]
    datasets.push({
      label, data: raw, borderColor: dim ? color + '40' : color,
      tension: 0, pointRadius: 0, borderWidth: dim ? 1 : 1.5, spanGaps: true,
    })
    if (emaOn) datasets.push({
      label: `${label} EMA`, data: ema(raw), borderColor: color,
      borderWidth: 2, pointRadius: 0, tension: 0.3, spanGaps: true,
    })
  }
  return (
    <div className="min-h-[200px]">
      <Line data={{ labels, datasets }} options={options} />
    </div>
  )
})

function BCLogTable() {
  const episodes = useRangedEpisodes()
  if (!episodes.length) return <NoData />
  return (
    <div className="max-h-96 overflow-y-auto">
      <Table>
        <Thead>
          <Th>Step</Th>
          <Th>Loss</Th>
          <Th>BT</Th>
          <Th>BD</Th>
          <Th>CD</Th>
          <Th>BRT</Th>
          <Th>MD</Th>
          <Th>Grad</Th>
          <Th>Samples</Th>
        </Thead>
        <tbody>
          {[...episodes].reverse().slice(0, 200).map(e => (
            <Tr key={e.optimizer_step ?? e.episode}>
              <Td>{e.optimizer_step ?? e.episode}</Td>
              <Td>{fmt(e.loss, 4)}</Td>
              <Td>{fmt(e.loss_bt, 4)}</Td>
              <Td>{fmt(e.loss_bd, 4)}</Td>
              <Td>{fmt(e.loss_cd, 4)}</Td>
              <Td>{fmt(e.loss_brt, 4)}</Td>
              <Td>{fmt(e.loss_md, 4)}</Td>
              <Td>{fmt(e.grad_norm, 3)}</Td>
              <Td>{e.n_samples?.toLocaleString()}</Td>
            </Tr>
          ))}
        </tbody>
      </Table>
    </div>
  )
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
  'train-win-confidence':   { title: 'Win Confidence',      subtitle: 'per round', hasLegend: true, render: () => <WinConfidenceChart /> },
  'train-prediction-acc':   { title: 'Prediction Accuracy', subtitle: 'per episode', hasEma: true, render: () => <WinPredictionAccuracyChart /> },
  'train-round-accuracy':   { title: 'Round Accuracy',      subtitle: 'by absolute round', render: () => <RoundAccuracyChart /> },
  'train-elo':              { title: 'League ELO',          render: () => <LeagueEloCard /> },
  'train-loss-chart':       { title: 'Loss',                defaultOpen: false, hasEma: true, render: () => <LossChart /> },
  'train-entropy':          { title: 'Entropy',             defaultOpen: false, hasEma: true, render: () => <EntropyChart /> },
  'train-entropy-heads':    { title: 'Entropy per Head',    defaultOpen: false, hasEma: true, hasLegend: true, render: () => <HeadEntropyChart /> },
  'train-ev':               { title: 'Explained Variance',  hasEma: true, render: () => <ExplVarChart /> },
  'train-kl':               { title: 'Approx KL',           defaultOpen: false, hasEma: true, render: () => <KLChart /> },
  'train-clip':             { title: 'Clip Fraction',       defaultOpen: false, hasEma: true, render: () => <ClipChart /> },
  'train-actor-grad-norm':  { title: 'Actor Grad Norm',     defaultOpen: false, hasEma: true, render: () => <ActorGradNormChart /> },
  'train-critic-grad-norm': { title: 'Critic Grad Norm',    defaultOpen: false, hasEma: true, render: () => <CriticGradNormChart /> },
  'train-critic-value-delta': { title: 'Critic Value Delta', defaultOpen: false, hasEma: true, render: () => <CriticValueDeltaChart /> },
  'train-return-range':       { title: 'Return Range',        defaultOpen: false, hasEma: true, hasLegend: true, render: () => <ReturnRangeChart /> },
  'train-ret-oob':            { title: 'Return OOB Frac',     defaultOpen: false, hasEma: true, render: () => <RetOobFracChart /> },
  'train-actor-lr':           { title: 'Actor LR',            defaultOpen: false, hasEma: true, render: () => <ActorLrChart /> },
  'train-temp-mean': { title: 'Temperature Mean',    defaultOpen: false, hasEma: true, render: () => <TempMeanChart /> },
  'train-temp-std':  { title: 'Temperature Std',     defaultOpen: false, hasEma: true, render: () => <TempStdChart /> },
  'train-log-table':        { title: 'Episode Log',         defaultOpen: false, render: () => <EpisodeLogTable /> },
  'train-replays':          { title: 'Training Replays',    render: () => <TrainingReplaysContent /> },
  'train-pretrain-loss':    { title: 'Pretrain Loss',       defaultOpen: false, hasEma: true, render: () => <BCLossChart /> },
  'train-head-loss':        { title: 'Per-Head Loss',       defaultOpen: false, hasEma: true, render: () => <BCHeadLossChart /> },
  'train-grad-norm':        { title: 'Grad Norm',           defaultOpen: false, hasEma: true, render: () => <BCGradNormChart /> },
  'train-pretrain-log':     { title: 'Pretrain Step Log',   defaultOpen: false, render: () => <BCLogTable /> },
}

const DEFAULT_LAYOUT = {
  nextId: 1,
  root: { type: 'leaf', id: 'v0', cards: Object.keys(TRAINING_CARDS) },
}

// ---- Main export ----

export default function TrainingView() {
  const session = useAtomValue(trainSessionAtom)
  const [episode, setEpisode] = useState(null)
  const [range, setRange] = useState([null, null])
  const ctx = useMemo(() => ({ episode, setEpisode, range, setRange }), [episode, range])

  return (
    <TrainingDataProvider session={session}>
      <SelectedEpisodeCtx.Provider value={ctx}>
        <EpisodeSelector />
        <DownsampledProvider>
          <TileLayout
            cards={TRAINING_CARDS}
            storageKey="training-tiles"
            defaultLayout={DEFAULT_LAYOUT}
          />
        </DownsampledProvider>
      </SelectedEpisodeCtx.Provider>
    </TrainingDataProvider>
  )
}
