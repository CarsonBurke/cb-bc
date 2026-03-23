import { readFileSync, statSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const SERVER_ROOT = dirname(fileURLToPath(import.meta.url))
const DASHBOARD_ROOT = resolve(SERVER_ROOT, '..')
const PROJECT_ROOT = resolve(DASHBOARD_ROOT, '..')
const HISTORY_PATH = resolve(PROJECT_ROOT, 'ablations/history.json')

let cachedHistory = null

function buildHistoryCache(parsed) {
  const allBots = new Set()
  const roundsById = new Map()
  const rounds = []
  const timeline = []

  parsed.rounds.forEach((round, index) => {
    const roundBots = round.bots || {}
    const botElos = {}

    for (const [name, bot] of Object.entries(roundBots)) {
      allBots.add(name)
      botElos[name] = bot.elo
    }

    rounds.push({
      round_id: round.round_id,
      timestamp: round.timestamp,
      bot_count: Object.keys(roundBots).length,
      match_count: round.matches?.length || 0,
    })

    timeline.push({
      round_id: round.round_id,
      timestamp: round.timestamp,
      bots: botElos,
    })

    roundsById.set(round.round_id, {
      round,
      prevBots: parsed.rounds[index - 1]?.bots || null,
    })
  })

  return {
    lineage: parsed.lineage || {},
    rounds,
    timeline,
    allBots: [...allBots].sort(),
    roundsById,
  }
}

function loadHistoryCache() {
  const stats = statSync(HISTORY_PATH)
  if (
    cachedHistory &&
    cachedHistory.mtimeMs === stats.mtimeMs &&
    cachedHistory.size === stats.size
  ) {
    return cachedHistory.value
  }

  const parsed = JSON.parse(readFileSync(HISTORY_PATH, 'utf8'))
  const value = buildHistoryCache(parsed)
  cachedHistory = {
    mtimeMs: stats.mtimeMs,
    size: stats.size,
    value,
  }
  return value
}

export function getDashboardOverview() {
  const history = loadHistoryCache()
  return {
    rounds: history.rounds,
    timeline: history.timeline,
    lineage: history.lineage,
    allBots: history.allBots,
  }
}

export function getDashboardRound(roundId) {
  const history = loadHistoryCache()
  const entry = history.roundsById.get(roundId)
  if (!entry) return null
  return {
    round: entry.round,
    prevBots: entry.prevBots,
  }
}

export function getProjectRoot() {
  return PROJECT_ROOT
}

export function getDashboardRoot() {
  return DASHBOARD_ROOT
}
