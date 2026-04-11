import { spawn, execSync } from 'node:child_process'
import { closeSync, existsSync, openSync, readFileSync, readSync, readdirSync, statSync } from 'node:fs'
import http from 'node:http'
import { extname, resolve } from 'node:path'
import { getDashboardOverview, getDashboardRound, getDashboardRoot, getProjectRoot } from './history.js'

const DASHBOARD_ROOT = getDashboardRoot()
const PROJECT_ROOT = getProjectRoot()
const DIST_DIR = resolve(DASHBOARD_ROOT, 'dist')
const PUBLIC_DIR = resolve(DASHBOARD_ROOT, 'public')

const REPLAYS_DIR = resolve(PROJECT_ROOT, 'replays')
const TRAINING_REPLAYS_DIR = resolve(PROJECT_ROOT, 'replays/training')
const SESSIONS_DIR = resolve(PROJECT_ROOT, 'bots/tensor/sessions')
const TENSOR_DIR = resolve(PROJECT_ROOT, 'bots/tensor')

const trainSessionLogCache = new Map()
const replayCache = new Map()
const trainLogCache = new Map()
const gameLogCache = new Map()
const leagueCache = new Map()
const jsonResponseCache = new Map()
const jsonResponseKeysByFile = new Map()
const trainSessionsListCache = { mtime: 0, data: null }
const dashboardFileCache = new Map()

const MIME_TYPES = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
  '.webp': 'image/webp',
}

function findVisualiserDist() {
  try {
    const result = execSync(
      `${resolve(PROJECT_ROOT, '.venv/bin/python3')} -c "from pathlib import Path; import cambc; p = Path(cambc.__file__).parent / 'data' / 'visualiser'; print(p if (p / 'index.html').exists() else '')"`,
      { encoding: 'utf8' },
    ).trim()
    return result || null
  } catch {
    return null
  }
}

const VISUALISER_DIST = findVisualiserDist()

function withCors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
}

function sendJson(res, payload, statusCode = 200) {
  withCors(res)
  res.statusCode = statusCode
  res.setHeader('Content-Type', 'application/json; charset=utf-8')
  res.end(JSON.stringify(payload))
}

function sendCachedJson(res, cacheKey, buildPayload, filePath = null) {
  const cached = jsonResponseCache.get(cacheKey)
  if (cached) {
    sendBuffer(res, cached, 'application/json; charset=utf-8')
    return
  }
  const buf = Buffer.from(JSON.stringify(buildPayload()))
  jsonResponseCache.set(cacheKey, buf)
  if (filePath) {
    let set = jsonResponseKeysByFile.get(filePath)
    if (!set) {
      set = new Set()
      jsonResponseKeysByFile.set(filePath, set)
    }
    set.add(cacheKey)
  }
  sendBuffer(res, buf, 'application/json; charset=utf-8')
}

function sendText(res, body, statusCode = 200, contentType = 'text/plain; charset=utf-8') {
  withCors(res)
  res.statusCode = statusCode
  res.setHeader('Content-Type', contentType)
  res.end(body)
}

function sendBuffer(res, body, contentType = 'application/octet-stream', statusCode = 200) {
  withCors(res)
  res.statusCode = statusCode
  res.setHeader('Content-Type', contentType)
  res.setHeader('Content-Length', body.length)
  res.end(body)
}

function serveFile(res, filePath, statusCode = 200) {
  const data = readFileSync(filePath)
  const type = MIME_TYPES[extname(filePath)] || 'application/octet-stream'
  sendBuffer(res, data, type, statusCode)
}

function safeRelativePath(pathname, prefix) {
  const raw = pathname.slice(prefix.length)
  if (!raw) return ''
  const value = decodeURIComponent(raw.replace(/^\/+/, ''))
  if (!value || value.includes('..') || value.includes('\\')) return null
  return value
}

function sessionDir(url) {
  const session = url.searchParams.get('session')
  if (!session || session.includes('..') || session.includes('\\')) return null
  return resolve(SESSIONS_DIR, session)
}

function getReplayListing(dir) {
  const dirMtime = statSync(dir).mtimeMs
  const cached = replayCache.get(dir)
  if (cached && cached.mtime === dirMtime) return cached.files

  const known = cached ? new Map(cached.byName) : new Map()
  const currentNames = new Set(readdirSync(dir).filter((name) => name.endsWith('.replay26')))

  for (const name of currentNames) {
    if (!known.has(name)) {
      const stats = statSync(resolve(dir, name))
      known.set(name, { name, size: stats.size, mtime: stats.mtimeMs })
    }
  }

  for (const name of [...known.keys()]) {
    if (!currentNames.has(name)) known.delete(name)
  }

  const files = [...known.values()].sort((a, b) => b.mtime - a.mtime)
  replayCache.set(dir, { mtime: dirMtime, byName: known, files })
  return files
}

function countJsonlEpisodes(filePath) {
  let stats
  try {
    stats = statSync(filePath)
  } catch {
    return null
  }
  const cached = trainSessionLogCache.get(filePath)
  if (cached && cached.mtime === stats.mtimeMs && cached.size === stats.size) {
    return cached.episodes
  }
  const buf = readFileSync(filePath)
  let newlines = 0
  for (let i = 0; i < buf.length; i++) if (buf[i] === 10) newlines++
  if (buf.length > 0 && buf[buf.length - 1] !== 10) newlines++
  const episodes = Math.max(0, newlines - 1)
  trainSessionLogCache.set(filePath, { mtime: stats.mtimeMs, size: stats.size, episodes })
  return episodes
}

function readTrainSessions() {
  const sessionsMtime = statSync(SESSIONS_DIR).mtimeMs
  if (trainSessionsListCache.data && trainSessionsListCache.mtime === sessionsMtime) {
    return trainSessionsListCache.data
  }

  const dirs = readdirSync(SESSIONS_DIR, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      const sessionPath = resolve(SESSIONS_DIR, entry.name)
      const metaPath = resolve(sessionPath, 'meta.json')
      let meta = { name: entry.name }
      try {
        meta = JSON.parse(readFileSync(metaPath, 'utf8'))
      } catch {
        // Ignore missing metadata.
      }

      let episodes = 0
      const jsonlPath = resolve(sessionPath, 'train_log.jsonl')
      const counted = countJsonlEpisodes(jsonlPath)
      if (counted != null) {
        episodes = counted
      } else {
        try {
          const jsonPath = resolve(sessionPath, 'train_log.json')
          const log = JSON.parse(readFileSync(jsonPath, 'utf8'))
          episodes = log.episodes?.length || 0
        } catch {
          // Ignore broken session logs.
        }
      }

      let sessionMtime = 0
      try {
        sessionMtime = statSync(sessionPath).mtimeMs
      } catch {
        // session dir vanished mid-iteration; leave mtime = 0
      }
      return {
        id: entry.name,
        ...meta,
        episodes,
        mtime: sessionMtime,
      }
    })
    .sort((a, b) => b.mtime - a.mtime)

  trainSessionsListCache.data = dirs
  trainSessionsListCache.mtime = sessionsMtime
  return dirs
}

// Python may emit NaN/Infinity which are not valid JSON — replace with null.
function safeJsonParse(line) {
  return JSON.parse(line.replace(/\bNaN\b/g, 'null').replace(/\b-?Infinity\b/g, 'null'))
}

function readIncrementalJsonl(cacheMap, filePath, parseFirstLineAsConfig = false) {
  const stats = statSync(filePath)
  const cached = cacheMap.get(filePath)
  const canIncrementallyRead = cached && stats.size > cached.size && cached.offset > 0

  if (!cached || cached.size !== stats.size) {
    // Evict serialized response cache — underlying data changed
    const affectedKeys = jsonResponseKeysByFile.get(filePath)
    if (affectedKeys) {
      for (const key of affectedKeys) jsonResponseCache.delete(key)
      affectedKeys.clear()
    }
    let updatedIncrementally = false
    if (canIncrementallyRead) {
      try {
        const newBytes = Buffer.alloc(stats.size - cached.offset)
        const fd = openSync(filePath, 'r')
        try {
          readSync(fd, newBytes, 0, newBytes.length, cached.offset)
        } finally {
          closeSync(fd)
        }

        const tail = newBytes.toString('utf8').trimEnd()
        if (tail) {
          const nextRows = tail.split('\n').map((line) => safeJsonParse(line))
          cached.rows.push(...nextRows)
        }
        cached.offset = stats.size
        cached.size = stats.size
        updatedIncrementally = true
      } catch {
        // Fall back to a full read when the file was rewritten.
      }
    }

    if (!updatedIncrementally) {
      const lines = readFileSync(filePath, 'utf8').trimEnd().split('\n').filter(Boolean)
      if (parseFirstLineAsConfig) {
        const first = lines[0] ? safeJsonParse(lines[0]) : {}
        cacheMap.set(filePath, {
          size: stats.size,
          offset: stats.size,
          rows: lines.slice(1).map((line) => safeJsonParse(line)),
          config: first.config || {},
        })
      } else {
        cacheMap.set(filePath, {
          size: stats.size,
          offset: stats.size,
          rows: lines.map((line) => safeJsonParse(line)),
        })
      }
    }
  }

  return cacheMap.get(filePath)
}

function readLeagueData(filePath) {
  const stats = statSync(filePath)
  const cached = leagueCache.get(filePath)
  if (cached && cached.mtime === stats.mtimeMs && cached.size === stats.size) {
    return cached.data
  }

  const data = readFileSync(filePath, 'utf8')
  leagueCache.set(filePath, { mtime: stats.mtimeMs, size: stats.size, data })
  return data
}

function tryServeViewerFile(res, pathname) {
  if (!VISUALISER_DIST) return false

  let relativePath
  if (pathname === '/viewer' || pathname === '/viewer/') {
    relativePath = 'index.html'
  } else if (pathname.startsWith('/viewer/')) {
    relativePath = pathname.replace(/^\/viewer\//, '') || 'index.html'
  } else {
    relativePath = pathname.replace(/^\/+/, '')
  }

  if (!relativePath || relativePath.includes('..') || relativePath.includes('\\')) return false

  const filePath = resolve(VISUALISER_DIST, relativePath)
  if (!existsSync(filePath)) return false

  let data = readFileSync(filePath)
  const ext = extname(filePath)
  if (ext === '.html') {
    const escapeBridge = [
      '<script>',
      '(function () {',
      "  function notifyParent(event) {",
      "    if (event.key === 'Escape' && window.parent && window.parent !== window) {",
      "      window.parent.postMessage({ type: 'dashboard-viewer-escape' }, '*')",
      '    }',
      '  }',
      "  window.addEventListener('keydown', notifyParent, true)",
      "  document.addEventListener('keydown', notifyParent, true)",
      '})()',
      '</script>',
    ].join('')
    data = Buffer.from(
      data.toString()
        .replace(/src="\/assets\//g, 'src="/viewer/assets/')
        .replace(/href="\/assets\//g, 'href="/viewer/assets/')
        .replace('</body>', `${escapeBridge}</body>`),
    )
  }
  sendBuffer(res, data, MIME_TYPES[ext] || 'application/octet-stream')
  return true
}

function statFileOrNull(filePath) {
  try {
    const st = statSync(filePath)
    return st.isFile() ? st : null
  } catch {
    return null
  }
}

function resolveDashboardPath(pathname) {
  const cached = dashboardFileCache.get(pathname)
  if (cached !== undefined) return cached

  const normalized = pathname === '/' ? '/index.html' : pathname
  const relative = normalized.slice(1)
  if (!relative || relative.includes('..') || relative.includes('\\')) {
    dashboardFileCache.set(pathname, null)
    return null
  }

  const distPath = resolve(DIST_DIR, relative)
  if (statFileOrNull(distPath)) {
    dashboardFileCache.set(pathname, distPath)
    return distPath
  }

  const publicPath = resolve(PUBLIC_DIR, relative)
  if (statFileOrNull(publicPath)) {
    dashboardFileCache.set(pathname, publicPath)
    return publicPath
  }

  const indexPath = resolve(DIST_DIR, 'index.html')
  if (statFileOrNull(indexPath)) {
    dashboardFileCache.set(pathname, indexPath)
    return indexPath
  }

  return null
}

function tryServeDashboardFile(res, pathname) {
  const filePath = resolveDashboardPath(pathname)
  if (!filePath) return false
  serveFile(res, filePath)
  return true
}

async function handleRequest(req, res) {
  withCors(res)

  if (req.method === 'OPTIONS') {
    res.statusCode = 204
    res.end()
    return
  }

  if (req.method !== 'GET') {
    sendJson(res, { error: 'Method not allowed' }, 405)
    return
  }

  const url = new URL(req.url, 'http://127.0.0.1')
  const pathname = decodeURIComponent(url.pathname)

  if (pathname === '/api/dashboard/overview') {
    sendJson(res, getDashboardOverview())
    return
  }

  if (pathname.startsWith('/api/dashboard/round/')) {
    const roundId = Number.parseInt(pathname.slice('/api/dashboard/round/'.length), 10)
    if (!Number.isInteger(roundId)) {
      sendJson(res, { error: 'Invalid round id' }, 400)
      return
    }
    const payload = getDashboardRound(roundId)
    if (!payload) {
      sendJson(res, { error: 'Round not found' }, 404)
      return
    }
    sendJson(res, payload)
    return
  }

  if (pathname === '/api/train-sessions') {
    try {
      sendJson(res, readTrainSessions())
    } catch {
      sendJson(res, [])
    }
    return
  }

  if (pathname === '/api/train-replays') {
    const sd = sessionDir(url)
    const dir = sd ? resolve(sd, 'replays') : TRAINING_REPLAYS_DIR
    const minEp = url.searchParams.has('min_ep') ? Number(url.searchParams.get('min_ep')) : null
    const maxEp = url.searchParams.has('max_ep') ? Number(url.searchParams.get('max_ep')) : null
    try {
      let replays = getReplayListing(dir)
      if (minEp != null || maxEp != null) {
        const baseDir = sd || TENSOR_DIR
        const logPath = resolve(baseDir, 'game_log.jsonl')
        try {
          const entry = readIncrementalJsonl(gameLogCache, logPath)
          const epByReplay = new Map()
          for (const g of entry.rows) {
            if (g.replay != null && g.episode != null) epByReplay.set(g.replay, g.episode)
          }
          replays = replays.filter(r => {
            const ep = epByReplay.get(r.name)
            if (ep == null) return true
            return (minEp == null || ep >= minEp) && (maxEp == null || ep <= maxEp)
          })
        } catch {
          // game log unavailable, return unfiltered
        }
      }
      sendJson(res, replays)
    } catch {
      sendJson(res, [])
    }
    return
  }

  if (pathname.startsWith('/api/train-replay/')) {
    const filename = safeRelativePath(pathname, '/api/train-replay/')
    if (!filename) {
      sendText(res, 'Bad request', 400)
      return
    }
    const dir = sessionDir(url) ? resolve(sessionDir(url), 'replays') : TRAINING_REPLAYS_DIR
    try {
      serveFile(res, resolve(dir, filename))
    } catch {
      sendText(res, 'Not found', 404)
    }
    return
  }

  if (pathname === '/api/train-log') {
    const after = Number.parseInt(url.searchParams.get('after') || '0', 10)
    const minEp = url.searchParams.has('min_ep') ? Number(url.searchParams.get('min_ep')) : null
    const maxEp = url.searchParams.has('max_ep') ? Number(url.searchParams.get('max_ep')) : null
    const baseDir = sessionDir(url) || TENSOR_DIR
    const jsonlPath = resolve(baseDir, 'train_log.jsonl')
    const jsonPath = resolve(baseDir, 'train_log.json')
    try {
      if (existsSync(jsonlPath)) {
        const entry = readIncrementalJsonl(trainLogCache, jsonlPath, true)
        const cacheKey = `train-log:${jsonlPath}:${entry.size}:${after}:${minEp}:${maxEp}`
        sendCachedJson(res, cacheKey, () => {
          let episodes = after > 0 && after <= entry.rows.length ? entry.rows.slice(after) : entry.rows
          if (minEp != null || maxEp != null) {
            episodes = episodes.filter(e =>
              (minEp == null || e.episode >= minEp) &&
              (maxEp == null || e.episode <= maxEp)
            )
          }
          return {
            episodes,
            config: entry.config,
            total: entry.rows.length,
          }
        }, jsonlPath)
      } else {
        sendBuffer(res, Buffer.from(readFileSync(jsonPath, 'utf8')), 'application/json; charset=utf-8')
      }
    } catch {
      sendJson(res, { episodes: [], config: {}, total: 0 })
    }
    return
  }

  if (pathname === '/api/game-log') {
    const after = Number.parseInt(url.searchParams.get('after') || '0', 10)
    const episode = url.searchParams.get('episode')
    const minEp = url.searchParams.has('min_ep') ? Number(url.searchParams.get('min_ep')) : null
    const maxEp = url.searchParams.has('max_ep') ? Number(url.searchParams.get('max_ep')) : null
    const baseDir = sessionDir(url) || TENSOR_DIR
    const logPath = resolve(baseDir, 'game_log.jsonl')
    try {
      const entry = readIncrementalJsonl(gameLogCache, logPath)
      const cacheKey = `game-log:${logPath}:${entry.size}:${after}:${episode}:${minEp}:${maxEp}`
      sendCachedJson(res, cacheKey, () => {
        let games
        if (episode != null) {
          const ep = Number(episode)
          games = entry.rows.filter(g => g.episode === ep)
        } else {
          games = after > 0 && after <= entry.rows.length ? entry.rows.slice(after) : entry.rows
        }
        if (minEp != null || maxEp != null) {
          games = games.filter(g =>
            (minEp == null || g.episode >= minEp) &&
            (maxEp == null || g.episode <= maxEp)
          )
        }
        return { games, total: entry.rows.length }
      }, logPath)
    } catch {
      sendJson(res, { games: [], total: 0 })
    }
    return
  }

  if (pathname === '/api/game-log-episodes') {
    const baseDir = sessionDir(url) || TENSOR_DIR
    const logPath = resolve(baseDir, 'game_log.jsonl')
    const minEp = url.searchParams.has('min_ep') ? Number(url.searchParams.get('min_ep')) : null
    const maxEp = url.searchParams.has('max_ep') ? Number(url.searchParams.get('max_ep')) : null
    try {
      const entry = readIncrementalJsonl(gameLogCache, logPath)
      const cacheKey = `game-log-episodes:${logPath}:${entry.size}:${minEp}:${maxEp}`
      sendCachedJson(res, cacheKey, () => {
        const byEp = new Map()
        for (const g of entry.rows) {
          if (g.episode == null) continue
          if (minEp != null && g.episode < minEp) continue
          if (maxEp != null && g.episode > maxEp) continue
          if (!byEp.has(g.episode)) byEp.set(g.episode, [])
          byEp.get(g.episode).push(g)
        }
        const summaries = []
        for (const [ep, games] of byEp) {
          let correct = 0, total = 0, confSum = 0, confCount = 0
          for (const g of games) {
            const wc = g.win_confidence
            if (!wc || !wc.length) continue
            const finalConf = wc[wc.length - 1]
            confSum += finalConf
            confCount++
            if (g.winner != null) {
              total++
              const pred = finalConf > 0.5 ? 'a' : 'b'
              if (pred === g.winner) correct++
            }
          }
          summaries.push({
            episode: ep,
            n_games: games.length,
            accuracy: total > 0 ? correct / total : null,
            mean_confidence: confCount > 0 ? confSum / confCount : null,
          })
        }
        summaries.sort((a, b) => a.episode - b.episode)
        const latestEp = summaries.length ? summaries[summaries.length - 1].episode : null
        const latestGames = latestEp != null ? (byEp.get(latestEp) || []) : []
        return { episodes: summaries.map(s => s.episode), summaries, latestEp, latestGames }
      }, logPath)
    } catch {
      sendJson(res, { episodes: [], summaries: [] })
    }
    return
  }

  if (pathname === '/api/league') {
    const sd2 = sessionDir(url)
    const baseDir = sd2 ? resolve(sd2, 'league') : resolve(TENSOR_DIR, 'league')
    const modelsPath = resolve(baseDir, 'models.json')
    try {
      const models = JSON.parse(readLeagueData(modelsPath))
      sendJson(res, { models })
    } catch {
      sendJson(res, { models: [] })
    }
    return
  }

  if (pathname === '/api/replays') {
    try {
      sendJson(res, getReplayListing(REPLAYS_DIR))
    } catch {
      sendJson(res, [])
    }
    return
  }

  if (pathname.startsWith('/api/replay/')) {
    const filename = safeRelativePath(pathname, '/api/replay/')
    if (!filename) {
      sendText(res, 'Bad request', 400)
      return
    }
    try {
      serveFile(res, resolve(REPLAYS_DIR, filename))
    } catch {
      sendText(res, 'Not found', 404)
    }
    return
  }

  if (pathname === '/api/watch') {
    const matchId = url.searchParams.get('match_id')
    if (!matchId || /[^a-zA-Z0-9_\-]/.test(matchId)) {
      sendJson(res, { error: 'match_id required (alphanumeric, hyphens, underscores only)' }, 400)
      return
    }

    const replayPath = resolve(REPLAYS_DIR, `match_${matchId}.replay26`)
    const cambc = resolve(PROJECT_ROOT, '.venv/bin/cambc')
    const proc = spawn(cambc, ['watch', replayPath], {
      cwd: PROJECT_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
    })

    let responded = false

    function finish(payload, statusCode = 200) {
      if (responded) return
      responded = true
      proc.kill()
      sendJson(res, payload, statusCode)
    }

    function respondWithUrl(chunk) {
      const match = chunk.toString().match(/https?:\/\/[^\s]+/)
      if (match) finish({ url: match[0] })
    }

    proc.stdout.on('data', respondWithUrl)
    proc.stderr.on('data', respondWithUrl)

    proc.on('error', () => {
      finish({ error: 'Failed to launch cambc watch' }, 500)
    })

    setTimeout(() => {
      finish({ error: 'Timeout waiting for visualizer' }, 504)
    }, 5000)
    return
  }

  if (pathname === '/viewer' || pathname.startsWith('/viewer/')) {
    if (tryServeViewerFile(res, pathname)) return
    sendText(res, 'Not found', 404)
    return
  }

  if (tryServeViewerFile(res, pathname)) return
  if (tryServeDashboardFile(res, pathname)) return

  sendText(res, 'Not found', 404)
}

export function createDashboardServer() {
  return http.createServer((req, res) => {
    Promise.resolve(handleRequest(req, res)).catch(() => {
      if (!res.headersSent) sendJson(res, { error: 'Internal server error' }, 500)
      else res.end()
    })
  })
}

export function startDashboardServer({ host = '127.0.0.1', port = 5174 } = {}) {
  const server = createDashboardServer()
  return new Promise((resolveServer, reject) => {
    server.once('error', reject)
    server.listen(port, host, () => resolveServer(server))
  })
}
