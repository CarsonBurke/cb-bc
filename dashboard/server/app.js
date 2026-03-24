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

function readTrainSessions() {
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
      try {
        const jsonlPath = resolve(sessionPath, 'train_log.jsonl')
        const jsonPath = resolve(sessionPath, 'train_log.json')
        if (existsSync(jsonlPath)) {
          const stats = statSync(jsonlPath)
          const cached = trainSessionLogCache.get(jsonlPath)
          if (cached && cached.mtime === stats.mtimeMs && cached.size === stats.size) {
            episodes = cached.episodes
          } else {
            const lines = readFileSync(jsonlPath, 'utf8').trimEnd().split('\n')
            episodes = Math.max(0, lines.length - 1)
            trainSessionLogCache.set(jsonlPath, { mtime: stats.mtimeMs, size: stats.size, episodes })
          }
        } else if (existsSync(jsonPath)) {
          const log = JSON.parse(readFileSync(jsonPath, 'utf8'))
          episodes = log.episodes?.length || 0
        }
      } catch {
        // Ignore broken session logs.
      }

      const stats = statSync(sessionPath)
      return {
        id: entry.name,
        ...meta,
        episodes,
        mtime: stats.mtimeMs,
      }
    })
    .sort((a, b) => b.mtime - a.mtime)

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

function tryServeDashboardFile(res, pathname) {
  const normalized = pathname === '/' ? '/index.html' : pathname
  const distPath = resolve(DIST_DIR, normalized.slice(1))
  if (existsSync(distPath) && statSync(distPath).isFile()) {
    serveFile(res, distPath)
    return true
  }

  const publicPath = resolve(PUBLIC_DIR, normalized.slice(1))
  if (existsSync(publicPath) && statSync(publicPath).isFile()) {
    serveFile(res, publicPath)
    return true
  }

  const indexPath = resolve(DIST_DIR, 'index.html')
  if (existsSync(indexPath)) {
    serveFile(res, indexPath)
    return true
  }

  return false
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
    const dir = sessionDir(url) ? resolve(sessionDir(url), 'replays') : TRAINING_REPLAYS_DIR
    try {
      sendJson(res, getReplayListing(dir))
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
    const baseDir = sessionDir(url) || TENSOR_DIR
    const jsonlPath = resolve(baseDir, 'train_log.jsonl')
    const jsonPath = resolve(baseDir, 'train_log.json')
    try {
      if (existsSync(jsonlPath)) {
        const entry = readIncrementalJsonl(trainLogCache, jsonlPath, true)
        const episodes = after > 0 && after <= entry.rows.length ? entry.rows.slice(after) : entry.rows
        sendJson(res, {
          episodes,
          config: entry.config,
          total: entry.rows.length,
        })
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
    const baseDir = sessionDir(url) || TENSOR_DIR
    const logPath = resolve(baseDir, 'game_log.jsonl')
    try {
      const entry = readIncrementalJsonl(gameLogCache, logPath)
      const games = after > 0 && after <= entry.rows.length ? entry.rows.slice(after) : entry.rows
      sendJson(res, { games, total: entry.rows.length })
    } catch {
      sendJson(res, { games: [], total: 0 })
    }
    return
  }

  if (pathname === '/api/league') {
    const baseDir = sessionDir(url) ? resolve(sessionDir(url), 'league') : resolve(TENSOR_DIR, 'league')
    const leaguePath = resolve(baseDir, 'league.json')
    try {
      sendBuffer(res, Buffer.from(readLeagueData(leaguePath)), 'application/json; charset=utf-8')
    } catch {
      sendJson(res, [])
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
    if (!matchId) {
      sendJson(res, { error: 'match_id required' }, 400)
      return
    }

    const replayPath = resolve(REPLAYS_DIR, `match_${matchId}.replay26`)
    const cambc = resolve(PROJECT_ROOT, '.venv/bin/cambc')
    const proc = spawn(cambc, ['watch', replayPath], {
      cwd: PROJECT_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
    })

    let responded = false

    function respondWithUrl(chunk) {
      const match = chunk.toString().match(/https?:\/\/[^\s]+/)
      if (match && !responded) {
        responded = true
        sendJson(res, { url: match[0] })
      }
    }

    proc.stdout.on('data', respondWithUrl)
    proc.stderr.on('data', respondWithUrl)

    proc.on('error', () => {
      if (!responded) {
        responded = true
        sendJson(res, { error: 'Failed to launch cambc watch' }, 500)
      }
    })

    setTimeout(() => {
      if (!responded) {
        responded = true
        sendJson(res, { error: 'Timeout waiting for visualizer' }, 504)
      }
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

export function startDashboardServer({ host = '127.0.0.1', port = 5173 } = {}) {
  const server = createDashboardServer()
  return new Promise((resolveServer, reject) => {
    server.once('error', reject)
    server.listen(port, host, () => resolveServer(server))
  })
}
