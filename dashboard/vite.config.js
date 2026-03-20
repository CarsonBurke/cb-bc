import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { spawn, execSync } from 'child_process'
import { resolve } from 'path'
import { existsSync, readdirSync, readFileSync, statSync } from 'fs'

const PROJECT_ROOT = resolve(import.meta.dirname, '..')

// Find the bundled visualiser dist directory
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

function replayApiPlugin() {
  return {
    name: 'replay-api',
    configureServer(server) {
      const replaysDir = resolve(PROJECT_ROOT, 'replays')
      const sessionsDir = resolve(PROJECT_ROOT, 'bots/tensor/sessions')
      const _trainSessionLogCache = new Map() // path -> { mtime, size, episodes }
      const _trainLogCache = new Map() // path -> { mtime, size, json }
      const _leagueCache = new Map() // path -> { mtime, size, data }

      // List training sessions
      server.middlewares.use('/api/train-sessions', (_req, res) => {
        res.setHeader('Content-Type', 'application/json')
        try {
          const dirs = readdirSync(sessionsDir, { withFileTypes: true })
            .filter(d => d.isDirectory())
            .map(d => {
              const metaPath = resolve(sessionsDir, d.name, 'meta.json')
              let meta = { name: d.name }
              try { meta = JSON.parse(readFileSync(metaPath, 'utf8')) } catch {}
              let episodes = 0
              try {
                const jsonlPath = resolve(sessionsDir, d.name, 'train_log.jsonl')
                const jsonPath = resolve(sessionsDir, d.name, 'train_log.json')
                if (existsSync(jsonlPath)) {
                  const st = statSync(jsonlPath)
                  const cached = _trainSessionLogCache.get(jsonlPath)
                  if (cached && cached.mtime === st.mtimeMs && cached.size === st.size) {
                    episodes = cached.episodes
                  } else {
                    const lines = readFileSync(jsonlPath, 'utf8').trimEnd().split('\n')
                    episodes = Math.max(0, lines.length - 1) // first line is config
                    _trainSessionLogCache.set(jsonlPath, { mtime: st.mtimeMs, size: st.size, episodes })
                  }
                } else {
                  const log = JSON.parse(readFileSync(jsonPath, 'utf8'))
                  episodes = log.episodes?.length || 0
                }
              } catch {}
              const st = statSync(resolve(sessionsDir, d.name))
              return { id: d.name, ...meta, episodes, mtime: st.mtimeMs }
            })
            .sort((a, b) => b.mtime - a.mtime)
          res.end(JSON.stringify(dirs))
        } catch {
          res.end(JSON.stringify([]))
        }
      })

      // Helper: resolve session directory from query param
      function sessionDir(req) {
        const url = new URL(req.url, 'http://localhost')
        const session = url.searchParams.get('session')
        if (session && !session.includes('..')) {
          return resolve(sessionsDir, session)
        }
        return null
      }

      // Training replays (session-aware)
      const _replayCache = new Map() // dir -> { mtime, files: Map<name, entry>, json }
      server.middlewares.use('/api/train-replays', (req, res) => {
        res.setHeader('Content-Type', 'application/json')
        const sd = sessionDir(req)
        const dir = sd ? resolve(sd, 'replays') : resolve(PROJECT_ROOT, 'replays/training')
        try {
          const dirMtime = statSync(dir).mtimeMs
          const cached = _replayCache.get(dir)
          if (cached && cached.mtime === dirMtime) {
            res.end(cached.json)
            return
          }
          // Incremental: only stat new files
          const known = cached ? cached.files : new Map()
          const currentNames = new Set(
            readdirSync(dir).filter(f => f.endsWith('.replay26'))
          )
          // Stat only new files
          for (const f of currentNames) {
            if (!known.has(f)) {
              const st = statSync(resolve(dir, f))
              known.set(f, { name: f, size: st.size, mtime: st.mtimeMs })
            }
          }
          // Remove deleted files
          for (const f of known.keys()) {
            if (!currentNames.has(f)) known.delete(f)
          }
          const files = [...known.values()].sort((a, b) => b.mtime - a.mtime)
          const json = JSON.stringify(files)
          _replayCache.set(dir, { mtime: dirMtime, files: known, json })
          res.end(json)
        } catch {
          res.end(JSON.stringify([]))
        }
      })

      // Serve training replay files (session-aware)
      server.middlewares.use('/api/train-replay/', (req, res) => {
        const url = new URL(req.url, 'http://localhost')
        const session = url.searchParams.get('session')
        const filename = decodeURIComponent(url.pathname.replace(/^\//, ''))
        if (!filename || filename.includes('..')) {
          res.statusCode = 400
          res.end('Bad request')
          return
        }
        let dir
        if (session && !session.includes('..')) {
          dir = resolve(sessionsDir, session, 'replays')
        } else {
          dir = resolve(PROJECT_ROOT, 'replays/training')
        }
        try {
          const data = readFileSync(resolve(dir, filename))
          res.setHeader('Content-Type', 'application/octet-stream')
          res.setHeader('Content-Length', data.length)
          res.end(data)
        } catch {
          res.statusCode = 404
          res.end('Not found')
        }
      })

      // Training log (session-aware) — reads JSONL (line 1 = config, rest = episodes)
      server.middlewares.use('/api/train-log', (req, res) => {
        res.setHeader('Content-Type', 'application/json')
        const sd = sessionDir(req)
        const base = sd || resolve(PROJECT_ROOT, 'bots/tensor')
        const jsonlPath = resolve(base, 'train_log.jsonl')
        const jsonPath = resolve(base, 'train_log.json')
        try {
          if (existsSync(jsonlPath)) {
            const st = statSync(jsonlPath)
            const cached = _trainLogCache.get(jsonlPath)
            if (cached && cached.mtime === st.mtimeMs && cached.size === st.size) {
              res.end(cached.json)
              return
            }
            const lines = readFileSync(jsonlPath, 'utf8').trimEnd().split('\n')
            const first = JSON.parse(lines[0])
            const config = first.config || {}
            const episodes = lines.slice(1).map(l => JSON.parse(l))
            const json = JSON.stringify({ episodes, config })
            _trainLogCache.set(jsonlPath, { mtime: st.mtimeMs, size: st.size, json })
            res.end(json)
          } else {
            // Fallback: legacy single-JSON format
            res.end(readFileSync(jsonPath, 'utf8'))
          }
        } catch {
          res.end(JSON.stringify({ episodes: [], config: {} }))
        }
      })

      // Game log (session-aware) — JSONL, one line per game
      const _gameLogCache = new Map() // path -> { mtime, json }
      server.middlewares.use('/api/game-log', (req, res) => {
        res.setHeader('Content-Type', 'application/json')
        const sd = sessionDir(req)
        const base = sd || resolve(PROJECT_ROOT, 'bots/tensor')
        const logPath = resolve(base, 'game_log.jsonl')
        try {
          const mtime = statSync(logPath).mtimeMs
          const cached = _gameLogCache.get(logPath)
          if (cached && cached.mtime === mtime) {
            res.end(cached.json)
            return
          }
          const lines = readFileSync(logPath, 'utf8').trimEnd().split('\n')
          const games = lines.map(l => JSON.parse(l))
          const json = JSON.stringify(games)
          _gameLogCache.set(logPath, { mtime, json })
          res.end(json)
        } catch {
          res.end(JSON.stringify([]))
        }
      })

      // League data (session-aware)
      server.middlewares.use('/api/league', (req, res) => {
        res.setHeader('Content-Type', 'application/json')
        const sd = sessionDir(req)
        const leaguePath = sd
          ? resolve(sd, 'league', 'league.json')
          : resolve(PROJECT_ROOT, 'bots/tensor/league/league.json')
        try {
          const st = statSync(leaguePath)
          const cached = _leagueCache.get(leaguePath)
          const data = (cached && cached.mtime === st.mtimeMs && cached.size === st.size)
            ? cached.data
            : readFileSync(leaguePath, 'utf8')
          if (!cached || cached.mtime !== st.mtimeMs || cached.size !== st.size) {
            _leagueCache.set(leaguePath, { mtime: st.mtimeMs, size: st.size, data })
          }
          res.end(data)
        } catch {
          res.end(JSON.stringify([]))
        }
      })

      // List available replay files
      server.middlewares.use('/api/replays', (_req, res) => {
        try {
          const files = readdirSync(replaysDir)
            .filter(f => f.endsWith('.replay26'))
            .map(f => {
              const st = statSync(resolve(replaysDir, f))
              return { name: f, size: st.size, mtime: st.mtimeMs }
            })
            .sort((a, b) => {
              const na = parseInt(a.name.match(/\d+/)?.[0] ?? '0', 10)
              const nb = parseInt(b.name.match(/\d+/)?.[0] ?? '0', 10)
              return nb - na
            })
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify(files))
        } catch {
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify([]))
        }
      })

      // Serve a specific replay file
      server.middlewares.use('/api/replay/', (req, res) => {
        const filename = decodeURIComponent(req.url.replace(/^\//, '').split('?')[0])
        if (!filename || filename.includes('..')) {
          res.statusCode = 400
          res.end('Bad request')
          return
        }
        try {
          const data = readFileSync(resolve(replaysDir, filename))
          res.setHeader('Content-Type', 'application/octet-stream')
          res.setHeader('Content-Length', data.length)
          res.end(data)
        } catch {
          res.statusCode = 404
          res.end('Not found')
        }
      })

      // Serve the bundled visualiser static files
      // The visualiser uses absolute paths (/assets/..., /*.png) so we
      // serve it at /viewer/ with rewritten HTML, and also serve its
      // assets/images as a fallback for any path not handled by Vite.
      if (VISUALISER_DIST) {
        const visTypes = {
          html: 'text/html', js: 'application/javascript', css: 'text/css',
          wasm: 'application/wasm', png: 'image/png', svg: 'image/svg+xml',
          json: 'application/json', ico: 'image/x-icon', webp: 'image/webp',
        }

        server.middlewares.use('/viewer/', (req, res) => {
          const urlPath = req.url.split('?')[0].replace(/^\//, '') || 'index.html'
          try {
            let data = readFileSync(resolve(VISUALISER_DIST, urlPath))
            const ext = urlPath.split('.').pop()
            // Rewrite asset paths in HTML to point to /viewer/
            if (ext === 'html') {
              data = Buffer.from(
                data.toString()
                  .replace(/src="\/assets\//g, 'src="/viewer/assets/')
                  .replace(/href="\/assets\//g, 'href="/viewer/assets/')
              )
            }
            res.setHeader('Content-Type', visTypes[ext] || 'application/octet-stream')
            res.end(data)
          } catch {
            res.statusCode = 404
            res.end('Not found')
          }
        })

        // Fallback: serve visualiser images/assets for paths the viewer JS
        // requests with absolute URLs (e.g. /base_gold.png, /bg.png)
        server.middlewares.use((req, res, next) => {
          const urlPath = req.url.split('?')[0].replace(/^\//, '')
          if (!urlPath || urlPath.startsWith('src/') || urlPath.startsWith('node_modules/') || urlPath.startsWith('@')) {
            return next()
          }
          const filePath = resolve(VISUALISER_DIST, urlPath)
          try {
            const data = readFileSync(filePath)
            const ext = urlPath.split('.').pop()
            if (visTypes[ext]) {
              res.setHeader('Content-Type', visTypes[ext])
              res.end(data)
              return
            }
          } catch {
            // Not a visualiser file, pass through
          }
          next()
        })
      }

      // Legacy: launch external cambc watch
      server.middlewares.use('/api/watch', (req, res) => {
        const url = new URL(req.url, 'http://localhost')
        const matchId = url.searchParams.get('match_id')
        if (matchId == null) {
          res.statusCode = 400
          res.end(JSON.stringify({ error: 'match_id required' }))
          return
        }

        const replayPath = resolve(replaysDir, `match_${matchId}.replay26`)
        const cambc = resolve(PROJECT_ROOT, '.venv/bin/cambc')

        const proc = spawn(cambc, ['watch', replayPath], {
          cwd: PROJECT_ROOT,
          stdio: ['ignore', 'pipe', 'pipe'],
        })

        let responded = false

        function sendUrl(data) {
          const text = data.toString()
          const match = text.match(/https?:\/\/[^\s]+/)
          if (match && !responded) {
            responded = true
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ url: match[0] }))
          }
        }

        proc.stdout.on('data', sendUrl)
        proc.stderr.on('data', sendUrl)

        proc.on('error', () => {
          if (!responded) {
            responded = true
            res.statusCode = 500
            res.end(JSON.stringify({ error: 'Failed to launch cambc watch' }))
          }
        })

        setTimeout(() => {
          if (!responded) {
            responded = true
            res.statusCode = 504
            res.end(JSON.stringify({ error: 'Timeout waiting for visualizer' }))
          }
        }, 5000)
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), replayApiPlugin()],
})
