import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { spawn } from 'child_process'
import { resolve } from 'path'

const PROJECT_ROOT = resolve(import.meta.dirname, '..')

function replayApiPlugin() {
  return {
    name: 'replay-api',
    configureServer(server) {
      server.middlewares.use('/api/watch', (req, res) => {
        const url = new URL(req.url, 'http://localhost')
        const matchId = url.searchParams.get('match_id')
        if (matchId == null) {
          res.statusCode = 400
          res.end(JSON.stringify({ error: 'match_id required' }))
          return
        }

        const replayPath = resolve(PROJECT_ROOT, 'replays', `match_${matchId}.replay26`)
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

        // Timeout after 5s
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
