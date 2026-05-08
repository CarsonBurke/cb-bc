import { spawn } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const SERVER_ROOT = dirname(fileURLToPath(import.meta.url))
const DASHBOARD_ROOT = resolve(SERVER_ROOT, '..')

const backend = spawn(process.execPath, [resolve(SERVER_ROOT, 'index.js')], {
  cwd: DASHBOARD_ROOT,
  stdio: 'inherit',
})

const vite = spawn(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', 'dev:web'], {
  cwd: DASHBOARD_ROOT,
  stdio: 'inherit',
})

let shuttingDown = false

function shutdown(code = 0) {
  if (shuttingDown) return
  shuttingDown = true
  backend.kill('SIGTERM')
  vite.kill('SIGTERM')
  setTimeout(() => process.exit(code), 50)
}

backend.on('exit', (code) => shutdown(code || 0))
vite.on('exit', (code) => shutdown(code || 0))

process.on('SIGINT', () => shutdown(0))
process.on('SIGTERM', () => shutdown(0))
