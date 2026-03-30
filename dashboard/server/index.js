import { startDashboardServer } from './app.js'

const host = process.env.DASHBOARD_HOST || '127.0.0.1'
const port = Number.parseInt(process.env.DASHBOARD_PORT || '5174', 10)

startDashboardServer({ host, port })
  .then(() => {
    console.log(`dashboard backend listening on http://${host}:${port}`)
  })
  .catch((error) => {
    console.error(error)
    process.exit(1)
  })
