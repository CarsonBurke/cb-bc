import historyData from '../../ablations/history.json'

export const DATA = historyData

export const BOT_COLORS = [
  '#89b4fa', '#a6e3a1', '#f9e2af', '#f38ba8', '#cba6f7',
  '#f5c2e7', '#74c7ec', '#94e2d5', '#fab387', '#eba0ac',
  '#b4befe', '#89dceb', '#f2cdcd', '#f5e0dc', '#585b70',
]

export function botColor(index) {
  return BOT_COLORS[index % BOT_COLORS.length]
}

export function getAllBots() {
  return [...new Set(DATA.rounds.flatMap(r => Object.keys(r.bots)))]
}

export function getLatestRound() {
  return DATA.rounds[DATA.rounds.length - 1] ?? null
}

export function formatDate(timestamp, opts) {
  return new Date(timestamp).toLocaleDateString('en-US', opts)
}

// Chart theme constants (Catppuccin Mocha)
export const CHART_THEME = {
  text: '#cdd6f4',
  subtext: '#a6adc8',
  grid: '#313244',
  surface: '#181825',
  border: '#45475a',
}
