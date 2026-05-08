export const BOT_COLORS = [
  '#89b4fa', '#a6e3a1', '#f9e2af', '#f38ba8', '#cba6f7',
  '#f5c2e7', '#74c7ec', '#94e2d5', '#fab387', '#eba0ac',
  '#b4befe', '#89dceb', '#f2cdcd', '#f5e0dc', '#585b70',
]

export function botColor(index) {
  return BOT_COLORS[index % BOT_COLORS.length]
}

import { formatDistanceToNow } from 'date-fns'

export function formatDate(timestamp, opts) {
  return new Date(timestamp).toLocaleDateString('en-US', opts)
}

export function relativeTime(ms) {
  return formatDistanceToNow(new Date(ms), { addSuffix: true })
}

// Chart theme constants (Catppuccin Mocha)
export const CHART_THEME = {
  text: '#cdd6f4',
  subtext: '#a6adc8',
  grid: '#313244',
  surface: '#181825',
  border: '#45475a',
}
