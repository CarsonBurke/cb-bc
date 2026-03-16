import { atom } from 'jotai'

// Persisted sidebar mode (code vs tensor)
const MODE_KEY = 'dashboard-mode'
const MODE_DEFAULTS = { code: 'all', tensor: 'training' }

function loadMode() {
  try { return localStorage.getItem(MODE_KEY) || 'code' } catch { return 'code' }
}

export const modeAtom = atom(loadMode())

export const modePersistAtom = atom(
  (get) => get(modeAtom),
  (get, set, next) => {
    set(modeAtom, next)
    try { localStorage.setItem(MODE_KEY, next) } catch {}
    // Reset view to mode default when switching
    const view = get(currentViewAtom)
    const codeViews = ['all', 'replays']
    const tensorViews = ['training']
    if (next === 'code' && !codeViews.includes(view) && typeof view !== 'number') {
      set(currentViewAtom, MODE_DEFAULTS.code)
      try { localStorage.setItem(VIEW_KEY, MODE_DEFAULTS.code) } catch {}
    } else if (next === 'tensor' && !tensorViews.includes(view)) {
      set(currentViewAtom, MODE_DEFAULTS.tensor)
      try { localStorage.setItem(VIEW_KEY, MODE_DEFAULTS.tensor) } catch {}
    }
  },
)

// Persisted sidebar tab selection
const VIEW_KEY = 'dashboard-current-view'

function loadView() {
  try { return localStorage.getItem(VIEW_KEY) || 'all' } catch { return 'all' }
}

export const currentViewAtom = atom(loadView())

export const currentViewPersistAtom = atom(
  (get) => get(currentViewAtom),
  (_get, set, next) => {
    set(currentViewAtom, next)
    try { localStorage.setItem(VIEW_KEY, next) } catch {}
  },
)

// Game viewer modal state: { url, title } | null, with minimized flag
export const viewerAtom = atom(null)
export const viewerMinimizedAtom = atom(false)

// Active training session (null = most recent)
const SESSION_KEY = 'dashboard-train-session'
function loadSession() {
  try { return localStorage.getItem(SESSION_KEY) || null } catch { return null }
}
export const trainSessionAtom = atom(loadSession())
export const trainSessionPersistAtom = atom(
  (get) => get(trainSessionAtom),
  (_get, set, next) => {
    set(trainSessionAtom, next)
    try {
      if (next) localStorage.setItem(SESSION_KEY, next)
      else localStorage.removeItem(SESSION_KEY)
    } catch {}
  },
)

// Persisted card collapse state: { [cardId]: boolean }
const STORAGE_KEY = 'dashboard-card-state'

function loadState() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}
  } catch {
    return {}
  }
}

export const cardStateAtom = atom(loadState())

export const cardOpenAtom = (id, defaultOpen) => atom(
  (get) => {
    const state = get(cardStateAtom)
    return id in state ? state[id] : defaultOpen
  },
  (get, set, next) => {
    const state = { ...get(cardStateAtom), [id]: next }
    set(cardStateAtom, state)
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)) } catch {}
  },
)
