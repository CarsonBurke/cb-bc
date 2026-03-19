import { atom } from 'jotai'

// ---- URL-based routing ----
// Hash format: #/{mode}/{view}?session={id}
// Examples: #/code/all, #/tensor/training?session=sp_20260318, #/code/3

const MODE_DEFAULTS = { code: 'all', tensor: 'training' }

function parseHash() {
  const hash = location.hash.replace(/^#\/?/, '')
  const [path, qs] = hash.split('?')
  const parts = path.split('/')
  const mode = (parts[0] === 'code' || parts[0] === 'tensor') ? parts[0] : 'code'
  let view = parts[1] || MODE_DEFAULTS[mode]
  // Numeric round IDs
  if (/^\d+$/.test(view)) view = parseInt(view, 10)
  const params = new URLSearchParams(qs || '')
  const session = params.get('session') || null
  return { mode, view, session }
}

function buildHash(mode, view, session) {
  let hash = `#/${mode}/${view}`
  if (session) hash += `?session=${encodeURIComponent(session)}`
  return hash
}

function setHash(mode, view, session) {
  const hash = buildHash(mode, view, session)
  if (location.hash !== hash) {
    history.pushState(null, '', hash)
  }
}

const initial = parseHash()

// Core atoms (internal)
export const modeAtom = atom(initial.mode)
export const currentViewAtom = atom(initial.view)
export const trainSessionAtom = atom(initial.session)

// Persist atoms that sync to URL
export const modePersistAtom = atom(
  (get) => get(modeAtom),
  (get, set, next) => {
    set(modeAtom, next)
    const view = MODE_DEFAULTS[next]
    set(currentViewAtom, view)
    setHash(next, view, next === 'tensor' ? get(trainSessionAtom) : null)
  },
)

export const currentViewPersistAtom = atom(
  (get) => get(currentViewAtom),
  (get, set, next) => {
    set(currentViewAtom, next)
    const mode = get(modeAtom)
    setHash(mode, next, mode === 'tensor' ? get(trainSessionAtom) : null)
  },
)

export const trainSessionPersistAtom = atom(
  (get) => get(trainSessionAtom),
  (get, set, next) => {
    set(trainSessionAtom, next)
    setHash(get(modeAtom), get(currentViewAtom), next)
  },
)

// Listen for back/forward navigation
if (typeof window !== 'undefined') {
  window.addEventListener('popstate', () => {
    // Atoms won't auto-update from this — the App component handles it
  })
}

// Export a hook-friendly updater for popstate
export const syncFromHash = (set) => {
  const { mode, view, session } = parseHash()
  set(modeAtom, mode)
  set(currentViewAtom, view)
  set(trainSessionAtom, session)
}

// Game viewer modal state: { url, title } | null, with minimized flag
export const viewerAtom = atom(null)
export const viewerMinimizedAtom = atom(false)

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

// Persisted card size state: { [cardId]: 'sm' | 'md' | 'lg' | 'xl' }
const SIZE_KEY = 'dashboard-card-sizes'

function loadSizes() {
  try {
    return JSON.parse(localStorage.getItem(SIZE_KEY)) || {}
  } catch {
    return {}
  }
}

export const cardSizeStateAtom = atom(loadSizes())

export const cardSizeAtom = (id) => atom(
  (get) => get(cardSizeStateAtom)[id] || 'md',
  (get, set, next) => {
    const state = { ...get(cardSizeStateAtom), [id]: next }
    set(cardSizeStateAtom, state)
    try { localStorage.setItem(SIZE_KEY, JSON.stringify(state)) } catch {}
  },
)

// Persisted card EMA toggle: { [cardId]: boolean }
const EMA_KEY = 'dashboard-card-ema'

function loadEma() {
  try {
    return JSON.parse(localStorage.getItem(EMA_KEY)) || {}
  } catch {
    return {}
  }
}

export const cardEmaStateAtom = atom(loadEma())

export const cardEmaAtom = (id) => atom(
  (get) => !!get(cardEmaStateAtom)[id],
  (get, set, next) => {
    const state = { ...get(cardEmaStateAtom), [id]: next }
    set(cardEmaStateAtom, state)
    try { localStorage.setItem(EMA_KEY, JSON.stringify(state)) } catch {}
  },
)
