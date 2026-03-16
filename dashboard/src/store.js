import { atom } from 'jotai'

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
