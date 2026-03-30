import { createContext, useContext } from 'react'

export const CardSettingsCtx = createContext({ ema: false, bands: false, legend: false })

export function useCardSettings() {
  return useContext(CardSettingsCtx)
}
