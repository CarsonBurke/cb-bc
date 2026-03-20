import { useMemo, useRef, createContext, useContext } from 'react'
import { useAtom } from 'jotai'
import { cardOpenAtom, cardSizeAtom, cardEmaAtom, cardBandsAtom } from '../store'

const SIZES = ['sm', 'md', 'lg', 'xl']
const SIZE_LABELS = { sm: 'S', md: 'M', lg: 'L', xl: 'XL' }
const SIZE_HEIGHTS = { sm: 150, md: null, lg: 450, xl: 780 }

const CardSettingsCtx = createContext({ ema: false, bands: false })
export function useCardSettings() { return useContext(CardSettingsCtx) }

export default function Card({ title, subtitle, children, className = '', defaultOpen = true, id, dragHandleProps, hasEma }) {
  const cardId = id || title || 'untitled'
  const openAtom = useMemo(() => cardOpenAtom(cardId, defaultOpen), [cardId, defaultOpen])
  const sizeAtom = useMemo(() => cardSizeAtom(cardId), [cardId])
  const emaAtom = useMemo(() => cardEmaAtom(cardId), [cardId])
  const bandsAtom = useMemo(() => cardBandsAtom(cardId), [cardId])
  const [open, setOpen] = useAtom(openAtom)
  const [size, setSize] = useAtom(sizeAtom)
  const [ema, setEma] = useAtom(emaAtom)
  const [bands, setBands] = useAtom(bandsAtom)
  const didDrag = useRef(false)

  const headerDragProps = dragHandleProps ? {
    draggable: true,
    onDragStart: (e) => {
      didDrag.current = true
      dragHandleProps.onDragStart?.(e)
    },
    onDragEnd: (e) => {
      dragHandleProps.onDragEnd?.(e)
      setTimeout(() => { didDrag.current = false }, 0)
    },
  } : {}

  const handleClick = () => {
    if (!didDrag.current) setOpen(!open)
  }

  const cycleSize = (e) => {
    e.stopPropagation()
    const idx = SIZES.indexOf(size)
    setSize(SIZES[(idx + 1) % SIZES.length])
  }

  const toggleEma = (e) => {
    e.stopPropagation()
    setEma(!ema)
  }

  const toggleBands = (e) => {
    e.stopPropagation()
    setBands(!bands)
  }

  return (
    <div className={`bg-base border border-surface0 mb-3 rounded-md overflow-hidden ${className}`}>
      {title && (
        <div
          {...headerDragProps}
          className={`flex items-center gap-2 px-5 py-3 border-surface0 transition-colors select-none ${
            open ? 'border-b' : ''
          } cursor-pointer hover:bg-surface0/30`}
          onClick={handleClick}
        >
          <svg
            className={`w-3 h-3 text-overlay0 transition-transform duration-150 shrink-0 ${open ? 'rotate-90' : ''}`}
            fill="currentColor"
            viewBox="0 0 8 10"
          >
            <path d="M0 0 L8 5 L0 10Z" />
          </svg>
          <span className="text-[13px] font-semibold text-text flex-1">
            {title}
            {subtitle && <span className="text-[11px] font-normal text-overlay0 ml-2">{subtitle}</span>}
          </span>
          {open && hasEma && (
            <>
              <button
                onClick={toggleBands}
                className={`px-1.5 py-0.5 rounded text-[10px] font-semibold transition-colors ${
                  bands ? 'text-blue bg-blue/10' : 'text-overlay0 hover:text-subtext0 hover:bg-surface0/50'
                }`}
                title={bands ? 'P5/P95 viewport on (click to disable)' : 'Scale viewport to current P5-P95'}
              >
                <svg className="w-3.5 h-3" viewBox="0 0 16 12" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round">
                  <path d="M1 3h14" strokeDasharray="2 2" />
                  <path d="M1 6h14" />
                  <path d="M1 9h14" strokeDasharray="2 2" />
                </svg>
              </button>
              <button
                onClick={toggleEma}
                className={`px-1.5 py-0.5 rounded text-[10px] font-semibold transition-colors ${
                  ema ? 'text-blue bg-blue/10' : 'text-overlay0 hover:text-subtext0 hover:bg-surface0/50'
                }`}
                title={ema ? 'EMA smoothing on (click to disable)' : 'Enable EMA smoothing'}
              >
                <svg className="w-3.5 h-3" viewBox="0 0 16 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                  <path d="M1 10C3 10 4 2 8 2S13 10 15 10" />
                </svg>
              </button>
            </>
          )}
          {open && (
            <button
              onClick={cycleSize}
              className="px-1.5 py-0.5 rounded text-[10px] font-semibold text-overlay0 hover:text-subtext0 hover:bg-surface0/50 transition-colors"
              title={`Size: ${SIZE_LABELS[size]} (click to cycle)`}
            >
              {SIZE_LABELS[size]}
            </button>
          )}
          {dragHandleProps && (
            <svg className="w-2.5 h-3.5 text-overlay0/30 shrink-0" viewBox="0 0 6 10" fill="currentColor">
              <circle cx="1.2" cy="1.5" r="0.9" />
              <circle cx="4.8" cy="1.5" r="0.9" />
              <circle cx="1.2" cy="5" r="0.9" />
              <circle cx="4.8" cy="5" r="0.9" />
              <circle cx="1.2" cy="8.5" r="0.9" />
              <circle cx="4.8" cy="8.5" r="0.9" />
            </svg>
          )}
        </div>
      )}
      <div
        className={`transition-all duration-150 ease-out overflow-hidden ${
          open ? 'max-h-[2000px] opacity-100' : 'max-h-0 opacity-0'
        }`}
      >
        <CardSettingsCtx.Provider value={{ ema, bands }}>
          <div
            className={`${title ? 'px-5 pt-4 pb-5' : 'p-5'} ${
              SIZE_HEIGHTS[size] ? 'flex flex-col overflow-hidden [&>div]:flex-1 [&>div]:min-h-0 [&>div]:max-h-none' : ''
            }`}
            style={SIZE_HEIGHTS[size] ? { height: SIZE_HEIGHTS[size] + 'px' } : undefined}
          >
            {children}
          </div>
        </CardSettingsCtx.Provider>
      </div>
    </div>
  )
}
