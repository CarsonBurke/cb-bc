import { useMemo } from 'react'
import { useAtom } from 'jotai'
import { cardOpenAtom } from '../store'

export default function Card({ title, children, className = '', defaultOpen = true, id }) {
  const cardId = id || title || 'untitled'
  const openAtom = useMemo(() => cardOpenAtom(cardId, defaultOpen), [cardId, defaultOpen])
  const [open, setOpen] = useAtom(openAtom)

  return (
    <div className={`bg-base border border-surface0 mb-4 ${className}`}>
      {title && (
        <button
          onClick={() => setOpen(!open)}
          className="flex items-center gap-2 w-full text-left px-5 py-3 hover:bg-surface0/30 transition-colors"
        >
          <svg
            className={`w-3 h-3 text-overlay0 transition-transform ${open ? 'rotate-90' : ''}`}
            fill="currentColor"
            viewBox="0 0 8 10"
          >
            <path d="M0 0 L8 5 L0 10Z" />
          </svg>
          <span className="text-[13px] font-semibold text-text">{title}</span>
        </button>
      )}
      {open && (
        <div className={title ? 'px-5 pb-5' : 'p-5'}>
          {children}
        </div>
      )}
    </div>
  )
}
