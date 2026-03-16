import { useEffect, useCallback, useRef } from 'react'
import { useAtom } from 'jotai'
import { viewerAtom, viewerMinimizedAtom } from '../store'

export default function GameViewer() {
  const [viewer, setViewer] = useAtom(viewerAtom)
  const [minimized, setMinimized] = useAtom(viewerMinimizedAtom)
  const iframeRef = useRef(null)

  const close = useCallback(() => { setViewer(null); setMinimized(false) }, [setViewer, setMinimized])

  useEffect(() => {
    if (!viewer || minimized) return
    function onKey(e) {
      if (e.key === 'Escape') close()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [viewer, minimized, close])

  // Attach Escape listener to iframe contentWindow (same-origin)
  function handleIframeLoad() {
    try {
      const iframeWin = iframeRef.current?.contentWindow
      if (iframeWin) {
        iframeWin.addEventListener('keydown', (e) => {
          if (e.key === 'Escape') close()
        })
      }
    } catch {
      // cross-origin — ignore
    }
  }

  if (!viewer) return null

  if (minimized) {
    return (
      <div className="fixed bottom-4 left-4 z-50 flex items-center gap-2 bg-mantle border border-surface0 rounded-lg shadow-lg px-3 py-2">
        <span className="text-[12px] font-medium text-text truncate max-w-48">
          {viewer.title}
        </span>
        <button
          onClick={() => setMinimized(false)}
          className="p-1 rounded hover:bg-surface0/60 text-overlay1 hover:text-text transition-colors"
          title="Restore"
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="2" y="2" width="12" height="12" rx="1" />
          </svg>
        </button>
        <button
          onClick={() => { setViewer(null); setMinimized(false) }}
          className="p-1 rounded hover:bg-red/10 text-overlay1 hover:text-red transition-colors"
          title="Close"
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 4l8 8M12 4l-8 8" />
          </svg>
        </button>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-crust/95">
      <div className="flex items-center justify-between px-4 py-2 bg-mantle border-b border-surface0">
        <span className="text-[13px] font-semibold text-text">{viewer.title}</span>
        <div className="flex items-center gap-1">
          <a
            href={viewer.url}
            target="_blank"
            rel="noopener noreferrer"
            className="p-1.5 rounded hover:bg-surface0/60 text-overlay1 hover:text-text transition-colors"
            title="Open in new tab"
          >
            <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M6 2H3a1 1 0 00-1 1v10a1 1 0 001 1h10a1 1 0 001-1v-3" />
              <path d="M9 2h5v5" />
              <path d="M14 2L7 9" />
            </svg>
          </a>
          <button
            onClick={() => setMinimized(true)}
            className="p-1.5 rounded hover:bg-surface0/60 text-overlay1 hover:text-text transition-colors"
            title="Minimize"
          >
            <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 12h10" />
            </svg>
          </button>
          <button
            onClick={close}
            className="p-1.5 rounded hover:bg-red/10 text-overlay1 hover:text-red transition-colors"
            title="Close"
          >
            <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 4l8 8M12 4l-8 8" />
            </svg>
          </button>
        </div>
      </div>
      <div className="flex-1 relative">
        <iframe
          ref={iframeRef}
          key={viewer.url}
          src={viewer.url}
          className="absolute inset-0 w-full h-full"
          allow="fullscreen"
          onLoad={handleIframeLoad}
        />
      </div>
    </div>
  )
}
