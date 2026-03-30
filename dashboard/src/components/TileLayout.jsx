import { createContext, useContext, useState, useCallback, useRef, useEffect, useMemo, memo } from 'react'
import Card from './Card'

// ---- Tree helpers (pure) ----

function transformTree(node, leafId, fn) {
  if (node.type === 'leaf') return node.id === leafId ? fn(node) : node
  const children = node.children.map(c => transformTree(c, leafId, fn))
  return { ...node, children }
}

function splitLeaf(root, leafId, direction, newId, splitId) {
  return transformTree(root, leafId, leaf => ({
    type: 'split', direction, splitId, ratio: 0.5,
    children: [leaf, { type: 'leaf', id: newId, cards: [] }],
  }))
}

function transformSplit(node, splitId, fn) {
  if (node.type === 'split') {
    if (node.splitId === splitId) return fn(node)
    const children = node.children.map(c => transformSplit(c, splitId, fn))
    return { ...node, children }
  }
  return node
}

function resizeSplit(root, splitId, ratio) {
  return transformSplit(root, splitId, node => ({ ...node, ratio: Math.max(0.15, Math.min(0.85, ratio)) }))
}

function mergeIntoFirst(node, cards) {
  if (!cards.length) return node
  if (node.type === 'leaf') return { ...node, cards: [...node.cards, ...cards] }
  return { ...node, children: [mergeIntoFirst(node.children[0], cards), node.children[1]] }
}

function closeLeaf(root, leafId) {
  if (root.type === 'leaf') return root
  for (let i = 0; i < 2; i++) {
    if (root.children[i].type === 'leaf' && root.children[i].id === leafId) {
      const orphans = root.children[i].cards
      return mergeIntoFirst(root.children[1 - i], orphans)
    }
  }
  return { ...root, children: root.children.map(c => closeLeaf(c, leafId)) }
}

function moveCard(root, cardId, fromId, toId, toIndex = -1) {
  let r = transformTree(root, fromId, leaf => ({
    ...leaf, cards: leaf.cards.filter(c => c !== cardId),
  }))
  r = transformTree(r, toId, leaf => {
    const cards = [...leaf.cards]
    cards.splice(toIndex >= 0 ? toIndex : cards.length, 0, cardId)
    return { ...leaf, cards }
  })
  return r
}

function reorderCard(root, leafId, cardId, toIndex) {
  return transformTree(root, leafId, leaf => {
    const cards = leaf.cards.filter(c => c !== cardId)
    cards.splice(toIndex, 0, cardId)
    return { ...leaf, cards }
  })
}

function allLeafCards(node) {
  if (node.type === 'leaf') return node.cards
  return node.children.flatMap(allLeafCards)
}

function firstLeafId(node) {
  if (node.type === 'leaf') return node.id
  return firstLeafId(node.children[0])
}

function removeCards(node, toRemove) {
  if (node.type === 'leaf') return { ...node, cards: node.cards.filter(c => !toRemove.has(c)) }
  return { ...node, children: node.children.map(c => removeCards(c, toRemove)) }
}

function syncLayout(root, registry) {
  const known = new Set(Object.keys(registry))
  const placed = new Set(allLeafCards(root))
  const missing = [...known].filter(id => !placed.has(id))
  const extra = [...placed].filter(id => !known.has(id))
  if (!extra.length && !missing.length) return root
  let cleaned = extra.length ? removeCards(root, new Set(extra)) : root
  if (missing.length) {
    const fid = firstLeafId(cleaned)
    cleaned = transformTree(cleaned, fid, leaf => ({
      ...leaf, cards: [...leaf.cards, ...missing],
    }))
  }
  return cleaned
}

// ---- Persistence ----

function usePersisted(key, defaultValue) {
  const [state, _setState] = useState(() => {
    try {
      const saved = localStorage.getItem(key)
      return saved ? JSON.parse(saved) : defaultValue
    } catch { return defaultValue }
  })
  const ref = useRef(state)

  useEffect(() => {
    ref.current = state
  }, [state])

  const setState = useCallback((updater) => {
    const next = typeof updater === 'function' ? updater(ref.current) : updater
    ref.current = next
    _setState(next)
  }, [])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        localStorage.setItem(key, JSON.stringify(state))
      } catch (err) {
        void err
      }
    }, 150)
    return () => window.clearTimeout(timer)
  }, [key, state])

  return [state, setState]
}

// ---- Context ----

const TileCtx = createContext(null)

// ---- Main component ----

export default function TileLayout({ cards, storageKey, defaultLayout }) {
  const [state, setState] = usePersisted(storageKey, defaultLayout)
  const nextIdRef = useRef(state.nextId || 1)
  const [dragCardId, setDragCardId] = useState(null)

  // Sync card registry with persisted layout
  const root = useMemo(() => syncLayout(state.root, cards), [state.root, cards])
  useEffect(() => {
    if (root !== state.root) {
      setState(s => ({ ...s, root: syncLayout(s.root, cards) }))
    }
  }, [cards, root, setState, state.root])

  const isOnly = root.type === 'leaf'

  const split = useCallback((leafId, direction) => {
      const newId = `v${nextIdRef.current++}`
      const sId = `s${nextIdRef.current++}`
      setState(s => ({
        root: splitLeaf(s.root, leafId, direction, newId, sId),
        nextId: nextIdRef.current,
      }))
    }, [setState])

  const resize = useCallback((splitId, ratio) => {
      setState(s => ({ ...s, root: resizeSplit(s.root, splitId, ratio) }))
    }, [setState])

  const close = useCallback((leafId) => {
      setState(s => ({ ...s, root: closeLeaf(s.root, leafId) }))
    }, [setState])

  const move = useCallback((cardId, fromId, toId, toIndex) => {
      setState(s => ({
        ...s,
        root: fromId === toId
          ? reorderCard(s.root, fromId, cardId, toIndex)
          : moveCard(s.root, cardId, fromId, toId, toIndex),
      }))
    }, [setState])

  const reset = useCallback(() => {
      nextIdRef.current = defaultLayout.nextId || 1
      setState(defaultLayout)
    }, [setState, defaultLayout])

  const actions = useMemo(() => ({ split, resize, close, move, reset }), [split, resize, close, move, reset])
  const ctxValue = useMemo(
    () => ({ cards, actions, isOnly, dragCardId, setDragCardId }),
    [cards, actions, isOnly, dragCardId],
  )

  return (
    <TileCtx.Provider value={ctxValue}>
      <div className="h-full">
        <TileNode node={root} />
      </div>
    </TileCtx.Provider>
  )
}

// ---- Recursive node renderer ----

const TileNode = memo(function TileNode({ node }) {
  if (node.type === 'split') {
    const isH = node.direction === 'h'
    const ratio = node.ratio ?? 0.5
    const pctA = (ratio * 100).toFixed(2) + '%'
    const pctB = ((1 - ratio) * 100).toFixed(2) + '%'
    return (
      <div className={`flex ${isH ? 'flex-row' : 'flex-col'} h-full`}>
        <div className="min-w-0 min-h-0 overflow-hidden" style={isH ? { width: pctA } : { height: pctA }}>
          <TileNode node={node.children[0]} />
        </div>
        <ResizeHandle splitId={node.splitId} direction={node.direction} />
        <div className="min-w-0 min-h-0 overflow-hidden" style={isH ? { width: pctB } : { height: pctB }}>
          <TileNode node={node.children[1]} />
        </div>
      </div>
    )
  }
  return <TileLeaf node={node} />
})

// ---- Viewport (leaf) ----

function TileLeaf({ node }) {
  const { cards, actions, isOnly, dragCardId, setDragCardId } = useContext(TileCtx)
  const [dropTarget, setDropTarget] = useState(null) // { cardId, pos } or 'empty'
  const leafRef = useRef()

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    const cardId = e.dataTransfer.getData('text/card-id')
    const fromLeaf = e.dataTransfer.getData('text/from-leaf')
    if (!cardId) return
    setDropTarget(null)
    setDragCardId(null)

    if (dropTarget && dropTarget !== 'empty' && dropTarget.cardId) {
      const targetIdx = node.cards.indexOf(dropTarget.cardId)
      const insertIdx = dropTarget.pos === 'after' ? targetIdx + 1 : targetIdx
      actions.move(cardId, fromLeaf, node.id, insertIdx)
    } else {
      actions.move(cardId, fromLeaf, node.id, node.cards.length)
    }
  }, [actions, node, dropTarget, setDragCardId])

  const handleEmptyDragOver = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDropTarget('empty')
  }, [])

  // Check if any visible cards exist
  const visibleCards = node.cards.filter(id => cards[id])

  return (
    <div
      ref={leafRef}
      className={`h-full flex flex-col relative group/leaf rounded-lg border transition-colors duration-150 ${
        dropTarget === 'empty' ? 'border-blue/40 bg-blue/5' : 'border-surface0/40 bg-mantle/30'
      }`}
      onDragOver={handleDragOver}
      onDragLeave={(e) => {
        if (!leafRef.current?.contains(e.relatedTarget)) setDropTarget(null)
      }}
      onDrop={handleDrop}
    >
      {/* Close button — top right, visible on hover */}
      {!isOnly && (
        <button
          onClick={() => actions.close(node.id)}
          className="absolute top-1.5 right-1.5 z-20 w-5 h-5 flex items-center justify-center rounded-md
            text-overlay0 hover:text-red hover:bg-red/10 bg-mantle/80
            opacity-0 group-hover/leaf:opacity-100 transition-all duration-150"
          title="Close viewport"
        >
          <svg className="w-2.5 h-2.5" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M2 2l6 6M8 2l-6 6" />
          </svg>
        </button>
      )}

      {/* Scrollable card area */}
      <div className="flex-1 overflow-y-auto p-2 min-h-0">
        {visibleCards.length === 0 ? (
          <div
            className="h-full flex items-center justify-center text-overlay0/60 text-[12px]"
            onDragOver={handleEmptyDragOver}
          >
            {dragCardId ? 'Drop here' : 'Drag cards here'}
          </div>
        ) : (
          visibleCards.map((cardId) => {
            const card = cards[cardId]
            return (
              <CardSlot
                key={cardId}
                cardId={cardId}
                leafId={node.id}
                card={card}
                isDragTarget={dropTarget?.cardId === cardId}
                dragPos={dropTarget?.pos}
                onDragOverCard={(pos) => setDropTarget({ cardId, pos })}
                onDragLeaveCard={() => {
                  setDropTarget(t => t?.cardId === cardId ? null : t)
                }}
              />
            )
          })
        )}
      </div>

      {/* Split buttons — edge strips */}
      <SplitEdge
        direction="v"
        onClick={() => actions.split(node.id, 'v')}
        className="absolute bottom-0 left-4 right-4 h-2 rounded-t-sm"
      />
      <SplitEdge
        direction="h"
        onClick={() => actions.split(node.id, 'h')}
        className="absolute top-4 right-0 bottom-4 w-2 rounded-l-sm"
      />
    </div>
  )
}

// ---- Draggable card with drop zone ----

function CardSlot({ cardId, leafId, card, isDragTarget, dragPos, onDragOverCard, onDragLeaveCard }) {
  const { setDragCardId } = useContext(TileCtx)
  const ref = useRef()
  const [dragging, setDragging] = useState(false)

  const dragHandleProps = {
    draggable: true,
    onDragStart: (e) => {
      e.dataTransfer.setData('text/card-id', cardId)
      e.dataTransfer.setData('text/from-leaf', leafId)
      e.dataTransfer.effectAllowed = 'move'
      setDragCardId(cardId)
      // Let browser capture drag image at full opacity, then dim
      requestAnimationFrame(() => setDragging(true))
    },
    onDragEnd: () => {
      setDragging(false)
      setDragCardId(null)
    },
  }

  return (
    <div
      ref={ref}
      className={`relative transition-opacity duration-150 ${dragging ? 'opacity-30' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        e.stopPropagation()
        const rect = ref.current.getBoundingClientRect()
        const pos = (e.clientY - rect.top) < rect.height / 2 ? 'before' : 'after'
        onDragOverCard(pos)
      }}
      onDragLeave={(e) => {
        if (!ref.current?.contains(e.relatedTarget)) onDragLeaveCard()
      }}
    >
      {/* Drop indicator — before */}
      <div className={`absolute top-0 left-3 right-3 h-0.5 rounded-full bg-blue z-10 transition-opacity duration-100 ${
        isDragTarget && dragPos === 'before' ? 'opacity-100' : 'opacity-0'
      }`} />

      <Card
        title={card.title}
        subtitle={card.subtitle}
        id={cardId}
        defaultOpen={card.defaultOpen ?? true}
        dragHandleProps={dragHandleProps}
        hasEma={card.hasEma}
        hasLegend={card.hasLegend}
        renderContent={card.render}
      />

      {/* Drop indicator — after */}
      <div className={`absolute bottom-0 left-3 right-3 h-0.5 rounded-full bg-blue z-10 transition-opacity duration-100 ${
        isDragTarget && dragPos === 'after' ? 'opacity-100' : 'opacity-0'
      }`} />
    </div>
  )
}

// ---- Resize handle between split panes ----

function ResizeHandle({ splitId, direction }) {
  const { actions } = useContext(TileCtx)
  const [active, setActive] = useState(false)
  const handleRef = useRef()
  const frameRef = useRef(0)
  const latestRatioRef = useRef(null)
  const isH = direction === 'h'

  const onPointerDown = useCallback((e) => {
    e.preventDefault()
    setActive(true)
    const parent = handleRef.current.parentElement
    const rect = parent.getBoundingClientRect()

    const onMove = (e) => {
      latestRatioRef.current = isH
        ? (e.clientX - rect.left) / rect.width
        : (e.clientY - rect.top) / rect.height
      if (frameRef.current) return
      frameRef.current = requestAnimationFrame(() => {
        frameRef.current = 0
        actions.resize(splitId, latestRatioRef.current)
      })
    }

    const onUp = () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current)
        frameRef.current = 0
      }
      if (latestRatioRef.current != null) {
        actions.resize(splitId, latestRatioRef.current)
        latestRatioRef.current = null
      }
      setActive(false)
      document.removeEventListener('pointermove', onMove)
      document.removeEventListener('pointerup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }

    document.body.style.cursor = isH ? 'col-resize' : 'row-resize'
    document.body.style.userSelect = 'none'
    document.addEventListener('pointermove', onMove)
    document.addEventListener('pointerup', onUp)
  }, [actions, splitId, isH])

  return (
    <div
      ref={handleRef}
      className={`shrink-0 flex items-center justify-center group/resize z-10
        ${isH ? 'w-1.5 cursor-col-resize' : 'h-1.5 cursor-row-resize'}
        ${active ? 'bg-blue/30' : 'hover:bg-blue/15'} transition-colors duration-100`}
      onPointerDown={onPointerDown}
    >
      <div className={`rounded-full bg-overlay0/30 group-hover/resize:bg-overlay1/50 transition-colors ${
        active ? 'bg-blue/60' : ''
      } ${isH ? 'w-0.5 h-6' : 'h-0.5 w-6'}`} />
    </div>
  )
}

// ---- Split edge button ----

function SplitEdge({ direction, onClick, className }) {
  const [hover, setHover] = useState(false)

  return (
    <div
      className={`${className} z-20 flex items-center justify-center cursor-pointer
        opacity-0 group-hover/leaf:opacity-100 transition-all duration-200`}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={(e) => { e.stopPropagation(); onClick() }}
    >
      <div className={`flex items-center justify-center rounded-sm transition-all duration-150 ${
        direction === 'h' ? 'w-3.5 h-7' : 'h-3.5 w-7'
      } ${hover ? 'bg-blue/20 scale-110' : 'bg-surface0/60'}`}>
        <svg className="w-2 h-2 text-overlay1" viewBox="0 0 8 8" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M4 1v6M1 4h6" />
        </svg>
      </div>
    </div>
  )
}
