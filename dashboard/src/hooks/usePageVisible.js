import { useEffect, useState } from 'react'

export function usePageVisible() {
  const [visible, setVisible] = useState(
    () => typeof document === 'undefined' || document.visibilityState !== 'hidden',
  )

  useEffect(() => {
    if (typeof document === 'undefined') return undefined

    const onVisibilityChange = () => {
      setVisible(document.visibilityState !== 'hidden')
    }

    document.addEventListener('visibilitychange', onVisibilityChange)
    return () => document.removeEventListener('visibilitychange', onVisibilityChange)
  }, [])

  return visible
}
