import { useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchJsonApi } from '../api'
import { usePageVisible } from './usePageVisible'

const TRAINING_POLL_MS = 2000
const SESSION_POLL_MS = 10000

function hiddenAwareInterval(visible, intervalMs) {
  return visible ? intervalMs : false
}

export function useDashboardOverviewQuery() {
  return useQuery({
    queryKey: ['dashboard-overview'],
    queryFn: ({ signal }) => fetchJsonApi('/api/dashboard/overview', { signal }),
    staleTime: 30_000,
  })
}

export function useDashboardRoundQuery(roundId) {
  return useQuery({
    queryKey: ['dashboard-round', roundId],
    queryFn: ({ signal }) => fetchJsonApi(`/api/dashboard/round/${roundId}`, { signal }),
    enabled: Number.isInteger(roundId),
    staleTime: 30_000,
  })
}

export function useTrainSessionsQuery() {
  const visible = usePageVisible()
  return useQuery({
    queryKey: ['train-sessions'],
    queryFn: ({ signal }) => fetchJsonApi('/api/train-sessions', { signal }),
    refetchInterval: hiddenAwareInterval(visible, SESSION_POLL_MS),
    staleTime: 5_000,
  })
}

export function useReplayListQuery() {
  return useQuery({
    queryKey: ['replays'],
    queryFn: ({ signal }) => fetchJsonApi('/api/replays', { signal }),
    staleTime: 5_000,
  })
}

export function useTrainLogQuery(session) {
  const visible = usePageVisible()
  const queryClient = useQueryClient()

  return useQuery({
    queryKey: ['train-log', session || null],
    queryFn: async ({ signal, queryKey }) => {
      const previous = queryClient.getQueryData(queryKey)
      const after = previous?.total || 0
      const params = new URLSearchParams()
      if (session) params.set('session', session)
      if (after > 0) params.set('after', String(after))
      const qs = params.toString() ? `?${params}` : ''
      const payload = await fetchJsonApi(`/api/train-log${qs}`, { signal })
      const newEpisodes = payload.episodes || []
      const serverTotal = payload.total ?? (after + newEpisodes.length)

      if (!previous || after === 0) {
        return {
          episodes: newEpisodes,
          config: payload.config || {},
          total: serverTotal,
        }
      }

      if (newEpisodes.length === 0 && serverTotal === previous.total) {
        return previous
      }

      return {
        episodes: previous.episodes.concat(newEpisodes),
        config: payload.config || previous.config,
        total: serverTotal,
      }
    },
    refetchInterval: hiddenAwareInterval(visible, TRAINING_POLL_MS),
    staleTime: 0,
  })
}

export function useLeagueQuery(session) {
  const visible = usePageVisible()

  return useQuery({
    queryKey: ['league', session || null],
    queryFn: async ({ signal }) => {
      const qs = session ? `?session=${encodeURIComponent(session)}` : ''
      const data = await fetchJsonApi(`/api/league${qs}`, { signal })
      return data.models || []
    },
    refetchInterval: hiddenAwareInterval(visible, TRAINING_POLL_MS),
    staleTime: 0,
  })
}

export function useTrainReplaysQuery(session, minEp, maxEp) {
  const visible = usePageVisible()

  return useQuery({
    queryKey: ['train-replays', session || null, minEp ?? null, maxEp ?? null],
    queryFn: ({ signal }) => {
      const params = new URLSearchParams()
      if (session) params.set('session', session)
      if (minEp != null) params.set('min_ep', String(minEp))
      if (maxEp != null) params.set('max_ep', String(maxEp))
      const qs = params.toString() ? `?${params}` : ''
      return fetchJsonApi(`/api/train-replays${qs}`, { signal })
    },
    refetchInterval: hiddenAwareInterval(visible, TRAINING_POLL_MS),
    staleTime: 0,
  })
}

export function useGameLogEpisodesQuery(session, minEp, maxEp) {
  const visible = usePageVisible()

  return useQuery({
    queryKey: ['game-log-episodes', session || null, minEp ?? null, maxEp ?? null],
    queryFn: ({ signal }) => {
      const params = new URLSearchParams()
      if (session) params.set('session', session)
      if (minEp != null) params.set('min_ep', String(minEp))
      if (maxEp != null) params.set('max_ep', String(maxEp))
      const qs = params.toString() ? `?${params}` : ''
      return fetchJsonApi(`/api/game-log-episodes${qs}`, { signal })
    },
    refetchInterval: hiddenAwareInterval(visible, TRAINING_POLL_MS),
    staleTime: 0,
  })
}

export function useGameLogForEpisodeQuery(session, episode) {
  const visible = usePageVisible()

  return useQuery({
    queryKey: ['game-log-episode', session || null, episode],
    queryFn: ({ signal }) => {
      const params = new URLSearchParams()
      if (session) params.set('session', session)
      if (episode != null) params.set('episode', String(episode))
      const qs = params.toString() ? `?${params}` : ''
      return fetchJsonApi(`/api/game-log${qs}`, { signal })
    },
    enabled: episode != null,
    refetchInterval: hiddenAwareInterval(visible, TRAINING_POLL_MS),
    staleTime: 0,
  })
}

export function useGameLogRangedQuery(session, minEp, maxEp) {
  const visible = usePageVisible()

  return useQuery({
    queryKey: ['game-log-ranged', session || null, minEp ?? null, maxEp ?? null],
    queryFn: ({ signal }) => {
      const params = new URLSearchParams()
      if (session) params.set('session', session)
      if (minEp != null) params.set('min_ep', String(minEp))
      if (maxEp != null) params.set('max_ep', String(maxEp))
      const qs = params.toString() ? `?${params}` : ''
      return fetchJsonApi(`/api/game-log${qs}`, { signal })
    },
    refetchInterval: hiddenAwareInterval(visible, TRAINING_POLL_MS),
    staleTime: 0,
  })
}

export function useGameLogQuery(session) {
  const visible = usePageVisible()
  const queryClient = useQueryClient()

  return useQuery({
    queryKey: ['game-log', session || null],
    queryFn: async ({ signal, queryKey }) => {
      const previous = queryClient.getQueryData(queryKey)
      const after = previous?.total || 0
      const params = new URLSearchParams()
      if (session) params.set('session', session)
      if (after > 0) params.set('after', String(after))
      const qs = params.toString() ? `?${params}` : ''
      const payload = await fetchJsonApi(`/api/game-log${qs}`, { signal })
      const newGames = Array.isArray(payload.games) ? payload.games : Array.isArray(payload) ? payload : []
      const serverTotal = payload.total ?? (after + newGames.length)

      if (!previous || after === 0) {
        const map = {}
        for (const game of newGames) {
          if (game.replay) map[game.replay] = game
        }
        return { map, total: serverTotal }
      }

      if (newGames.length === 0 && serverTotal === previous.total) {
        return previous
      }

      const map = { ...previous.map }
      for (const game of newGames) {
        if (game.replay) map[game.replay] = game
      }
      return { map, total: serverTotal }
    },
    refetchInterval: hiddenAwareInterval(visible, TRAINING_POLL_MS),
    staleTime: 0,
  })
}
