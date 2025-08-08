import { useCallback, useEffect, useRef, useState } from 'react'
import config from '../utils/config'
import { useTrainingStore } from '../stores/trainingStore'

export interface Checkpoint {
  id: string
  nickname: string
  episode: number
  created_at: string
  training_duration: number
  model_config: {
    model_size: 'lightning' | 'base' | 'expert' | string
    learning_rate: number
    n_experts: number
    n_layers: number
    d_model: number
    n_heads: number
  }
  performance_metrics: {
    best_score: number
    avg_score: number
    final_loss: number
    training_speed: number
  }
  file_size: number
  tags: string[]
}

export interface CheckpointStats {
  total_checkpoints: number
  total_size: number
  best_score: number
  latest_episode: number
  total_training_time: number
}

export interface UseCheckpointsOptions {
  onNavigateToTab?: (tab: string) => void
}

export const useCheckpoints = (options?: UseCheckpointsOptions) => {
  const { onNavigateToTab } = options || {}

  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [stats, setStats] = useState<CheckpointStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Configuration state
  const [checkpointInterval, setCheckpointInterval] = useState<number>(50)
  const [longRunMode, setLongRunMode] = useState<boolean>(false)
  const [configLoading, setConfigLoading] = useState<boolean>(false)

  // Guards for request ordering and intervals
  const latestRequestIdRef = useRef<number>(0)
  const checkpointRefreshIntervalIdRef = useRef<number | null>(null)

  // ===== Helpers =====
  const fetchWithTimeout = useCallback(async (url: string, init?: RequestInit, timeoutMs: number = 15000) => {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs)
    try {
      const res = await fetch(url, { ...(init || {}), signal: controller.signal })
      return res
    } finally {
      clearTimeout(timeoutId)
    }
  }, [])

  // ===== Core data loaders =====
  const refresh = useCallback(async (silent: boolean = false) => {
    const thisRequestId = ++latestRequestIdRef.current
    try {
      if (!silent) setLoading(true)

      const [checkpointsRes, statsRes] = await Promise.all([
        fetchWithTimeout(`${config.api.baseUrl}/checkpoints`),
        fetchWithTimeout(`${config.api.baseUrl}/checkpoints/stats`)
      ])

      // Ignore out-of-order responses
      if (thisRequestId !== latestRequestIdRef.current) {
        return
      }

      const checkpointsData: Checkpoint[] = checkpointsRes.ok ? await checkpointsRes.json() : []
      const statsData: CheckpointStats | null = statsRes.ok ? await statsRes.json() : null

      setCheckpoints(checkpointsData)
      setStats(statsData)
      setError(null)
    } catch (err: any) {
      // Simple retry for background refreshes
      if (silent) {
        try {
          await new Promise(r => setTimeout(r, 500))
          const retryRes = await fetchWithTimeout(`${config.api.baseUrl}/checkpoints`, undefined, 8000)
          if (retryRes.ok) {
            const data = await retryRes.json()
            if (thisRequestId === latestRequestIdRef.current) {
              setCheckpoints(data)
              setError(null)
            }
            return
          }
        } catch (_) {
          // keep existing data on silent failure
          return
        }
        return
      }
      setError(err instanceof Error ? err.message : 'Failed to load checkpoints')
    } finally {
      if (!silent) setLoading(false)
    }
  }, [fetchWithTimeout])

  const loadCheckpointConfig = useCallback(async () => {
    try {
      const res = await fetchWithTimeout(`${config.api.baseUrl}/training/checkpoint/config`)
      if (res.ok) {
        const cfg = await res.json()
        setCheckpointInterval(cfg.interval)
        setLongRunMode(!!cfg.long_run_mode)
      }
    } catch (err) {
      // Non-blocking
      console.error('Error loading checkpoint config:', err)
    }
  }, [fetchWithTimeout])

  const saveCheckpointConfig = useCallback(async (interval: number, longRun: boolean) => {
    try {
      setConfigLoading(true)
      const res = await fetchWithTimeout(`${config.api.baseUrl}/training/checkpoint/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ interval, long_run_mode: longRun })
      })
      if (!res.ok) {
        throw new Error('Failed to save checkpoint configuration')
      }
      setCheckpointInterval(interval)
      setLongRunMode(longRun)
      setError(null)
    } catch (err: any) {
      setError(err instanceof Error ? err.message : 'Failed to save checkpoint configuration')
      throw err
    } finally {
      setConfigLoading(false)
    }
  }, [fetchWithTimeout])

  // ===== Mutations =====
  const updateNickname = useCallback(async (checkpointId: string, nickname: string) => {
    const res = await fetchWithTimeout(`${config.api.baseUrl}/checkpoints/${checkpointId}/nickname`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ nickname })
    })
    if (!res.ok) {
      throw new Error('Failed to update nickname')
    }
    setCheckpoints(prev => prev.map(cp => cp.id === checkpointId ? { ...cp, nickname } : cp))
  }, [fetchWithTimeout])

  const deleteCheckpoint = useCallback(async (checkpointId: string) => {
    const res = await fetchWithTimeout(`${config.api.baseUrl}/checkpoints/${checkpointId}`, { method: 'DELETE' })
    if (!res.ok) {
      throw new Error('Failed to delete checkpoint')
    }
    setCheckpoints(prev => prev.filter(cp => cp.id !== checkpointId))
    await refresh(true)
  }, [fetchWithTimeout, refresh])

  // ===== Training/Playback integration =====
  const loadCheckpointForTraining = useCallback(async (checkpointId: string) => {
    const store = useTrainingStore.getState()
    try {
      setLoading(true)
      store.setCheckpointLoadingState({
        isCheckpointLoading: true,
        checkpointId: checkpointId,
        loadingMessage: 'Starting checkpoint load...',
        loadingProgress: 0
      })

      const res = await fetchWithTimeout(`${config.api.baseUrl}/checkpoints/${checkpointId}/load`, { method: 'POST' })
      if (!res.ok) {
        throw new Error('Failed to load checkpoint')
      }
      await res.json()
      onNavigateToTab?.('dashboard')
    } catch (err: any) {
      store.setCheckpointLoadingError(err instanceof Error ? err.message : 'Failed to load checkpoint')
      setError(err instanceof Error ? err.message : 'Failed to load checkpoint')
      throw err
    } finally {
      setLoading(false)
    }
  }, [fetchWithTimeout, onNavigateToTab])

  const startPlayback = useCallback(async (checkpointId: string) => {
    const store = useTrainingStore.getState()
    try {
      const loadingSteps = [
        'Loading checkpoint metadata...',
        'Validating checkpoint integrity...',
        'Initializing playback environment...',
        'Starting game simulation...',
        'Waiting for first game data...'
      ]
      store.startLoadingOperation('playback', loadingSteps)
      onNavigateToTab?.('game')

      // staged UI feedback
      setTimeout(() => store.updateLoadingProgress(20, loadingSteps[1]), 500)
      setTimeout(() => store.updateLoadingProgress(40, loadingSteps[2]), 1000)
      setTimeout(() => store.updateLoadingProgress(60, loadingSteps[3]), 1500)

      const res = await fetchWithTimeout(`${config.api.baseUrl}/checkpoints/${checkpointId}/playback/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      if (!res.ok) {
        store.setLoadingState('isPlaybackStarting', false)
        store.setLoadingState('loadingMessage', null)
        const errorMessage = res.status === 404
          ? 'Checkpoint not found. It may have been deleted or the backend is not running.'
          : res.status === 500
          ? 'Server error occurred. Please check if the backend is running and try again.'
          : res.status === 503
          ? 'Service temporarily unavailable. Please wait a moment and try again.'
          : `Failed to start playback: ${res.status} ${res.statusText}`
        throw new Error(errorMessage)
      }
      store.updateLoadingProgress(80, loadingSteps[4], 5)
    } catch (err: any) {
      const store = useTrainingStore.getState()
      store.setLoadingState('isPlaybackStarting', false)
      store.setLoadingState('loadingMessage', null)
      setError(err instanceof Error ? err.message : 'Failed to start playback')
      throw err
    }
  }, [fetchWithTimeout, onNavigateToTab])

  // ===== Effects =====
  useEffect(() => {
    refresh()
    loadCheckpointConfig()

    if (checkpointRefreshIntervalIdRef.current) {
      clearInterval(checkpointRefreshIntervalIdRef.current)
    }
    // Silent background refresh
    checkpointRefreshIntervalIdRef.current = setInterval(() => {
      refresh(true)
    }, 10000) as unknown as number

    return () => {
      if (checkpointRefreshIntervalIdRef.current) {
        clearInterval(checkpointRefreshIntervalIdRef.current)
        checkpointRefreshIntervalIdRef.current = null
      }
    }
  }, [refresh, loadCheckpointConfig])

  return {
    // Data
    checkpoints,
    stats,
    loading,
    error,
    // Config
    checkpointInterval,
    longRunMode,
    configLoading,
    setLongRunMode,
    setCheckpointInterval,
    saveCheckpointConfig,
    // Actions
    refresh,
    updateNickname,
    deleteCheckpoint,
    loadCheckpointForTraining,
    startPlayback,
    setError,
  }
}


