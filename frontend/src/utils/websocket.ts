import { useEffect, useRef, useState, useCallback } from 'react'
import { useTrainingStore } from '../stores/trainingStore'
import config from './config'
import { 
  isMobile,
  isMobileSafari, 
  getMobileOptimizedWebSocketURL, 
  getConnectionRetryDelay, 
  getMaxReconnectAttempts,
  detectNetworkQuality,
  getAdaptivePollingInterval,
  getAdaptiveRetryStrategy,
  NetworkQuality
} from './mobile-detection'

export const useWebSocket = () => {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)
  const pollingIntervalRef = useRef<number | null>(null)
  const sseRef = useRef<EventSource | null>(null)
  const maxReconnectAttempts = getMaxReconnectAttempts()
  const reconnectDelay = getConnectionRetryDelay()
  const reconnectingRef = useRef(false) // NEW: race-condition guard for reconnect attempts
  const networkQualityRef = useRef<NetworkQuality | null>(null)
  const lastNetworkCheckRef = useRef<number>(0)
  
  // Playback health monitoring
  const playbackHealthRef = useRef({
    lastHeartbeat: 0,
    consecutiveFailures: 0,
    isHealthy: true,
    lastHealthCheck: 0
  })
  
  // Connection quality monitoring
  const connectionStatsRef = useRef({
    latency: 0,
    messagesReceived: 0,
    messagesSent: 0,
    lastHeartbeat: 0,
    connectionStartTime: 0,
    dataTransferred: 0
  })

  const { 
    setConnected, 
    setConnectionError, 
    updateTrainingData,
    isConnected
  } = useTrainingStore()

  const [reconnectAttempts, setReconnectAttempts] = useState(0)
  const [connectionHealth, setConnectionHealth] = useState<'healthy' | 'degraded' | 'poor' | 'critical'>('healthy')
  const [lastSuccessfulConnection, setLastSuccessfulConnection] = useState<number>(Date.now())
  const [consecutiveFailures, setConsecutiveFailures] = useState(0)
  const [isInRecoveryMode, setIsInRecoveryMode] = useState(false)

  // Connection stability monitoring
  const monitorConnectionHealth = useCallback(() => {
    const timeSinceLastSuccess = Date.now() - lastSuccessfulConnection
    
    // Determine connection health based on multiple factors
    if (consecutiveFailures >= 5 || timeSinceLastSuccess > 60000) {
      setConnectionHealth('critical')
    } else if (consecutiveFailures >= 3 || timeSinceLastSuccess > 30000) {
      setConnectionHealth('poor')
    } else if (consecutiveFailures >= 1 || timeSinceLastSuccess > 15000) {
      setConnectionHealth('degraded')
    } else {
      setConnectionHealth('healthy')
    }
  }, [lastSuccessfulConnection, consecutiveFailures])

  // Centralized reconnect scheduler (idempotent)
  const scheduleReconnect = useCallback(() => {
    if (reconnectingRef.current) return
    if (reconnectAttempts >= maxReconnectAttempts) return
    if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current)

    if (connectionHealth === 'critical' && !isInRecoveryMode) {
      setIsInRecoveryMode(true)
      reconnectTimeoutRef.current = setTimeout(() => {
        setIsInRecoveryMode(false)
        setConsecutiveFailures(0)
        setReconnectAttempts(0)
        reconnectTimeoutRef.current = null
        connect()
      }, 30000)
      return
    }

    const baseDelay = isMobile() ? getConnectionRetryDelay() * 1.5 : getConnectionRetryDelay()
    const backoff = Math.min(baseDelay * Math.pow(2, reconnectAttempts), 30000)
    reconnectTimeoutRef.current = setTimeout(() => {
      reconnectTimeoutRef.current = null
      setReconnectAttempts(prev => prev + 1)
      connect()
    }, backoff)
  }, [reconnectAttempts, maxReconnectAttempts, connectionHealth, isInRecoveryMode])

  // Enhanced message processing with health tracking
  const processMessageData = useCallback((data: any) => {
    try {
      console.log('Received WebSocket message:', data.type)

      if (data.type === 'training_update') {
        console.log('Received training update:', data)
        useTrainingStore.getState().updateTrainingData(data)
        const currentState = useTrainingStore.getState()
        if (currentState.loadingStates.isTrainingStarting) {
          useTrainingStore.getState().completeLoadingOperation()
        }
      } else if (data.type === 'checkpoint_playback') {
        console.log('Received checkpoint playback data:', data)
        useTrainingStore.getState().updateCheckpointPlaybackData(data)
        const currentState = useTrainingStore.getState()
        if (currentState.loadingStates.isPlaybackStarting || currentState.loadingStates.isNewGameStarting) {
          useTrainingStore.getState().completeLoadingOperation()
        }
      } else if (data.type === 'training_status_update') {
        console.log('Received training status update:', data)
        useTrainingStore.getState().setTrainingStatus(data.is_training, data.is_paused)
        if (data.current_episode !== undefined) {
          useTrainingStore.getState().setEpisode(data.current_episode)
        }
      } else if (data.type === 'training_reset') {
        console.log('Received training reset:', data)
        useTrainingStore.getState().setTrainingStatus(false, false)
        useTrainingStore.getState().setEpisode(0)
      } else if (data.type === 'training_start') {
        console.log('Received training start:', data)
        useTrainingStore.getState().setTrainingStatus(true, false)
        useTrainingStore.getState().setShowingGameOver(false)
        useTrainingStore.getState().setGameCompletionData(null)
        useTrainingStore.getState().setLoadingState('isTrainingStarting', false)
        useTrainingStore.getState().setLoadingState('loadingMessage', null)
        if (data.model_config && data.model_config.model_size) {
          const size = data.model_config.model_size as 'lightning' | 'base' | 'expert'
          if (size === 'lightning' || size === 'base' || size === 'expert') {
            useTrainingStore.getState().setModelSize(size)
          }
        }
      } else if (data.type === 'training_complete') {
        console.log('Received training complete:', data)
        useTrainingStore.getState().setTrainingStatus(false, false)
        useTrainingStore.getState().setLoadingState('isTrainingStarting', false)
        useTrainingStore.getState().setLoadingState('loadingMessage', null)
      } else if (data.type === 'connection_status') {
        console.log('Received connection status:', data)
        useTrainingStore.getState().setConnected(data.connected)
        if (data.error) {
          useTrainingStore.getState().setConnectionError(data.error)
        } else {
          useTrainingStore.getState().setConnectionError(null)
        }
        if (data.connected) {
          const state = useTrainingStore.getState()
          if (state.lastTrainingData && state.isTraining) {
            console.log('Restoring training data on reconnection:', state.lastTrainingData)
            useTrainingStore.getState().updateTrainingData(state.lastTrainingData)
          }
        }
      } else if (data.type === 'playback_status') {
        console.log('Playback status:', data.message)
        if (data.status === 'starting') {
          const currentLoadingStates = useTrainingStore.getState().loadingStates
          if (!currentLoadingStates.isPlaybackStarting) {
            useTrainingStore.getState().setLoadingState('isPlaybackStarting', true)
            useTrainingStore.getState().setLoadingState('loadingMessage', 'Loading checkpoint model...')
          }
        } else if (data.status === 'stopped') {
          useTrainingStore.getState().setPlayingCheckpoint(false)
          useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
          useTrainingStore.getState().setLoadingState('loadingMessage', null)
        }
      } else if (data.type === 'new_game_started') {
        useTrainingStore.getState().setLoadingState('isNewGameStarting', false)
        useTrainingStore.getState().setLoadingState('loadingMessage', null)
        useTrainingStore.getState().setShowingGameOver(false)
        useTrainingStore.getState().setGameCompletionData(null)
      } else if (data.type === 'game_completed') {
        console.log('Received game completion:', data)
        useTrainingStore.getState().setGameCompletionData({
          final_score: data.final_score,
          total_steps: data.total_steps,
          max_tile: data.max_tile,
          final_board_state: [],
          checkpoint_id: data.checkpoint_id,
          game_number: data.game_number
        })
        useTrainingStore.getState().setShowingGameOver(true)
      } else if (data.type === 'new_episode_started') {
        console.log('Received new episode started:', data)
        if (data.episode !== undefined) {
          useTrainingStore.getState().setEpisode(data.episode)
        }
      } else if (data.type === 'checkpoint_loading_status') {
        console.log('Received checkpoint loading status:', data)
        const { checkpoint_id, status, message } = data
        if (status === 'loading') {
          useTrainingStore.getState().setCheckpointLoadingState({
            isCheckpointLoading: true,
            checkpointId: checkpoint_id,
            loadingMessage: message || 'Loading checkpoint...',
            loadingProgress: 10
          })
        } else if (status === 'config_loaded') {
          useTrainingStore.getState().updateCheckpointLoadingProgress(30, message || 'Checkpoint data loaded, initializing model...')
        } else if (status === 'trainer_created') {
          useTrainingStore.getState().updateCheckpointLoadingProgress(50, message || 'Model initialized, loading checkpoint weights...')
        } else if (status === 'weights_loaded') {
          useTrainingStore.getState().updateCheckpointLoadingProgress(70, message || 'Checkpoint weights loaded, preparing training environment...')
        } else if (status === 'starting_training') {
          useTrainingStore.getState().updateCheckpointLoadingProgress(90, message || 'Starting training session...')
        } else if (status === 'complete') {
          useTrainingStore.getState().completeCheckpointLoading(message || 'Checkpoint loaded successfully')
        } else if (status === 'error') {
          useTrainingStore.getState().setCheckpointLoadingError(message || 'Failed to load checkpoint')
        }
      } else if (data.type === 'checkpoint_created') {
        console.log('Received checkpoint created:', data)
      } else if (data.type === 'evaluation_metrics') {
        console.log('Received evaluation metrics:', data)
        try {
          useTrainingStore.getState().updateEvaluationMetrics({ metrics: data.metrics || {} })
        } catch (e) {
          console.warn('Failed to update evaluation metrics in store:', e)
        }
      } else {
        console.log('Unknown message type:', data.type)
      }
    } catch (error) {
      console.error('Error processing WebSocket message:', error)
    }
  }, [])

  // SSE fallback – preferred over plain polling when available
  const startSSEFallback = useCallback(() => {
    if (sseRef.current) {
      try { sseRef.current.close() } catch {}
      sseRef.current = null
    }
    try {
      const sseUrl = `${config.api.baseUrl}/events?t=${Date.now()}`
      const es = new EventSource(sseUrl, { withCredentials: false })
      sseRef.current = es
      setConnectionError('Using SSE fallback due to connection issues...')

      es.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data)
          // Reuse message pipeline
          if (data.type === 'training_status_update') {
            useTrainingStore.getState().setTrainingStatus(data.is_training, data.is_paused)
            if (data.current_episode !== undefined) {
              useTrainingStore.getState().setEpisode(Number(data.current_episode) || 0)
            }
          }
          processMessageData(data)
        } catch (e) {
          console.warn('SSE parse error:', e)
        }
      }

      es.onerror = () => {
        // Fallback to HTTP polling if SSE also fails
        try { es.close() } catch {}
        sseRef.current = null
        startPollingFallback()
      }
    } catch (e) {
      console.warn('Failed to start SSE fallback, using polling:', e)
      startPollingFallback()
    }
  }, [processMessageData])

  // Enhanced polling fallback with better error handling
  const startPollingFallback = useCallback(async () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current)
    }
    
    console.log('Starting enhanced polling fallback')
    setConnectionError('Using polling fallback due to connection issues...')
    
    const networkQuality = networkQualityRef.current || await detectAndUpdateNetworkQuality()
    let currentPollingInterval = getAdaptivePollingInterval(networkQuality)
    let pollingConsecutiveFailures = 0
    
    const createPollingInterval = (interval: number) => {
      return setInterval(async () => {
        try {
          // Enhanced request with better timeout handling
          const controller = new AbortController()
          const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout
          
          const response = await fetch(`${config.api.baseUrl}/training/status`, {
            signal: controller.signal,
            headers: {
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache'
            }
          })
          
          clearTimeout(timeoutId)
          
          if (response.ok) {
            pollingConsecutiveFailures = 0
            setConsecutiveFailures(0)
            setLastSuccessfulConnection(Date.now())
            setConnected(true)
            
            // Process response data
            const data = await response.json()
            
            // Sync training status with backend to avoid stale localStorage state
            if (data.is_training !== undefined || data.is_paused !== undefined) {
              console.log('Syncing training status via polling:', data)
              useTrainingStore.getState().setTrainingStatus(data.is_training, data.is_paused)
              if (data.current_episode !== undefined) {
                useTrainingStore.getState().setEpisode(data.current_episode)
              }
            }
            
            updateTrainingData(data)
            
            // Check if we can upgrade back to WebSocket
            if (pollingConsecutiveFailures === 0 && connectionHealth !== 'critical') {
              console.log('Polling stable, attempting WebSocket upgrade')
              setTimeout(() => {
                if (pollingConsecutiveFailures === 0) {
                  console.log('Upgrading from fallback to WebSocket')
                  clearInterval(pollingIntervalRef.current!)
                  pollingIntervalRef.current = null
                  setReconnectAttempts(0)
                  connect()
                }
              }, 10000)
            }
          } else {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`)
          }
          
        } catch (error) {
          pollingConsecutiveFailures++
          setConsecutiveFailures(prev => prev + 1)
          console.error('Polling failed:', error)
          
          // Adaptive polling interval based on failures
          if (pollingConsecutiveFailures >= 3) {
            currentPollingInterval = Math.min(currentPollingInterval * 1.5, 10000)
            console.log(`Polling failures detected, increasing interval to ${currentPollingInterval}ms`)
            
            // Restart polling with new interval
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current)
              pollingIntervalRef.current = createPollingInterval(currentPollingInterval)
            }
          }
          
          // If polling fails too many times, show critical error
          if (pollingConsecutiveFailures >= 10) {
            setConnected(false)
            setConnectionError('Server appears to be down. Please refresh the page or check if the backend is running.')
            setConnectionHealth('critical')
          }
        }
      }, interval)
    }
    
    pollingIntervalRef.current = createPollingInterval(currentPollingInterval)
     }, [networkQualityRef, connectionHealth, updateTrainingData])


  // Network quality detection
  const detectAndUpdateNetworkQuality = async () => {
    try {
      const quality = await detectNetworkQuality()
      networkQualityRef.current = quality
      
      // Update connection error message if network quality is poor
      if (quality.level === 'poor' && !isConnected) {
        setConnectionError(`Poor network connection (${quality.latency}ms latency) - trying fallback...`)
      }
      
      console.log('Network quality detected:', quality)
      return quality
    } catch (error) {
      console.error('Failed to detect network quality:', error)
      return {
        level: 'fair' as const,
        latency: 1000,
        bandwidth: 0,
        isOnline: navigator.onLine
      }
    }
  }

  // Connection quality monitoring
  const measureConnectionLatency = () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return
    }
    
    const startTime = Date.now()
    const pingMessage = JSON.stringify({
      type: 'ping',
      timestamp: startTime
    })
    
    try {
      wsRef.current.send(pingMessage)
      connectionStatsRef.current.messagesSent++
      connectionStatsRef.current.dataTransferred += pingMessage.length
    } catch (error) {
      console.error('Failed to send ping:', error)
    }
  }

  const updateConnectionStats = (messageSize: number) => {
    connectionStatsRef.current.messagesReceived++
    connectionStatsRef.current.dataTransferred += messageSize
    connectionStatsRef.current.lastHeartbeat = Date.now()
  }

  const getConnectionQuality = () => {
    const stats = connectionStatsRef.current
    const currentTime = Date.now()
    const connectionDuration = currentTime - stats.connectionStartTime
    
    if (connectionDuration < 1000) return 'connecting'
    
    const avgLatency = stats.latency
    const timeSinceLastMessage = currentTime - stats.lastHeartbeat
    
    if (timeSinceLastMessage > 30000) return 'poor'
    if (avgLatency > 2000) return 'poor'
    if (avgLatency > 1000) return 'fair'
    if (avgLatency > 500) return 'good'
    return 'excellent'
  }

  // Removed unused checkPlaybackHealth function

  // Removed unused attemptPlaybackRestart function

  const connect = useCallback(async () => {
    // Prevent duplicate connections
    if (wsRef.current || reconnectingRef.current) {
      return
    }

    reconnectingRef.current = true // guard — set immediately to avoid races

    // Clear any scheduled reconnect to avoid duplicate timers
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    // Check network quality periodically (every 30 seconds)
    const now = Date.now()
    if (now - lastNetworkCheckRef.current > 30000) {
      await detectAndUpdateNetworkQuality()
      lastNetworkCheckRef.current = now
    }

    try {
      // Include last_seq for lightweight resume
      const lastSeq = useTrainingStore.getState().lastMessageSeq || 0
      const baseUrl = getMobileOptimizedWebSocketURL(config.websocket.url)
      const websocketUrl = lastSeq > 0 ? `${baseUrl}?last_seq=${lastSeq}` : baseUrl
      console.log('Connecting to WebSocket:', websocketUrl)
      
      const ws = new WebSocket(websocketUrl)
      wsRef.current = ws
      
      // CRITICAL FIX: Add connection timeout for all devices during training
      let connectionTimeout: number | null = null
      const timeoutDuration = isMobileSafari() ? 10000 : isMobile() ? 8000 : 5000 // Longer timeouts for mobile
      connectionTimeout = setTimeout(() => {
        if (ws.readyState === WebSocket.CONNECTING) {
          console.log(`WebSocket connection timeout (${timeoutDuration}ms), trying fallback`)
          ws.close()
          setConnectionError(`Connection timeout (${timeoutDuration}ms) - switching to polling...`)
          startPollingFallback()
        }
      }, timeoutDuration)
      
      // Reset connection stats
      connectionStatsRef.current = {
        latency: 0,
        messagesReceived: 0,
        messagesSent: 0,
        lastHeartbeat: Date.now(),
        connectionStartTime: Date.now(),
        dataTransferred: 0
      }

      ws.onopen = () => {
        console.log('WebSocket connected successfully')
        reconnectingRef.current = false // connection succeeded – clear guard
        
        // Clear connection timeout if it exists
        if (connectionTimeout) {
          clearTimeout(connectionTimeout)
          connectionTimeout = null
        }
        
        setConnected(true)
        setConnectionError(null)
        setReconnectAttempts(0)
        setLastSuccessfulConnection(Date.now())
        setConsecutiveFailures(0)
        
        // Start periodic latency monitoring
        const latencyInterval = setInterval(() => {
          if (wsRef.current === ws && ws.readyState === WebSocket.OPEN) {
            measureConnectionLatency()
          } else {
            clearInterval(latencyInterval)
          }
        }, 10000) // Check every 10 seconds
        
        // Clear any polling fallback
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current)
          pollingIntervalRef.current = null
        }

        // We now expect the server to push a snapshot immediately after connect;
        // keep the HTTP sync as a safety net with no-store headers.
        fetch(`${config.api.baseUrl}/training/status?t=${Date.now()}`, { cache: 'no-store', headers: { 'Cache-Control': 'no-store, no-cache', 'Pragma': 'no-cache' } })
          .then((res) => res.json())
          .then((status) => {
            console.log('Syncing training status with backend:', status)
            useTrainingStore.getState().setTrainingStatus(status.is_training, status.is_paused)
            useTrainingStore.getState().setEpisode(Number(status.current_episode) || 0)
          })
          .catch((error) => {
            console.warn('Failed to sync training status:', error)
          })

        // If we were in playback mode before losing connection, ask backend for status so we can resync
        if (useTrainingStore.getState().isPlayingCheckpoint) {
          fetch(`${config.api.baseUrl}/checkpoints/playback/status`)
            .then((res) => res.json())
            .then((status) => {
              if (!status.is_playing) {
                // Playback stopped on server – update local state
                useTrainingStore.getState().setPlayingCheckpoint(false)
              }
            })
            .catch(() => {/* ignore */})
        }
      }

      ws.onmessage = (event) => {
        try {
          updateConnectionStats(event.data.length)
          const parsed = JSON.parse(event.data)

          // Handle ping/pong for latency measurement
          if (parsed.type === 'pong' && parsed.timestamp) {
            const latency = Date.now() - parsed.timestamp
            connectionStatsRef.current.latency = latency
            console.log(`WebSocket latency: ${latency}ms`)
            return
          }

          // Handle message batches
          if (parsed.type === 'message_batch' && Array.isArray(parsed.messages)) {
            for (const message of parsed.messages) {
              processMessageData(message)
            }
            return
          }

          // Normalize heartbeats
          if (parsed.type === 'playback_heartbeat' || parsed.type === 'heartbeat') {
            playbackHealthRef.current.lastHeartbeat = Date.now()
            playbackHealthRef.current.consecutiveFailures = parsed.consecutive_failures || 0
            playbackHealthRef.current.isHealthy = parsed.is_healthy !== false
            playbackHealthRef.current.lastHealthCheck = Date.now()
            return
          }

          processMessageData(parsed)
        } catch (error) {
          console.error('Error processing WebSocket message:', error)
          setConsecutiveFailures(prev => prev + 1)
        }
      }
        
        ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.reason)
        wsRef.current = null
        setConnected(false)
        
        // Use adaptive retry strategy based on network quality and connection quality
        const networkQuality = networkQualityRef.current
        const connectionQuality = getConnectionQuality()
        
        console.log(`Connection quality: ${connectionQuality}, Network quality: ${networkQuality?.level}`)
        
        const retryStrategy = networkQuality 
          ? getAdaptiveRetryStrategy(networkQuality)
          : { maxAttempts: maxReconnectAttempts, baseDelay: reconnectDelay, maxDelay: 30000 }
        
        // Adjust retry strategy based on connection quality
        if (connectionQuality === 'poor') {
          retryStrategy.maxAttempts = Math.max(1, retryStrategy.maxAttempts - 2)
          retryStrategy.baseDelay *= 2
        }
        
        // Avoid multiple concurrent reconnect timers
        if (reconnectTimeoutRef.current || reconnectAttempts >= maxReconnectAttempts) {
          return
        }
        setConnectionError(`Connection lost. Reconnecting... (${reconnectAttempts + 1}/${maxReconnectAttempts})`)
        scheduleReconnect()
        if (reconnectAttempts >= maxReconnectAttempts - 1) {
          // After all retries failed, try polling fallback on mobile
          if (isMobileSafari() || isMobile()) {
            console.log('All WebSocket retries failed, switching to SSE fallback')
            setConnectionError('WebSocket failed, switching to SSE mode...')
            startSSEFallback()
          } else {
            setConnectionError('Connection lost. Please refresh the page.')
          }
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        console.log('User agent:', navigator.userAgent)
        console.log('Is mobile device:', isMobile())
        console.log('Is mobile Safari:', isMobileSafari())
        
        if (isMobile()) {
          // More specific error messages for mobile
          const errorMessage = isMobileSafari() 
            ? 'Mobile Safari connection issue - trying fallback...'
            : 'Mobile connection issue - trying fallback...'
          setConnectionError(errorMessage)
          
          // Try SSE fallback after a shorter delay for mobile
          setTimeout(() => {
            if (!isConnected) {
              console.log('Attempting SSE fallback for mobile device')
              startSSEFallback()
            }
          }, 500) // Even shorter delay for mobile
        } else {
          setConnectionError('Connection error. Please check if the server is running.')
        }
      }
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      setConnectionError('Failed to connect to server.')
    }
  }, [maxReconnectAttempts, reconnectDelay, startPollingFallback])

  const disconnect = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current)
      pollingIntervalRef.current = null
    }
    
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    
    setConnected(false)
  }

  // Enhanced connection monitoring
  useEffect(() => {
    const healthMonitorInterval = setInterval(monitorConnectionHealth, 5000) // Check every 5 seconds
    return () => clearInterval(healthMonitorInterval)
  }, [monitorConnectionHealth])

  useEffect(() => {
    connect()
    
    // Add offline/online event listeners
    const handleOffline = () => {
      console.log('Network went offline')
      setConnectionError('Network connection lost - you are offline')
      setConnected(false)
    }
    
    const handleOnline = () => {
      console.log('Network came back online')
      setConnectionError('Network connection restored - reconnecting...')
      
      // Attempt to reconnect after a short delay
      setTimeout(() => {
        if (!isConnected) {
          scheduleReconnect()
        }
      }, 300)
    }
    
    // Add event listeners for network state changes
    window.addEventListener('offline', handleOffline)
    window.addEventListener('online', handleOnline)
    
    // Also listen for visibility changes to handle mobile app switching
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        // Page became visible again, check connection
        if (!isConnected && navigator.onLine) {
          console.log('Page became visible and we are online - attempting reconnect')
          setTimeout(() => {
            if (!isConnected) {
              scheduleReconnect()
            }
          }, 300)
        }
      } else {
        // Page became hidden, conserve resources
        console.log('Page became hidden - pausing connection monitoring')
      }
    }
    
    document.addEventListener('visibilitychange', handleVisibilityChange)
    
    return () => {
      disconnect()
      window.removeEventListener('offline', handleOffline)
      window.removeEventListener('online', handleOnline)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [])

  return {
    isConnected,
    reconnect: () => scheduleReconnect(),
    disconnect,
  }
}

export const sendWebSocketMessage = (message: any) => {
  // This is a placeholder - in a real implementation, we'd manage the WebSocket reference
  console.log('Sending WebSocket message:', message)
} 