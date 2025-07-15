import { useEffect, useRef } from 'react'
import { useTrainingStore } from '../stores/trainingStore'
import config from './config'
import { 
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
  const reconnectAttempts = useRef(0)
  const pollingIntervalRef = useRef<number | null>(null)
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
    updateCheckpointPlaybackData,
    isConnected 
  } = useTrainingStore()

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

  const checkPlaybackHealth = () => {
    const currentTime = Date.now()
    const health = playbackHealthRef.current
    
    // If we're not in playback mode, consider healthy
    if (!useTrainingStore.getState().isPlayingCheckpoint) {
      return { isHealthy: true, reason: 'not_in_playback' }
    }
    
    // Check if we've received recent heartbeat
    if (currentTime - health.lastHeartbeat > 30000) { // 30 seconds
      return { isHealthy: false, reason: 'no_heartbeat' }
    }
    
    // Check if playback system reports unhealthy
    if (!health.isHealthy) {
      return { isHealthy: false, reason: 'system_unhealthy' }
    }
    
    // Check consecutive failures
    if (health.consecutiveFailures >= 5) {
      return { isHealthy: false, reason: 'too_many_failures' }
    }
    
    return { isHealthy: true, reason: 'healthy' }
  }

  const attemptPlaybackRestart = async () => {
    try {
      console.log('Attempting to restart playback automatically...')
      
      // First, try to get the current playback status
      const statusResp = await fetch(`${config.api.baseUrl}/checkpoints/playback/status`)
      if (!statusResp.ok) {
        console.warn('Could not get playback status for restart')
        return
      }
      
      const status = await statusResp.json()
      
      // If there's a checkpoint loaded but not playing, try to restart
      if (status.current_checkpoint && !status.is_playing) {
        console.log(`Restarting playback for checkpoint: ${status.current_checkpoint}`)
        
        // Set loading state
        useTrainingStore.getState().setLoadingState('isPlaybackStarting', true)
        useTrainingStore.getState().setLoadingState('loadingMessage', 'Restarting playback...')
        
        // Start playback
        const restartResp = await fetch(`${config.api.baseUrl}/checkpoints/${status.current_checkpoint}/playback/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ speed: 1.0 })
        })
        
        if (restartResp.ok) {
          console.log('Playback restart successful')
          // Reset health monitoring
          playbackHealthRef.current.lastHeartbeat = Date.now()
          playbackHealthRef.current.consecutiveFailures = 0
          playbackHealthRef.current.isHealthy = true
        } else {
          console.warn('Failed to restart playback:', restartResp.status)
          // Clear loading state on failure
          useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
          useTrainingStore.getState().setLoadingState('loadingMessage', null)
        }
      }
    } catch (error) {
      console.error('Error attempting playback restart:', error)
      // Clear loading state on error
      useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
      useTrainingStore.getState().setLoadingState('loadingMessage', null)
    }
  }

  const connect = async () => {
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
      const websocketUrl = getMobileOptimizedWebSocketURL(config.websocket.url)
      console.log('Connecting to WebSocket:', websocketUrl)
      
      const ws = new WebSocket(websocketUrl)
      wsRef.current = ws
      
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
        setConnected(true)
        setConnectionError(null)
        reconnectAttempts.current = 0
        
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
          const data = JSON.parse(event.data)
          updateConnectionStats(event.data.length)
          
          // Handle ping/pong for latency measurement
          if (data.type === 'pong' && data.timestamp) {
            const latency = Date.now() - data.timestamp
            connectionStatsRef.current.latency = latency
            console.log(`WebSocket latency: ${latency}ms`)
            return
          }
          
          // Handle playback heartbeat
          if (data.type === 'playback_heartbeat') {
            playbackHealthRef.current.lastHeartbeat = Date.now()
            playbackHealthRef.current.consecutiveFailures = data.consecutive_failures || 0
            playbackHealthRef.current.isHealthy = data.is_healthy !== false
            playbackHealthRef.current.lastHealthCheck = Date.now()
            
            console.log(`Playback heartbeat: healthy=${data.is_healthy}, failures=${data.consecutive_failures}`)
            
            // If playback is unhealthy, show warning
            if (!data.is_healthy) {
              console.warn('Playback system is unhealthy, may need restart')
            }
            return
          }
          
          // Monitor playback health in all messages
          if (data.playback_health) {
            playbackHealthRef.current.consecutiveFailures = data.playback_health.consecutive_failures || 0
            playbackHealthRef.current.isHealthy = data.playback_health.is_healthy !== false
            playbackHealthRef.current.lastHealthCheck = Date.now()
          }
          
          if (data.type === 'training_update') {
            // Only update graphs if we have valid history arrays to avoid accidental wipe
            if (data.loss_history && data.score_history) {
              updateTrainingData(data)
            }
          } else if (data.type === 'checkpoint_playback') {
            updateCheckpointPlaybackData(data)
          } else if (data.type === 'playback_status') {
            console.log('Playback status:', data.message)
            // Handle playback loading states
            if (data.status === 'starting') {
              // Only set loading state if it's not already set (to avoid overriding existing loading message)
              const currentLoadingStates = useTrainingStore.getState().loadingStates
              if (!currentLoadingStates.isPlaybackStarting) {
                useTrainingStore.getState().setLoadingState('isPlaybackStarting', true)
                useTrainingStore.getState().setLoadingState('loadingMessage', 'Loading checkpoint model...')
              }
            } else if (data.status === 'recovering') {
              // Show recovery status
              useTrainingStore.getState().setLoadingState('loadingMessage', 'Reconnecting to playback...')
              console.warn('Playback system is recovering from connection issues')
            } else if (data.status === 'stopped') {
              useTrainingStore.getState().setPlayingCheckpoint(false)
              
              // Clear all loading states when playback stops
              useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
              useTrainingStore.getState().setLoadingState('isNewGameStarting', false)
              useTrainingStore.getState().setLoadingState('loadingMessage', null)
              
              // If stopped due to errors, show error history
              if (data.error_history && data.error_history.length > 0) {
                console.error('Playback stopped with errors:', data.error_history)
                const lastError = data.error_history[data.error_history.length - 1]
                setConnectionError(`Playback failed: ${lastError.error}`)
              }
            }
          } else if (data.type === 'playback_recovery') {
            console.log('Playback recovery:', data.message)
            // Update connection status to show recovery
            setConnectionError(null) // Clear any previous errors
            useTrainingStore.getState().setLoadingState('loadingMessage', 'Recovering playback connection...')
          } else if (data.type === 'training_reset') {
            console.log('Training reset:', data.message)
            // Update training state
            useTrainingStore.getState().setTrainingStatus(false, false)
            useTrainingStore.getState().setEpisode(0)
            // Clear loading state
            useTrainingStore.getState().setLoadingState('isTrainingResetting', false)
            useTrainingStore.getState().setLoadingState('loadingMessage', null)
          } else if (data.type === 'training_status_update') {
            // Handle training status updates (e.g., from checkpoint loading)
            const { setTrainingStatus, setEpisode } = useTrainingStore.getState()
            setTrainingStatus(data.is_training, data.is_paused)
            if (data.current_episode !== undefined) {
              setEpisode(data.current_episode)
            }
            console.log('Training status updated:', data.message)
          } else if (data.type === 'playback_error') {
            console.error('Playback error:', data.error)
            // Clear loading states on error
            useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
            useTrainingStore.getState().setLoadingState('loadingMessage', null)
            // Show error in connection status
            setConnectionError(`Playback error: ${data.error}`)
          } else if (data.type === 'game_completed') {
            console.log(`Game completed: Score ${data.final_score}, Steps ${data.total_steps}, Max tile ${data.max_tile}`)
            // Set loading state for new game starting
            useTrainingStore.getState().setLoadingState('isNewGameStarting', true)
            useTrainingStore.getState().setLoadingState('loadingMessage', 'Starting new game...')
          } else if (data.type === 'new_game_started') {
            // Clear new game loading state
            useTrainingStore.getState().setLoadingState('isNewGameStarting', false)
            useTrainingStore.getState().setLoadingState('loadingMessage', null)
          } else if (data.type === 'training_started') {
            console.log('Training started:', data.message)
            // The training start loading state will be cleared when first training data arrives
          } else if (data.type === 'new_episode_started') {
            console.log('New episode started:', data.message)
            // Set loading state for new episode briefly during training
            if (useTrainingStore.getState().isTraining) {
              useTrainingStore.getState().setLoadingState('isNewGameStarting', true)
              useTrainingStore.getState().setLoadingState('loadingMessage', `Starting episode ${data.episode}...`)
              // Clear it after a short delay since training data will arrive soon
              setTimeout(() => {
                useTrainingStore.getState().setLoadingState('isNewGameStarting', false)
                useTrainingStore.getState().setLoadingState('loadingMessage', null)
              }, 1000)
            }
          } else if (data.type === 'connection_established') {
            console.log('Connection established:', data.message)
          } else if (data.type === 'training_complete') {
            console.log('Training completed:', data.message)
            setConnected(true) // Keep connection alive
          } else if (data.type === 'heartbeat') {
            // Heartbeat received, connection is alive
            if (data.mobile_optimized) {
              console.log('Mobile-optimized heartbeat received')
            }
          } else {
            console.log('Received message:', data)
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
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
        if (reconnectTimeoutRef.current || reconnectAttempts.current >= retryStrategy.maxAttempts) {
          return
        }

        if (reconnectAttempts.current < retryStrategy.maxAttempts) {
          reconnectAttempts.current++
          setConnectionError(`Connection lost. Reconnecting... (${reconnectAttempts.current}/${retryStrategy.maxAttempts})`)
          
          // Adaptive backoff based on network quality
          const backoff = Math.min(
            retryStrategy.baseDelay * Math.pow(2, reconnectAttempts.current - 1), 
            retryStrategy.maxDelay
          )
          
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectTimeoutRef.current = null
            reconnectingRef.current = false // allow new attempt
            connect()
          }, backoff)
        } else {
          setConnectionError('Connection lost. Please refresh the page.')
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        console.log('User agent:', navigator.userAgent)
        console.log('Is mobile Safari:', isMobileSafari())
        
        if (isMobileSafari()) {
          setConnectionError('Connection issue on mobile Safari - trying fallback...')
          // Try polling fallback after a delay
          setTimeout(() => {
            if (!isConnected) {
              console.log('Attempting polling fallback for mobile Safari')
              startPollingFallback()
            }
          }, 2000)
        } else {
          setConnectionError('Connection error. Please check if the server is running.')
        }
      }
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      setConnectionError('Failed to connect to server.')
    }
  }

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

  const startPollingFallback = async () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current)
    }
    
    console.log('Starting polling fallback for mobile Safari')
    setConnectionError('Using polling fallback for mobile Safari...')
    
    // Detect network quality to determine initial polling interval
    const networkQuality = networkQualityRef.current || await detectAndUpdateNetworkQuality()
    let currentPollingInterval = getAdaptivePollingInterval(networkQuality)
    
    console.log(`Using adaptive polling interval: ${currentPollingInterval}ms (network: ${networkQuality.level})`)
    
    const createPollingInterval = (interval: number) => {
      return setInterval(async () => {
        try {
          // Periodically re-check network quality and adjust polling interval
          const now = Date.now()
          if (now - lastNetworkCheckRef.current > 15000) { // Check every 15 seconds
            const updatedQuality = await detectAndUpdateNetworkQuality()
            const newInterval = getAdaptivePollingInterval(updatedQuality)
            
            if (newInterval !== currentPollingInterval) {
              console.log(`Network quality changed from ${networkQuality.level} to ${updatedQuality.level}`)
              console.log(`Updating polling interval from ${currentPollingInterval}ms to ${newInterval}ms`)
              
              currentPollingInterval = newInterval
              
              // Restart polling with new interval
              if (pollingIntervalRef.current) {
                clearInterval(pollingIntervalRef.current)
                pollingIntervalRef.current = createPollingInterval(newInterval)
              }
              return
            }
          }
          
          // Always poll training status
          const trainingResp = await fetch(`${config.api.baseUrl}/training/status`)
          if (trainingResp.ok) {
            const data = await trainingResp.json()
            if (!isConnected) {
              setConnected(true)
              setConnectionError(null)
            }
            if (data.is_training) {
              updateTrainingData({
                type: 'training_update',
                ...data
              })
            }
          }
          
          // Enhanced checkpoint playback polling - get actual playback data
          if (useTrainingStore.getState().isPlayingCheckpoint) {
            const playbackResp = await fetch(`${config.api.baseUrl}/checkpoints/playback/current`)
            if (playbackResp.ok) {
              const response = await playbackResp.json()
              
              if (response.has_data && response.playback_data) {
                // Update with actual playback data (game moves/visuals)
                updateCheckpointPlaybackData(response.playback_data)
                
                // Update health status if available
                if (response.playback_data.health_status) {
                  playbackHealthRef.current.isHealthy = response.playback_data.health_status.is_healthy !== false
                  playbackHealthRef.current.consecutiveFailures = response.playback_data.health_status.consecutive_failures || 0
                  playbackHealthRef.current.lastHealthCheck = Date.now()
                }
              } else {
                // No data available - check if we should show loading states
                const currentLoadingStates = useTrainingStore.getState().loadingStates
                if (!currentLoadingStates.isPlaybackStarting && !currentLoadingStates.isNewGameStarting) {
                  // Check if playback should be active but we're not getting data
                  const status = response.status
                  if (status && status.is_playing && !status.is_paused) {
                    // Server says it's playing but we're not getting data - show loading
                    useTrainingStore.getState().setLoadingState('loadingMessage', 'Waiting for game data...')
                  }
                }
              }
              
              // Also check status for state changes
              const status = response.status
              if (status && !status.is_playing) {
                // Playback ended or server stopped – update store
                useTrainingStore.getState().setPlayingCheckpoint(false)
                
                // Clear all loading states
                useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
                useTrainingStore.getState().setLoadingState('isNewGameStarting', false)
                useTrainingStore.getState().setLoadingState('loadingMessage', null)
                
                // If there were errors, show them
                if (status.error_count > 0) {
                  console.warn(`Playback stopped with ${status.error_count} errors`)
                  setConnectionError(`Playback stopped due to errors`)
                }
              }
              
              // Check playback health
              const healthCheck = checkPlaybackHealth()
              if (!healthCheck.isHealthy) {
                console.warn(`Playback health check failed: ${healthCheck.reason}`)
                if (healthCheck.reason === 'no_heartbeat') {
                  setConnectionError('Playback connection lost - restarting may be needed')
                  
                  // Try to restart playback if we haven't received data in a while
                  const timeSinceLastData = Date.now() - playbackHealthRef.current.lastHeartbeat
                  if (timeSinceLastData > 45000) { // 45 seconds
                    console.log('Attempting to restart playback due to no heartbeat')
                    await attemptPlaybackRestart()
                  }
                } else if (healthCheck.reason === 'system_unhealthy') {
                  setConnectionError('Playback system unhealthy - performance may be degraded')
                } else if (healthCheck.reason === 'too_many_failures') {
                  setConnectionError('Playback experiencing too many failures - restart recommended')
                }
              }
            } else {
              console.error('Failed to fetch playback data:', playbackResp.status)
              // If playback endpoint is failing, this might indicate a problem
              if (playbackResp.status >= 500) {
                setConnectionError('Playback server error - may need restart')
              } else if (playbackResp.status === 404) {
                // Playback might have stopped on server
                console.log('Playback not found on server, stopping local playback')
                useTrainingStore.getState().setPlayingCheckpoint(false)
                useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
                useTrainingStore.getState().setLoadingState('isNewGameStarting', false)
                useTrainingStore.getState().setLoadingState('loadingMessage', null)
              }
            }
          }
        } catch (error) {
          console.error('Polling fallback error:', error)
          
          // On error, increase polling interval to reduce server load
          if (currentPollingInterval < 10000) { // Max 10 seconds
            const newInterval = Math.min(currentPollingInterval * 1.5, 10000)
            console.log(`Polling error - increasing interval to ${newInterval}ms`)
            
            currentPollingInterval = newInterval
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current)
              pollingIntervalRef.current = createPollingInterval(newInterval)
            }
          }
          
          if (isConnected) {
            setConnected(false)
            setConnectionError('Polling fallback failed - server may be down')
          }
        }
      }, interval)
    }
    
    pollingIntervalRef.current = createPollingInterval(currentPollingInterval)
  }

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
          connect()
        }
      }, 1000)
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
              connect()
            }
          }, 500)
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
    reconnect: connect,
    disconnect,
  }
}

export const sendWebSocketMessage = (message: any) => {
  // This is a placeholder - in a real implementation, we'd manage the WebSocket reference
  console.log('Sending WebSocket message:', message)
} 