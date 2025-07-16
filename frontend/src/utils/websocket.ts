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

  // Enhanced reconnection with exponential backoff and circuit breaker
  const reconnect = useCallback(async () => {
    if (reconnectingRef.current || reconnectAttempts >= maxReconnectAttempts) {
      return
    }
    
    // Implement circuit breaker pattern
    if (connectionHealth === 'critical' && !isInRecoveryMode) {
      console.log('Connection health is critical, entering recovery mode')
      setIsInRecoveryMode(true)
      
      // Wait longer before attempting recovery
      setTimeout(() => {
        setIsInRecoveryMode(false)
        setConsecutiveFailures(0)
        setReconnectAttempts(0)
      }, 30000) // 30 second recovery period
      
      return
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    
    const backoff = Math.min(getConnectionRetryDelay() * Math.pow(2, reconnectAttempts), 30000)
    console.log(`Reconnecting in ${backoff}ms (attempt ${reconnectAttempts + 1}/${maxReconnectAttempts})`)
    
    reconnectTimeoutRef.current = setTimeout(async () => {
      try {
        setReconnectAttempts(prev => prev + 1)
        await connect()
      } catch (error) {
        console.error('Reconnection failed:', error)
        setConsecutiveFailures(prev => prev + 1)
        
                 if (reconnectAttempts < maxReconnectAttempts) {
           setTimeout(() => reconnect(), 1000) // Try again with exponential backoff
         } else {
           console.log('Max reconnection attempts reached, switching to polling fallback')
           setConnectionError('Connection lost after multiple attempts. Switching to polling...')
           startPollingFallback()
         }
      }
    }, backoff)
     }, [reconnectAttempts, connectionHealth, isInRecoveryMode, maxReconnectAttempts])

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
                  console.log('Upgrading from polling to WebSocket')
                  clearInterval(pollingIntervalRef.current!)
                  pollingIntervalRef.current = null
                  setReconnectAttempts(0)
                  connect()
                }
              }, 10000) // Wait 10 seconds before upgrade
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

  // Enhanced message processing with health tracking
  const processMessage = useCallback((event: MessageEvent) => {
    const data = JSON.parse(event.data)
    console.log('[WebSocket] Received message:', data)
    setLastSuccessfulConnection(Date.now())
    setConsecutiveFailures(0)
    updateConnectionStats(JSON.stringify(data).length)
    
    // Handle pong responses for latency measurement
    if (data.type === 'pong') {
      const latency = Date.now() - data.timestamp
      connectionStatsRef.current.latency = latency
      return
    }
    
    // Record successful message processing
    playbackHealthRef.current.lastHeartbeat = Date.now()
    playbackHealthRef.current.consecutiveFailures = 0
    playbackHealthRef.current.isHealthy = true
    
         if (data.type === 'training_update') {
       if (data.loss_history && data.score_history) {
         updateTrainingData(data)
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
       // Clear any loading states
       useTrainingStore.getState().setLoadingState('isTrainingStarting', false)
       useTrainingStore.getState().setLoadingState('loadingMessage', null)
     } else if (data.type === 'checkpoint_playback' || data.type === 'checkpoint_playback_light') {
       updateCheckpointPlaybackData(data)
     } else if (data.type === 'playback_status') {
       console.log('Playback status:', data.message)
       // Handle playback loading states
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
     } else if (data.type === 'game_completed') {
       useTrainingStore.getState().setLoadingState('isNewGameStarting', true)
       useTrainingStore.getState().setLoadingState('loadingMessage', 'Starting new game...')
     } else if (data.type === 'new_game_started') {
       useTrainingStore.getState().setLoadingState('isNewGameStarting', false)
       useTrainingStore.getState().setLoadingState('loadingMessage', null)
     } else if (data.type === 'new_episode_started') {
       console.log('Received new episode started:', data)
       if (data.episode !== undefined) {
         useTrainingStore.getState().setEpisode(data.episode)
       }
     }
  }, [updateTrainingData, updateCheckpointPlaybackData])

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
      const websocketUrl = getMobileOptimizedWebSocketURL(config.websocket.url)
      console.log('Connecting to WebSocket:', websocketUrl)
      
      const ws = new WebSocket(websocketUrl)
      wsRef.current = ws
      
      // Add connection timeout for mobile devices
      let connectionTimeout: number | null = null
      if (isMobile()) {
        connectionTimeout = setTimeout(() => {
          if (ws.readyState === WebSocket.CONNECTING) {
            console.log('WebSocket connection timeout on mobile, trying fallback')
            ws.close()
            setConnectionError('Connection timeout - switching to polling...')
            startPollingFallback()
          }
        }, 5000) // 5 second timeout for mobile
      }
      
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

        // Sync training status with backend to avoid stale localStorage state
        fetch(`${config.api.baseUrl}/training/status`)
          .then((res) => res.json())
          .then((status) => {
            console.log('Syncing training status with backend:', status)
            // Update the training store with the actual backend status
            useTrainingStore.getState().setTrainingStatus(status.is_training, status.is_paused)
            useTrainingStore.getState().setEpisode(status.current_episode)
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
          const data = JSON.parse(event.data)
          updateConnectionStats(event.data.length)
          
          // Handle ping/pong for latency measurement
          if (data.type === 'pong' && data.timestamp) {
            const latency = Date.now() - data.timestamp
            connectionStatsRef.current.latency = latency
            console.log(`WebSocket latency: ${latency}ms`)
            return
          }
          
          // Handle message batches
          if (data.type === 'message_batch') {
            // Process each message in the batch
            for (const message of data.messages) {
              processMessage(message)
            }
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
          
          processMessage(event)
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }
              
        ws.onmessage = (event) => {
          try {
            processMessage(event)
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

        if (reconnectAttempts < maxReconnectAttempts) {
          setReconnectAttempts(prev => prev + 1)
          setConnectionError(`Connection lost. Reconnecting... (${reconnectAttempts + 1}/${maxReconnectAttempts})`)
          
          // Adaptive backoff based on network quality
          const backoff = Math.min(
            reconnectDelay * Math.pow(2, reconnectAttempts), 
            30000
          )
          
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectTimeoutRef.current = null
            reconnectingRef.current = false // allow new attempt
            connect()
          }, backoff)
        } else {
          // After all retries failed, try polling fallback on mobile
          if (isMobileSafari() || isMobile()) {
            console.log('All WebSocket retries failed, switching to polling fallback')
            setConnectionError('WebSocket failed, switching to polling mode...')
            startPollingFallback()
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
          setConnectionError('Connection issue on mobile device - trying fallback...')
          // Try polling fallback after a shorter delay for mobile
          setTimeout(() => {
            if (!isConnected) {
              console.log('Attempting polling fallback for mobile device')
              startPollingFallback()
            }
          }, 1000) // Reduced delay for mobile
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