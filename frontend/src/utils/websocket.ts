import { useEffect, useRef } from 'react'
import { useTrainingStore } from '../stores/trainingStore'
import config from './config'
import { 
  isMobileSafari, 
  getMobileOptimizedWebSocketURL, 
  getConnectionRetryDelay, 
  getMaxReconnectAttempts
} from './mobile-detection'

export const useWebSocket = () => {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)
  const reconnectAttempts = useRef(0)
  const pollingIntervalRef = useRef<number | null>(null)
  const maxReconnectAttempts = getMaxReconnectAttempts()
  const reconnectDelay = getConnectionRetryDelay()
  const reconnectingRef = useRef(false) // NEW: race-condition guard for reconnect attempts

  const { 
    setConnected, 
    setConnectionError, 
    updateTrainingData,
    updateCheckpointPlaybackData,
    isConnected 
  } = useTrainingStore()

  const connect = () => {
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

    try {
      const websocketUrl = getMobileOptimizedWebSocketURL(config.websocket.url)
      console.log('Connecting to WebSocket:', websocketUrl)
      
      const ws = new WebSocket(websocketUrl)
      wsRef.current = ws

      ws.onopen = () => {
        console.log('WebSocket connected successfully')
        reconnectingRef.current = false // connection succeeded – clear guard
        setConnected(true)
        setConnectionError(null)
        reconnectAttempts.current = 0
        
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
            } else if (data.status === 'stopped') {
              useTrainingStore.getState().setPlayingCheckpoint(false)
            }
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
          } else if (data.type === 'game_completed') {
            console.log(`Game completed: Score ${data.final_score}, Steps ${data.total_steps}, Max tile ${data.max_tile}`)
            // Set loading state for new game starting
            useTrainingStore.getState().setLoadingState('isNewGameStarting', true)
            useTrainingStore.getState().setLoadingState('loadingMessage', 'Starting new game...')
          } else if (data.type === 'new_game_started') {
            // Clear new game loading state
            useTrainingStore.getState().setLoadingState('isNewGameStarting', false)
            useTrainingStore.getState().setLoadingState('loadingMessage', null)
          } else if (data.type === 'training_start') {
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
        
        // Avoid multiple concurrent reconnect timers
        if (reconnectTimeoutRef.current || reconnectAttempts.current >= maxReconnectAttempts) {
          return
        }

        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++
          setConnectionError(`Connection lost. Reconnecting... (${reconnectAttempts.current}/${maxReconnectAttempts})`)
          
          // Exponential backoff capped at 30s
          const backoff = Math.min(reconnectDelay * Math.pow(2, reconnectAttempts.current - 1), 30000)
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

  const startPollingFallback = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current)
    }
    
    console.log('Starting polling fallback for mobile Safari')
    setConnectionError('Using polling fallback for mobile Safari...')
    
    pollingIntervalRef.current = setInterval(async () => {
      try {
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
        
        // -------- NEW: checkpoint playback polling --------
        if (useTrainingStore.getState().isPlayingCheckpoint) {
          const playbackResp = await fetch(`${config.api.baseUrl}/checkpoints/playback/status`)
          if (playbackResp.ok) {
            const status = await playbackResp.json()
            if (!status.is_playing) {
              // Playback ended or server stopped – update store
              useTrainingStore.getState().setPlayingCheckpoint(false)
            }
          }
        }
        //---------------------------------------------------
      } catch (error) {
        console.error('Polling fallback error:', error)
        if (isConnected) {
          setConnected(false)
          setConnectionError('Polling fallback failed - server may be down')
        }
      }
    }, 2000) // Poll every 2 seconds
  }

  useEffect(() => {
    connect()
    
    return () => {
      disconnect()
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