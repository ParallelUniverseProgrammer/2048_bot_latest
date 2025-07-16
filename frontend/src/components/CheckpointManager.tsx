import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Archive, 
  Play, 
  Trash2, 
  Edit, 
  Check, 
  X, 
  Clock, 
  Zap, 
  HardDrive,
  Search,
  RefreshCw,
  ChevronDown,
  ChevronUp
} from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'
import { useDeviceDetection } from '../utils/deviceDetection'
import config from '../utils/config'

interface Checkpoint {
  id: string
  nickname: string
  episode: number
  created_at: string
  training_duration: number
  model_config: {
    model_size: string
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

interface CheckpointStats {
  total_checkpoints: number
  total_size: number
  best_score: number
  latest_episode: number
  total_training_time: number
}

interface CheckpointManagerProps {
  onNavigateToTab?: (tab: string) => void
}

const CheckpointManager: React.FC<CheckpointManagerProps> = ({ onNavigateToTab }) => {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [stats, setStats] = useState<CheckpointStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  // Device detection for mobile optimization
  useDeviceDetection()
  
  // UI state
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string | null>(null)
  const [editingNickname, setEditingNickname] = useState<string | null>(null)
  const [newNickname, setNewNickname] = useState('')
  const [expandedCheckpoint, setExpandedCheckpoint] = useState<string | null>(null)

  // Cleanup loading states on component unmount
  useEffect(() => {
    return () => {
      // Clear loading states to prevent stuck states after navigation
      useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
      useTrainingStore.getState().setLoadingState('loadingMessage', null)
    }
  }, [])

  // Load checkpoints and stats
  const loadCheckpoints = async (silent: boolean = false) => {
    try {
      if (!silent) setLoading(true)
      
      // Use AbortController for better timeout handling on mobile
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 15000) // 15 second timeout for mobile
      
      try {
        const [checkpointsRes, statsRes] = await Promise.all([
          fetch(`${config.api.baseUrl}/checkpoints`, { signal: controller.signal }),
          fetch(`${config.api.baseUrl}/checkpoints/stats`, { signal: controller.signal })
        ])
        
        clearTimeout(timeoutId)
        
        if (!checkpointsRes.ok) {
          console.error('Failed to load checkpoints:', checkpointsRes.status)
        }
        if (!statsRes.ok) {
          console.error('Failed to load stats:', statsRes.status)
        }
        
        const checkpointsData = checkpointsRes.ok ? await checkpointsRes.json() : []
        const statsData = statsRes.ok ? await statsRes.json() : null
        
        setCheckpoints(checkpointsData)
        setStats(statsData)
        setError(null)
      } catch (fetchErr) {
        clearTimeout(timeoutId)
        
        // Handle abort errors more gracefully
        if (fetchErr instanceof Error && fetchErr.name === 'AbortError') {
          console.warn('Request timed out, using cached data if available')
          if (silent && checkpoints.length > 0) {
            // For silent refreshes, keep existing data if timeout occurs
            return
          }
          throw new Error('Request timed out - please check your connection')
        }
        throw fetchErr
      }
    } catch (err) {
      console.error('Error loading checkpoints:', err)
      
      // For silent refreshes, don't show error if we have existing data
      if (silent && checkpoints.length > 0) {
        console.warn('Silent refresh failed, keeping existing data')
        return
      }
      
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      // Always clear loading state, even for silent refreshes that fail
      setLoading(false)
    }
  }

  useEffect(() => {
    loadCheckpoints()
    
    // Poll for playback status updates
    const playbackInterval = setInterval(async () => {
      try {
        const res = await fetch(`${config.api.baseUrl}/checkpoints/playback/status`)
        if (res.ok) {
          // setPlaybackStatus(data) // This state is removed
        }
      } catch (err) {
        // Ignore polling errors
      }
    }, 1000)

    // NEW: periodically refresh checkpoint list so UI stays up-to-date
    // Use longer interval for mobile to reduce network load
    const checkpointInterval = setInterval(() => {
      loadCheckpoints(true)
    }, 10000) // Increased to 10 seconds for better mobile performance
    
    return () => {
      clearInterval(playbackInterval)
      clearInterval(checkpointInterval)
    }
  }, [])

  // Filter and search checkpoints
  const filteredCheckpoints = checkpoints.filter(cp => {
    const matchesSearch = cp.nickname.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         cp.id.toLowerCase().includes(searchTerm.toLowerCase())
    return matchesSearch
  })

  // Format file size
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`
    } else if (bytes < 1024 * 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    } else {
      return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
    }
  }

  // Format duration
  const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`
    } else {
      return `${secs}s`
    }
  }

  // Update checkpoint nickname
  const updateNickname = async (checkpointId: string, nickname: string) => {
    try {
      const res = await fetch(`${config.api.baseUrl}/checkpoints/${checkpointId}/nickname`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nickname })
      })
      
      if (!res.ok) {
        throw new Error('Failed to update nickname')
      }
      
      // Update local state
      setCheckpoints(prev => prev.map(cp => 
        cp.id === checkpointId ? { ...cp, nickname } : cp
      ))
      
      setEditingNickname(null)
      setNewNickname('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update nickname')
    }
  }

  // Delete checkpoint
  const deleteCheckpoint = async (checkpointId: string) => {
    if (!confirm('Are you sure you want to delete this checkpoint?')) {
      return
    }
    
    try {
      const res = await fetch(`${config.api.baseUrl}/checkpoints/${checkpointId}`, {
        method: 'DELETE'
      })
      
      if (!res.ok) {
        throw new Error('Failed to delete checkpoint')
      }
      
      // Remove from local state
      setCheckpoints(prev => prev.filter(cp => cp.id !== checkpointId))
      
      // Reload stats
      loadCheckpoints()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete checkpoint')
    }
  }

  // Load checkpoint for training
  const loadCheckpointForTraining = async (checkpointId: string) => {
    try {
      setLoading(true)
      
      // Initialize checkpoint loading state
      useTrainingStore.getState().setCheckpointLoadingState({
        isCheckpointLoading: true,
        checkpointId: checkpointId,
        loadingMessage: 'Starting checkpoint load...',
        loadingProgress: 0
      })
      
      const res = await fetch(`${config.api.baseUrl}/checkpoints/${checkpointId}/load`, {
        method: 'POST'
      })
      
      if (!res.ok) {
        throw new Error('Failed to load checkpoint')
      }
      
      await res.json() // Response contains status info, but we rely on WebSocket for progress
      
      // The backend will send WebSocket messages with progress updates
      // The loading state will be managed by the WebSocket message handler
      console.log(`Checkpoint load initiated for ${checkpointId}`)
      
      // Navigate to training dashboard to show the loading progress
      onNavigateToTab?.('dashboard')
      
    } catch (err) {
      // Set error state
      useTrainingStore.getState().setCheckpointLoadingError(
        err instanceof Error ? err.message : 'Failed to load checkpoint'
      )
      setError(err instanceof Error ? err.message : 'Failed to load checkpoint')
    } finally {
      setLoading(false)
    }
  }

  // Start playback function (simplified - just starts playback and navigates)
  const startPlayback = async (checkpointId: string) => {
    try {
      // Start enhanced loading operation
      const loadingSteps = [
        'Loading checkpoint metadata...',
        'Validating checkpoint integrity...',
        'Initializing playback environment...',
        'Starting game simulation...',
        'Waiting for first game data...'
      ]
      
      useTrainingStore.getState().startLoadingOperation('playback', loadingSteps)
      
      // Navigate to game tab first to show loading state
      setSelectedCheckpoint(checkpointId)
      onNavigateToTab?.('game')
      
      // Small delay to ensure loading state is visible
      await new Promise(resolve => setTimeout(resolve, 100))
      
      // Simulate step progression
      setTimeout(() => useTrainingStore.getState().updateLoadingProgress(20, loadingSteps[1]), 500)
      setTimeout(() => useTrainingStore.getState().updateLoadingProgress(40, loadingSteps[2]), 1000)
      setTimeout(() => useTrainingStore.getState().updateLoadingProgress(60, loadingSteps[3]), 1500)
      
      const res = await fetch(`${config.api.baseUrl}/checkpoints/${checkpointId}/playback/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      
      if (!res.ok) {
        // Clear loading state on API error
        useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
        useTrainingStore.getState().setLoadingState('loadingMessage', null)
        const errorMessage = res.status === 404 
          ? 'Checkpoint not found. It may have been deleted or the backend is not running.'
          : res.status === 500
          ? 'Server error occurred. Please check if the backend is running and try again.'
          : res.status === 503
          ? 'Service temporarily unavailable. Please wait a moment and try again.'
          : `Failed to start playback: ${res.status} ${res.statusText}`
        throw new Error(errorMessage)
      }
      
      // Update to final step
      useTrainingStore.getState().updateLoadingProgress(80, loadingSteps[4], 5)
      
      // Loading state will be cleared when first playback data arrives or timeout expires
      console.log('Playback API call successful. Waiting for WebSocket data...')
      
    } catch (err) {
      // Clear loading state on any error
      useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
      useTrainingStore.getState().setLoadingState('loadingMessage', null)
      
      const errorMessage = err instanceof Error ? err.message : 'Failed to start playback'
      setError(errorMessage)
      console.error('Playback start error:', err)
    }
  }

  // Check if we should show loading state
  const shouldShowLoading = loading

  if (shouldShowLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="flex items-center space-x-3">
          <RefreshCw className="w-5 h-5 animate-spin text-blue-400" />
          <span className="text-gray-400">Loading checkpoints...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col space-y-2 pb-6">
      {/* Error display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-red-500/20 border border-red-500/50 rounded-2xl p-4 flex-shrink-0"
        >
          <div className="flex items-center space-x-2">
            <X className="w-4 h-4 text-red-400" />
            <span className="text-red-400 text-sm flex-1">{error}</span>
            <button
              onClick={() => setError(null)}
              className="text-red-400 hover:text-red-300"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </motion.div>
      )}

      {/* Stats Overview */}
      {stats && (
        <motion.div
          className="card-glass p-4 rounded-2xl flex-shrink-0"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="grid grid-cols-2 gap-3">
            <div className="text-center">
              <div className="text-lg font-bold text-blue-400">{stats.total_checkpoints}</div>
              <div className="text-xs text-gray-400">Checkpoints</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-green-400">{stats.best_score.toLocaleString()}</div>
              <div className="text-xs text-gray-400">Best Score</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-purple-400">{formatFileSize(stats.total_size)}</div>
              <div className="text-xs text-gray-400">Total Size</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-orange-400">{formatDuration(stats.total_training_time)}</div>
              <div className="text-xs text-gray-400">Training Time</div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Search and Controls */}
      <motion.div
        className="card-glass p-4 rounded-2xl flex-shrink-0"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center space-x-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search checkpoints..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-gray-700 text-white rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
            />
          </div>
          <button
            onClick={() => loadCheckpoints()}
            className="p-2 bg-gray-700 text-gray-400 rounded-xl hover:bg-gray-600"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </motion.div>

      {/* Checkpoints List */}
      <div className="flex-1 overflow-y-auto space-y-2">
        {filteredCheckpoints.map((checkpoint, index) => (
          <motion.div
            key={checkpoint.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className={`card-glass rounded-2xl p-4 ${
              selectedCheckpoint === checkpoint.id ? 'ring-2 ring-blue-500' : ''
            }`}
          >
            {/* Checkpoint Header */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex-1 min-w-0">
                {editingNickname === checkpoint.id ? (
                  <div className="flex items-center space-x-2">
                    <input
                      type="text"
                      value={newNickname}
                      onChange={(e) => setNewNickname(e.target.value)}
                      className="flex-1 bg-gray-700 text-white rounded-lg px-3 py-1.5 text-sm"
                      autoFocus
                    />
                    <button
                      onClick={() => updateNickname(checkpoint.id, newNickname)}
                      className="text-green-400 hover:text-green-300"
                    >
                      <Check className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => {
                        setEditingNickname(null)
                        setNewNickname('')
                      }}
                      className="text-red-400 hover:text-red-300"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center space-x-2">
                    <h3 className="font-semibold text-white truncate text-sm">{checkpoint.nickname}</h3>
                    <button
                      onClick={() => {
                        setEditingNickname(checkpoint.id)
                        setNewNickname(checkpoint.nickname)
                      }}
                      className="text-gray-400 hover:text-gray-300 flex-shrink-0"
                    >
                      <Edit className="w-3 h-3" />
                    </button>
                  </div>
                )}
              </div>
              
              <button
                onClick={() => setExpandedCheckpoint(
                  expandedCheckpoint === checkpoint.id ? null : checkpoint.id
                )}
                className="text-gray-400 hover:text-gray-300 ml-2"
              >
                {expandedCheckpoint === checkpoint.id ? 
                  <ChevronUp className="w-4 h-4" /> : 
                  <ChevronDown className="w-4 h-4" />
                }
              </button>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-3 gap-2 mb-3">
              <div className="text-center">
                <div className="text-sm font-bold text-green-400">
                  {checkpoint.performance_metrics.best_score.toLocaleString()}
                </div>
                <div className="text-xs text-gray-400">Score</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-bold text-blue-400">
                  {checkpoint.episode.toLocaleString()}
                </div>
                <div className="text-xs text-gray-400">Episode</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-bold text-purple-400">
                  {checkpoint.model_config.model_size}
                </div>
                <div className="text-xs text-gray-400">Size</div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex space-x-2">
              <button
                onClick={() => startPlayback(checkpoint.id)}
                className="flex-1 flex items-center justify-center space-x-2 bg-green-500/20 text-green-400 rounded-xl py-2.5 text-sm font-medium"
              >
                <Play className="w-4 h-4" />
                <span>Watch</span>
              </button>
              <button
                onClick={() => loadCheckpointForTraining(checkpoint.id)}
                className="flex-1 flex items-center justify-center space-x-2 bg-blue-500/20 text-blue-400 rounded-xl py-2.5 text-sm font-medium"
              >
                <Zap className="w-4 h-4" />
                <span>Resume</span>
              </button>
              <button
                onClick={() => deleteCheckpoint(checkpoint.id)}
                className="flex items-center justify-center bg-red-500/20 text-red-400 rounded-xl py-2.5 px-3 text-sm font-medium"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>

            {/* Expanded Details */}
            {expandedCheckpoint === checkpoint.id && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-3 pt-3 border-t border-gray-700"
              >
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <div className="text-gray-400 mb-1">Model Config</div>
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <span>Experts:</span>
                        <span className="text-white">{checkpoint.model_config.n_experts}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Layers:</span>
                        <span className="text-white">{checkpoint.model_config.n_layers}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Dimensions:</span>
                        <span className="text-white">{checkpoint.model_config.d_model}</span>
                      </div>
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400 mb-1">Performance</div>
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <span>Avg Score:</span>
                        <span className="text-white">{checkpoint.performance_metrics.avg_score.toFixed(0)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Final Loss:</span>
                        <span className="text-white">{checkpoint.performance_metrics.final_loss.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Speed:</span>
                        <span className="text-white">{checkpoint.performance_metrics.training_speed.toFixed(1)} ep/min</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center justify-between mt-3 pt-2 border-t border-gray-700">
                  <div className="flex items-center space-x-1 text-xs text-gray-400">
                    <Clock className="w-3 h-3" />
                    <span>{new Date(checkpoint.created_at).toLocaleDateString()}</span>
                  </div>
                  <div className="flex items-center space-x-1 text-xs text-gray-400">
                    <HardDrive className="w-3 h-3" />
                    <span>{formatFileSize(checkpoint.file_size)}</span>
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>
        ))}

        {filteredCheckpoints.length === 0 && (
          <div className="text-center py-12">
            <Archive className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-400">No checkpoints found</p>
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                className="text-blue-400 hover:text-blue-300 text-sm mt-2"
              >
                Clear search
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default CheckpointManager 