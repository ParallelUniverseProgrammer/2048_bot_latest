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
  Grid,
  List,
  RefreshCw
} from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'
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
  
  // Get training store loading states to prevent conflicts
  const { loadingStates } = useTrainingStore()
  
  // UI state
  const [searchTerm, setSearchTerm] = useState('')
  const [filterTag, setFilterTag] = useState<string>('all')
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string | null>(null)
  const [editingNickname, setEditingNickname] = useState<string | null>(null)
  const [newNickname, setNewNickname] = useState('')

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

    // ------------------------------------------------------------
    // NEW: periodically refresh checkpoint list so UI stays up-to-date
    // Use longer interval for mobile to reduce network load
    const checkpointInterval = setInterval(() => {
      loadCheckpoints(true)
    }, 10000) // Increased to 10 seconds for better mobile performance
    // ------------------------------------------------------------
    
    return () => {
      clearInterval(playbackInterval)
      clearInterval(checkpointInterval)
    }
  }, [])

  // Filter and search checkpoints
  const filteredCheckpoints = checkpoints.filter(cp => {
    const matchesSearch = cp.nickname.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         cp.id.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesFilter = filterTag === 'all' || cp.tags.includes(filterTag)
    return matchesSearch && matchesFilter
  })

  // Get all unique tags
  const allTags = Array.from(new Set(checkpoints.flatMap(cp => cp.tags)))

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
      return `${hours}h ${minutes}m ${secs}s`
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
      
      // Set loading state in training store
      useTrainingStore.getState().setLoadingState('isTrainingStarting', true)
      useTrainingStore.getState().setLoadingState('loadingMessage', 'Loading checkpoint and resuming training...')
      
              const res = await fetch(`${config.api.baseUrl}/checkpoints/${checkpointId}/load`, {
        method: 'POST'
      })
      
      if (!res.ok) {
        throw new Error('Failed to load checkpoint')
      }
      
      const result = await res.json()
      
      // Update training store state to reflect that training has started
      useTrainingStore.getState().setTrainingStatus(true, false)
      useTrainingStore.getState().setEpisode(result.episode)
      
      // Navigate to training dashboard automatically
      onNavigateToTab?.('dashboard')
      
      // Show success message
      console.log(`Checkpoint loaded successfully (Episode ${result.episode}) and training resumed`)
      
    } catch (err) {
      // Clear loading states on error
      useTrainingStore.getState().setLoadingState('isTrainingStarting', false)
      useTrainingStore.getState().setLoadingState('loadingMessage', null)
      setError(err instanceof Error ? err.message : 'Failed to load checkpoint')
    } finally {
      setLoading(false)
    }
  }

  // Playback controls
  // startPlayback, pausePlayback, resumePlayback, stopPlayback, setPlaybackSpeedAPI
  // These functions are now handled by GameBoard.
  
  // Start playback function (simplified - just starts playback and navigates)
  const startPlayback = async (checkpointId: string) => {
    try {
      // Set loading state for playback start
      useTrainingStore.getState().setLoadingState('isPlaybackStarting', true)
      useTrainingStore.getState().setLoadingState('loadingMessage', 'Loading checkpoint and starting playback...')
      
      // Navigate to game tab first to show loading state
      setSelectedCheckpoint(checkpointId)
      onNavigateToTab?.('game')
      
      // Small delay to ensure loading state is visible
      await new Promise(resolve => setTimeout(resolve, 100))
      
              const res = await fetch(`${config.api.baseUrl}/checkpoints/${checkpointId}/playback/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed: 1.0 })
      })
      
      if (!res.ok) {
        // Clear loading state on error
        useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
        useTrainingStore.getState().setLoadingState('loadingMessage', null)
        throw new Error('Failed to start playback')
      }
      
      // Loading state will be cleared when first playback data arrives
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start playback')
      // Clear loading state on error
      useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
      useTrainingStore.getState().setLoadingState('loadingMessage', null)
    }
  }

  // Check if we should show loading state
  const shouldShowLoading = loading || 
    loadingStates.isTrainingStarting || 
    loadingStates.isPlaybackStarting || 
    loadingStates.isNewGameStarting ||
    loadingStates.isTrainingStopping ||
    loadingStates.isTrainingResetting

  if (shouldShowLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center space-x-2">
          <RefreshCw className="w-5 h-5 animate-spin text-blue-400" />
          <span className="text-gray-400">
            {loadingStates.isTrainingStarting ? 'Starting training...' :
             loadingStates.isPlaybackStarting ? 'Starting playback...' :
             loadingStates.isNewGameStarting ? 'Starting new game...' :
             loadingStates.isTrainingStopping ? 'Stopping training...' :
             loadingStates.isTrainingResetting ? 'Resetting training...' :
             'Loading checkpoints...'}
          </span>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Error display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-500/20 border border-red-500/50 rounded-lg p-4"
        >
          <div className="flex items-center space-x-2">
            <X className="w-5 h-5 text-red-400" />
            <span className="text-red-400">{error}</span>
            <button
              onClick={() => setError(null)}
              className="ml-auto text-red-400 hover:text-red-300"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </motion.div>
      )}

      {/* Stats Overview */}
      {stats && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card-glass p-6 rounded-xl"
        >
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <Archive className="w-6 h-6 mr-2 text-blue-400" />
            Checkpoint Library
          </h2>
          
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-400">{stats.total_checkpoints}</div>
              <div className="text-sm text-gray-400">Total Checkpoints</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400">{formatFileSize(stats.total_size)}</div>
              <div className="text-sm text-gray-400">Total Size</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-400">{stats.best_score.toLocaleString()}</div>
              <div className="text-sm text-gray-400">Best Score</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-400">{stats.latest_episode.toLocaleString()}</div>
              <div className="text-sm text-gray-400">Latest Episode</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-400">{formatDuration(stats.total_training_time)}</div>
              <div className="text-sm text-gray-400">Total Training</div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Search and Filter Controls */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card-glass p-4 rounded-xl"
      >
        <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0 md:space-x-4">
          <div className="flex flex-col sm:flex-row sm:items-center space-y-3 sm:space-y-0 sm:space-x-4">
            <div className="relative flex-1 sm:flex-none">
              <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search checkpoints..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full sm:w-64 pl-10 pr-4 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <select
              value={filterTag}
              onChange={(e) => setFilterTag(e.target.value)}
              className="w-full sm:w-auto bg-gray-700 text-white rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Tags</option>
              {allTags.map(tag => (
                <option key={tag} value={tag}>{tag}</option>
              ))}
            </select>
          </div>
          
          <div className="flex items-center justify-center sm:justify-start space-x-2">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 rounded ${viewMode === 'grid' ? 'bg-blue-500 text-white' : 'bg-gray-700 text-gray-400'}`}
            >
              <Grid className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 rounded ${viewMode === 'list' ? 'bg-blue-500 text-white' : 'bg-gray-700 text-gray-400'}`}
            >
              <List className="w-4 h-4" />
            </button>
            <button
              onClick={() => loadCheckpoints()}
              className="p-2 bg-gray-700 text-gray-400 rounded hover:bg-gray-600"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </motion.div>

      {/* Checkpoints Grid/List */}
      <div className={viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6' : 'space-y-4'}>
        {filteredCheckpoints.map((checkpoint, index) => (
          <motion.div
            key={checkpoint.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`card-glass p-6 rounded-xl ${
              selectedCheckpoint === checkpoint.id ? 'ring-2 ring-blue-500' : ''
            }`}
          >
            {/* Checkpoint Header */}
            <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between mb-4 space-y-3 sm:space-y-0">
              <div className="flex-1 min-w-0">
                {editingNickname === checkpoint.id ? (
                  <div className="flex items-center space-x-2">
                    <input
                      type="text"
                      value={newNickname}
                      onChange={(e) => setNewNickname(e.target.value)}
                      className="flex-1 bg-gray-700 text-white rounded px-2 py-1 text-sm"
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
                    <h3 className="text-lg font-semibold text-white truncate">{checkpoint.nickname}</h3>
                    <button
                      onClick={() => {
                        setEditingNickname(checkpoint.id)
                        setNewNickname(checkpoint.nickname)
                      }}
                      className="text-gray-400 hover:text-gray-300 flex-shrink-0"
                    >
                      <Edit className="w-4 h-4" />
                    </button>
                  </div>
                )}
                <p className="text-sm text-gray-400">Episode {checkpoint.episode.toLocaleString()}</p>
              </div>
              
              <div className="flex items-center justify-center sm:justify-end space-x-1">
                <button
                  onClick={() => startPlayback(checkpoint.id)}
                  className="text-green-400 hover:text-green-300 flex items-center justify-center w-8 h-8 sm:w-auto sm:h-auto sm:px-2 sm:py-1 rounded"
                  title="Watch AI Play"
                >
                  <Play className="w-4 h-4" />
                  <span className="text-xs hidden sm:inline sm:ml-1">Watch</span>
                </button>
                <button
                  onClick={() => loadCheckpointForTraining(checkpoint.id)}
                  className="text-blue-400 hover:text-blue-300 flex items-center justify-center w-8 h-8 sm:w-auto sm:h-auto sm:px-2 sm:py-1 rounded"
                  title="Resume Training from this Checkpoint"
                >
                  <Zap className="w-4 h-4" />
                  <span className="text-xs hidden sm:inline sm:ml-1">Resume</span>
                </button>
                <button
                  onClick={() => deleteCheckpoint(checkpoint.id)}
                  className="text-red-400 hover:text-red-300 flex items-center justify-center w-8 h-8 sm:w-auto sm:h-auto sm:px-2 sm:py-1 rounded"
                  title="Delete Checkpoint"
                >
                  <Trash2 className="w-4 h-4" />
                  <span className="text-xs hidden sm:inline sm:ml-1">Delete</span>
                </button>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="text-center">
                <div className="text-xl font-bold text-green-400">
                  {checkpoint.performance_metrics.best_score.toLocaleString()}
                </div>
                <div className="text-xs text-gray-400">Best Score</div>
              </div>
              <div className="text-center">
                <div className="text-xl font-bold text-blue-400">
                  {checkpoint.performance_metrics.final_loss.toFixed(3)}
                </div>
                <div className="text-xs text-gray-400">Final Loss</div>
              </div>
            </div>

            {/* Model Config */}
            <div className="space-y-2 mb-4">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Model Size:</span>
                <span className="text-white">{checkpoint.model_config.model_size}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Experts:</span>
                <span className="text-white">{checkpoint.model_config.n_experts}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Layers:</span>
                <span className="text-white">{checkpoint.model_config.n_layers}</span>
              </div>
            </div>

            {/* Tags */}
            <div className="flex flex-wrap gap-2 mb-4">
              {checkpoint.tags.map(tag => (
                <span
                  key={tag}
                  className="px-2 py-1 bg-gray-700 text-gray-300 rounded-full text-xs"
                >
                  {tag}
                </span>
              ))}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between text-xs text-gray-400">
              <div className="flex items-center space-x-1">
                <Clock className="w-3 h-3" />
                <span>{new Date(checkpoint.created_at).toLocaleDateString()}</span>
              </div>
              <div className="flex items-center space-x-1">
                <HardDrive className="w-3 h-3" />
                <span>{formatFileSize(checkpoint.file_size)}</span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {filteredCheckpoints.length === 0 && (
        <div className="text-center py-12">
          <Archive className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-400">No checkpoints found</p>
        </div>
      )}
    </div>
  )
}

export default CheckpointManager 