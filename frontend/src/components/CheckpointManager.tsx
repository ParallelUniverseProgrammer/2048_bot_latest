import React, { useState, useEffect, useRef, useMemo } from 'react'
import { motion } from 'framer-motion'
import { 
  Archive, 
  Play, 
  Trash2, 
  Edit, 
  Check, 
  X, 
  Zap, 
  Search,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Settings,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Activity,
  Target,
  Calendar,
  Layers,
  Cpu,
  Database,
  SortAsc,
  SortDesc
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
  // ===== STATE MANAGEMENT =====
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [stats, setStats] = useState<CheckpointStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  // ===== DEVICE DETECTION =====
  const { displayMode } = useDeviceDetection()
  const isMobile = displayMode === 'mobile'
  
  // ===== UI STATE =====
  const [searchTerm, setSearchTerm] = useState('')
  const [editingNickname, setEditingNickname] = useState<string | null>(null)
  const [newNickname, setNewNickname] = useState('')
  const [expandedCheckpoint, setExpandedCheckpoint] = useState<string | null>(null)
  
  // ===== FILTER & SORT STATE =====
  const [sortBy, setSortBy] = useState<'date' | 'score' | 'episode' | 'size'>('date')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  const [filterBy, setFilterBy] = useState<'all' | 'recent' | 'high-score' | 'large'>('all')
  
  // ===== CONFIGURATION STATE =====
  const [checkpointInterval, setCheckpointInterval] = useState(50)
  const [checkpointIntervalInput, setCheckpointIntervalInput] = useState('50')
  const [checkpointIntervalError, setCheckpointIntervalError] = useState<string | null>(null)
  const [longRunMode, setLongRunMode] = useState(false)
  const [showConfigPanel, setShowConfigPanel] = useState(false)
  const [configLoading, setConfigLoading] = useState(false)
  
  // ===== REFS =====
  const editingInputRef = useRef<HTMLInputElement>(null)

  // Cleanup loading states on component unmount
  useEffect(() => {
    return () => {
      useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
      useTrainingStore.getState().setLoadingState('loadingMessage', null)
    }
  }, [])

  // Focus management for mobile keyboard
  useEffect(() => {
    if (editingNickname && editingInputRef.current) {
      const timer = setTimeout(() => {
        if (editingInputRef.current) {
          editingInputRef.current.focus()
          editingInputRef.current.select()
          editingInputRef.current.click()
        }
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [editingNickname])

  // Load checkpoints and stats
  const loadCheckpoints = async (silent: boolean = false) => {
    try {
      if (!silent) setLoading(true)
      
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 15000)
      
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
        
        if (fetchErr instanceof Error && fetchErr.name === 'AbortError') {
          console.warn('Request timed out, using cached data if available')
          if (silent && checkpoints.length > 0) {
            return
          }
          throw new Error('Request timed out - please check your connection')
        }
        throw fetchErr
      }
    } catch (err) {
      console.error('Error loading checkpoints:', err)
      
      if (silent && checkpoints.length > 0) {
        console.warn('Silent refresh failed, keeping existing data')
        return
      }
      
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadCheckpoints()
    
    const playbackInterval = setInterval(async () => {
      try {
        const res = await fetch(`${config.api.baseUrl}/checkpoints/playback/status`)
        if (res.ok) {
          // Status updates handled via WebSocket
        }
      } catch (err) {
        // Ignore polling errors
      }
    }, 1000)

    const checkpointInterval = setInterval(() => {
      loadCheckpoints(true)
    }, 10000)
    
    loadCheckpointConfig()
    
    return () => {
      clearInterval(playbackInterval)
      clearInterval(checkpointInterval)
    }
  }, [])

  // Load checkpoint configuration
  const loadCheckpointConfig = async () => {
    try {
      const res = await fetch(`${config.api.baseUrl}/training/checkpoint/config`)
      if (res.ok) {
        const config = await res.json()
        setCheckpointInterval(config.interval)
        setCheckpointIntervalInput(config.interval.toString())
        setLongRunMode(config.long_run_mode)
      }
    } catch (err) {
      console.error('Error loading checkpoint config:', err)
    }
  }

  // Validate checkpoint interval
  const validateCheckpointInterval = (value: string): number | null => {
    const num = parseInt(value)
    if (isNaN(num) || num < 1) {
      return null
    }
    if (num > 1000) {
      return null
    }
    return num
  }

  // Handle checkpoint interval input changes
  const handleCheckpointIntervalChange = (value: string) => {
    setCheckpointIntervalInput(value)
    setCheckpointIntervalError(null)
  }

  // Handle checkpoint interval blur (validation)
  const handleCheckpointIntervalBlur = () => {
    const validatedValue = validateCheckpointInterval(checkpointIntervalInput)
    if (validatedValue === null) {
      setCheckpointIntervalError('Please enter a number between 1 and 1000')
      setCheckpointIntervalInput(checkpointInterval.toString())
    } else {
      setCheckpointInterval(validatedValue)
      setCheckpointIntervalInput(validatedValue.toString())
      setCheckpointIntervalError(null)
    }
  }

  // Save checkpoint configuration
  const saveCheckpointConfig = async () => {
    // Validate before saving
    const validatedValue = validateCheckpointInterval(checkpointIntervalInput)
    if (validatedValue === null) {
      setCheckpointIntervalError('Please enter a valid number between 1 and 1000')
      return
    }

    try {
      setConfigLoading(true)
      const res = await fetch(`${config.api.baseUrl}/training/checkpoint/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          interval: validatedValue,
          long_run_mode: longRunMode
        })
      })
      
      if (!res.ok) {
        throw new Error('Failed to save checkpoint configuration')
      }
      
      setCheckpointInterval(validatedValue)
      setShowConfigPanel(false)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save checkpoint configuration')
    } finally {
      setConfigLoading(false)
    }
  }

  // Enhanced filtering and sorting
  const filteredAndSortedCheckpoints = useMemo(() => {
    let filtered = checkpoints.filter(cp => {
      const matchesSearch = cp.nickname.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           cp.id.toLowerCase().includes(searchTerm.toLowerCase())
      
      if (!matchesSearch) return false
      
      // Apply additional filters
      switch (filterBy) {
        case 'recent':
          const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000)
          return new Date(cp.created_at) > oneDayAgo
        case 'high-score':
          return cp.performance_metrics.best_score > 1000
        case 'large':
          return cp.file_size > 10 * 1024 * 1024 // > 10MB
        default:
          return true
      }
    })
    
    // Sort checkpoints
    filtered.sort((a, b) => {
      let aValue: any, bValue: any
      
      switch (sortBy) {
        case 'date':
          aValue = new Date(a.created_at).getTime()
          bValue = new Date(b.created_at).getTime()
          break
        case 'score':
          aValue = a.performance_metrics.best_score
          bValue = b.performance_metrics.best_score
          break
        case 'episode':
          aValue = a.episode
          bValue = b.episode
          break
        case 'size':
          aValue = a.file_size
          bValue = b.file_size
          break
        default:
          return 0
      }
      
      return sortOrder === 'desc' ? bValue - aValue : aValue - bValue
    })
    
    return filtered
  }, [checkpoints, searchTerm, filterBy, sortBy, sortOrder])

  // ===== UTILITY FUNCTIONS =====
  
  /**
   * Format file size in bytes to human readable format
   */
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`
    } else if (bytes < 1024 * 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    } else {
      return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
    }
  }

  /**
   * Format duration in seconds to human readable format
   */
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

  /**
   * Get appropriate trend icon based on score performance
   */
  const getScoreTrendIcon = (score: number) => {
    if (score > 1000) return <TrendingUp className="w-4 h-4 text-green-400" />
    if (score > 500) return <BarChart3 className="w-4 h-4 text-yellow-400" />
    return <TrendingDown className="w-4 h-4 text-red-400" />
  }

  /**
   * Get text color for model size indicator
   */
  const getModelSizeColor = (size: string) => {
    switch (size.toLowerCase()) {
      case 'lightning': return 'text-blue-400'
      case 'base': return 'text-green-400'
      case 'expert': return 'text-purple-400'
      default: return 'text-gray-400'
    }
  }

  // ===== ACTION FUNCTIONS =====
  
  /**
   * Update checkpoint nickname via API
   */
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
      
      setCheckpoints(prev => prev.map(cp => 
        cp.id === checkpointId ? { ...cp, nickname } : cp
      ))
      
      setEditingNickname(null)
      setNewNickname('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update nickname')
    }
  }

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
      
      setCheckpoints(prev => prev.filter(cp => cp.id !== checkpointId))
      loadCheckpoints()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete checkpoint')
    }
  }

  const loadCheckpointForTraining = async (checkpointId: string) => {
    try {
      setLoading(true)
      
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
      
      await res.json()
      console.log(`Checkpoint load initiated for ${checkpointId}`)
      onNavigateToTab?.('dashboard')
      
    } catch (err) {
      useTrainingStore.getState().setCheckpointLoadingError(
        err instanceof Error ? err.message : 'Failed to load checkpoint'
      )
      setError(err instanceof Error ? err.message : 'Failed to load checkpoint')
    } finally {
      setLoading(false)
    }
  }

  const startPlayback = async (checkpointId: string) => {
    try {
      const loadingSteps = [
        'Loading checkpoint metadata...',
        'Validating checkpoint integrity...',
        'Initializing playback environment...',
        'Starting game simulation...',
        'Waiting for first game data...'
      ]
      
      useTrainingStore.getState().startLoadingOperation('playback', loadingSteps)
      onNavigateToTab?.('game')
      
      await new Promise(resolve => setTimeout(resolve, 100))
      
      setTimeout(() => useTrainingStore.getState().updateLoadingProgress(20, loadingSteps[1]), 500)
      setTimeout(() => useTrainingStore.getState().updateLoadingProgress(40, loadingSteps[2]), 1000)
      setTimeout(() => useTrainingStore.getState().updateLoadingProgress(60, loadingSteps[3]), 1500)
      
      const res = await fetch(`${config.api.baseUrl}/checkpoints/${checkpointId}/playback/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      
      if (!res.ok) {
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
      
      useTrainingStore.getState().updateLoadingProgress(80, loadingSteps[4], 5)
      console.log('Playback API call successful. Waiting for WebSocket data...')
      
    } catch (err) {
      useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
      useTrainingStore.getState().setLoadingState('loadingMessage', null)
      
      const errorMessage = err instanceof Error ? err.message : 'Failed to start playback'
      setError(errorMessage)
      console.error('Playback start error:', err)
    }
  }

  // Loading state
  if (loading) {
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
    <div className="safe-area h-full flex flex-col gap-2 pb-6 px-4">
      {/* Error Display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="card-glass p-4 rounded-2xl border border-red-500/30 bg-red-500/5 flex-shrink-0"
        >
          <div className="flex items-center space-x-3">
            <X className="w-5 h-5 text-ui-state-danger flex-shrink-0" />
            <span className="text-ui-state-danger text-sm flex-1">{error}</span>
            <button
              onClick={() => setError(null)}
              className="text-ui-state-danger hover:text-white p-2 rounded-lg hover:bg-ui-state-danger/20 transition-colors"
              aria-label="Dismiss error"
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
          <div className={`grid ${isMobile ? 'grid-cols-2' : 'grid-cols-4'} gap-3`}>
            <div className="text-center">
              <div className="text-lg font-bold text-ui-brand-primary numeric">{stats.total_checkpoints}</div>
              <div className="text-xs text-ui-text-secondary">Checkpoints</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-ui-state-success numeric">{stats.best_score.toLocaleString()}</div>
              <div className="text-xs text-ui-text-secondary">Best Score</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-ui-state-info numeric">{formatFileSize(stats.total_size)}</div>
              <div className="text-xs text-ui-text-secondary">Total Size</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-ui-state-warning numeric">{formatDuration(stats.total_training_time)}</div>
              <div className="text-xs text-ui-text-secondary">Training Time</div>
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
        {/* Search Bar */}
        <div className="relative mb-4">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-ui-text-secondary" />
          <input
            type="text"
            placeholder="Search checkpoints..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 bg-ui-surface-elevated text-ui-text-primary rounded-xl focus:outline-none focus:ring-2 focus:ring-ui-focus text-sm border border-ui-border-muted"
            inputMode="text"
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
            spellCheck="false"
            style={{ fontSize: '16px' }}
          />
        </div>

        {/* Filter and Sort Controls */}
        <div className="flex flex-wrap items-center gap-2 mb-4">
          {/* Filter Buttons */}
          <div className="flex space-x-1">
            {[
              { key: 'all', label: 'All' },
              { key: 'recent', label: 'Recent' },
              { key: 'high-score', label: 'High Score' }
            ].map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setFilterBy(key as any)}
                className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                  filterBy === key 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Sort Controls */}
          <div className="flex items-center space-x-1 ml-auto">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="bg-gray-700 text-white rounded-lg px-2 py-1.5 text-xs border border-gray-600 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="date">Date</option>
              <option value="score">Score</option>
              <option value="episode">Episode</option>
              <option value="size">Size</option>
            </select>
            <button
              onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
              className="p-2 bg-gray-700 text-gray-400 rounded-lg hover:bg-gray-600 transition-colors"
              aria-label={`Sort ${sortOrder === 'desc' ? 'ascending' : 'descending'}`}
            >
              {sortOrder === 'desc' ? <SortDesc className="w-3 h-3" /> : <SortAsc className="w-3 h-3" />}
            </button>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => loadCheckpoints()}
            className="flex items-center space-x-2 px-3 py-1.5 bg-ui-surface-elevated text-ui-text-secondary rounded-lg hover:bg-gray-700/50 transition-colors text-sm"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
          <button
            onClick={() => setShowConfigPanel(!showConfigPanel)}
            className="flex items-center space-x-2 px-3 py-1.5 bg-ui-surface-elevated text-ui-text-secondary rounded-lg hover:bg-gray-700/50 transition-colors text-sm"
          >
            <Settings className="w-4 h-4" />
            <span>Configuration</span>
          </button>
        </div>

        {/* Configuration Panel */}
        {showConfigPanel && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 pt-4 border-t border-ui-border-muted"
          >
            <h3 className="text-sm font-semibold text-white mb-3 flex items-center">
              <Settings className="w-4 h-4 mr-2 text-ui-brand-primary" />
              Checkpoint Configuration
            </h3>
            
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-ui-text-secondary mb-1">
                  Checkpoint Interval (episodes)
                </label>
                <input
                  type="number"
                  min="1"
                  max="1000"
                  value={checkpointIntervalInput}
                  onChange={(e) => handleCheckpointIntervalChange(e.target.value)}
                  onBlur={handleCheckpointIntervalBlur}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.currentTarget.blur()
                    }
                  }}
                  className={`w-full px-3 py-2 bg-ui-surface-elevated text-ui-text-primary rounded-lg focus:outline-none focus:ring-2 focus:ring-ui-focus text-sm border ${
                    checkpointIntervalError ? 'ring-2 ring-ui-state-danger border-ui-state-danger' : 'border-ui-border-muted'
                  }`}
                  inputMode="numeric"
                  autoComplete="off"
                  autoCorrect="off"
                  autoCapitalize="off"
                  spellCheck="false"
                  style={{ fontSize: '16px' }}
                />
                {checkpointIntervalError ? (
                  <p className="text-xs text-ui-state-danger mt-1">{checkpointIntervalError}</p>
                ) : (
                  <p className="text-xs text-ui-text-secondary mt-1">
                    Save a checkpoint every {checkpointInterval} episodes
                  </p>
                )}
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-xs text-ui-text-secondary block">Long Run Mode</label>
                  <p className="text-xs text-ui-text-secondary">
                    Only keep the latest checkpoint from this training run
                  </p>
                </div>
                <button
                  onClick={() => setLongRunMode(!longRunMode)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    longRunMode ? 'bg-ui-brand-primary' : 'bg-gray-600'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-ui-text-primary transition-transform ${
                      longRunMode ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
              
              <div className="flex space-x-2 pt-2">
                <button
                  onClick={saveCheckpointConfig}
                  disabled={configLoading}
                  className="flex-1 flex items-center justify-center space-x-2 bg-ui-brand-primary text-white rounded-xl py-2 text-sm font-medium disabled:opacity-50 hover:brightness-110 transition-colors"
                >
                  {configLoading ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    <Check className="w-4 h-4" />
                  )}
                  <span>{configLoading ? 'Saving...' : 'Save Configuration'}</span>
                </button>
                <button
                  onClick={() => setShowConfigPanel(false)}
                  className="flex items-center justify-center bg-ui-surface-elevated text-ui-text-secondary rounded-xl py-2 px-3 text-sm font-medium hover:bg-gray-700/50 transition-colors"
                  aria-label="Close configuration"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </motion.div>

      {/* Checkpoints List */}
      <div className="flex-1 overflow-y-auto space-y-2">
        {filteredAndSortedCheckpoints.map((checkpoint: Checkpoint, index: number) => (
          <motion.div
            key={checkpoint.id}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.05 }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="card-glass rounded-2xl p-4"
          >
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1 min-w-0">
                {editingNickname === checkpoint.id ? (
                  <div className="flex items-center space-x-2">
                    <input
                      ref={editingInputRef}
                      type="text"
                      value={newNickname}
                      onChange={(e) => setNewNickname(e.target.value)}
                      className="flex-1 bg-gray-700 text-white rounded-lg px-3 py-1.5 text-sm"
                      autoFocus
                      inputMode="text"
                      autoComplete="off"
                      autoCorrect="off"
                      autoCapitalize="off"
                      spellCheck="false"
                      style={{ fontSize: '16px' }}
                    />
                    <button
                      onClick={() => updateNickname(checkpoint.id, newNickname)}
                      className="text-green-400 hover:text-green-300 p-2 rounded-lg hover:bg-green-500/20 transition-colors"
                      aria-label="Save nickname"
                    >
                      <Check className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => {
                        setEditingNickname(null)
                        setNewNickname('')
                      }}
                      className="text-red-400 hover:text-red-300 p-2 rounded-lg hover:bg-red-500/20 transition-colors"
                      aria-label="Cancel edit"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center space-x-2">
                    <h3 className="font-semibold text-ui-text-primary text-sm truncate">{checkpoint.nickname}</h3>
                    <button
                      onClick={() => {
                        setEditingNickname(checkpoint.id)
                        setNewNickname(checkpoint.nickname)
                      }}
                      className="text-ui-text-secondary hover:text-ui-text-primary p-2 rounded-lg hover:bg-gray-700/50 transition-colors"
                      aria-label="Edit nickname"
                    >
                      <Edit className="w-3 h-3" />
                    </button>
                  </div>
                )}
                
                {/* Quick Stats */}
                <div className="flex items-center space-x-3 mt-2 text-xs">
                  <div className="flex items-center space-x-1">
                    <Target className="w-3 h-3 text-ui-state-success" />
                    <span className="text-ui-state-success font-medium numeric">
                      {checkpoint.performance_metrics.best_score.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <BarChart3 className="w-3 h-3 text-ui-brand-primary" />
                    <span className="text-ui-brand-primary font-medium numeric">
                      {checkpoint.episode.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Cpu className={`w-3 h-3 ${getModelSizeColor(checkpoint.model_config.model_size)}`} />
                    <span className={`font-medium ${getModelSizeColor(checkpoint.model_config.model_size)}`}>
                      {checkpoint.model_config.model_size}
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="flex items-center space-x-2 ml-2">
                {getScoreTrendIcon(checkpoint.performance_metrics.best_score)}
                <button
                  onClick={() => setExpandedCheckpoint(
                    expandedCheckpoint === checkpoint.id ? null : checkpoint.id
                  )}
                  onTouchEnd={(e) => {
                    e.preventDefault()
                    setExpandedCheckpoint(
                      expandedCheckpoint === checkpoint.id ? null : checkpoint.id
                    )
                  }}
                  className="text-ui-text-secondary hover:text-ui-text-primary p-2 rounded-lg hover:bg-gray-700/50 transition-colors"
                  aria-label={expandedCheckpoint === checkpoint.id ? "Collapse details" : "Expand details"}
                >
                  {expandedCheckpoint === checkpoint.id ? 
                    <ChevronUp className="w-4 h-4" /> : 
                    <ChevronDown className="w-4 h-4" />
                  }
                </button>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex space-x-2">
              <button
                onClick={() => startPlayback(checkpoint.id)}
                className="flex-1 flex items-center justify-center space-x-2 bg-ui-state-success/20 text-ui-state-success rounded-xl py-2.5 text-sm font-medium hover:bg-ui-state-success/30 transition-colors"
              >
                <Play className="w-4 h-4" />
                <span>Watch</span>
              </button>
              <button
                onClick={() => loadCheckpointForTraining(checkpoint.id)}
                className="flex-1 flex items-center justify-center space-x-2 bg-ui-brand-primary/20 text-ui-brand-primary rounded-xl py-2.5 text-sm font-medium hover:bg-ui-brand-primary/30 transition-colors"
              >
                <Zap className="w-4 h-4" />
                <span>Resume</span>
              </button>
              <button
                onClick={() => deleteCheckpoint(checkpoint.id)}
                className="flex items-center justify-center bg-ui-state-danger/20 text-ui-state-danger rounded-xl py-2.5 px-3 text-sm font-medium hover:bg-ui-state-danger/30 transition-colors"
                aria-label="Delete checkpoint"
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
                className="mt-3 pt-3 border-t border-ui-border-muted"
              >
                <div className="grid grid-cols-2 gap-4 text-xs">
                  <div>
                    <div className="text-ui-text-secondary mb-2 flex items-center">
                      <Layers className="w-3 h-3 mr-1" />
                      Model Config
                    </div>
                    <div className="space-y-1.5">
                      <div className="flex justify-between">
                        <span>Experts:</span>
                        <span className="text-ui-text-primary font-medium numeric">{checkpoint.model_config.n_experts}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Layers:</span>
                        <span className="text-ui-text-primary font-medium numeric">{checkpoint.model_config.n_layers}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Dimensions:</span>
                        <span className="text-ui-text-primary font-medium numeric">{checkpoint.model_config.d_model}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Heads:</span>
                        <span className="text-ui-text-primary font-medium numeric">{checkpoint.model_config.n_heads}</span>
                      </div>
                    </div>
                  </div>
                  <div>
                    <div className="text-ui-text-secondary mb-2 flex items-center">
                      <Activity className="w-3 h-3 mr-1" />
                      Performance
                    </div>
                    <div className="space-y-1.5">
                      <div className="flex justify-between">
                        <span>Avg Score:</span>
                        <span className="text-ui-text-primary font-medium numeric">{checkpoint.performance_metrics.avg_score.toFixed(0)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Final Loss:</span>
                        <span className="text-ui-text-primary font-medium numeric">{checkpoint.performance_metrics.final_loss.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Speed:</span>
                        <span className="text-ui-text-primary font-medium numeric">{checkpoint.performance_metrics.training_speed.toFixed(1)} ep/min</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Duration:</span>
                        <span className="text-ui-text-primary font-medium numeric">{formatDuration(checkpoint.training_duration)}</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center justify-between mt-3 pt-2 border-t border-ui-border-muted">
                  <div className="flex items-center space-x-1 text-xs text-ui-text-secondary">
                    <Calendar className="w-3 h-3" />
                    <span>{new Date(checkpoint.created_at).toLocaleDateString()}</span>
                  </div>
                  <div className="flex items-center space-x-1 text-xs text-ui-text-secondary">
                    <Database className="w-3 h-3" />
                    <span className="numeric">{formatFileSize(checkpoint.file_size)}</span>
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>
        ))}

        {filteredAndSortedCheckpoints.length === 0 && (
          <motion.div
            className="text-center py-12"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <Archive className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-ui-text-secondary mb-2">No checkpoints found</p>
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                className="text-ui-brand-primary hover:brightness-110 text-sm"
              >
                Clear search
              </button>
            )}
          </motion.div>
        )}
      </div>
    </div>
  )
}

export default CheckpointManager 