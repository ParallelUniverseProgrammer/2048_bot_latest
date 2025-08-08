import React, { useMemo, useRef, useState, useDeferredValue } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
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
import { useDeviceDetection } from '../utils/deviceDetection'
import ConfirmDialog from './ConfirmDialog'
import { useCheckpoints, Checkpoint } from '../hooks/useCheckpoints'

type SortBy = 'date' | 'score' | 'episode' | 'size'
type SortOrder = 'asc' | 'desc'
type FilterBy = 'all' | 'recent' | 'high-score' | 'large'

interface CheckpointManagerProps {
  onNavigateToTab?: (tab: string) => void
}

const CheckpointManager: React.FC<CheckpointManagerProps> = ({ onNavigateToTab }) => {
  // ===== STATE MANAGEMENT =====
  const shouldReduceMotion = useReducedMotion()
  const {
    checkpoints,
    stats,
    loading,
    error,
    checkpointInterval,
    longRunMode,
    configLoading,
    setLongRunMode,
    setCheckpointInterval,
    saveCheckpointConfig,
    refresh,
    updateNickname,
    deleteCheckpoint,
    loadCheckpointForTraining,
    startPlayback,
    setError,
  } = useCheckpoints({ onNavigateToTab })
  
  // ===== DEVICE DETECTION =====
  const { displayMode } = useDeviceDetection()
  const isMobile = displayMode === 'mobile'
  
  // ===== UI STATE =====
  const [searchTerm, setSearchTerm] = useState('')
  const deferredSearch = useDeferredValue(searchTerm)
  const [editingNickname, setEditingNickname] = useState<string | null>(null)
  const [newNickname, setNewNickname] = useState('')
  const [expandedCheckpoint, setExpandedCheckpoint] = useState<string | null>(null)
  
  // ===== FILTER & SORT STATE =====
  const [sortBy, setSortBy] = useState<SortBy>('date')
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc')
  const [filterBy, setFilterBy] = useState<FilterBy>('all')
  
  // ===== CONFIGURATION STATE =====
  const [checkpointIntervalInput, setCheckpointIntervalInput] = useState('50')
  const [checkpointIntervalError, setCheckpointIntervalError] = useState<string | null>(null)
  const [showConfigPanel, setShowConfigPanel] = useState(false)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  
  // ===== REFS =====
  const editingInputRef = useRef<HTMLInputElement>(null)

  // Cleanup handled in store/hook

  // Focus management for mobile keyboard
  React.useEffect(() => {
    if (editingNickname && editingInputRef.current) {
      editingInputRef.current.focus()
      editingInputRef.current.select()
    }
  }, [editingNickname])

  // Polling removed; handled by centralized hook and WebSocket

  React.useEffect(() => {
    setCheckpointIntervalInput(String(checkpointInterval))
  }, [checkpointInterval])

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
  const saveCheckpointConfigLocal = async () => {
    // Validate before saving
    const validatedValue = validateCheckpointInterval(checkpointIntervalInput)
    if (validatedValue === null) {
      setCheckpointIntervalError('Please enter a valid number between 1 and 1000')
      return
    }

    try {
      await saveCheckpointConfig(validatedValue, longRunMode)
      setShowConfigPanel(false)
    } catch (_) {}
  }

  // Handlers
  const handleUpdateNickname = async (checkpointId: string, nickname: string) => {
    const trimmed = nickname.trim()
    if (!trimmed) {
      setError('Nickname cannot be empty')
      return
    }
    try {
      await updateNickname(checkpointId, trimmed)
      setEditingNickname(null)
      setNewNickname('')
    } catch (_) {
      // Error already handled by hook
    }
  }

  const confirmAndDelete = async (checkpointId: string) => {
    try {
      await deleteCheckpoint(checkpointId)
    } finally {
      setConfirmDeleteId(null)
    }
  }

  // Enhanced filtering and sorting
  const FILESIZE_LARGE_THRESHOLD = 10 * 1024 * 1024 // 10MB
  const filteredAndSortedCheckpoints = useMemo(() => {
    let filtered = checkpoints.filter(cp => {
      const name = (cp.nickname || '').toLowerCase()
      const id = (cp.id || '').toLowerCase()
      const term = deferredSearch.toLowerCase()
      const matchesSearch = name.includes(term) || id.includes(term)
      
      if (!matchesSearch) return false
      
      // Apply additional filters
      switch (filterBy) {
        case 'recent':
          const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000)
          return new Date(cp.created_at) > oneDayAgo
        case 'high-score':
          return (cp.performance_metrics?.best_score ?? 0) > TREND_HIGH_SCORE
        case 'large':
          return cp.file_size > FILESIZE_LARGE_THRESHOLD
        default:
          return true
      }
    })
    
    // Sort checkpoints
    filtered.sort((a, b) => {
      let aValue: number = 0, bValue: number = 0
      
      switch (sortBy) {
        case 'date':
          aValue = new Date(a.created_at).getTime()
          bValue = new Date(b.created_at).getTime()
          break
        case 'score':
          aValue = a.performance_metrics?.best_score ?? 0
          bValue = b.performance_metrics?.best_score ?? 0
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
  }, [checkpoints, deferredSearch, filterBy, sortBy, sortOrder])

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
  const TREND_HIGH_SCORE = 1000
  const TREND_MEDIUM_SCORE = 500
  const getScoreTrendIcon = (score: number) => {
    if (score > TREND_HIGH_SCORE) return <TrendingUp className="w-4 h-4 text-ui-state-success" />
    if (score > TREND_MEDIUM_SCORE) return <BarChart3 className="w-4 h-4 text-ui-state-warning" />
    return <TrendingDown className="w-4 h-4 text-ui-state-danger" />
  }

  /**
   * Get text color for model size indicator
   */
  const getModelSizeColor = (size?: string) => {
    if (!size) return 'text-ui-text-secondary'
    switch (size.toLowerCase()) {
      case 'lightning': return 'text-ui-brand-primary'
      case 'base': return 'text-ui-state-success'
      case 'expert': return 'text-ui-state-info'
      default: return 'text-ui-text-secondary'
    }
  }

  // ===== ACTION FUNCTIONS =====
  
  /**
   * Update checkpoint nickname via API
   */
  // moved to hook

  // moved to hook

  // moved to hook

  // moved to hook

  // Loading state
  if (loading) {
    return (
      <div className="safe-area h-full flex flex-col gap-2 pb-6 px-4">
        <div className="card-glass p-4 rounded-2xl">
          <div className="grid grid-cols-4 gap-3 animate-pulse">
            {[0,1,2,3].map(i => (
              <div key={i} className="h-10 rounded bg-token-surface-70" />
            ))}
          </div>
        </div>
        <div className="flex-1 overflow-y-auto space-y-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="card-glass rounded-2xl p-4 animate-pulse">
              <div className="h-4 w-1/3 rounded bg-token-surface-70 mb-3" />
              <div className="grid grid-cols-3 gap-2">
                <div className="h-3 rounded bg-token-surface-70" />
                <div className="h-3 rounded bg-token-surface-70" />
                <div className="h-3 rounded bg-token-surface-70" />
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="safe-area h-full flex flex-col gap-2 pb-6 px-4">
      {/* Error Display */}
      {error && (
        <motion.div
          initial={shouldReduceMotion ? false : { opacity: 0, scale: 0.95 }}
          animate={shouldReduceMotion ? {} : { opacity: 1, scale: 1 }}
          className="card-glass p-4 rounded-2xl border border-ui-state-danger/30 bg-ui-state-danger/5 flex-shrink-0"
          aria-live="polite"
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
        initial={shouldReduceMotion ? false : { opacity: 0, y: -10 }}
        animate={shouldReduceMotion ? {} : { opacity: 1, y: 0 }}
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
            aria-label="Search checkpoints"
          />
        </div>

        {/* Filter and Sort Controls */}
        <div className="flex flex-wrap items-center gap-2 mb-4">
          {/* Filter Buttons */}
          <div className="flex space-x-1">
              {([
                { key: 'all', label: 'All' },
                { key: 'recent', label: 'Recent' },
                { key: 'high-score', label: 'High Score' }
              ] as { key: FilterBy, label: string }[]).map(({ key, label }) => (
              <button
                key={key}
                  onClick={() => setFilterBy(key)}
                className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                  filterBy === key 
                    ? 'bg-ui-brand-primary text-white' 
                    : 'bg-ui-surface-elevated text-ui-text-secondary hover:bg-ui-surface-elevated/80'
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
              onChange={(e) => setSortBy(e.target.value as SortBy)}
              className="bg-ui-surface-elevated text-ui-text-primary rounded-lg px-2 py-1.5 text-xs border border-ui-border-muted focus:outline-none focus:ring-1 focus:ring-ui-focus"
            >
              <option value="date">Date</option>
              <option value="score">Score</option>
              <option value="episode">Episode</option>
              <option value="size">Size</option>
            </select>
            <button
              onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
              className="p-2 bg-ui-surface-elevated text-ui-text-secondary rounded-lg hover:bg-ui-surface-elevated/80 transition-colors"
              aria-label={`Sort ${sortOrder === 'desc' ? 'ascending' : 'descending'}`}
            >
              {sortOrder === 'desc' ? <SortDesc className="w-3 h-3" /> : <SortAsc className="w-3 h-3" />}
            </button>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => refresh()}
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
            initial={shouldReduceMotion ? false : { opacity: 0, height: 0 }}
            animate={shouldReduceMotion ? {} : { opacity: 1, height: 'auto' }}
            exit={shouldReduceMotion ? {} : { opacity: 0, height: 0 }}
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
                  onClick={saveCheckpointConfigLocal}
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
      <div role="list" className="flex-1 overflow-y-auto space-y-2">
        {filteredAndSortedCheckpoints.map((checkpoint: Checkpoint, index: number) => (
          <motion.div
            key={checkpoint.id}
            role="listitem"
            initial={shouldReduceMotion ? false : { opacity: 0, scale: 0.95 }}
            animate={shouldReduceMotion ? {} : { opacity: 1, scale: 1 }}
            transition={shouldReduceMotion ? undefined : { delay: Math.min(index * 0.03, 0.12) }}
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
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          handleUpdateNickname(checkpoint.id, newNickname)
                        }
                        if (e.key === 'Escape') {
                          setEditingNickname(null)
                          setNewNickname('')
                        }
                      }}
                      className="flex-1 bg-gray-700 text-white rounded-lg px-3 py-1.5 text-sm"
                      inputMode="text"
                      autoComplete="off"
                      autoCorrect="off"
                      autoCapitalize="off"
                      spellCheck="false"
                      style={{ fontSize: '16px' }}
                    />
                    <button
                      onClick={() => handleUpdateNickname(checkpoint.id, newNickname)}
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
                  aria-expanded={expandedCheckpoint === checkpoint.id}
                  aria-controls={`cp-details-${checkpoint.id}`}
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
                onClick={() => setConfirmDeleteId(checkpoint.id)}
                className="flex items-center justify-center bg-ui-state-danger/20 text-ui-state-danger rounded-xl py-2.5 px-3 text-sm font-medium hover:bg-ui-state-danger/30 transition-colors"
                aria-label="Delete checkpoint"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>

            {/* Expanded Details */}
            {expandedCheckpoint === checkpoint.id && (
              <motion.div
                id={`cp-details-${checkpoint.id}`}
                role="region"
                aria-label="Checkpoint details"
                initial={shouldReduceMotion ? false : { opacity: 0, height: 0 }}
                animate={shouldReduceMotion ? {} : { opacity: 1, height: 'auto' }}
                exit={shouldReduceMotion ? {} : { opacity: 0, height: 0 }}
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
            initial={shouldReduceMotion ? false : { opacity: 0 }}
            animate={shouldReduceMotion ? {} : { opacity: 1 }}
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

      {/* Confirm Delete Dialog */}
      <ConfirmDialog
        open={!!confirmDeleteId}
        title="Delete checkpoint?"
        description="This action cannot be undone. The checkpoint file will be permanently removed."
        confirmLabel="Delete"
        cancelLabel="Cancel"
        confirmVariant="danger"
        onConfirm={() => confirmAndDelete(confirmDeleteId!)}
        onCancel={() => setConfirmDeleteId(null)}
      />
    </div>
  )
}

export default CheckpointManager 