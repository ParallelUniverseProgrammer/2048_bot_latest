import React, { useMemo } from 'react'
import { motion } from 'framer-motion'
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Eye, EyeOff, PlayCircle, PauseCircle, StopCircle, Gauge, Loader2, RefreshCw } from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'
import config from '../utils/config'

const GameBoard: React.FC = () => {
  const { 
    trainingData, 
    isTraining, 
    lastPolicyLoss, 
    lastValueLoss, 
    checkpointPlaybackData, 
    isPlayingCheckpoint,
    loadingStates
  } = useTrainingStore()
  const [showAttention, setShowAttention] = React.useState(false)
  const [playbackSpeed, setPlaybackSpeed] = React.useState(1.0)
  const [playbackStatus, setPlaybackStatus] = React.useState<any>(null)
  
  // Load playback status
  React.useEffect(() => {
    const loadPlaybackStatus = async () => {
      try {
        const res = await fetch(`${config.api.baseUrl}/checkpoints/playback/status`)
        if (res.ok) {
          const data = await res.json()
          setPlaybackStatus(data)
        }
      } catch (err) {
        // Ignore errors
      }
    }
    
    loadPlaybackStatus()
    const interval = setInterval(loadPlaybackStatus, 1000)
    return () => clearInterval(interval)
  }, [])

  // Playback control functions
  const pausePlayback = async () => {
    try {
              const res = await fetch(`${config.api.baseUrl}/checkpoints/playback/pause`, { method: 'POST' })
      if (!res.ok) throw new Error('Failed to pause playback')
    } catch (err) {
      console.error('Error pausing playback:', err)
    }
  }

  const resumePlayback = async () => {
    try {
              const res = await fetch(`${config.api.baseUrl}/checkpoints/playback/resume`, { method: 'POST' })
      if (!res.ok) throw new Error('Failed to resume playback')
    } catch (err) {
      console.error('Error resuming playback:', err)
    }
  }

  const stopPlayback = async () => {
    try {
      const res = await fetch(`${config.api.baseUrl}/checkpoints/playback/stop`, { method: 'POST' })
      if (!res.ok) throw new Error('Failed to stop playback')
    } catch (err) {
      console.error('Error stopping playback:', err)
    }
  }

  const recoverPlayback = async () => {
    try {
      console.log('Attempting to recover from stuck playback state...')
      
      // Clear all loading states first
      useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
      useTrainingStore.getState().setLoadingState('isNewGameStarting', false)
      useTrainingStore.getState().setLoadingState('loadingMessage', null)
      
      // Stop any existing playback
      await stopPlayback()
      
      // Wait a moment for cleanup
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // Check if we have a checkpoint to restart
      if (playbackStatus?.current_checkpoint) {
        console.log(`Restarting playback for checkpoint: ${playbackStatus.current_checkpoint}`)
        
        // Set loading state for recovery
        useTrainingStore.getState().setLoadingState('isPlaybackStarting', true)
        useTrainingStore.getState().setLoadingState('loadingMessage', 'Recovering playback...')
        
        // Start playback again
        const res = await fetch(`${config.api.baseUrl}/checkpoints/${playbackStatus.current_checkpoint}/playback/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ speed: playbackSpeed })
        })
        
        if (!res.ok) {
          throw new Error('Failed to restart playback')
        }
        
        console.log('Playback recovery successful')
      } else {
        // No checkpoint available, just clear the playback state
        useTrainingStore.getState().setPlayingCheckpoint(false)
        console.log('No checkpoint available for recovery, cleared playback state')
      }
      
    } catch (err) {
      console.error('Error recovering playback:', err)
      // Clear loading states on error
      useTrainingStore.getState().setLoadingState('isPlaybackStarting', false)
      useTrainingStore.getState().setLoadingState('isNewGameStarting', false)
      useTrainingStore.getState().setLoadingState('loadingMessage', null)
      // Also clear playback state
      useTrainingStore.getState().setPlayingCheckpoint(false)
    }
  }

  const startNewGame = async () => {
    try {
      if (!playbackStatus?.current_checkpoint) {
        console.error('No checkpoint loaded for new game')
        return
      }
      
      // Set loading state for new game
      useTrainingStore.getState().setLoadingState('isNewGameStarting', true)
      useTrainingStore.getState().setLoadingState('loadingMessage', 'Starting new game...')
      
              const res = await fetch(`${config.api.baseUrl}/checkpoints/${playbackStatus.current_checkpoint}/playback/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed: playbackSpeed })
      })
      
      if (!res.ok) {
        // Clear loading state on error
        useTrainingStore.getState().setLoadingState('isNewGameStarting', false)
        useTrainingStore.getState().setLoadingState('loadingMessage', null)
        throw new Error('Failed to start new game')
      }
      
      // Loading state will be cleared when first game data arrives
      
    } catch (err) {
      console.error('Error starting new game:', err)
      // Clear loading state on error
      useTrainingStore.getState().setLoadingState('isNewGameStarting', false)
      useTrainingStore.getState().setLoadingState('loadingMessage', null)
    }
  }

  const setPlaybackSpeedAPI = async (speed: number) => {
    try {
              const res = await fetch(`${config.api.baseUrl}/checkpoints/playback/speed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed })
      })
      if (!res.ok) throw new Error('Failed to set playback speed')
      setPlaybackSpeed(speed)
    } catch (err) {
      console.error('Error setting playback speed:', err)
    }
  }
  
  // Simplified state management - just track the current board
  const [displayBoard, setDisplayBoard] = React.useState<number[][]>([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
  
  // Get current data (either training or checkpoint playback)
  const currentData = React.useMemo(() => {
    if (isPlayingCheckpoint && checkpointPlaybackData) {
      return {
        board_state: checkpointPlaybackData.step_data.board_state,
        score: checkpointPlaybackData.step_data.score,
        action: checkpointPlaybackData.step_data.action,
        action_probs: checkpointPlaybackData.step_data.action_probs,
        attention_weights: checkpointPlaybackData.step_data.attention_weights,
        episode: 0, // Not applicable for playback
        is_playback: true,
        checkpoint_id: checkpointPlaybackData.checkpoint_id
      }
    } else if (trainingData) {
      return {
        board_state: trainingData.board_state,
        score: trainingData.score,
        action: null, // TrainingData doesn't have a single action
        action_probs: trainingData.actions, // TrainingData has actions array
        attention_weights: trainingData.attention_weights,
        episode: trainingData.episode,
        is_playback: false,
        checkpoint_id: null
      }
    }
    return null
  }, [isPlayingCheckpoint, checkpointPlaybackData, trainingData])

  // Simple board update - just update when data changes
  React.useEffect(() => {
    if (currentData?.board_state) {
      setDisplayBoard(currentData.board_state)
    }
  }, [currentData?.board_state])

  // Get tile color based on value
  const getTileColor = (value: number) => {
    if (value === 0) return 'bg-gray-700/30'
    if (value <= 4) return 'bg-game-2'
    if (value <= 8) return 'bg-game-4'
    if (value <= 16) return 'bg-game-8'
    if (value <= 32) return 'bg-game-16'
    if (value <= 64) return 'bg-game-32'
    if (value <= 128) return 'bg-game-64'
    if (value <= 256) return 'bg-game-128'
    if (value <= 512) return 'bg-game-256'
    if (value <= 1024) return 'bg-game-512'
    if (value <= 2048) return 'bg-game-1024'
    return 'bg-game-2048'
  }

  // Get text color based on value
  const getTextColor = (value: number) => {
    if (value === 0) return 'text-transparent'
    if (value <= 4) return 'text-gray-800'
    return 'text-white'
  }

  // Action probabilities display
  const actionProbabilities = useMemo(() => {
    if (!currentData?.action_probs) return []
    
    const actions = ['Up', 'Down', 'Left', 'Right']
    const icons = [ArrowUp, ArrowDown, ArrowLeft, ArrowRight]
    const colors = ['text-blue-400', 'text-green-400', 'text-yellow-400', 'text-red-400']
    
    return currentData.action_probs.map((prob: number, index: number) => ({
      name: actions[index],
      probability: prob,
      icon: icons[index],
      color: colors[index],
    }))
  }, [currentData?.action_probs])

  // Get attention opacity for a cell
  const getAttentionOpacity = (row: number, col: number) => {
    if (!showAttention || !currentData?.attention_weights) return 0
    
    const weight = currentData.attention_weights[row]?.[col] || 0
    return Math.min(weight * 5, 0.8) // Scale up for visibility
  }

  return (
    <div className="space-y-8">
      {/* Loading Indicators */}
      {(loadingStates.isPlaybackStarting || loadingStates.isNewGameStarting) && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className={`card-glass p-4 rounded-xl border ${
            loadingStates.isPlaybackStarting 
              ? 'border-purple-500/30 bg-purple-500/10' 
              : 'border-green-500/30 bg-green-500/10'
          }`}
        >
          <div className="flex items-center space-x-3">
            <Loader2 className={`w-5 h-5 animate-spin ${
              loadingStates.isPlaybackStarting ? 'text-purple-400' : 'text-green-400'
            }`} />
            <div className="flex-1">
              <div className={`text-sm font-medium ${
                loadingStates.isPlaybackStarting ? 'text-purple-300' : 'text-green-300'
              }`}>
                {loadingStates.isPlaybackStarting ? 'Loading Checkpoint Playback' : 'Starting New Game'}
              </div>
              <div className={`text-xs ${
                loadingStates.isPlaybackStarting ? 'text-purple-400/80' : 'text-green-400/80'
              }`}>
                {loadingStates.loadingMessage || 
                  (loadingStates.isPlaybackStarting 
                    ? 'Loading checkpoint model and initializing playback...' 
                    : 'Preparing new game board and resetting state...'
                  )
                }
              </div>
            </div>
            <div className="flex items-center space-x-1">
              {[0, 1, 2].map(i => (
                <div 
                  key={i}
                  className={`w-2 h-2 rounded-full animate-pulse ${
                    loadingStates.isPlaybackStarting ? 'bg-purple-400' : 'bg-green-400'
                  }`}
                  style={{ animationDelay: `${i * 200}ms` }}
                />
              ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* Status Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">
          {currentData?.is_playback ? 'Checkpoint Playback' : 'Live Training'}
        </h2>
        
        <div className="flex items-center space-x-2">
          {currentData?.is_playback && (
            <div className="px-3 py-1 bg-purple-500/20 text-purple-400 rounded-full text-sm">
              {currentData.checkpoint_id}
            </div>
          )}
          <div className={`px-3 py-1 rounded-full text-sm ${
            currentData?.is_playback 
              ? 'bg-purple-500/20 text-purple-400' 
              : isTraining 
                ? 'bg-green-500/20 text-green-400' 
                : 'bg-gray-500/20 text-gray-400'
          }`}>
            {loadingStates.isPlaybackStarting ? 'Loading...' :
             loadingStates.isNewGameStarting ? 'Starting...' :
             currentData?.is_playback ? 'Playback' : 
             isTraining ? 'Training' : 'Idle'}
          </div>
        </div>
      </div>

      {/* Playback Controls - Show when checkpoint playback is active OR when loading playback */}
      {(isPlayingCheckpoint || loadingStates.isPlaybackStarting) && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card-glass p-4 rounded-xl border border-green-500/20"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className={`w-2 h-2 rounded-full ${
                loadingStates.isPlaybackStarting ? 'bg-purple-400 animate-pulse' :
                playbackStatus?.is_playing && !playbackStatus?.is_paused 
                  ? 'bg-green-400 animate-pulse' 
                  : playbackStatus?.is_paused 
                    ? 'bg-yellow-400' 
                    : 'bg-gray-400'
              }`}></div>
              <h3 className="text-lg font-semibold">Playback Controls</h3>
              <span className={`text-sm px-2 py-1 rounded ${
                loadingStates.isPlaybackStarting ? 'bg-purple-500/20 text-purple-400' :
                playbackStatus?.is_playing && !playbackStatus?.is_paused ? 'bg-green-500/20 text-green-400' :
                playbackStatus?.is_paused ? 'bg-yellow-500/20 text-yellow-400' : 'bg-gray-500/20 text-gray-400'
              }`}>
                {loadingStates.isPlaybackStarting ? 'Loading' :
                 playbackStatus?.is_playing && !playbackStatus?.is_paused ? 'Playing' :
                 playbackStatus?.is_paused ? 'Paused' : 'Stopped'}
              </span>
            </div>
            
            <div className="text-sm text-gray-400">
              {playbackStatus?.current_checkpoint || 'Loading...'}
            </div>
          </div>
          
          <div className="flex items-center justify-between">
            {/* Speed Control */}
            <div className="flex items-center space-x-2">
              <Gauge className="w-4 h-4 text-gray-400" />
              <select
                value={playbackSpeed}
                onChange={(e) => setPlaybackSpeedAPI(parseFloat(e.target.value))}
                disabled={loadingStates.isPlaybackStarting}
                className={`bg-gray-700 text-white rounded px-3 py-1 text-sm border border-gray-600 focus:border-blue-500 focus:outline-none ${
                  loadingStates.isPlaybackStarting ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                <option value={0.5}>0.5x</option>
                <option value={1.0}>1.0x</option>
                <option value={1.5}>1.5x</option>
                <option value={2.0}>2.0x</option>
                <option value={3.0}>3.0x</option>
              </select>
            </div>
            
            {/* Playback Buttons */}
            <div className="flex items-center space-x-3">
              {!loadingStates.isPlaybackStarting && playbackStatus && (
                <>
                  {playbackStatus.is_playing && !playbackStatus.is_paused ? (
                    <button
                      onClick={pausePlayback}
                      className="flex items-center space-x-2 px-4 py-2 bg-yellow-500/20 text-yellow-400 rounded-lg hover:bg-yellow-500/30 transition-colors"
                    >
                      <PauseCircle className="w-4 h-4" />
                      <span>Pause</span>
                    </button>
                  ) : playbackStatus.is_paused ? (
                    <button
                      onClick={resumePlayback}
                      className="flex items-center space-x-2 px-4 py-2 bg-green-500/20 text-green-400 rounded-lg hover:bg-green-500/30 transition-colors"
                    >
                      <PlayCircle className="w-4 h-4" />
                      <span>Resume</span>
                    </button>
                  ) : null}
                  
                  <div className="flex items-center space-x-2">
                    <>
                      {playbackStatus.is_playing || playbackStatus.is_paused ? (
                        <button
                          onClick={stopPlayback}
                          className="flex items-center space-x-2 px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
                        >
                          <StopCircle className="w-4 h-4" />
                          <span>Stop</span>
                        </button>
                      ) : (
                        <button
                          onClick={startNewGame}
                          className="flex items-center space-x-2 px-4 py-2 bg-yellow-500/20 text-yellow-400 rounded-lg hover:bg-yellow-500/30 transition-colors"
                        >
                          <PlayCircle className="w-4 h-4" />
                          <span>New Game</span>
                        </button>
                      )}
                      
                      {/* Recovery button - show when loading states are active for too long or system appears stuck */}
                      {(loadingStates.isPlaybackStarting || loadingStates.isNewGameStarting || 
                        (playbackStatus.is_playing && !isPlayingCheckpoint)) && (
                        <button
                          onClick={recoverPlayback}
                          className="flex items-center space-x-2 px-3 py-2 bg-blue-500/20 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors text-sm"
                          title="Recover from stuck playback state"
                        >
                          <RefreshCw className="w-4 h-4" />
                          <span>Recover</span>
                        </button>
                      )}
                    </>
                  </div>
                </>
              )}
              
              {/* Show loading message when starting playback */}
              {loadingStates.isPlaybackStarting && (
                <div className="flex items-center space-x-2 px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg">
                  <div className="w-4 h-4 border-2 border-purple-400 border-t-transparent rounded-full animate-spin" />
                  <span>Loading checkpoint...</span>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* Game Info */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <motion.div
            className="card-glass p-4 rounded-xl"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
          >
            <h3 className="text-sm font-medium text-gray-400 mb-1">Current Score</h3>
            <p className="text-2xl font-bold text-green-400">
              {currentData?.score?.toLocaleString() || '0'}
            </p>
          </motion.div>
          
          <motion.div
            className="card-glass p-4 rounded-xl"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <h3 className="text-sm font-medium text-gray-400 mb-1">
              {currentData?.is_playback ? 'Step' : 'Episode'}
            </h3>
            <p className="text-2xl font-bold text-blue-400">
              {currentData?.is_playback 
                ? (checkpointPlaybackData?.step_data?.step?.toLocaleString() || '0')
                : (currentData?.episode?.toLocaleString() || '0')
              }
            </p>
          </motion.div>
        </div>

        {/* Attention Toggle */}
        <motion.button
          onClick={() => setShowAttention(!showAttention)}
          className={`btn-secondary flex items-center space-x-2 ${
            showAttention ? 'bg-blue-500/20 text-blue-400' : ''
          }`}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {showAttention ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          <span>Attention</span>
        </motion.button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Game Board */}
        <motion.div
          className="lg:col-span-2 card-glass p-6 rounded-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <div className="w-5 h-5 mr-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded" />
            Game Board
            {isTraining && (
              <div className="ml-2 w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            )}
          </h3>
          
          <div className="grid grid-cols-4 gap-2 max-w-xs mx-auto">
            {displayBoard.map((row, rowIndex) =>
              row.map((value, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className={`game-tile aspect-square ${getTileColor(value)} ${getTextColor(value)} relative overflow-hidden transition-all duration-300 ease-in-out`}
                  style={showAttention ? {
                    border: `3px solid rgba(34, 197, 94, ${getAttentionOpacity(rowIndex, colIndex)})`,
                    transition: 'border 300ms ease-in-out'
                  } : {}}
                >
                  {/* Tile value */}
                  <span className="text-sm sm:text-base font-bold">
                    {value > 0 ? value : ''}
                  </span>
                </div>
              ))
            ) || (
              // Empty board placeholder
              Array.from({ length: 16 }).map((_, index) => (
                <div
                  key={index}
                  className="game-tile aspect-square bg-gray-700/30 text-transparent transition-all duration-300 ease-in-out"
                >
                  0
                </div>
              ))
            )}
          </div>
        </motion.div>

        {/* Action Probabilities */}
        <motion.div
          className="card-glass p-6 rounded-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <ArrowUp className="w-5 h-5 mr-2 text-blue-400" />
            {currentData?.is_playback && checkpointPlaybackData?.step_data?.done && checkpointPlaybackData?.step_data?.action === null 
              ? 'Game Over' 
              : 'Next Action'
            }
          </h3>
          
          <div className="space-y-3">          
            {currentData?.is_playback && checkpointPlaybackData?.step_data?.done && checkpointPlaybackData?.step_data?.action === null ? (
              <motion.div
                className="flex items-center justify-center p-6 rounded-lg bg-red-500/20 border border-red-500/30"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <div className="text-center">
                  <div className="text-red-400 font-bold text-lg mb-2">Game Over</div>
                  <div className="text-gray-400 text-sm">No legal moves remaining</div>
                  <div className="text-blue-400 font-medium mt-2">
                    Final Score: {currentData?.score?.toLocaleString() || '0'}
                  </div>
                </div>
              </motion.div>
            ) : (
              actionProbabilities.map((action: any, index: number) => {
                const IconComponent = action.icon
                const isHighest = action.probability === Math.max(...actionProbabilities.map((a: any) => a.probability))
                
                return (
                  <motion.div
                    key={action.name}
                    className={`flex items-center justify-between p-3 rounded-lg transition-all duration-300 ${
                      isHighest ? 'bg-blue-500/20 border border-blue-500/30' : 'bg-gray-700/30'
                    }`}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: 0.4 + index * 0.1 }}
                  >
                    <div className="flex items-center space-x-3">
                      <IconComponent className={`w-4 h-4 ${action.color}`} />
                      <span className="font-medium">{action.name}</span>
                    </div>
                    <div className="text-right">
                      <span className={`font-bold ${action.color}`}>
                        {(action.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  </motion.div>
                )
              })
            )}
          </div>
        </motion.div>
      </div>

      {/* Training Status */}
      {trainingData && (
        <motion.div
          className="card-glass p-4 rounded-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <div className="flex items-center justify-between text-sm text-gray-400">
            <div className="flex items-center space-x-4">
              <span>Reward: {trainingData.reward?.toFixed(3) || '0.000'}</span>
              <span>Policy Loss: {trainingData.policy_loss != null ? trainingData.policy_loss.toFixed(4) : lastPolicyLoss != null ? lastPolicyLoss.toFixed(4) : 'N/A'}</span>
              <span>Value Loss: {trainingData.value_loss != null ? trainingData.value_loss.toFixed(4) : lastValueLoss != null ? lastValueLoss.toFixed(4) : 'N/A'}</span>
            </div>
            <div className="hidden sm:block">
              <span>Updated: {new Date(trainingData.timestamp * 1000).toLocaleTimeString()}</span>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}

export default GameBoard 