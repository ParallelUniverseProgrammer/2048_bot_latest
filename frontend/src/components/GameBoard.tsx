import React, { useMemo } from 'react'
import { motion } from 'framer-motion'
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Eye, EyeOff, PlayCircle, PauseCircle, StopCircle, Loader2, RefreshCw } from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'
import { useDeviceDetection } from '../utils/deviceDetection'
import config from '../utils/config'

const GameBoard: React.FC = () => {
  const { 
    trainingData, 
    isTraining, 
    lastPolicyLoss, 
    lastValueLoss, 
    checkpointPlaybackData, 
    isPlayingCheckpoint,
    loadingStates,
    gameCompletionData,
    isShowingGameOver
  } = useTrainingStore()
  
  // Device detection for mobile optimization
  useDeviceDetection()
  // Set attention ON by default
  const [showAttention, setShowAttention] = React.useState(true)
  const [playbackStatus, setPlaybackStatus] = React.useState<any>(null)
  const [restartCountdown, setRestartCountdown] = React.useState(3)
  
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

  // Handle restart countdown
  React.useEffect(() => {
    if (isShowingGameOver) {
      setRestartCountdown(3)
      const countdownInterval = setInterval(() => {
        setRestartCountdown(prev => {
          if (prev <= 1) {
            clearInterval(countdownInterval)
            return 0
          }
          return prev - 1
        })
      }, 1000)
      
      return () => clearInterval(countdownInterval)
    }
  }, [isShowingGameOver])

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

  const startNewGame = async () => {
    try {
      if (!playbackStatus?.current_checkpoint) {
        console.error('No checkpoint loaded for new game')
        return
      }
      
      // Clear game over state if showing
      if (useTrainingStore.getState().isShowingGameOver) {
        useTrainingStore.getState().setShowingGameOver(false)
        useTrainingStore.getState().setGameCompletionData(null)
      }
      
      // Start enhanced loading operation
      const loadingSteps = [
        'Resetting game state...',
        'Initializing new game...',
        'Starting game simulation...',
        'Waiting for first move...'
      ]
      
      useTrainingStore.getState().startLoadingOperation('newGame', loadingSteps)
      
      // Optimistically set playing state to prevent brief idle display
      useTrainingStore.getState().setPlayingCheckpoint(true)
      
      // Simulate step progression
      setTimeout(() => useTrainingStore.getState().updateLoadingProgress(25, loadingSteps[1]), 300)
      setTimeout(() => useTrainingStore.getState().updateLoadingProgress(50, loadingSteps[2]), 600)
      
      const res = await fetch(`${config.api.baseUrl}/checkpoints/${playbackStatus.current_checkpoint}/playback/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      
      if (!res.ok) {
        // Clear loading state on error
        useTrainingStore.getState().setLoadingState('isNewGameStarting', false)
        useTrainingStore.getState().setLoadingState('loadingMessage', null)
        useTrainingStore.getState().setPlayingCheckpoint(false)
        throw new Error('Failed to start new game')
      }
      
      // Update to final step
      useTrainingStore.getState().updateLoadingProgress(75, loadingSteps[3], 2)
      
      // Loading state will be cleared when first game data arrives
      
    } catch (err) {
      console.error('Error starting new game:', err)
      // Clear loading state on error
      useTrainingStore.getState().setLoadingState('isNewGameStarting', false)
      useTrainingStore.getState().setLoadingState('loadingMessage', null)
      useTrainingStore.getState().setPlayingCheckpoint(false)
    }
  }

  // Simplified state management - just track the current board
  const [displayBoard, setDisplayBoard] = React.useState<number[][]>([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
  
  // Speed control state
  const [playbackSpeed, setPlaybackSpeed] = React.useState<number>(2.5) // Default 2.5x speed
  const [speedOptions] = React.useState<number[]>([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
  
  // Speed control functions
  const changePlaybackSpeed = async (newSpeed: number) => {
    try {
      const res = await fetch(`${config.api.baseUrl}/checkpoints/playback/speed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed: newSpeed })
      })
      
      if (!res.ok) {
        throw new Error('Failed to change playback speed')
      }
      
      await res.json()
      setPlaybackSpeed(newSpeed)
      console.log(`Playback speed changed to ${newSpeed}x`)
    } catch (err) {
      console.error('Error changing playback speed:', err)
    }
  }

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
        checkpoint_id: checkpointPlaybackData.checkpoint_id,
        done: checkpointPlaybackData.step_data.done
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
        checkpoint_id: null,
        done: false
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
    <div className="h-full flex flex-col space-y-2 pb-6">
      {/* Loading State - Sticky positioned at top of content */}
      {(loadingStates.isPlaybackStarting || loadingStates.isNewGameStarting) && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="sticky top-0 z-40 card-glass p-4 rounded-2xl border border-blue-500/30 bg-blue-500/5 shadow-lg mb-2"
        >
          <div className="space-y-3">
            {/* Header */}
            <div className="flex items-center space-x-3">
              <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-blue-300 truncate">
                  {loadingStates.isPlaybackStarting ? 'Loading Playback' : 'Starting Game'}
                </div>
                <div className="text-xs text-blue-400/80 truncate">
                  {loadingStates.loadingStep || loadingStates.loadingMessage || 'Please wait...'}
                </div>
              </div>
              <div className="text-xs text-blue-400">
                {loadingStates.estimatedTimeRemaining !== null 
                  ? `${Math.ceil(loadingStates.estimatedTimeRemaining)}s`
                  : ''
                }
              </div>
            </div>
            
            {/* Progress Bar */}
            <div className="w-full bg-gray-700 rounded-full h-2">
              <motion.div
                className="bg-blue-400 h-2 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${loadingStates.loadingProgress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
            
            {/* Step Progress */}
            {loadingStates.loadingSteps.length > 0 && (
              <div className="flex items-center space-x-2">
                <div className="text-xs text-gray-400">
                  Step {loadingStates.currentStepIndex + 1} of {loadingStates.loadingSteps.length}
                </div>
                <div className="flex-1 flex space-x-1">
                  {loadingStates.loadingSteps.map((_, index) => (
                    <div
                      key={index}
                      className={`flex-1 h-1 rounded-full ${
                        index <= loadingStates.currentStepIndex 
                          ? 'bg-blue-400' 
                          : 'bg-gray-600'
                      }`}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Game Over Screen */}
      {isShowingGameOver && gameCompletionData && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="card-glass p-4 rounded-2xl border border-red-500/30 bg-red-500/5 flex-shrink-0"
        >
          <div className="text-center space-y-4">
            {/* Game Over Header */}
            <div className="space-y-2">
              <h2 className="text-2xl font-bold text-red-400">Game Over!</h2>
              <p className="text-sm text-gray-400">Game #{gameCompletionData.game_number} completed</p>
            </div>
            
            {/* Final Stats */}
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-lg font-bold text-green-400">
                  {gameCompletionData.final_score.toLocaleString()}
                </div>
                <div className="text-xs text-gray-400">Final Score</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-blue-400">
                  {gameCompletionData.total_steps.toLocaleString()}
                </div>
                <div className="text-xs text-gray-400">Total Steps</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-purple-400">
                  {gameCompletionData.max_tile.toLocaleString()}
                </div>
                <div className="text-xs text-gray-400">Max Tile</div>
              </div>
            </div>
            
            {/* Final Board State */}
            <div className="space-y-2">
              <p className="text-sm text-gray-400">Final Board State</p>
              <div className="grid grid-cols-4 gap-2 max-w-sm mx-auto">
                {gameCompletionData.final_board_state.map((row, rowIndex) =>
                  row.map((value, colIndex) => (
                    <div
                      key={`${rowIndex}-${colIndex}`}
                      className={`game-tile aspect-square ${getTileColor(value)} ${getTextColor(value)}`}
                    >
                      <span className="text-sm font-bold">
                        {value > 0 ? value : ''}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
            
            {/* Auto-restart notice and manual restart button */}
            <div className="space-y-2">
              <div className="text-xs text-gray-500">
                Starting new game in {restartCountdown} seconds...
              </div>
              <button
                onClick={startNewGame}
                className="flex items-center justify-center space-x-2 bg-blue-500/20 text-blue-400 rounded-xl py-2.5 px-4 text-sm font-medium hover:bg-blue-500/30 transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Start New Game Now</span>
              </button>
            </div>
          </div>
        </motion.div>
      )}

      {/* Header with Status */}
      <motion.div
        className="card-glass p-4 rounded-2xl flex-shrink-0"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              currentData?.done ? 'bg-red-400 animate-pulse' :
              currentData?.is_playback ? 'bg-purple-400' : 
              isTraining ? 'bg-green-400 animate-pulse' : 'bg-gray-400'
            }`} />
            <span className="text-sm font-medium text-white">
              {currentData?.done ? 'Game Over' :
               currentData?.is_playback ? 'Playback' : isTraining ? 'Training' : 'Idle'}
            </span>
          </div>
          
          {currentData?.is_playback && (
            <div className="text-xs text-purple-400 bg-purple-500/20 px-2 py-1 rounded">
              {currentData.checkpoint_id}
            </div>
          )}
        </div>
      </motion.div>

      {/* Playback Controls */}
      {isPlayingCheckpoint && (
        <motion.div
          className="card-glass p-4 rounded-2xl flex-shrink-0"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm font-medium text-white">Playback Controls</div>
            <div className="text-xs text-gray-400">
              Step {checkpointPlaybackData?.step_data?.step || 0}
            </div>
          </div>
          
          <div className="flex items-center space-x-2 mb-2">
            {playbackStatus?.is_playing && !playbackStatus?.is_paused ? (
                          <button
              onClick={pausePlayback}
              className="flex-1 flex items-center justify-center space-x-2 bg-yellow-500/20 text-yellow-400 rounded-xl py-2.5 text-sm font-medium"
            >
                <PauseCircle className="w-4 h-4" />
                <span>Pause</span>
              </button>
            ) : (
              <button
                onClick={resumePlayback}
                className="flex-1 flex items-center justify-center space-x-2 bg-green-500/20 text-green-400 rounded-xl py-2.5 text-sm font-medium"
              >
                <PlayCircle className="w-4 h-4" />
                <span>Resume</span>
              </button>
            )}
            
            <button
              onClick={stopPlayback}
              className="flex-1 flex items-center justify-center space-x-2 bg-red-500/20 text-red-400 rounded-xl py-2.5 text-sm font-medium"
            >
              <StopCircle className="w-4 h-4" />
              <span>Stop</span>
            </button>
            
            <button
              onClick={startNewGame}
              className="flex-1 flex items-center justify-center space-x-2 bg-blue-500/20 text-blue-400 rounded-xl py-2.5 text-sm font-medium"
            >
              <RefreshCw className="w-4 h-4" />
              <span>New</span>
            </button>
          </div>
          
          {/* Speed Control */}
          <div className="flex items-center justify-between">
            <div className="text-xs text-gray-400">Speed:</div>
            <div className="flex items-center space-x-1">
              {speedOptions.map((speed) => (
                <button
                  key={speed}
                  onClick={() => changePlaybackSpeed(speed)}
                  className={`px-2 py-1 text-xs rounded-lg ${
                    playbackSpeed === speed
                      ? 'bg-purple-500/30 text-purple-300 border border-purple-500/50'
                      : 'bg-gray-700/50 text-gray-400 hover:bg-gray-600/50'
                  }`}
                >
                  {speed}x
                </button>
              ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* Game Stats */}
      <motion.div
        className="card-glass p-4 rounded-2xl flex-shrink-0"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="grid grid-cols-3 gap-3">
          <div className="text-center">
            <div className="text-lg font-bold text-green-400">
              {currentData?.score?.toLocaleString() || '0'}
            </div>
            <div className="text-xs text-gray-400">Score</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-blue-400">
              {currentData?.is_playback 
                ? (checkpointPlaybackData?.step_data?.step?.toLocaleString() || '0')
                : (currentData?.episode?.toLocaleString() || '0')
              }
            </div>
            <div className="text-xs text-gray-400">
              {currentData?.is_playback ? 'Step' : 'Episode'}
            </div>
          </div>
          <div className="text-center">
            <button
              onClick={() => setShowAttention(!showAttention)}
              className={`text-lg font-bold ${showAttention ? 'text-blue-400' : 'text-gray-400'}`}
            >
              {showAttention ? <EyeOff className="w-5 h-5 mx-auto" /> : <Eye className="w-5 h-5 mx-auto" />}
            </button>
            <div className="text-xs text-gray-400">Attention</div>
          </div>
        </div>
      </motion.div>

              {/* Main Game Area */}
        <div className="flex-1 flex flex-col space-y-2 min-h-0">
          {/* Game Board */}
          <motion.div
            className={`card-glass p-4 rounded-2xl flex-shrink-0 relative ${
              currentData?.done ? 'border border-red-500/30 bg-red-500/5' : ''
            }`}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
          >
          {currentData?.done && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-xl z-10">
              <div className="text-center">
                <div className="text-lg font-bold text-red-400 mb-1">Game Over</div>
                <div className="text-xs text-gray-400">Final Score: {currentData.score?.toLocaleString()}</div>
              </div>
            </div>
          )}
          <div className="grid grid-cols-4 gap-2 max-w-sm mx-auto">
            {displayBoard.map((row, rowIndex) =>
              row.map((value, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className={`game-tile aspect-square ${getTileColor(value)} ${getTextColor(value)} relative overflow-hidden`}
                  style={showAttention ? {
                    border: `2px solid rgba(34, 197, 94, ${getAttentionOpacity(rowIndex, colIndex)})`,
                  } : {}}
                >
                  <span className="text-sm font-bold">
                    {value > 0 ? value : ''}
                  </span>
                </div>
              ))
            )}
          </div>
        </motion.div>

        {/* Compact Next Action Bar */}
        {actionProbabilities.length > 0 && (
          <div className="flex items-center justify-between mt-1 px-3 py-2 bg-gray-800/80 rounded-xl max-w-xs mx-auto">
            {/* Most likely action */}
            {(() => {
              const maxProb = Math.max(...actionProbabilities.map((a: any) => a.probability))
              const mainAction = actionProbabilities.find((a: any) => a.probability === maxProb)
              if (!mainAction) return null
              const Icon = mainAction.icon
              return (
                <div className="flex items-center space-x-2">
                  <Icon className={`w-5 h-5 ${mainAction.color}`} />
                  <span className={`font-bold text-sm ${mainAction.color}`}>{(mainAction.probability * 100).toFixed(0)}%</span>
                </div>
              )
            })()}
            {/* Dots for other actions */}
            <div className="flex items-center space-x-1 ml-2">
              {actionProbabilities.map((action: any) => {
                if (action.probability === Math.max(...actionProbabilities.map((a: any) => a.probability))) return null
                return (
                  <span key={action.name} className={`w-2 h-2 rounded-full ${action.color} opacity-70`} title={`${action.name}: ${(action.probability * 100).toFixed(1)}%`} />
                )
              })}
            </div>
          </div>
        )}
      </div>

      {/* Training Status Footer */}
      {trainingData && (
        <motion.div
          className="card-glass p-3 rounded-2xl flex-shrink-0"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center justify-between text-xs text-gray-400">
            <span>R: {trainingData.reward?.toFixed(2) || '0.00'}</span>
            <span>PL: {trainingData.policy_loss?.toFixed(3) || lastPolicyLoss?.toFixed(3) || 'N/A'}</span>
            <span>VL: {trainingData.value_loss?.toFixed(3) || lastValueLoss?.toFixed(3) || 'N/A'}</span>
            <span>{new Date(trainingData.timestamp * 1000).toLocaleTimeString()}</span>
          </div>
        </motion.div>
      )}
    </div>
  )
}

export default GameBoard 