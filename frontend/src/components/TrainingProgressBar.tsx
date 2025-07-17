import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Loader2, Clock, Brain } from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'

/**
 * LoadingProgressBar – shows a minimal, non-intrusive progress bar at the very top of the screen
 * for training-start, checkpoint-load, and new game operations. It replaces the previous floating pop-ups.
 */
const TrainingProgressBar: React.FC = () => {
  const { loadingStates, isWaitingForFirstData } = useTrainingStore()
  const {
    isTrainingStarting,
    isPlaybackStarting,
    isNewGameStarting,
    isCheckpointLoading,
    checkpointLoadingProgress,
    checkpointLoadingMessage,
    checkpointLoadingError,
    loadingProgress,
    loadingStep,
    estimatedTimeRemaining,
  } = loadingStates

  // Show bar if waiting for first data, or if any loading operation is active
  const isActive = isWaitingForFirstData || 
                   isTrainingStarting || 
                   isPlaybackStarting || 
                   isNewGameStarting || 
                   isCheckpointLoading || 
                   !!checkpointLoadingMessage || 
                   !!checkpointLoadingError

  if (!isActive) {
    return null
  }

  // Determine message & progress based on operation type
  let progress = loadingProgress
  let message = 'Initializing...'
  let icon = <Loader2 className="w-3 h-3 animate-spin" />
  let showTime = estimatedTimeRemaining !== null

  // Handle different loading states
  if (isWaitingForFirstData) {
    progress = Math.max(progress, 80) // Show at least 80% while waiting
    message = 'Waiting for first training data... (This may take 10-30 seconds)'
    icon = <Brain className="w-3 h-3 animate-pulse" />
    showTime = true
  } else if (isTrainingStarting) {
    if (loadingStep) {
      message = loadingStep
      if (loadingStep.includes('Waiting for first training data')) {
        icon = <Brain className="w-3 h-3 animate-pulse" />
        message = 'Waiting for first training data... (This may take 10-30 seconds)'
      } else if (loadingStep.includes('GPU/CPU optimization')) {
        icon = <Brain className="w-3 h-3 animate-spin" />
      } else if (loadingStep.includes('Establishing training loop')) {
        icon = <Loader2 className="w-3 h-3 animate-spin" />
      } else {
        icon = <Loader2 className="w-3 h-3 animate-spin" />
      }
    } else {
      message = 'Initializing training...'
    }
  } else if (isPlaybackStarting) {
    if (loadingStep) {
      message = loadingStep
    } else {
      message = 'Starting playback...'
    }
  } else if (isNewGameStarting) {
    if (loadingStep) {
      message = loadingStep
      if (loadingStep.includes('Waiting for first game data')) {
        icon = <Brain className="w-3 h-3 animate-pulse" />
        message = 'Waiting for first game data...'
      } else {
        icon = <Loader2 className="w-3 h-3 animate-spin" />
      }
    } else {
      message = 'Starting new game...'
    }
  } else if (isCheckpointLoading) {
    progress = checkpointLoadingProgress
    message = checkpointLoadingError || checkpointLoadingMessage || 'Loading checkpoint…'
    icon = <Loader2 className="w-3 h-3 animate-spin" />
  }

  // Color scheme based on state
  const barColor = checkpointLoadingError
    ? 'bg-red-400'
    : progress === 100
    ? 'bg-green-400'
    : isWaitingForFirstData
    ? 'bg-purple-400'
    : isNewGameStarting
    ? 'bg-blue-400'
    : isPlaybackStarting
    ? 'bg-green-400'
    : 'bg-blue-400'
  const bgTint = checkpointLoadingError
    ? 'bg-red-500/20'
    : progress === 100
    ? 'bg-green-500/20'
    : isWaitingForFirstData
    ? 'bg-purple-500/20'
    : isNewGameStarting
    ? 'bg-blue-500/20'
    : isPlaybackStarting
    ? 'bg-green-500/20'
    : 'bg-blue-500/20'

  return (
    <AnimatePresence>
      {/* Container fixed to top, spans full width */}
      <motion.div
        key="training-progress-bar"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        transition={{ duration: 0.25 }}
        className={`fixed top-0 left-0 right-0 z-[99999] pointer-events-none ${bgTint}`}
      >
        {/* Progress bar background */}
        <div className="h-1 w-full bg-transparent">
          {/* Animated foreground bar */}
          <motion.div
            className={`h-full ${barColor}`}
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(progress, 100)}%` }}
            transition={{ ease: 'linear', duration: 0.2 }}
          />
        </div>

        {/* Optional status line */}
        <div className="flex items-center justify-center gap-2 py-1 text-xs text-blue-200 backdrop-blur-sm bg-gray-900/40">
          {icon}
          <span className="truncate max-w-[50vw]">
            {message}
          </span>
          {showTime && estimatedTimeRemaining !== null && (
            <span className="whitespace-nowrap flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {Math.ceil(estimatedTimeRemaining)}s
            </span>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  )
}

export default TrainingProgressBar 