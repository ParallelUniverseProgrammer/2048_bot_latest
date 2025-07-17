import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Loader2 } from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'

/**
 * LoadingProgressBar – shows a minimal, non-intrusive progress bar at the very top of the screen
 * for both training-start and checkpoint-load operations. It replaces the previous floating pop-ups.
 */
const TrainingProgressBar: React.FC = () => {
  const { loadingStates } = useTrainingStore()
  const {
    isTrainingStarting,
    isCheckpointLoading,
    checkpointLoadingProgress,
    checkpointLoadingMessage,
    checkpointLoadingError,
    loadingProgress,
    loadingStep,
    estimatedTimeRemaining,
  } = loadingStates

  const isActive = isTrainingStarting || isCheckpointLoading || !!checkpointLoadingMessage || !!checkpointLoadingError

  if (!isActive) {
    return null
  }

  // Determine message & progress
  const progress = isTrainingStarting ? loadingProgress : checkpointLoadingProgress
  const message = isTrainingStarting
    ? loadingStep || 'Initializing…'
    : checkpointLoadingError || checkpointLoadingMessage || 'Loading checkpoint…'

  // Color scheme based on state
  const barColor = checkpointLoadingError
    ? 'bg-red-400'
    : progress === 100
    ? 'bg-green-400'
    : 'bg-blue-400'
  const bgTint = checkpointLoadingError
    ? 'bg-red-500/20'
    : progress === 100
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
          <Loader2 className="w-3 h-3 animate-spin" />
          <span className="truncate max-w-[50vw]">
            {message}
          </span>
          {isTrainingStarting && estimatedTimeRemaining !== null && (
            <span className="whitespace-nowrap">
              {Math.ceil(estimatedTimeRemaining)}s
            </span>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  )
}

export default TrainingProgressBar 