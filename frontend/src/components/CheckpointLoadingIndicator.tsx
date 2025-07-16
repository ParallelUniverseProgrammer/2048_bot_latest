import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Loader2, CheckCircle, XCircle, Archive } from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'

const CheckpointLoadingIndicator: React.FC = () => {
  const { loadingStates } = useTrainingStore()
  const { 
    isCheckpointLoading, 
    checkpointId, 
    checkpointLoadingMessage, 
    checkpointLoadingProgress,
    checkpointLoadingError 
  } = loadingStates

  // Don't render if no checkpoint loading activity
  if (!isCheckpointLoading && !checkpointLoadingError && !checkpointLoadingMessage) {
    return null
  }

  const getStatusIcon = () => {
    if (checkpointLoadingError) {
      return <XCircle className="w-5 h-5 text-red-400" />
    }
    if (checkpointLoadingProgress === 100) {
      return <CheckCircle className="w-5 h-5 text-green-400" />
    }
    return <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
  }

  const getStatusColor = () => {
    if (checkpointLoadingError) {
      return 'border-red-500/30 bg-red-500/10'
    }
    if (checkpointLoadingProgress === 100) {
      return 'border-green-500/30 bg-green-500/10'
    }
    return 'border-blue-500/30 bg-blue-500/10'
  }

  const getProgressColor = () => {
    if (checkpointLoadingError) {
      return 'bg-red-400'
    }
    if (checkpointLoadingProgress === 100) {
      return 'bg-green-400'
    }
    return 'bg-blue-400'
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.9 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -20, scale: 0.9 }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
        className={`
          fixed bottom-4 right-4 z-50 max-w-sm w-full
          card-glass p-4 rounded-xl border-2 shadow-2xl
          ${getStatusColor()}
        `}
      >
        <div className="flex items-start space-x-3">
          {/* Icon */}
          <div className="flex-shrink-0 mt-0.5">
            {getStatusIcon()}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            {/* Header */}
            <div className="flex items-center space-x-2 mb-2">
              <Archive className="w-4 h-4 text-gray-400" />
              <span className="text-sm font-medium text-white">
                {checkpointLoadingError ? 'Checkpoint Error' : 'Loading Checkpoint'}
              </span>
              {checkpointId && (
                <span className="text-xs text-gray-400 bg-gray-700/50 px-2 py-0.5 rounded">
                  {checkpointId}
                </span>
              )}
            </div>

            {/* Message */}
            <div className="text-sm text-gray-300 mb-3">
              {checkpointLoadingError || checkpointLoadingMessage || 'Processing checkpoint...'}
            </div>

            {/* Progress Bar */}
            {!checkpointLoadingError && (
              <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                <motion.div
                  className={`h-2 rounded-full ${getProgressColor()}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${checkpointLoadingProgress}%` }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                />
              </div>
            )}

            {/* Progress Text */}
            {!checkpointLoadingError && checkpointLoadingProgress < 100 && (
              <div className="text-xs text-gray-400">
                {checkpointLoadingProgress}% complete
              </div>
            )}
          </div>
        </div>

        {/* Auto-dismiss indicator */}
        {(checkpointLoadingProgress === 100 || checkpointLoadingError) && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute top-2 right-2 text-xs text-gray-500"
          >
            Auto-dismiss
          </motion.div>
        )}
      </motion.div>
    </AnimatePresence>
  )
}

export default CheckpointLoadingIndicator 