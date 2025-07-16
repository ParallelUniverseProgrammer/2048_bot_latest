import React from 'react'
import { motion } from 'framer-motion'
import { useTrainingStore } from '../stores/trainingStore'

const ConnectionStatus: React.FC = () => {
  const { isConnected, connectionError } = useTrainingStore()
  
  // Check if we're in polling fallback mode
  const isPollingFallback = connectionError?.includes('polling fallback')
  const isOffline = connectionError?.includes('offline')
  const isNetworkPoor = connectionError?.includes('Poor network')

  // Determine status and styling
  let status = 'Connecting...'
  let bgColor = 'bg-gray-500'
  let textColor = 'text-gray-300'
  let glowColor = 'shadow-gray-500/30'

  if (isConnected && !isPollingFallback) {
    status = 'Connected'
    bgColor = 'bg-green-600'
    textColor = 'text-white'
    glowColor = 'shadow-green-500/40'
  } else if (isConnected && isPollingFallback) {
    status = 'Polling'
    bgColor = 'bg-yellow-600'
    textColor = 'text-white'
    glowColor = 'shadow-yellow-500/40'
  } else if (isOffline) {
    status = 'Offline'
    bgColor = 'bg-red-600'
    textColor = 'text-white'
    glowColor = 'shadow-red-500/40'
  } else if (isNetworkPoor) {
    status = 'Poor Network'
    bgColor = 'bg-orange-600'
    textColor = 'text-white'
    glowColor = 'shadow-orange-500/40'
  } else if (connectionError) {
    status = 'Error'
    bgColor = 'bg-red-600'
    textColor = 'text-white'
    glowColor = 'shadow-red-500/40'
  }

  return (
    <motion.div
      className={`
                      px-3 py-1 rounded-xl text-xs font-medium
        ${bgColor} ${textColor} ${glowColor}
        shadow-lg transition-all duration-300
      `}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.2 }}
      whileHover={{ scale: 1.05 }}
    >
      {status}
    </motion.div>
  )
}

export default ConnectionStatus 