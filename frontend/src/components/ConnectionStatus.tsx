import React from 'react'
import { motion } from 'framer-motion'
import { Wifi, WifiOff, AlertCircle } from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'

const ConnectionStatus: React.FC = () => {
  const { isConnected, connectionError } = useTrainingStore()

  if (isConnected) {
    return (
      <motion.div
        className="flex items-center space-x-1 text-xs text-green-400/80"
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
      >
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        >
          <Wifi className="w-3 h-3" />
        </motion.div>
        <span className="hidden md:inline">Connected</span>
      </motion.div>
    )
  }

  if (connectionError) {
    return (
      <motion.div
        className="flex items-center space-x-1 text-xs text-red-400/80"
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
      >
        <motion.div
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <AlertCircle className="w-3 h-3" />
        </motion.div>
        <span className="hidden md:inline">
          {connectionError.length > 15 ? 'Error' : connectionError}
        </span>
      </motion.div>
    )
  }

  return (
    <motion.div
      className="flex items-center space-x-1 text-xs text-gray-400/60"
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <motion.div
        animate={{ opacity: [0.3, 1, 0.3] }}
        transition={{ duration: 2, repeat: Infinity }}
      >
        <WifiOff className="w-3 h-3" />
      </motion.div>
      <span className="hidden md:inline">Connecting...</span>
    </motion.div>
  )
}

export default ConnectionStatus 