import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Loader2, AlertTriangle } from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'
import { isMobileSafari } from '../utils/mobile-detection'

interface LoadingFallbackProps {
  children: React.ReactNode
}

const LoadingFallback: React.FC<LoadingFallbackProps> = ({ children }) => {
  const [showFallback, setShowFallback] = useState(false)
  const [loadingTimeout, setLoadingTimeout] = useState(false)
  const { isConnected, connectionError } = useTrainingStore()

  useEffect(() => {
    // If we're on mobile Safari and haven't connected within 10 seconds, show fallback
    if (isMobileSafari()) {
      const timeout = setTimeout(() => {
        if (!isConnected) {
          setLoadingTimeout(true)
          setShowFallback(true)
        }
      }, 10000) // 10 second timeout

      return () => clearTimeout(timeout)
    }
  }, [isConnected])

  useEffect(() => {
    // If we get connected later, hide the fallback
    if (isConnected && showFallback) {
      setShowFallback(false)
      setLoadingTimeout(false)
    }
  }, [isConnected, showFallback])

  // Show fallback if we're on mobile Safari and either timed out or have a connection error
  if (isMobileSafari() && (loadingTimeout || connectionError)) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-gray-800 rounded-lg p-6 max-w-md w-full text-center"
        >
          <div className="flex items-center justify-center mb-4">
            <AlertTriangle className="h-8 w-8 text-yellow-400 mr-2" />
            <h2 className="text-xl font-bold text-white">Connection Issue</h2>
          </div>
          
          <p className="text-gray-300 mb-4">
            Having trouble connecting to the training server from mobile Safari.
          </p>
          
          <div className="space-y-2 text-sm text-gray-400 mb-4">
            <p>• Make sure you're on the same Wi-Fi network as the server</p>
            <p>• Try refreshing the page</p>
            <p>• Check if the server is still running</p>
            <p>• Consider using a different browser</p>
          </div>

          <div className="space-y-2">
            <button
              onClick={() => window.location.reload()}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition-colors"
            >
              Refresh Page
            </button>
            
            <button
              onClick={() => setShowFallback(false)}
              className="w-full bg-gray-600 hover:bg-gray-700 text-white font-medium py-2 px-4 rounded-md transition-colors"
            >
              Try Again
            </button>
          </div>

          {connectionError && (
            <div className="mt-4 p-3 bg-red-900/50 border border-red-700 rounded-md">
              <p className="text-red-200 text-sm">
                Error: {connectionError}
              </p>
            </div>
          )}
        </motion.div>
      </div>
    )
  }

  return (
    <>
      {children}
      {/* Show a loading overlay for mobile Safari if taking too long */}
      {isMobileSafari() && !isConnected && !loadingTimeout && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gray-800 rounded-lg p-6 text-center"
          >
            <Loader2 className="h-8 w-8 text-blue-400 animate-spin mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">
              Connecting to Training Server
            </h3>
            <p className="text-gray-300 text-sm">
              Setting up connection for mobile Safari...
            </p>
          </motion.div>
        </div>
      )}
    </>
  )
}

export default LoadingFallback 