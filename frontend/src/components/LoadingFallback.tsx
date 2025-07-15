import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Loader2, AlertTriangle, WifiOff } from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'
import { 
  isMobileSafari, 
  getLoadingFallbackTimeout, 
  getMaxReconnectAttempts, 
  getConnectionRetryDelay
} from '../utils/mobile-detection'

interface LoadingFallbackProps {
  children: React.ReactNode
}

const LoadingFallback: React.FC<LoadingFallbackProps> = ({ children }) => {
  const [showFallback, setShowFallback] = useState(false)
  const [loadingTimeout, setLoadingTimeout] = useState(false)
  const [retryAttempt, setRetryAttempt] = useState(0)
  const [timeRemaining, setTimeRemaining] = useState(0)
  const { isConnected, connectionError } = useTrainingStore()

  useEffect(() => {
    if (isMobileSafari()) {
      const fallbackTimeout = getLoadingFallbackTimeout()
      
      // Set up the main fallback timeout
      const mainTimeout = setTimeout(() => {
        if (!isConnected) {
          setLoadingTimeout(true)
          setShowFallback(true)
        }
      }, fallbackTimeout)

      // Set up retry progress tracking
      const maxAttempts = getMaxReconnectAttempts()
      const baseDelay = getConnectionRetryDelay()
      let currentTime = 0
      
      const trackRetries = () => {
        for (let i = 1; i <= maxAttempts; i++) {
          const delay = Math.min(baseDelay * Math.pow(2, i - 1), 30000)
          currentTime += delay
          
          setTimeout(() => {
            if (!isConnected && !loadingTimeout) {
              setRetryAttempt(i)
              setTimeRemaining(Math.max(0, (fallbackTimeout - currentTime) / 1000))
            }
          }, currentTime)
        }
      }
      
      trackRetries()

      return () => {
        clearTimeout(mainTimeout)
      }
    }
  }, [isConnected, loadingTimeout])

  useEffect(() => {
    // If we get connected later, hide the fallback
    if (isConnected && showFallback) {
      setShowFallback(false)
      setLoadingTimeout(false)
      setRetryAttempt(0)
      setTimeRemaining(0)
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
            {loadingTimeout 
              ? "Could not connect to the training server after multiple attempts." 
              : "Having trouble connecting to the training server from mobile Safari."
            }
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
              onClick={() => {
                setShowFallback(false)
                setLoadingTimeout(false)
                setRetryAttempt(0)
                setTimeRemaining(0)
              }}
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
      {/* Enhanced loading overlay for mobile Safari with retry progress */}
      {isMobileSafari() && !isConnected && !loadingTimeout && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gray-800 rounded-lg p-6 text-center max-w-sm w-full mx-4"
          >
            <div className="flex items-center justify-center mb-4">
              {retryAttempt > 0 ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                >
                  <WifiOff className="h-8 w-8 text-yellow-400" />
                </motion.div>
              ) : (
                <Loader2 className="h-8 w-8 text-blue-400 animate-spin" />
              )}
            </div>
            
            <h3 className="text-lg font-semibold text-white mb-2">
              {retryAttempt > 0 ? `Retrying Connection` : 'Connecting to Training Server'}
            </h3>
            
            <p className="text-gray-300 text-sm mb-4">
              {retryAttempt > 0 
                ? `Attempt ${retryAttempt} of ${getMaxReconnectAttempts()}` 
                : 'Setting up connection for mobile Safari...'
              }
            </p>
            
            {retryAttempt > 0 && timeRemaining > 0 && (
              <div className="mb-4">
                <div className="bg-gray-700 rounded-full h-2 overflow-hidden">
                  <motion.div
                    className="bg-blue-500 h-full rounded-full"
                    initial={{ width: '0%' }}
                    animate={{ width: `${((getMaxReconnectAttempts() - retryAttempt) / getMaxReconnectAttempts()) * 100}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <p className="text-gray-400 text-xs mt-2">
                  {Math.round(timeRemaining)}s remaining before fallback
                </p>
              </div>
            )}
          </motion.div>
        </div>
      )}
    </>
  )
}

export default LoadingFallback 