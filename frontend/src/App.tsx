import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Activity, Brain, GamepadIcon, Archive, Palette } from 'lucide-react'

import TrainingDashboard from './components/TrainingDashboard'
import GameBoard from './components/GameBoard'
import ConnectionStatus from './components/ConnectionStatus'

import CheckpointManager from './components/CheckpointManager'
import ModelStudioTab from './components/ModelStudioTab'
import TrainingProgressBar from './components/TrainingProgressBar'
import { useTrainingStore } from './stores/trainingStore'
import { useWebSocket } from './utils/websocket'
import { useDeviceDetection } from './utils/deviceDetection'
import config from './utils/config'

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'game' | 'checkpoints' | 'model-studio'>('dashboard')
  const [showIOSTooltip, setShowIOSTooltip] = useState(false)
  
  // Intelligent device detection
  const { displayMode } = useDeviceDetection()
  const isMobile = displayMode === 'mobile'
  
  // Show iOS tooltip for iOS Safari users (only when not already installed as PWA)
  useEffect(() => {
    const isIOSSafari = /iPad|iPhone|iPod/.test(navigator.userAgent) && 
                       /Safari/.test(navigator.userAgent) && 
                       !/Chrome/.test(navigator.userAgent)
    
    // Check if app is already running in standalone mode (installed as PWA)
    const isStandalone = (window.navigator as any).standalone === true
    
    if (isIOSSafari && isMobile && !isStandalone) {
      // Show tooltip after a short delay
      const timer = setTimeout(() => {
        setShowIOSTooltip(true)
      }, 3000)
      
      // Hide tooltip after 8 seconds
      const hideTimer = setTimeout(() => {
        setShowIOSTooltip(false)
      }, 8000)
      
      return () => {
        clearTimeout(timer)
        clearTimeout(hideTimer)
      }
    }
  }, [isMobile])
  
  const { 
    isTraining, 
    isPaused,
    currentEpisode, 
    trainingData
  } = useTrainingStore()
  
  // Initialize WebSocket connection
  useWebSocket()
  
  // Sync training status with backend on page load to avoid stale localStorage state
  useEffect(() => {
    const syncTrainingStatus = async () => {
      try {
        const response = await fetch(`${config.api.baseUrl}/training/status`)
        if (response.ok) {
          const status = await response.json()
          console.log('Initial training status sync:', status)
          useTrainingStore.getState().setTrainingStatus(status.is_training, status.is_paused)
          useTrainingStore.getState().setEpisode(status.current_episode)
        }
      } catch (error) {
        console.warn('Failed to sync initial training status:', error)
      }
    }
    
    // Sync after a short delay to ensure the store is hydrated
    const timer = setTimeout(syncTrainingStatus, 100)
    return () => clearTimeout(timer)
  }, [])
  
  // Navigation handler for child components
  const handleNavigateToTab = (tab: string) => {
    const validTabs = ['dashboard', 'game', 'checkpoints', 'model-studio'] as const
    if (validTabs.includes(tab as any)) {
      setActiveTab(tab as typeof activeTab)
    }
  }

  const tabs = [
    { id: 'dashboard', label: 'Training', icon: Activity },
    { id: 'game', label: 'Game', icon: GamepadIcon },
    { id: 'checkpoints', label: 'Checkpoints', icon: Archive },
    { id: 'model-studio', label: 'Studio', icon: Palette },
  ]

  return (
    <div className={`h-screen flex flex-col relative transition-all duration-1000 ${
      isTraining && !isPaused 
        ? 'bg-gradient-to-br from-gray-900 via-purple-900/20 to-gray-900' 
        : isPaused 
        ? 'bg-gradient-to-br from-gray-900 via-yellow-900/10 to-gray-900'
        : 'bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900'
    }`}>
      {/* Overarching Training Animation */}
      {isTraining && !isPaused && (
        <>
          {/* Animated background gradient */}
          <motion.div
            className="absolute inset-0 pointer-events-none opacity-20"
            animate={{
              opacity: [0.1, 0.3, 0.1],
              scale: [1, 1.1, 1],
            }}
            transition={{
              duration: 8,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            style={{
              background: 'radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.5) 0%, transparent 70%)',
            } as React.CSSProperties}
          />
          
          {/* Floating particles */}
          <div className="absolute inset-0 pointer-events-none overflow-hidden">
            {[...Array(6)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-2 h-2 bg-blue-400 rounded-full opacity-60"
                animate={{
                  x: [0, Math.random() * window.innerWidth],
                  y: [0, Math.random() * window.innerHeight],
                  scale: [0, 1, 0],
                  opacity: [0, 0.6, 0],
                }}
                transition={{
                  duration: 8 + Math.random() * 4,
                  repeat: Infinity,
                  delay: Math.random() * 5,
                  ease: "easeInOut"
                }}
                style={{
                  left: Math.random() * 100 + '%',
                  top: Math.random() * 100 + '%',
                }}
              />
            ))}
          </div>
        </>
      )}
      
      {/* Pause state overlay */}
      {isPaused && (
        <motion.div
          className="absolute inset-0 pointer-events-none bg-yellow-400/5"
          animate={{
            opacity: [0.3, 0.7, 0.3],
            scale: [1, 1.02, 1],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      )}

      {/* Header - Minimalist and centered */}
      <motion.header 
        className={`card-glass flex-shrink-0 z-40 ${isMobile ? 'px-4 py-3' : 'px-6 py-4 sm:px-8 lg:px-10'}`}
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <div className="flex items-center justify-between">
          {/* Logo - Centered */}
          <div className="flex items-center space-x-3">
            <motion.div
              className={`flex items-center justify-center ${isMobile ? 'w-8 h-8' : 'w-10 h-10'} bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Brain className={`${isMobile ? 'w-4 h-4' : 'w-5 h-5'} text-white`} />
            </motion.div>
            <div>
              <h1 className={`font-bold text-gradient ${isMobile ? 'text-base' : 'text-lg sm:text-xl'}`}>
                {isMobile ? '2048 AI' : '2048 Bot'}
              </h1>
              {!isMobile && (
                <p className="text-xs text-gray-400">
                  Episode {currentEpisode.toLocaleString()}
                </p>
              )}
            </div>
          </div>

          {/* Status - Right side */}
          <div className="flex items-center space-x-3">
            {/* Connection Status */}
            <ConnectionStatus />

            {/* Training Status Indicator */}
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                isTraining && !isPaused ? 'bg-green-400 animate-pulse' :
                isPaused ? 'bg-yellow-400' :
                'bg-gray-400'
              }`} />
              <span className="text-xs text-gray-300 font-medium">
                {isTraining && !isPaused ? 'Training' :
                 isPaused ? 'Paused' : 'Idle'}
              </span>
            </div>
          </div>
        </div>

        {/* Compact Status Bar - Only show essential info */}
        {trainingData && !isMobile && (
          <motion.div 
            className="mt-3 flex items-center justify-center text-xs text-gray-400"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <div className="flex items-center space-x-6">
              <span>Episode {currentEpisode.toLocaleString()}</span>
              <span>Score: {trainingData.score?.toLocaleString()}</span>
              <span>GPU: {trainingData.gpu_memory?.toFixed(1)}GB</span>
            </div>
          </motion.div>
        )}
      </motion.header>

      {/* Navigation Tabs - More compact */}
      <motion.nav 
        className={`card-glass flex-shrink-0 mt-1 rounded-2xl ${isMobile ? 'mx-2' : 'mx-4 sm:mx-6 lg:mx-8'}`}
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <div className="flex justify-center space-x-1 p-1">
          {tabs.map((tab) => {
            const IconComponent = tab.icon
            return (
              <motion.button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`relative flex flex-col items-center space-y-0.5 rounded-xl font-medium transition-all duration-200 ${
                  isMobile ? 'px-2 py-1' : 'px-3 py-1'
                } ${
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <IconComponent className="w-3 h-3" />
                <span className="text-xs">{tab.label}</span>
                {activeTab === tab.id && (
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl -z-10"
                    layoutId="activeTab"
                    initial={false}
                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                  />
                )}
              </motion.button>
            )
          })}
        </div>
      </motion.nav>

      {/* Main Content - More generous padding */}
      <motion.main 
        className={`flex-1 overflow-hidden ${isMobile ? 'mobile-main pb-4' : 'px-6 py-4 sm:px-8 lg:px-10 pb-4'}`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="h-full"
          >
            {activeTab === 'dashboard' && <TrainingDashboard />}
            {activeTab === 'game' && <GameBoard />}
            {activeTab === 'checkpoints' && <CheckpointManager onNavigateToTab={handleNavigateToTab} />}
            {activeTab === 'model-studio' && <ModelStudioTab />}
          </motion.div>
        </AnimatePresence>
      </motion.main>

      {/* iOS Tooltip - Only show for iOS Safari users when not already installed as PWA */}
      <AnimatePresence>
        {showIOSTooltip && (
          <motion.div
            className="fixed bottom-4 left-4 right-4 z-50 bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4 rounded-lg shadow-lg"
            initial={{ y: 100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 100, opacity: 0 }}
            transition={{ duration: 0.3 }}
            style={{
              paddingBottom: 'max(16px, env(safe-area-inset-bottom))',
              paddingLeft: 'max(16px, env(safe-area-inset-left))',
              paddingRight: 'max(16px, env(safe-area-inset-right))'
            }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center">
                  📱
                </div>
                <div>
                  <p className="font-semibold text-sm">Add to Home Screen</p>
                  <p className="text-xs opacity-90">Tap the share button (📤) then "Add to Home Screen"</p>
                </div>
              </div>
              <button
                onClick={() => setShowIOSTooltip(false)}
                className="text-white opacity-70 hover:opacity-100 transition-opacity min-w-[32px] min-h-[32px] flex items-center justify-center"
              >
                ✕
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Global Loading Indicator (training + checkpoint) handled above */}

      {/* Training Loading Indicator */}
      <TrainingProgressBar />

      {/* Footer - Removed for mobile to save space */}
      {!isMobile && (
        <motion.footer 
          className="card-glass mt-4 mx-4 sm:mx-6 lg:mx-8 p-2 text-center text-xs text-gray-400"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          <p>2048 Bot Training • Real-time AI Visualization</p>
        </motion.footer>
      )}
    </div>
  )
}

export default App