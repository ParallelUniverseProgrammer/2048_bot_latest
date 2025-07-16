import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Activity, Brain, GamepadIcon, Network, Play, Pause, AlertTriangle, Archive } from 'lucide-react'

import TrainingDashboard from './components/TrainingDashboard'
import GameBoard from './components/GameBoard'
import NetworkVisualizer from './components/NetworkVisualizer'
import ConnectionStatus from './components/ConnectionStatus'

import CheckpointManager from './components/CheckpointManager'
import { useTrainingStore } from './stores/trainingStore'
import { useWebSocket } from './utils/websocket'
import { useDeviceDetection } from './utils/deviceDetection'
import config from './utils/config'

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'game' | 'network' | 'checkpoints'>('dashboard')
  const [showIOSTooltip, setShowIOSTooltip] = useState(false)
  
  // Intelligent device detection
  const { displayMode } = useDeviceDetection()
  const isMobile = displayMode === 'mobile'
  
  // Show iOS tooltip for iOS Safari users
  useEffect(() => {
    const isIOSSafari = /iPad|iPhone|iPod/.test(navigator.userAgent) && 
                       /Safari/.test(navigator.userAgent) && 
                       !/Chrome/.test(navigator.userAgent)
    
    if (isIOSSafari && isMobile) {
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
    isConnected, 
    isTraining, 
    isPaused,
    currentEpisode, 
    trainingData,
    loadingStates,
    startTraining,
    pauseTraining,
    resumeTraining,
    stopTraining,
    resetTraining
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
    const validTabs = ['dashboard', 'game', 'network', 'checkpoints'] as const
    if (validTabs.includes(tab as any)) {
      setActiveTab(tab as typeof activeTab)
    }
  }

  const tabs = [
    { id: 'dashboard', label: 'Training', icon: Activity },
    { id: 'game', label: 'Game', icon: GamepadIcon },
    { id: 'network', label: 'Network', icon: Network },
    { id: 'checkpoints', label: 'Checkpoints', icon: Archive },
  ]

  const handleTrainingControl = async (action: 'start' | 'pause' | 'resume' | 'stop' | 'reset') => {
    try {
      switch (action) {
        case 'start':
          await startTraining()
          break
        case 'pause':
          await pauseTraining()
          break
        case 'resume':
          await resumeTraining()
          break
        case 'stop':
          await stopTraining()
          break
        case 'reset':
          await resetTraining()
          break
      }
    } catch (error) {
      console.error('Training control error:', error)
    }
  }

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

      {/* Header */}
      <motion.header 
        className={`card-glass flex-shrink-0 z-40 ${isMobile ? 'px-3 py-2' : 'px-4 py-3 sm:px-6 lg:px-8'}`}
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <div className="flex items-center justify-between">
          {/* Logo and Title - Simplified */}
          <div className="flex items-center space-x-3">
            <motion.div
              className={`flex items-center justify-center ${isMobile ? 'w-8 h-8' : 'w-10 h-10'} bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Brain className={`${isMobile ? 'w-4 h-4' : 'w-6 h-6'} text-white`} />
            </motion.div>
            <div>
              <h1 className={`font-bold text-gradient ${isMobile ? 'text-sm' : 'text-lg sm:text-xl'}`}>
                {isMobile ? '2048 AI' : '2048 Bot'}
              </h1>
              {!isMobile && (
                <p className="text-xs text-gray-400">
                  Episode {currentEpisode.toLocaleString()}
                </p>
              )}
            </div>
          </div>

          {/* Right side - Connection + Main Control */}
          <div className="flex items-center space-x-3">
            {/* Connection Status */}
            <ConnectionStatus />

            {/* Single Primary Control Button */}
            <motion.button
              onClick={() => handleTrainingControl(
                isPaused ? 'resume' : 
                isTraining && !isPaused ? 'pause' : 
                'start'
              )}
              className={`
                relative flex items-center justify-center rounded-xl font-bold transition-all duration-300 
                ${isMobile ? 'w-10 h-10' : 'px-6 py-3 space-x-2'}
                ${loadingStates.isTrainingStarting 
                  ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-lg shadow-blue-500/30' 
                  : isTraining && !isPaused 
                  ? 'bg-gradient-to-r from-orange-500 to-red-500 text-white shadow-lg shadow-orange-500/30' 
                  : isPaused 
                  ? 'bg-gradient-to-r from-yellow-500 to-orange-500 text-white shadow-lg shadow-yellow-500/30'
                  : 'bg-gradient-to-r from-green-500 to-blue-500 text-white shadow-lg shadow-green-500/30'
                }
                ${!isConnected || loadingStates.isTrainingStarting ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105 active:scale-95'}
              `}
              whileHover={isConnected && !loadingStates.isTrainingStarting ? { scale: 1.05 } : {}}
              whileTap={isConnected && !loadingStates.isTrainingStarting ? { scale: 0.95 } : {}}
              disabled={!isConnected || loadingStates.isTrainingStarting}
            >
              {/* Button content */}
              <div className="relative z-10 flex items-center space-x-2">
                {loadingStates.isTrainingStarting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    {!isMobile && <span className="text-sm">Starting...</span>}
                  </>
                ) : isPaused ? (
                  <>
                    <Play className="w-4 h-4" />
                    {!isMobile && <span className="text-sm">Resume</span>}
                  </>
                ) : isTraining && !isPaused ? (
                  <>
                    <Pause className="w-4 h-4" />
                    {!isMobile && <span className="text-sm">Pause</span>}
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    {!isMobile && <span className="text-sm">Start</span>}
                  </>
                )}
              </div>
            </motion.button>

            {/* Secondary Actions Menu (Stop/Reset) - Only show when needed */}
            <AnimatePresence>
              {(isTraining || isPaused) && (
                <motion.button
                  onClick={() => handleTrainingControl('stop')}
                  disabled={loadingStates.isTrainingStopping}
                  className={`
                    relative flex items-center justify-center rounded-xl font-bold transition-all duration-200
                    ${isMobile ? 'w-8 h-8' : 'px-4 py-3 space-x-1'}
                    bg-gradient-to-r from-red-600 to-red-800 text-white 
                    shadow-lg shadow-red-600/40
                    hover:from-red-700 hover:to-red-900
                    active:scale-95
                    ${loadingStates.isTrainingStopping ? 'opacity-75 cursor-not-allowed' : ''}
                  `}
                  initial={{ opacity: 0, scale: 0.8, x: 20 }}
                  animate={{ opacity: 1, scale: 1, x: 0 }}
                  exit={{ opacity: 0, scale: 0.8, x: 20 }}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <AlertTriangle className="w-4 h-4" />
                  {!isMobile && <span className="text-xs">Stop</span>}
                </motion.button>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Compact Status Bar - Only show essential info */}
        {trainingData && !isMobile && (
          <motion.div 
            className="mt-2 flex items-center justify-between text-xs text-gray-400"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <div className="flex items-center space-x-4">
              <span>Episode {currentEpisode.toLocaleString()}</span>
              <span>Score: {trainingData.score?.toLocaleString()}</span>
              <span>GPU: {trainingData.gpu_memory?.toFixed(1)}GB</span>
            </div>
          </motion.div>
        )}
      </motion.header>

      {/* Navigation Tabs - Compact with Labels */}
      <motion.nav 
        className={`card-glass flex-shrink-0 mt-2 ${isMobile ? 'mx-3' : 'mx-4 sm:mx-6 lg:mx-8'}`}
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <div className="flex justify-center space-x-1 p-2">
          {tabs.map((tab) => {
            const IconComponent = tab.icon
            return (
              <motion.button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`relative flex flex-col items-center space-y-1 rounded-lg font-medium transition-all duration-200 ${
                  isMobile ? 'px-3 py-2' : 'px-4 py-2'
                } ${
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <IconComponent className="w-4 h-4" />
                <span className="text-xs">{tab.label}</span>
                {activeTab === tab.id && (
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg -z-10"
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

      {/* Main Content - Scrollable */}
      <motion.main 
        className={`flex-1 overflow-auto ${isMobile ? 'mobile-main pb-6' : 'px-4 py-4 sm:px-6 lg:px-8 pb-6'}`}
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
            {activeTab === 'network' && <NetworkVisualizer />}
            {activeTab === 'checkpoints' && <CheckpointManager onNavigateToTab={handleNavigateToTab} />}
          </motion.div>
        </AnimatePresence>
      </motion.main>

      {/* iOS Tooltip - Only show for iOS Safari users */}
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

      {/* Footer */}
      {!isMobile && (
        <motion.footer 
          className="card-glass mt-8 mx-4 sm:mx-6 lg:mx-8 p-4 text-center text-sm text-gray-400"
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