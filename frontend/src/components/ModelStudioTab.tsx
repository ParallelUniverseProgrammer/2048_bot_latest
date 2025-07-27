import React, { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { 
  Plus, 
  Play, 
  CheckCircle, 
  AlertCircle, 
  Zap,
  Palette,
  Cpu,
  MemoryStick,
  Network
} from 'lucide-react'

import { useDesignStore, ModelArchitecture } from '../stores/designStore'
import { useDeviceDetection } from '../utils/deviceDetection'
import config from '../utils/config'
import ModelStudioCanvas from './Canvas/ModelStudioCanvas'
import ModelStudioPalette from './Palette/ModelStudioPalette'

const ModelStudioTab: React.FC = () => {
  const { displayMode } = useDeviceDetection()
  const isMobile = displayMode === 'mobile'
  
  const {
    currentDesign,
    validation,
    paramCount,
    estimatedMemory,
    isCompiling,
    isTraining,
    compileResult,
    setDesign,
    setValidation,
    setIsCompiling,
    setIsTraining,
  } = useDesignStore()

  const [designName, setDesignName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const workerRef = useRef<Worker | null>(null)

  // Initialize web worker
  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        workerRef.current = new Worker(
          new URL('../workers/design_worker.ts', import.meta.url)
        )
        
        workerRef.current.onmessage = (e) => {
          setValidation(e.data)
        }
        
        workerRef.current.onerror = (error) => {
          console.error('Design worker error:', error)
          setError('Validation worker failed to initialize')
        }
        
        // Store reference globally for store subscription
        ;(window as any).designWorker = workerRef.current
      } catch (err) {
        console.error('Failed to create design worker:', err)
        setError('Failed to initialize validation system')
      }
    }

    return () => {
      if (workerRef.current) {
        workerRef.current.terminate()
      }
    }
  }, [])

  // Create new design
  const createNewDesign = () => {
    const newDesign: ModelArchitecture = {
      id: `design_${Date.now()}`,
      name: designName || 'Untitled Design',
      components: [],
      edges: [],
      meta: {
        d_model: 128,
        n_heads: 4,
        n_experts: 4
      }
    }
    setDesign(newDesign)
    setDesignName('')
    setError(null)
  }

  // Compile design
  const compileDesign = async () => {
    if (!currentDesign) return

    setIsCompiling(true)
    setError(null)
    
    try {
      const response = await fetch(
        `${config.api.baseUrl}/api/designs/${currentDesign.id}/compile`, 
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(currentDesign)
        }
      )

      if (response.ok) {
        const result = await response.json()
        useDesignStore.getState().setCompileResult(result)
      } else {
        const errorData = await response.json().catch(() => ({ 
          message: 'Compilation failed' 
        }))
        setError(errorData.message || 'Compilation failed')
      }
    } catch (err) {
      console.error('Compilation error:', err)
      setError('Network error during compilation')
    } finally {
      setIsCompiling(false)
    }
  }

  // Train design
  const trainDesign = async () => {
    if (!currentDesign) return

    setIsTraining(true)
    setError(null)
    
    try {
      const response = await fetch(
        `${config.api.baseUrl}/api/designs/${currentDesign.id}/train`, 
        {
          method: 'POST'
        }
      )

      if (response.ok) {
        console.log('Training started successfully')
      } else {
        const errorData = await response.json().catch(() => ({ 
          message: 'Training failed to start' 
        }))
        setError(errorData.message || 'Training failed to start')
        setIsTraining(false)
      }
    } catch (err) {
      console.error('Training error:', err)
      setError('Network error during training start')
      setIsTraining(false)
    }
  }

  // Metrics data for stats display
  const metrics = [
    {
      title: 'Components',
      value: currentDesign?.components.length || 0,
      icon: Cpu,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/10'
    },
    {
      title: 'Parameters',
      value: `${paramCount.toFixed(1)}M`,
      icon: Zap,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/10'
    },
    {
      title: 'Memory',
      value: `${estimatedMemory}MB`,
      icon: MemoryStick,
      color: 'text-orange-400',
      bgColor: 'bg-orange-500/10'
    },
    {
      title: 'Connections',
      value: currentDesign?.edges.length || 0,
      icon: Network,
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-500/10'
    }
  ]

  return (
    <div className="safe-area h-full flex flex-col md:flex-row p-2 md:p-4 space-y-2 md:space-y-0 md:space-x-2">
      {/* Palette Sidebar - 25% width on desktop */}
      <aside className="md:w-1/4 lg:w-1/5 flex-shrink-0">
        <ModelStudioPalette />
      </aside>

      {/* Main Content Area - Follows style guide layout pattern */}
      <main className="flex-1 flex flex-col space-y-2 pb-6">
        {/* Error Display */}
        {(error || (validation && !validation.valid)) && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="card-glass p-4 rounded-2xl border border-red-500/30 bg-red-500/5 flex-shrink-0"
          >
            <div className="flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <div className="min-w-0">
                <div className="text-sm font-medium text-red-400 mb-1">
                  {error ? 'Error' : 'Validation Issues'}
                </div>
                {error ? (
                  <p className="text-sm text-red-300">{error}</p>
                ) : (
                  <div className="space-y-1">
                    {validation?.errors.slice(0, 3).map((err, i) => (
                      <p key={i} className="text-xs text-red-300">• {err}</p>
                    ))}
                    {validation && validation.errors.length > 3 && (
                      <p className="text-xs text-red-400">
                        +{validation.errors.length - 3} more issues
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}

        {/* Design Controls Section */}
        <motion.div
          className="card-glass p-4 rounded-2xl flex-shrink-0"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {/* Design Name Input */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Design Name
            </label>
            <input
              type="text"
              placeholder="Enter design name..."
              value={designName}
              onChange={(e) => setDesignName(e.target.value)}
              className="w-full bg-gray-700 text-white rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-gray-600 transition-colors"
              style={{ fontSize: '16px', minHeight: '44px' }}
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck="false"
            />
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-2">
            <button
              onClick={createNewDesign}
              className="flex-1 flex items-center justify-center space-x-2 bg-blue-500 text-white rounded-xl py-2.5 text-sm font-medium hover:bg-blue-600 transition-colors"
              style={{ minHeight: '44px' }}
            >
              <Plus className="w-4 h-4" />
              <span>Create New Design</span>
            </button>

            <button
              onClick={compileDesign}
              disabled={!currentDesign || !validation?.valid || isCompiling}
              className="flex-1 flex items-center justify-center space-x-2 bg-green-500/20 text-green-400 rounded-xl py-2.5 text-sm font-medium hover:bg-green-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              style={{ minHeight: '44px' }}
            >
              <Zap className="w-4 h-4" />
              <span>{isCompiling ? 'Compiling...' : 'Compile'}</span>
            </button>

            <button
              onClick={trainDesign}
              disabled={!compileResult || isTraining}
              className="flex-1 flex items-center justify-center space-x-2 bg-purple-500/20 text-purple-400 rounded-xl py-2.5 text-sm font-medium hover:bg-purple-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              style={{ minHeight: '44px' }}
            >
              <Play className="w-4 h-4" />
              <span>{isTraining ? 'Training...' : 'Train'}</span>
            </button>
          </div>

          {/* Status Indicators */}
          {(validation?.valid || compileResult) && (
            <div className="flex items-center justify-center space-x-4 mt-3 pt-3 border-t border-gray-700/50">
              {validation?.valid && (
                <div className="flex items-center space-x-1">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span className="text-xs text-green-400">Valid Design</span>
                </div>
              )}
              {compileResult && (
                <div className="flex items-center space-x-1">
                  <Zap className="w-4 h-4 text-blue-400" />
                  <span className="text-xs text-blue-400">Compiled</span>
                </div>
              )}
            </div>
          )}
        </motion.div>

        {/* Stats Overview */}
        <motion.div
          className="card-glass p-4 rounded-2xl flex-shrink-0"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className={`grid ${isMobile ? 'grid-cols-2 gap-2' : 'grid-cols-4 gap-3'}`}>
            {metrics.map((metric, index) => (
              <motion.div
                key={metric.title}
                className="flex items-center space-x-2 p-3 bg-gray-800/30 rounded-xl"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.2 + index * 0.05 }}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className={`p-1 rounded ${metric.bgColor}`}>
                  <metric.icon className={`w-3 h-3 ${metric.color}`} />
                </div>
                <div className="min-w-0">
                  <div className="text-xs text-gray-400 font-medium truncate">
                    {metric.title}
                  </div>
                  <div className={`font-bold ${metric.color} text-sm truncate`}>
                    {metric.value}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Canvas Area - 70%+ height allocation */}
        <div className="flex-1 card-glass rounded-2xl overflow-hidden min-h-0">
          {currentDesign ? (
            <ModelStudioCanvas />
          ) : (
            <motion.div 
              className="h-full flex items-center justify-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              <div className="text-center text-gray-400 px-4">
                <motion.div
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.4, type: "spring", stiffness: 200 }}
                >
                  <Palette className="w-16 h-16 mx-auto mb-4 opacity-50" />
                </motion.div>
                <h3 className="text-lg font-medium text-gray-300 mb-2">
                  Ready to Design
                </h3>
                <p className="text-sm text-gray-400 mb-2">
                  Create a new design to start building your model
                </p>
                <p className="text-xs text-gray-500">
                  Large canvas optimized for touch drag & drop
                </p>
              </div>
            </motion.div>
          )}
        </div>
      </main>
    </div>
  )
}

export default ModelStudioTab