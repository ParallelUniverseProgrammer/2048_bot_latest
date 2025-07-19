import React, { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { 
  Plus, 
  Play, 
  CheckCircle, 
  AlertCircle, 
  Zap,
  Palette
} from 'lucide-react'

import { useDesignStore, ModelArchitecture, ModelComponent } from '../stores/designStore'
import { useDeviceDetection } from '../utils/deviceDetection'
import config from '../utils/config'

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
    addComponent,

  } = useDesignStore()

  const [showBlockPalette, setShowBlockPalette] = useState(false)
  const [designName, setDesignName] = useState('')
  const workerRef = useRef<Worker | null>(null)

  // Initialize web worker
  useEffect(() => {
    if (typeof window !== 'undefined') {
      workerRef.current = new Worker(new URL('../workers/design_worker.ts', import.meta.url))
      
      workerRef.current.onmessage = (e) => {
        setValidation(e.data)
      }
      
      // Store reference globally for store subscription
      ;(window as any).designWorker = workerRef.current
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
    setShowBlockPalette(false)
  }

  // Add a block to the design
  const addBlock = (type: string) => {
    if (!currentDesign) return

    const newComponent: ModelComponent = {
      id: `${type}_${Date.now()}`,
      type,
      props: getDefaultProps(type),
      position: { x: 100, y: 100 }
    }

    addComponent(newComponent)
    setShowBlockPalette(false)
  }

  // Get default props for component type
  const getDefaultProps = (type: string): Record<string, number | string | boolean> => {
    switch (type) {
      case 'TRANSFORMER_LAYER':
        return { d_model: 128, n_heads: 4 }
      case 'MOE_LAYER':
        return { d_model: 128, n_experts: 4 }
      case 'BOARD_INPUT':
        return { d_model: 128 }
      case 'ACTION_OUTPUT':
        return { d_model: 128 }
      case 'VALUE_HEAD':
        return { d_model: 128 }
      default:
        return {}
    }
  }

  // Compile design
  const compileDesign = async () => {
    if (!currentDesign) return

    setIsCompiling(true)
    try {
      const response = await fetch(`${config.api.baseUrl}/api/designs/${currentDesign.id}/compile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentDesign)
      })

      if (response.ok) {
        const result = await response.json()
        useDesignStore.getState().setCompileResult(result)
      } else {
        console.error('Compilation failed')
      }
    } catch (error) {
      console.error('Compilation error:', error)
    } finally {
      setIsCompiling(false)
    }
  }

  // Train design
  const trainDesign = async () => {
    if (!currentDesign) return

    setIsTraining(true)
    try {
      const response = await fetch(`${config.api.baseUrl}/api/designs/${currentDesign.id}/train`, {
        method: 'POST'
      })

      if (response.ok) {
        console.log('Training started')
      } else {
        console.error('Training failed to start')
      }
    } catch (error) {
      console.error('Training error:', error)
    }
  }

  // Block palette component
  const BlockPalette = () => (
    <motion.div
      className="absolute top-full right-0 mt-1 z-50 bg-gray-800 border border-gray-700 rounded-xl p-2 shadow-lg min-w-48"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
    >
      <div className="grid grid-cols-2 gap-1">
        {[
          { type: 'BOARD_INPUT', label: 'Input', icon: '📥' },
          { type: 'TRANSFORMER_LAYER', label: 'Transformer', icon: '⚡' },
          { type: 'MOE_LAYER', label: 'MoE', icon: '🧠' },
          { type: 'ACTION_OUTPUT', label: 'Actions', icon: '🎯' },
          { type: 'VALUE_HEAD', label: 'Value', icon: '💰' }
        ].map(({ type, label, icon }) => (
          <button
            key={type}
            onClick={() => addBlock(type)}
            className="flex flex-col items-center p-1.5 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          >
            <span className="text-sm mb-0.5">{icon}</span>
            <span className="text-xs text-gray-300">{label}</span>
          </button>
        ))}
      </div>
    </motion.div>
  )

  return (
    <div className="h-full flex flex-col space-y-2 pb-6">
      {/* Compact Header with Error Display */}
      <div className="flex-shrink-0 space-y-2">
        {/* Error Display - Compact */}
        {validation && !validation.valid && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="card-glass p-3 rounded-xl border border-red-500/30 bg-red-500/5"
          >
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
              <span className="text-sm font-medium text-red-400">Validation Errors</span>
              <span className="text-sm text-red-300">({validation.errors.length})</span>
            </div>
          </motion.div>
        )}

        {/* Controls Section - Mobile Optimized */}
        <motion.div
          className="card-glass p-3 rounded-xl flex-shrink-0"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {/* Design Name Input - Full Width */}
          <div className="mb-3">
            <input
              type="text"
              placeholder="Design name..."
              value={designName}
              onChange={(e) => setDesignName(e.target.value)}
              className="w-full bg-gray-700 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              style={{ fontSize: '16px' }}
            />
          </div>

          {/* Control Buttons - Responsive Layout */}
          <div className={`flex ${isMobile ? 'flex-col space-y-2' : 'items-center justify-between'}`}>
            {/* Left Side: Create & Add Block */}
            <div className={`flex ${isMobile ? 'space-x-2' : 'items-center space-x-2'}`}>
              <button
                onClick={createNewDesign}
                className="flex items-center justify-center space-x-2 bg-blue-500 text-white rounded-lg px-3 py-2 text-sm font-medium hover:bg-blue-600 transition-colors"
              >
                <Plus className="w-4 h-4" />
                <span>Create New</span>
              </button>

              <div className="relative">
                <button
                  onClick={() => setShowBlockPalette(!showBlockPalette)}
                  className="flex items-center justify-center space-x-2 bg-gray-700 text-gray-300 rounded-lg px-3 py-2 text-sm hover:bg-gray-600 transition-colors"
                >
                  <Palette className="w-4 h-4" />
                  <span>Add Block</span>
                </button>
                {showBlockPalette && <BlockPalette />}
              </div>
            </div>

            {/* Right Side: Status & Action Buttons */}
            <div className={`flex ${isMobile ? 'space-x-2' : 'items-center space-x-2'}`}>
              {/* Status Indicators */}
              <div className="flex items-center space-x-2">
                {validation?.valid && (
                  <CheckCircle className="w-4 h-4 text-green-400" />
                )}
                {compileResult && (
                  <Zap className="w-4 h-4 text-blue-400" />
                )}
              </div>

              {/* Action Buttons */}
              <button
                onClick={compileDesign}
                disabled={!currentDesign || !validation?.valid || isCompiling}
                className="flex items-center justify-center space-x-2 bg-green-500/20 text-green-400 rounded-lg px-3 py-2 text-sm font-medium hover:bg-green-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Zap className="w-4 h-4" />
                <span>{isCompiling ? 'Compiling...' : 'Compile'}</span>
              </button>

              <button
                onClick={trainDesign}
                disabled={!compileResult || isTraining}
                className="flex items-center justify-center space-x-2 bg-purple-500/20 text-purple-400 rounded-lg px-3 py-2 text-sm font-medium hover:bg-purple-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Play className="w-4 h-4" />
                <span>{isTraining ? 'Training...' : 'Train'}</span>
              </button>
            </div>
          </div>
        </motion.div>

        {/* Stats Bar - Compact */}
        <motion.div
          className="card-glass p-3 rounded-xl flex-shrink-0"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className={`grid ${isMobile ? 'grid-cols-4 gap-2' : 'grid-cols-4 gap-3'}`}>
            <div className="text-center p-2 bg-gray-800/30 rounded-lg">
              <div className="text-sm font-bold text-blue-400">
                {currentDesign?.components.length || 0}
              </div>
              <div className="text-xs text-gray-400">Components</div>
            </div>
            <div className="text-center p-2 bg-gray-800/30 rounded-lg">
              <div className="text-sm font-bold text-purple-400">
                {paramCount.toFixed(1)}M
              </div>
              <div className="text-xs text-gray-400">Parameters</div>
            </div>
            <div className="text-center p-2 bg-gray-800/30 rounded-lg">
              <div className="text-sm font-bold text-orange-400">
                {estimatedMemory}MB
              </div>
              <div className="text-xs text-gray-400">Memory</div>
            </div>
            <div className="text-center p-2 bg-gray-800/30 rounded-lg">
              <div className="text-sm font-bold text-cyan-400">
                {currentDesign?.edges.length || 0}
              </div>
              <div className="text-xs text-gray-400">Connections</div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Canvas Area - Optimized for Mobile */}
      <div className="flex-1 card-glass rounded-2xl overflow-hidden min-h-0">
        {currentDesign ? (
          <div className="h-full p-4 pb-6">
            <div className="text-center text-gray-400 text-sm">
              <p>Canvas coming in Week 1</p>
              <p className="text-xs mt-1">Drag & drop blocks, connect components</p>
              <div className="mt-4 text-xs text-gray-500">
                <p>Available space: {Math.round(window.innerHeight * 0.7)}px height</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center pb-6">
            <div className="text-center text-gray-400">
              <Palette className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-sm">Create a new design to get started</p>
              <p className="text-xs mt-2 text-gray-500">Large canvas area ready for drag & drop</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default ModelStudioTab 