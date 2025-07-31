import React, { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { DndProvider } from 'react-dnd'
import { TouchBackend } from 'react-dnd-touch-backend'
import { 
  Plus, 
  Play, 
  CheckCircle, 
  AlertCircle, 
  Zap,
  Palette,
  Cpu,
  MemoryStick,
  Network,
  Shield,
  Settings,
  BarChart3,
  Layers
} from 'lucide-react'

import { useDesignStore, ModelArchitecture } from '../stores/designStore'
import { useDeviceDetection } from '../utils/deviceDetection'
import config from '../utils/config'
import ModelStudioCanvas from './Canvas/ModelStudioCanvas'
import ModelStudioPalette from './Palette/ModelStudioPalette'

type TabType = 'design' | 'validation' | 'build' | 'metrics'

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

  const [activeTab, setActiveTab] = useState<TabType>('design')
  const [designName, setDesignName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [workerError, setWorkerError] = useState<string | null>(null)
  const workerRef = useRef<Worker | null>(null)

  // Initialize web worker with better error handling
  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        console.log('Initializing design worker...')
        workerRef.current = new Worker(
          new URL('../workers/design_worker.ts', import.meta.url)
        )
        
        workerRef.current.onmessage = (e) => {
          console.log('Worker message received:', e.data)
          setValidation(e.data)
        }
        
        workerRef.current.onerror = (error) => {
          console.error('Design worker error:', error)
          setWorkerError('Validation worker failed to initialize: ' + error.message)
        }
        
        // Store reference globally for store subscription
        ;(window as any).designWorker = workerRef.current
        console.log('Design worker initialized successfully')
      } catch (err) {
        console.error('Failed to create design worker:', err)
        setWorkerError('Failed to initialize validation system: ' + (err as Error).message)
        
        // Try fallback inline worker
        try {
          console.log('Trying fallback inline worker...')
          const workerCode = `
            self.onmessage = function(e) {
              const { type, design } = e.data;
              if (type === 'validate' && design) {
                // Simple validation logic
                const errors = [];
                if (!design.components || design.components.length === 0) {
                  errors.push("At least one component required");
                }
                const result = {
                  valid: errors.length === 0,
                  errors: errors,
                  paramCount: 0,
                  estimatedMemory: 0
                };
                postMessage(result);
              }
            };
          `;
          const blob = new Blob([workerCode], { type: 'application/javascript' });
          workerRef.current = new Worker(URL.createObjectURL(blob));
          
          workerRef.current.onmessage = (e) => {
            console.log('Fallback worker message received:', e.data);
            setValidation(e.data);
          };
          
          workerRef.current.onerror = (error) => {
            console.error('Fallback worker error:', error);
            setWorkerError('Fallback worker also failed: ' + error.message);
          };
          
          ;(window as any).designWorker = workerRef.current;
          console.log('Fallback worker initialized successfully');
        } catch (fallbackErr) {
          console.error('Fallback worker also failed:', fallbackErr);
          setWorkerError('Both worker attempts failed: ' + (fallbackErr as Error).message);
        }
      }
    }

    return () => {
      if (workerRef.current) {
        console.log('Terminating design worker...')
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

  // Tab configuration
  const tabs = [
    {
      id: 'design' as TabType,
      label: 'Design',
      icon: Layers,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/10'
    },
    {
      id: 'validation' as TabType,
      label: 'Validation',
      icon: Shield,
      color: 'text-green-400',
      bgColor: 'bg-green-500/10'
    },
    {
      id: 'build' as TabType,
      label: 'Build',
      icon: Settings,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/10'
    },
    {
      id: 'metrics' as TabType,
      label: 'Metrics',
      icon: BarChart3,
      color: 'text-orange-400',
      bgColor: 'bg-orange-500/10'
    }
  ]

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

  // Render tab content
  const renderTabContent = () => {
    switch (activeTab) {
      case 'design':
        return (
          <DndProvider
            backend={TouchBackend}
            options={{ 
              enableMouseEvents: true, 
              delayTouchStart: 0,
              enableHoverOutsideTarget: true,
              enableKeyboardEvents: false
            }}
          >
            <div className="h-full flex flex-col space-y-2 pb-6">
              {/* Design Controls - Ultra Compact */}
              <motion.div
                className="card-glass p-3 rounded-xl flex-shrink-0"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <div className="flex items-center space-x-2">
                  <input
                    type="text"
                    placeholder="Design name..."
                    value={designName}
                    onChange={(e) => setDesignName(e.target.value)}
                    className="flex-1 bg-gray-700 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-gray-600 transition-colors"
                    style={{ fontSize: '16px', minHeight: '40px' }}
                    autoComplete="off"
                    autoCorrect="off"
                    autoCapitalize="off"
                    spellCheck="false"
                  />
                  <button
                    onClick={createNewDesign}
                    className="flex items-center justify-center space-x-2 bg-blue-500 text-white rounded-lg px-4 py-2 text-sm font-medium hover:bg-blue-600 transition-colors"
                    style={{ minHeight: '40px' }}
                  >
                    <Plus className="w-4 h-4" />
                    <span className="hidden sm:inline">Create</span>
                  </button>
                </div>
              </motion.div>

              {/* Palette - Horizontal Scrollable */}
              <motion.div
                className="card-glass p-3 rounded-xl flex-shrink-0"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-gray-300">Block Palette</h3>
                  <div className="flex items-center space-x-1">
                    {(validation?.valid || compileResult) && (
                      <>
                        {validation?.valid && (
                          <div className="flex items-center space-x-1">
                            <CheckCircle className="w-3 h-3 text-green-400" />
                            <span className="text-xs text-green-400">Valid</span>
                          </div>
                        )}
                        {compileResult && (
                          <div className="flex items-center space-x-1">
                            <Zap className="w-3 h-3 text-blue-400" />
                            <span className="text-xs text-blue-400">Compiled</span>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <ModelStudioPalette />
                </div>
              </motion.div>

              {/* Canvas Area - Constrained height */}
              <div className="card-glass rounded-xl overflow-hidden flex-shrink-0" style={{ height: '30vh' }}>
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
                      <p className="text-xs text-gray-400">
                        Drag blocks from the palette above to get started
                      </p>
                    </div>
                  </motion.div>
                )}
              </div>
            </div>
          </DndProvider>
        )

      case 'validation':
        return (
          <div className="h-full flex flex-col space-y-2 pb-6">
            {/* Error Display - Fixed height */}
            {(error || (validation && !validation.valid) || workerError) && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="card-glass p-3 rounded-xl border border-red-500/30 bg-red-500/5 flex-shrink-0"
              >
                <div className="flex items-start space-x-2">
                  <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-red-400 mb-1">
                      {error || workerError || 'Validation Issues'}
                    </div>
                    {error ? (
                      <p className="text-xs text-red-300">{error}</p>
                    ) : (
                      <>
                        {workerError && <p className="text-xs text-red-300">{workerError}</p>}
                        {validation?.errors.slice(0, 2).map((err, i) => (
                          <p key={i} className="text-xs text-red-300">• {err}</p>
                        ))}
                        {validation && validation.errors.length > 2 && (
                          <p className="text-xs text-red-400">
                            +{validation.errors.length - 2} more issues
                          </p>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </motion.div>
            )}

            {/* Validation Results - Fixed height */}
            <motion.div
              className="card-glass p-3 rounded-xl flex-shrink-0"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <h3 className="text-sm font-medium text-gray-300 mb-2">Design Analysis</h3>
              <div className={`grid ${isMobile ? 'grid-cols-2 gap-2' : 'grid-cols-4 gap-2'}`}>
                {metrics.map((metric, index) => (
                  <motion.div
                    key={metric.title}
                    className="flex items-center space-x-2 p-2 bg-gray-800/30 rounded-lg"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: 0.1 + index * 0.05 }}
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

            {/* Validation Details - Scrollable content */}
            <div className="flex-1 overflow-y-auto space-y-2">
              <div className="card-glass rounded-xl p-3">
                <h3 className="text-sm font-medium text-gray-300 mb-2">Validation Details</h3>
                {validation ? (
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <CheckCircle className={`w-4 h-4 ${validation.valid ? 'text-green-400' : 'text-red-400'}`} />
                      <span className={`text-sm ${validation.valid ? 'text-green-400' : 'text-red-400'}`}>
                        {validation.valid ? 'Design is valid' : 'Design has issues'}
                      </span>
                    </div>
                    {validation.errors.length > 0 && (
                      <div className="space-y-1">
                        {validation.errors.map((err, i) => (
                          <div key={i} className="text-xs text-red-300 bg-red-500/10 p-2 rounded">
                            • {err}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-xs text-gray-400">No validation data available</p>
                )}
              </div>
            </div>
          </div>
        )

      case 'build':
        return (
          <div className="h-full flex flex-col space-y-2 pb-6">
            {/* Build Controls - Fixed height */}
            <motion.div
              className="card-glass p-3 rounded-xl flex-shrink-0"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <h3 className="text-sm font-medium text-gray-300 mb-2">Build Pipeline</h3>
              <div className="flex flex-col sm:flex-row gap-2">
                <button
                  onClick={compileDesign}
                  disabled={!currentDesign || !validation?.valid || isCompiling}
                  className="flex-1 flex items-center justify-center space-x-2 bg-green-500/20 text-green-400 rounded-lg py-2 text-sm font-medium hover:bg-green-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ minHeight: '40px' }}
                >
                  <Zap className="w-4 h-4" />
                  <span>{isCompiling ? 'Compiling...' : 'Compile'}</span>
                </button>

                <button
                  onClick={trainDesign}
                  disabled={!compileResult || isTraining}
                  className="flex-1 flex items-center justify-center space-x-2 bg-purple-500/20 text-purple-400 rounded-lg py-2 text-sm font-medium hover:bg-purple-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ minHeight: '40px' }}
                >
                  <Play className="w-4 h-4" />
                  <span>{isTraining ? 'Training...' : 'Train'}</span>
                </button>
              </div>
            </motion.div>

            {/* Build Status - Fixed height */}
            <motion.div
              className="card-glass p-3 rounded-xl flex-shrink-0"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <h3 className="text-sm font-medium text-gray-300 mb-2">Build Status</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Compilation</span>
                  <div className="flex items-center space-x-1">
                    {compileResult ? (
                      <>
                        <CheckCircle className="w-3 h-3 text-green-400" />
                        <span className="text-xs text-green-400">Success</span>
                      </>
                    ) : (
                      <>
                        <div className="w-3 h-3 rounded-full bg-gray-600" />
                        <span className="text-xs text-gray-400">Pending</span>
                      </>
                    )}
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Training</span>
                  <div className="flex items-center space-x-1">
                    {isTraining ? (
                      <>
                        <div className="w-3 h-3 rounded-full bg-yellow-400 animate-pulse" />
                        <span className="text-xs text-yellow-400">Running</span>
                      </>
                    ) : (
                      <>
                        <div className="w-3 h-3 rounded-full bg-gray-600" />
                        <span className="text-xs text-gray-400">Idle</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Build Logs - Scrollable content */}
            <div className="flex-1 overflow-y-auto space-y-2">
              <div className="card-glass rounded-xl p-3">
                <h3 className="text-sm font-medium text-gray-300 mb-2">Build Logs</h3>
                <div className="space-y-1">
                  <div className="text-xs text-gray-400 bg-gray-800/30 p-2 rounded">
                    {compileResult ? '✓ Compilation completed successfully' : 'Waiting for compilation...'}
                  </div>
                  <div className="text-xs text-gray-400 bg-gray-800/30 p-2 rounded">
                    {isTraining ? '🔄 Training in progress...' : 'Training ready to start'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )

      case 'metrics':
        return (
          <div className="h-full flex flex-col space-y-2 pb-6">
            {/* Performance Overview - Fixed height */}
            <motion.div
              className="card-glass p-3 rounded-xl flex-shrink-0"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <h3 className="text-sm font-medium text-gray-300 mb-2">Performance Metrics</h3>
              <div className={`grid ${isMobile ? 'grid-cols-2 gap-2' : 'grid-cols-4 gap-2'}`}>
                {metrics.map((metric, index) => (
                  <motion.div
                    key={metric.title}
                    className="flex items-center space-x-2 p-2 bg-gray-800/30 rounded-lg"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: 0.1 + index * 0.05 }}
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

            {/* Memory Analysis - Fixed height */}
            <motion.div
              className="card-glass p-3 rounded-xl flex-shrink-0"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <h3 className="text-sm font-medium text-gray-300 mb-2">Memory Analysis</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Estimated Memory</span>
                  <span className="text-sm font-medium text-orange-400">{estimatedMemory}MB</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-orange-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min((estimatedMemory / 2048) * 100, 100)}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-gray-400">
                  <span>0MB</span>
                  <span>2GB</span>
                </div>
              </div>
            </motion.div>

            {/* Optimization Tips - Scrollable content */}
            <div className="flex-1 overflow-y-auto space-y-2">
              <div className="card-glass rounded-xl p-3">
                <h3 className="text-sm font-medium text-gray-300 mb-2">Optimization Tips</h3>
                <div className="space-y-2">
                  <div className="text-xs text-gray-400 bg-blue-500/10 p-2 rounded border border-blue-500/20">
                    💡 Reduce model size for faster training
                  </div>
                  <div className="text-xs text-gray-400 bg-green-500/10 p-2 rounded border border-green-500/20">
                    ⚡ Use fewer components for better performance
                  </div>
                  <div className="text-xs text-gray-400 bg-purple-500/10 p-2 rounded border border-purple-500/20">
                    🔧 Optimize connections for efficiency
                  </div>
                </div>
              </div>
            </div>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="safe-area h-full flex flex-col p-2 md:p-4">
      {/* Tab Navigation */}
      <motion.div
        className="card-glass p-1 rounded-xl flex-shrink-0 mb-2"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className={`flex ${isMobile ? 'overflow-x-auto' : 'justify-center'} space-x-1`}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 whitespace-nowrap ${
                activeTab === tab.id
                  ? `${tab.bgColor} ${tab.color}`
                  : 'text-gray-400 hover:text-gray-300 hover:bg-gray-700/50'
              }`}
              style={{ minHeight: '40px' }}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </motion.div>

      {/* Tab Content */}
      <div className="flex-1 min-h-0">
        {renderTabContent()}
      </div>
    </div>
  )
}

export default ModelStudioTab