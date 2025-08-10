import React, { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Brain, Layers, GitBranch, BarChart3, Activity, Zap } from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'
import { useDeviceDetection } from '../utils/deviceDetection'
import { AnimatePresence } from 'framer-motion'

const NetworkVisualizer: React.FC = () => {
  const { trainingData } = useTrainingStore()
  const { displayMode } = useDeviceDetection()
  const isMobile = displayMode === 'mobile'
  
  const [activeTab, setActiveTab] = useState<'architecture' | 'usage' | 'performance'>('architecture')

  // Generate expert nodes with enhanced data
  const expertNodes = useMemo(() => {
    if (!trainingData?.expert_usage) return []
    
    const numExperts = trainingData.expert_usage.length
    const centerX = 150
    const centerY = 80
    const radius = Math.min(60, 40 + numExperts * 6)
    
    return trainingData.expert_usage.map((usage, index) => {
      const angle = (index * 2 * Math.PI) / numExperts - Math.PI / 2
      const x = centerX + radius * Math.cos(angle)
      const y = centerY + radius * Math.sin(angle)
      
      // Generate mock performance metrics
      const baseEfficiency = 0.4 + (usage * 0.6) + (Math.sin(index * 1.7) * 0.1)
      const efficiency = Math.max(0.1, Math.min(1.0, baseEfficiency))
      const activationFreq = usage * (0.8 + Math.cos(index * 2.1) * 0.2)
      const contribution = usage * efficiency
      
      const specializations = [
        { name: 'Early', icon: '🚀', color: 'text-ui-brand-primary' },
        { name: 'Mid', icon: '⚡', color: 'text-yellow-400' },
        { name: 'Late', icon: '🎯', color: 'text-green-400' },
        { name: 'Recovery', icon: '🛡️', color: 'text-red-400' },
        { name: 'Optimal', icon: '💎', color: 'text-purple-400' },
        { name: 'Aggressive', icon: '🔥', color: 'text-orange-400' },
        { name: 'Conservative', icon: '🧭', color: 'text-cyan-400' },
        { name: 'Strategic', icon: '🎭', color: 'text-pink-400' },
      ]
      
      return {
        id: index,
        name: `E${index + 1}`,
        usage: usage,
        efficiency: efficiency,
        activationFreq: activationFreq,
        contribution: contribution,
        specialization: specializations[index % specializations.length],
        x,
        y,
        color: `hsl(${(index * 360) / numExperts}, 70%, 60%)`,
        size: 16 + usage * 30,
        health: efficiency * activationFreq,
      }
    })
  }, [trainingData?.expert_usage])

  // Calculate load balancing score
  const loadBalancingScore = useMemo(() => {
    if (!expertNodes.length) return 0
    const usages = expertNodes.map(node => node.usage)
    const mean = usages.reduce((a, b) => a + b, 0) / usages.length
    const variance = usages.reduce((acc, usage) => acc + Math.pow(usage - mean, 2), 0) / usages.length
    return Math.max(0, 1 - Math.sqrt(variance) * 2)
  }, [expertNodes])

  // Identify starved experts (usage below threshold)
  const STARVATION_THRESHOLD = 0.05
  const starvedExperts = useMemo(() => expertNodes.filter((e) => e.usage < STARVATION_THRESHOLD), [expertNodes])

  // Attention visualizer removed – heatmap generation no longer required

  // Architecture layers
  const layers = [
    { name: 'Input', description: 'Board State', color: 'bg-blue-500' },
    { name: 'Embed', description: 'Encoding', color: 'bg-green-500' },
    { name: 'Attention', description: 'Multi-Head', color: 'bg-purple-500' },
    { name: 'MoE', description: 'Experts', color: 'bg-orange-500' },
    { name: 'Output', description: 'Actions', color: 'bg-red-500' },
  ]

  const tabs = [
    { id: 'architecture', label: 'Architecture', icon: Brain },
    { id: 'usage', label: 'Usage', icon: GitBranch },
    { id: 'performance', label: 'Performance', icon: Activity },
  ]

  return (
    <div className="h-full flex flex-col space-y-3 pb-6">
      {/* Sub Navigation Tabs */}
      <motion.div
        className="card-glass p-2 rounded-2xl flex-shrink-0"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex justify-center space-x-1">
          {tabs.map((tab) => {
            const IconComponent = tab.icon
            return (
              <motion.button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`relative flex flex-col items-center space-y-1 rounded-xl font-medium transition-all duration-200 ${
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
                    className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl -z-10"
                    layoutId="activeSubTab"
                    initial={false}
                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                  />
                )}
              </motion.button>
            )
          })}
        </div>
      </motion.div>

      {/* Content Area */}
      <div className="flex-1 min-h-0">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="h-full"
          >
            {activeTab === 'architecture' && (
              <div className="grid grid-cols-1 gap-4 h-full">
                {/* Network Architecture */}
                <motion.div
                  className="card-glass p-4 rounded-xl flex flex-col"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5 }}
                >
                  <h3 className="text-base font-semibold mb-3 flex items-center">
                    <Brain className="w-4 h-4 mr-2 text-blue-400" />
                    Architecture
                  </h3>
                  
                  <div className="flex items-center justify-between space-x-2 overflow-x-auto flex-1">
                    {layers.map((layer, index) => (
                      <motion.div
                        key={layer.name}
                        className="flex-shrink-0 text-center"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.3, delay: index * 0.1 }}
                      >
                        <motion.div
                          className={`w-12 h-12 ${layer.color} rounded-xl flex items-center justify-center mb-2 mx-auto`}
                          whileHover={{ scale: 1.1 }}
                        >
                          <Layers className="w-6 h-6 text-white" />
                        </motion.div>
                        <p className="text-xs font-medium">{layer.name}</p>
                        <p className="text-xs text-gray-400">{layer.description}</p>
                        
                        {index < layers.length - 1 && (
                          <div className="flex justify-center mt-2">
                            <div className="w-6 h-0.5 bg-gray-600" />
                          </div>
                        )}
                      </motion.div>
                    ))}
                  </div>

                  {/* Numeric summary of architecture */}
                  <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                    <div className="flex items-center justify-between bg-gray-800/40 rounded-md p-2">
                      <span>Layers</span>
                      <span className="font-semibold text-blue-400">{layers.length}</span>
                    </div>
                    <div className="flex items-center justify-between bg-gray-800/40 rounded-md p-2">
                      <span>Experts</span>
                      <span className="font-semibold text-purple-400">{expertNodes.length}</span>
                    </div>
                    <div className="flex items-center justify-between bg-gray-800/40 rounded-md p-2">
                      <span>Params</span>
                      <span className="font-semibold text-green-400">{typeof trainingData?.model_params === 'number' ? `${trainingData.model_params.toFixed(1)}M` : '—'}</span>
                    </div>
                    <div className="flex items-center justify-between bg-gray-800/40 rounded-md p-2">
                      <span>Memory</span>
                      <span className="font-semibold text-orange-400">{trainingData?.gpu_memory?.toFixed(1) || '0.0'}GB</span>
                    </div>
                  </div>
                </motion.div>
              </div>
            )}

            {activeTab === 'usage' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
                {/* Quick Stats */}
                <motion.div
                  className="card-glass p-4 rounded-xl flex flex-col"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                >
                  <h3 className="text-base font-semibold mb-3 flex items-center">
                    <BarChart3 className="w-4 h-4 mr-2 text-green-400" />
                    Quick Stats
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-gray-800/40 rounded-xl p-3">
                      <div className="text-xs text-gray-400 mb-1">Experts</div>
                      <div className="text-lg font-bold text-blue-400">{expertNodes.length || '0'}</div>
                    </div>
                    <div className="bg-gray-800/40 rounded-xl p-3">
                      <div className="text-xs text-gray-400 mb-1">Load Balance</div>
                      <div className="text-lg font-bold text-green-400">{(loadBalancingScore * 100).toFixed(0)}%</div>
                    </div>
                    <div className="bg-gray-800/40 rounded-xl p-3">
                      <div className="text-xs text-gray-400 mb-1">GPU Memory</div>
                      <div className="text-lg font-bold text-purple-400">{trainingData?.gpu_memory?.toFixed(1) || '0.0'}GB</div>
                    </div>
                    <div className="bg-gray-800/40 rounded-xl p-3">
                      <div className="text-xs text-gray-400 mb-1">Learning Rate</div>
                      <div className="text-lg font-bold text-orange-400">{trainingData?.learning_rate?.toFixed(6) || '0.000000'}</div>
                    </div>
                  </div>
                </motion.div>

                {/* Expert Usage */}
                <motion.div
                  className="card-glass p-4 rounded-xl flex flex-col space-y-4"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                >
                  <div className="flex items-center justify-between">
                    <h3 className="text-base font-semibold mb-3 flex items-center">
                      <GitBranch className="w-4 h-4 mr-2 text-purple-400" />
                      Expert Usage
                    </h3>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-gray-400">Load Bal:</span>
                      <div className={`px-2 py-1 rounded text-xs font-medium ${
                        loadBalancingScore > 0.7 ? 'bg-green-500/20 text-green-400' :
                        loadBalancingScore > 0.4 ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {(loadBalancingScore * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  {/* Layer overview */}
                  <div className="flex flex-col items-center space-y-6">
                    {/* Input → Attention */}
                    <div className="flex flex-col items-center">
                      <Layers className="w-8 h-8 text-blue-400 mb-1" />
                      <span className="text-xs text-gray-300">Input → Attention</span>
                    </div>

                    {/* MoE Experts */}
                    <div className="w-full">
                      <h4 className="text-xs text-gray-400 mb-2">Mixture of Experts Usage</h4>
                      <div className="space-y-2">
                        {expertNodes.map((expert, idx) => (
                          <motion.div
                            key={expert.id}
                            className="flex items-center"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.3, delay: idx * 0.03 }}
                          >
                            <div
                              className="w-4 h-4 rounded-full mr-2 flex-shrink-0"
                              style={{ backgroundColor: expert.color }}
                            />
                            <span className="text-xs w-10 flex-shrink-0">{expert.name}</span>
                             <div className="flex-1 h-2 bg-ui-surface-elevated rounded overflow-hidden relative">
                              <motion.div
                                className="h-full rounded"
                                style={{
                                  width: `${(expert.usage * 100).toFixed(1)}%`,
                                   backgroundColor: expert.usage < STARVATION_THRESHOLD ? 'var(--ui-danger)' : 'var(--ui-success)',
                                }}
                                initial={{ width: 0 }}
                                animate={{ width: `${(expert.usage * 100).toFixed(1)}%` }}
                                transition={{ duration: 0.6, delay: 0.1 }}
                              />
                            </div>
                            <span className={`text-xs ml-2 w-12 text-right ${expert.usage < STARVATION_THRESHOLD ? 'text-red-400' : 'text-gray-200'}`}>
                              {(expert.usage * 100).toFixed(1)}%
                            </span>
                          </motion.div>
                        ))}
                      </div>

                      {starvedExperts.length > 0 && (
                        <div className="mt-3 text-xs text-red-400">
                          {starvedExperts.length} expert{starvedExperts.length > 1 ? 's are' : ' is'} under-utilized (&lt;5% usage)
                        </div>
                      )}
                    </div>

                    {/* Output Layer */}
                    <div className="flex flex-col items-center">
                      <Zap className="w-8 h-8 text-yellow-400 mb-1" />
                      <span className="text-xs text-gray-300">Output Layer</span>
                    </div>
                  </div>
                </motion.div>
              </div>
            )}

            {activeTab === 'performance' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
                {/* Performance Metrics */}
                <motion.div
                  className="card-glass p-4 rounded-xl flex flex-col"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5 }}
                >
                  <h3 className="text-base font-semibold mb-3 flex items-center">
                    <Activity className="w-4 h-4 mr-2 text-blue-400" />
                    Performance
                  </h3>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-gray-800/40 rounded-xl p-3">
                      <div className="text-xs text-gray-400 mb-1">Top Performer</div>
                      <div className="text-sm font-semibold text-green-400">
                        {expertNodes.length > 0 
                          ? expertNodes.reduce((best, expert) => 
                              expert.efficiency > best.efficiency ? expert : best
                            ).name
                          : 'N/A'}
                      </div>
                    </div>
                    <div className="bg-gray-800/40 rounded-xl p-3">
                      <div className="text-xs text-gray-400 mb-1">Most Active</div>
                      <div className="text-sm font-semibold text-blue-400">
                        {expertNodes.length > 0
                          ? expertNodes.reduce((most, expert) => 
                              expert.usage > most.usage ? expert : most
                            ).name
                          : 'N/A'}
                      </div>
                    </div>
                    <div className="bg-gray-800/40 rounded-xl p-3">
                      <div className="text-xs text-gray-400 mb-1">Avg Efficiency</div>
                      <div className="text-sm font-semibold text-purple-400">
                        {expertNodes.length > 0
                          ? ((expertNodes.reduce((sum, expert) => sum + expert.efficiency, 0) / expertNodes.length) * 100).toFixed(1)
                          : '0.0'}%
                      </div>
                    </div>
                    <div className="bg-gray-800/40 rounded-xl p-3">
                      <div className="text-xs text-gray-400 mb-1">Load Balance</div>
                      <div className="text-sm font-semibold text-orange-400">{(loadBalancingScore * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                </motion.div>

                {/* Model Details */}
                <motion.div
                  className="card-glass p-4 rounded-xl flex flex-col"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                >
                  <h3 className="text-base font-semibold mb-3 flex items-center">
                    <Zap className="w-4 h-4 mr-2 text-yellow-400" />
                    Model Details
                  </h3>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-gray-800/40 rounded-xl">
                      <span className="text-sm text-gray-400">Parameters:</span>
                      <span className="text-sm font-semibold text-blue-400">{typeof trainingData?.model_params === 'number' ? `${trainingData.model_params.toFixed(1)}M` : '—'}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-gray-800/40 rounded-xl">
                      <span className="text-sm text-gray-400">GPU Memory:</span>
                      <span className="text-sm font-semibold text-green-400">{trainingData?.gpu_memory?.toFixed(1) || '0.0'}GB</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-gray-800/40 rounded-xl">
                      <span className="text-sm text-gray-400">Learning Rate:</span>
                      <span className="text-sm font-semibold text-purple-400">{trainingData?.learning_rate?.toFixed(6) || '0.000000'}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-gray-800/40 rounded-xl">
                      <span className="text-sm text-gray-400">Total Experts:</span>
                      <span className="text-sm font-semibold text-orange-400">{expertNodes.length}</span>
                    </div>
                  </div>
                </motion.div>
              </div>
            )}

            {/* Subtabs reorganized into Architecture, Usage, and Performance */}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}

export default NetworkVisualizer 