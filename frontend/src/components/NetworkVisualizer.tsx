import React, { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Brain, Layers, GitBranch, Eye, BarChart3, Network, Activity, Zap } from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'
import { useDeviceDetection } from '../utils/deviceDetection'
import { AnimatePresence } from 'framer-motion'

const NetworkVisualizer: React.FC = () => {
  const { trainingData } = useTrainingStore()
  const { displayMode } = useDeviceDetection()
  const isMobile = displayMode === 'mobile'
  
  const [activeTab, setActiveTab] = useState<'overview' | 'experts' | 'attention' | 'stats'>('overview')
  const [hoveredExpert, setHoveredExpert] = useState<number | null>(null)

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
        { name: 'Early', icon: '🚀', color: 'text-blue-400' },
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

  // Generate connection strengths
  const connectionStrengths = useMemo(() => {
    if (!expertNodes.length) return []
    
    const connections = []
    for (let i = 0; i < expertNodes.length; i++) {
      for (let j = i + 1; j < expertNodes.length; j++) {
        const source = expertNodes[i]
        const target = expertNodes[j]
        const strength = Math.min(1, (source.usage + target.usage) / 2 + 0.1)
        const opacity = strength * 0.6
        
        connections.push({
          source: i,
          target: j,
          strength,
          opacity,
          x1: source.x,
          y1: source.y,
          x2: target.x,
          y2: target.y,
        })
      }
    }
    return connections
  }, [expertNodes])

  // Generate attention heatmap data
  const attentionHeatmap = useMemo(() => {
    if (!trainingData?.attention_weights) return []
    
    const heatmap = []
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        const weight = trainingData.attention_weights[i]?.[j] || 0
        heatmap.push({
          x: j,
          y: i,
          weight: weight,
          intensity: Math.min(weight * 10, 1),
        })
      }
    }
    return heatmap
  }, [trainingData?.attention_weights])

  // Architecture layers
  const layers = [
    { name: 'Input', description: 'Board State', color: 'bg-blue-500' },
    { name: 'Embed', description: 'Encoding', color: 'bg-green-500' },
    { name: 'Attention', description: 'Multi-Head', color: 'bg-purple-500' },
    { name: 'MoE', description: 'Experts', color: 'bg-orange-500' },
    { name: 'Output', description: 'Actions', color: 'bg-red-500' },
  ]

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Network },
    { id: 'experts', label: 'Experts', icon: GitBranch },
    { id: 'attention', label: 'Attention', icon: Eye },
    { id: 'stats', label: 'Stats', icon: BarChart3 },
  ]

  return (
    <div className="h-full flex flex-col space-y-3 pb-6">
      {/* Sub Navigation Tabs */}
      <motion.div
        className="card-glass p-2 rounded-xl flex-shrink-0"
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
            {activeTab === 'overview' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
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
                          className={`w-12 h-12 ${layer.color} rounded-lg flex items-center justify-center mb-2 mx-auto`}
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
                </motion.div>

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
                  
                  <div className="grid grid-cols-2 gap-3 flex-1">
                    <div className="bg-gray-800/40 rounded-lg p-3">
                      <div className="text-xs text-gray-400 mb-1">Experts</div>
                      <div className="text-lg font-bold text-blue-400">{expertNodes.length || '0'}</div>
                    </div>
                    <div className="bg-gray-800/40 rounded-lg p-3">
                      <div className="text-xs text-gray-400 mb-1">Load Balance</div>
                      <div className="text-lg font-bold text-green-400">{(loadBalancingScore * 100).toFixed(0)}%</div>
                    </div>
                    <div className="bg-gray-800/40 rounded-lg p-3">
                      <div className="text-xs text-gray-400 mb-1">GPU Memory</div>
                      <div className="text-lg font-bold text-purple-400">{trainingData?.gpu_memory?.toFixed(1) || '0.0'}GB</div>
                    </div>
                    <div className="bg-gray-800/40 rounded-lg p-3">
                      <div className="text-xs text-gray-400 mb-1">Learning Rate</div>
                      <div className="text-lg font-bold text-orange-400">{trainingData?.learning_rate?.toFixed(6) || '0.000000'}</div>
                    </div>
                  </div>
                </motion.div>
              </div>
            )}

            {activeTab === 'experts' && (
              <div className="h-full flex flex-col">
                {/* Expert Network Visualization */}
                <motion.div
                  className="card-glass p-4 rounded-xl flex-1 flex flex-col"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                >
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-base font-semibold flex items-center">
                      <GitBranch className="w-4 h-4 mr-2 text-purple-400" />
                      Expert Network
                    </h3>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-gray-400">Balance:</span>
                      <div className={`px-2 py-1 rounded text-xs font-medium ${
                        loadBalancingScore > 0.7 ? 'bg-green-500/20 text-green-400' :
                        loadBalancingScore > 0.4 ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {(loadBalancingScore * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  
                  <div className="relative flex-1 bg-gray-800/30 rounded-lg overflow-hidden">
                    {/* Background grid */}
                    <div className="absolute inset-0 opacity-10">
                      <svg width="100%" height="100%">
                        <defs>
                          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" strokeWidth="1"/>
                          </pattern>
                        </defs>
                        <rect width="100%" height="100%" fill="url(#grid)" />
                      </svg>
                    </div>

                    {/* Connection lines */}
                    <svg className="absolute inset-0 w-full h-full">
                      {connectionStrengths.map((connection, index) => (
                        <motion.line
                          key={`connection-${connection.source}-${connection.target}`}
                          x1={`${(connection.x1 / 300) * 100}%`}
                          y1={`${(connection.y1 / 160) * 100}%`}
                          x2={`${(connection.x2 / 300) * 100}%`}
                          y2={`${(connection.y2 / 160) * 100}%`}
                          stroke="rgba(147, 51, 234, 0.4)"
                          strokeWidth={connection.strength * 2}
                          opacity={connection.opacity}
                          initial={{ pathLength: 0 }}
                          animate={{ pathLength: 1 }}
                          transition={{ duration: 1, delay: index * 0.05 }}
                        />
                      ))}
                    </svg>

                    {/* Expert nodes */}
                    {expertNodes.map((expert, index) => (
                      <motion.div
                        key={expert.id}
                        className="absolute cursor-pointer"
                        style={{
                          left: `${(expert.x / 300) * 100}%`,
                          top: `${(expert.y / 160) * 100}%`,
                          transform: 'translate(-50%, -50%)',
                        } as React.CSSProperties}
                        initial={{ scale: 0, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.5, delay: index * 0.1 }}
                        onMouseEnter={() => setHoveredExpert(expert.id)}
                        onMouseLeave={() => setHoveredExpert(null)}
                      >
                        <motion.div
                          className="relative rounded-full flex items-center justify-center border-2 border-white/20"
                          style={{
                            width: `${expert.size}px`,
                            height: `${expert.size}px`,
                            backgroundColor: expert.color,
                            boxShadow: `0 0 ${expert.health * 20}px ${expert.color}`,
                          } as React.CSSProperties}
                          whileHover={{ scale: 1.2 }}
                          animate={hoveredExpert === expert.id ? { scale: 1.1 } : { scale: 1 }}
                        >
                          <div className="text-white text-xs font-bold">
                            {expert.specialization.icon}
                          </div>
                          
                          <div className="absolute inset-0 rounded-full border-2 border-white/40"
                               style={{
                                 background: `conic-gradient(from 0deg, ${expert.color} 0deg, ${expert.color} ${expert.efficiency * 360}deg, transparent ${expert.efficiency * 360}deg, transparent 360deg)`,
                                 padding: '2px',
                               } as React.CSSProperties}>
                          </div>
                        </motion.div>

                        <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-xs text-white whitespace-nowrap">
                          {expert.name}
                        </div>
                        
                        <div className="absolute -bottom-5 left-1/2 transform -translate-x-1/2 text-xs text-gray-400">
                          {(expert.usage * 100).toFixed(0)}%
                        </div>

                        {/* Tooltip */}
                        {hoveredExpert === expert.id && (
                          <motion.div
                            className="absolute z-10 bg-gray-900/95 border border-gray-600 rounded-lg p-2 text-xs"
                            style={{
                              left: expert.x > 150 ? 'auto' : '100%',
                              right: expert.x > 150 ? '100%' : 'auto',
                              top: expert.y > 80 ? 'auto' : '100%',
                              bottom: expert.y > 80 ? '100%' : 'auto',
                              transform: expert.y > 80 ? 'translateY(10px)' : 'translateY(-10px)',
                              minWidth: '160px',
                              marginLeft: expert.x > 150 ? '-10px' : '10px',
                              marginTop: expert.y > 80 ? '-10px' : '10px',
                            } as React.CSSProperties}
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.2 }}
                          >
                            <div className="font-semibold mb-1 flex items-center">
                              <span className="mr-1">{expert.specialization.icon}</span>
                              {expert.name}
                            </div>
                            <div className="space-y-1">
                              <div className="flex justify-between">
                                <span className="text-gray-400">Usage:</span>
                                <span className="text-white">{(expert.usage * 100).toFixed(1)}%</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Efficiency:</span>
                                <span className={`${expert.efficiency > 0.7 ? 'text-green-400' : expert.efficiency > 0.4 ? 'text-yellow-400' : 'text-red-400'}`}>
                                  {(expert.efficiency * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Specialization:</span>
                                <span className={expert.specialization.color}>{expert.specialization.name}</span>
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              </div>
            )}

            {activeTab === 'attention' && (
              <div className="h-full flex flex-col">
                {/* Attention Heatmap */}
                <motion.div
                  className="card-glass p-4 rounded-xl flex-1 flex flex-col"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                >
                  <h3 className="text-base font-semibold mb-3 flex items-center">
                    <Eye className="w-4 h-4 mr-2 text-green-400" />
                    Attention Heatmap
                  </h3>
                  
                  <div className="flex-1 flex flex-col items-center justify-center">
                    <div className="grid grid-cols-4 gap-1 max-w-xs">
                      {attentionHeatmap.map((cell, index) => (
                        <motion.div
                          key={index}
                          className="aspect-square rounded-sm flex items-center justify-center text-xs font-bold"
                          style={{
                            backgroundColor: `rgba(34, 197, 94, ${cell.intensity})`,
                            color: cell.intensity > 0.5 ? 'white' : 'rgba(34, 197, 94, 0.8)',
                          }}
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.3, delay: index * 0.02 }}
                          whileHover={{ scale: 1.1 }}
                        >
                          {cell.weight.toFixed(2)}
                        </motion.div>
                      ))}
                    </div>
                    
                    <div className="mt-4 flex items-center justify-between text-xs text-gray-400 w-full max-w-xs">
                      <span>Low</span>
                      <div className="flex items-center space-x-1">
                        <div className="w-8 h-2 bg-gradient-to-r from-transparent to-green-400 rounded" />
                      </div>
                      <span>High</span>
                    </div>
                  </div>
                </motion.div>
              </div>
            )}

            {activeTab === 'stats' && (
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
                  
                  <div className="grid grid-cols-2 gap-3 flex-1">
                    <div className="bg-gray-800/40 rounded-lg p-3">
                      <div className="text-xs text-gray-400 mb-1">Top Performer</div>
                      <div className="text-sm font-semibold text-green-400">
                        {expertNodes.length > 0 
                          ? expertNodes.reduce((best, expert) => 
                              expert.efficiency > best.efficiency ? expert : best
                            ).name
                          : 'N/A'
                        }
                      </div>
                    </div>
                    <div className="bg-gray-800/40 rounded-lg p-3">
                      <div className="text-xs text-gray-400 mb-1">Most Active</div>
                      <div className="text-sm font-semibold text-blue-400">
                        {expertNodes.length > 0
                          ? expertNodes.reduce((most, expert) => 
                              expert.usage > most.usage ? expert : most
                            ).name
                          : 'N/A'
                        }
                      </div>
                    </div>
                    <div className="bg-gray-800/40 rounded-lg p-3">
                      <div className="text-xs text-gray-400 mb-1">Avg Efficiency</div>
                      <div className="text-sm font-semibold text-purple-400">
                        {expertNodes.length > 0
                          ? ((expertNodes.reduce((sum, expert) => sum + expert.efficiency, 0) / expertNodes.length) * 100).toFixed(1)
                          : '0.0'
                        }%
                      </div>
                    </div>
                    <div className="bg-gray-800/40 rounded-lg p-3">
                      <div className="text-xs text-gray-400 mb-1">Load Balance</div>
                      <div className="text-sm font-semibold text-orange-400">
                        {(loadBalancingScore * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </motion.div>

                {/* Model Details */}
                <motion.div
                  className="card-glass p-4 rounded-xl flex flex-col"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                >
                  <h3 className="text-base font-semibold mb-3 flex items-center">
                    <Zap className="w-4 h-4 mr-2 text-yellow-400" />
                    Model Details
                  </h3>
                  
                  <div className="space-y-3 flex-1">
                    <div className="flex items-center justify-between p-3 bg-gray-800/40 rounded-lg">
                      <span className="text-sm text-gray-400">Parameters:</span>
                      <span className="text-sm font-semibold text-blue-400">{trainingData?.model_params || '0'}M</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-gray-800/40 rounded-lg">
                      <span className="text-sm text-gray-400">GPU Memory:</span>
                      <span className="text-sm font-semibold text-green-400">{trainingData?.gpu_memory?.toFixed(1) || '0.0'}GB</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-gray-800/40 rounded-lg">
                      <span className="text-sm text-gray-400">Learning Rate:</span>
                      <span className="text-sm font-semibold text-purple-400">{trainingData?.learning_rate?.toFixed(6) || '0.000000'}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-gray-800/40 rounded-lg">
                      <span className="text-sm text-gray-400">Total Experts:</span>
                      <span className="text-sm font-semibold text-orange-400">{expertNodes.length}</span>
                    </div>
                  </div>
                </motion.div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}

export default NetworkVisualizer 