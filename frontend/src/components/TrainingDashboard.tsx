import React, { useMemo } from 'react'
import { motion } from 'framer-motion'
import { Line, Doughnut, Bar } from 'react-chartjs-2'
import { 
  TrendingDown, Brain, Zap, Target, Activity, Settings, Loader2,
  Clock, Gauge, BarChart3, TrendingUpIcon, TrendingDownIcon, 
  CheckCircle, AlertTriangle, Info, Star
} from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'
import { useDeviceDetection } from '../utils/deviceDetection'

const TrainingDashboard: React.FC = () => {
  const { 
    trainingData, 
    lossHistory, 
    scoreHistory, 
    isTraining, 
    lastPolicyLoss, 
    lastValueLoss, 
    modelSize, 
    setModelSize, 
    loadingStates 
  } = useTrainingStore()

  const { displayMode } = useDeviceDetection()
  const isMobile = displayMode === 'mobile'

  // Chart data preparation with fallback mock data
  const createMockData = (episodes: number[], baseValue: number, variance: number = 0.2) => {
    return episodes.map((_, index) => {
      const progress = index / episodes.length
      const trend = baseValue * (0.5 + progress * 0.5)
      const noise = (Math.random() - 0.5) * baseValue * variance
      return Math.max(baseValue * 0.1, trend + noise)
    })
  }

  const lossChartData = useMemo(() => {
    const episodes = lossHistory.episodes.length > 0 
      ? lossHistory.episodes 
      : Array.from({length: 20}, (_, i) => i + 1)
    
    const dataPoints = isMobile ? Math.min(15, episodes.length) : episodes.length
    const startIndex = Math.max(0, episodes.length - dataPoints)
    const selectedEpisodes = episodes.slice(startIndex)
    
    const values = lossHistory.values.length > 0 
      ? lossHistory.values.slice(startIndex)
      : createMockData(selectedEpisodes, 2.5, 0.3)

    return {
      labels: selectedEpisodes,
      datasets: [
        {
          label: 'Training Loss',
          data: values,
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          borderWidth: isMobile ? 2 : 3,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: isMobile ? 4 : 6,
          pointHoverBorderWidth: 2,
          pointHoverBorderColor: '#ffffff',
          pointHoverBackgroundColor: '#ef4444',
        },
      ],
    }
  }, [lossHistory, isMobile])

  const scoreChartData = useMemo(() => {
    const episodes = scoreHistory.episodes.length > 0 
      ? scoreHistory.episodes 
      : Array.from({length: 20}, (_, i) => i + 1)
    
    const dataPoints = isMobile ? Math.min(15, episodes.length) : episodes.length
    const startIndex = Math.max(0, episodes.length - dataPoints)
    const selectedEpisodes = episodes.slice(startIndex)
    
    const values = scoreHistory.values.length > 0 
      ? scoreHistory.values.slice(startIndex)
      : createMockData(selectedEpisodes, 1000, 0.4)

    return {
      labels: selectedEpisodes,
      datasets: [
        {
          label: 'Game Score',
          data: values,
          borderColor: '#22c55e',
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          borderWidth: isMobile ? 2 : 3,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: isMobile ? 4 : 6,
          pointHoverBorderWidth: 2,
          pointHoverBorderColor: '#ffffff',
          pointHoverBackgroundColor: '#22c55e',
        },
      ],
    }
  }, [scoreHistory, isMobile])

  const actionDistributionData = useMemo(() => {
    const actions = trainingData?.actions || [0.25, 0.25, 0.25, 0.25]

    return {
      labels: isMobile ? ['↑', '↓', '←', '→'] : ['Up', 'Down', 'Left', 'Right'],
      datasets: [
        {
          data: actions.map(action => action * 100),
          backgroundColor: [
            '#3b82f6',
            '#06b6d4',
            '#8b5cf6',
            '#f59e0b',
          ],
          borderColor: '#ffffff',
          borderWidth: 2,
          hoverBorderWidth: 3,
          hoverBorderColor: '#ffffff',
        },
      ],
    }
  }, [trainingData?.actions, isMobile])

  const expertUsageData = useMemo(() => {
    const expertUsage = trainingData?.expert_usage || [0.2, 0.2, 0.2, 0.2, 0.2]

    return {
      labels: expertUsage.map((_, index) => isMobile ? `E${index + 1}` : `Expert ${index + 1}`),
      datasets: [
        {
          label: 'Expert Usage %',
          data: expertUsage.map(usage => usage * 100),
          backgroundColor: '#a855f7',
          borderColor: '#ffffff',
          borderWidth: 2,
          borderRadius: 4,
          hoverBackgroundColor: '#9333ea',
          hoverBorderWidth: 3,
        },
      ],
    }
  }, [trainingData?.expert_usage, isMobile])









  // Compact chart options for tiny rectangles
  const compactChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'nearest' as const,
      intersect: false,
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: true,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: '#374151',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: false,
        padding: 8,
        titleFont: { size: 10 },
        bodyFont: { size: 9 },
      },
    },
    scales: {
      x: {
        display: false,
        grid: {
          display: false,
        },
      },
      y: {
        display: false,
        grid: {
          display: false,
        },
      },
    },
    elements: {
      point: {
        radius: 0,
        hoverRadius: 3,
      },
      line: {
        borderJoinStyle: 'round' as const,
        borderCapStyle: 'round' as const,
        borderWidth: 1,
      },
    },
  }

  const compactDoughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: true,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: '#374151',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: false,
        padding: 8,
        titleFont: { size: 10 },
        bodyFont: { size: 9 },
        callbacks: {
          label: function(context: any) {
            return `${context.label}: ${context.parsed.toFixed(1)}%`;
          }
        }
      },
    },
    cutout: '70%',
  }

  const compactBarOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: true,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: '#374151',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: false,
        padding: 8,
        titleFont: { size: 10 },
        bodyFont: { size: 9 },
        callbacks: {
          label: function(context: any) {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
          }
        }
      },
    },
    scales: {
      x: {
        display: false,
        grid: {
          display: false,
        },
      },
      y: {
        display: false,
        grid: {
          display: false,
        },
      },
    },
  }

  // Enhanced metrics calculations
  const getScoreTrendIcon = () => {
    const trend = trainingData?.score_trend || 0
    if (trend > 100) return <TrendingUpIcon className="w-4 h-4 text-green-400" />
    if (trend < -100) return <TrendingDownIcon className="w-4 h-4 text-red-400" />
    return <BarChart3 className="w-4 h-4 text-yellow-400" />
  }

  const getLossTrendIcon = () => {
    const trend = trainingData?.loss_trend || 0
    if (trend < -0.1) return <TrendingDownIcon className="w-4 h-4 text-green-400" />
    if (trend > 0.1) return <TrendingUpIcon className="w-4 h-4 text-red-400" />
    return <BarChart3 className="w-4 h-4 text-yellow-400" />
  }

  const getEfficiencyStatus = () => {
    const efficiency = trainingData?.training_efficiency
    if (!efficiency) return { icon: <Info className="w-4 h-4 text-gray-400" />, color: 'text-gray-400' }
    
    const avgEfficiency = (efficiency.score_consistency + efficiency.loss_stability + 
                          efficiency.improvement_rate + efficiency.plateau_detection) / 4
    
    if (avgEfficiency > 0.7) return { icon: <CheckCircle className="w-4 h-4 text-green-400" />, color: 'text-green-400' }
    if (avgEfficiency > 0.4) return { icon: <AlertTriangle className="w-4 h-4 text-yellow-400" />, color: 'text-yellow-400' }
    return { icon: <AlertTriangle className="w-4 h-4 text-red-400" />, color: 'text-red-400' }
  }

  const metrics = [
    {
      title: 'Current Score',
      value: trainingData?.score?.toLocaleString() || '0',
      icon: Target,
      color: 'text-green-400',
      bgColor: 'bg-green-500/20',
    },
    {
      title: 'Training Loss',
      value: (() => {
        if (trainingData?.loss != null) {
          return trainingData.loss.toFixed(4)
        }
        if (lastPolicyLoss != null && lastValueLoss != null) {
          return (lastPolicyLoss + lastValueLoss).toFixed(4)
        }
        return 'N/A'
      })(),
      icon: TrendingDown,
      color: 'text-red-400',
      bgColor: 'bg-red-500/20',
    },
    {
      title: 'Learning Rate',
      value: trainingData?.learning_rate?.toFixed(6) || '0.000000',
      icon: Zap,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/20',
    },
    {
      title: 'Training Speed',
      value: `${trainingData?.training_speed?.toFixed(1) || '0.0'} ep/min`,
      icon: Gauge,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/20',
    },
    {
      title: 'Avg Game Length',
      value: trainingData?.avg_game_length?.toFixed(0) || '0',
      icon: Clock,
      color: 'text-orange-400',
      bgColor: 'bg-orange-500/20',
    },
    {
      title: 'GPU Memory',
      value: `${trainingData?.gpu_memory?.toFixed(1) || '0.0'}GB`,
      icon: Brain,
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-500/20',
    },
  ]

  const enhancedMetrics = [
    {
      title: 'Score Trend',
      value: trainingData?.score_trend ? `${trainingData.score_trend > 0 ? '+' : ''}${trainingData.score_trend.toFixed(0)}` : 'N/A',
      icon: getScoreTrendIcon(),
      color: trainingData?.score_trend ? (trainingData.score_trend > 100 ? 'text-green-400' : trainingData.score_trend < -100 ? 'text-red-400' : 'text-yellow-400') : 'text-gray-400',
      bgColor: 'bg-gray-500/20',
    },
    {
      title: 'Loss Trend',
      value: trainingData?.loss_trend ? `${trainingData.loss_trend > 0 ? '+' : ''}${trainingData.loss_trend.toFixed(3)}` : 'N/A',
      icon: getLossTrendIcon(),
      color: trainingData?.loss_trend ? (trainingData.loss_trend < -0.1 ? 'text-green-400' : trainingData.loss_trend > 0.1 ? 'text-red-400' : 'text-yellow-400') : 'text-gray-400',
      bgColor: 'bg-gray-500/20',
    },
    {
      title: 'Training Efficiency',
      value: trainingData?.training_efficiency ? `${((trainingData.training_efficiency.score_consistency + trainingData.training_efficiency.loss_stability + trainingData.training_efficiency.improvement_rate + trainingData.training_efficiency.plateau_detection) / 4 * 100).toFixed(0)}%` : 'N/A',
      icon: getEfficiencyStatus().icon,
      color: getEfficiencyStatus().color,
      bgColor: 'bg-gray-500/20',
    },
    {
      title: 'Best Max Tile',
      value: (() => {
        const frequency = trainingData?.max_tile_frequency
        if (!frequency || Object.keys(frequency).length === 0) return 'N/A'
        const maxTile = Math.max(...Object.keys(frequency).map(Number))
        return maxTile.toString()
      })(),
      icon: <Star className="w-4 h-4 text-yellow-400" />,
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-500/20',
    },
  ]

  return (
    <div className="h-full flex flex-col space-y-3 pb-6">
      {/* Loading Indicator */}
      {loadingStates.isTrainingStarting && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="card-glass p-3 rounded-xl border border-blue-500/30 bg-blue-500/10 flex-shrink-0"
        >
          <div className="flex items-center space-x-3">
            <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
            <div className="flex-1">
              <div className="text-sm font-medium text-blue-300">Starting Training Session</div>
              <div className="text-xs text-blue-400/80">
                {loadingStates.loadingMessage || 'Initializing model and training environment...'}
              </div>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '0ms' }} />
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '200ms' }} />
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '400ms' }} />
            </div>
          </div>
        </motion.div>
      )}

      {/* Compact Model Configuration */}
      <motion.div
        className="card-glass p-3 rounded-xl flex-shrink-0"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold flex items-center">
            <Settings className="w-4 h-4 mr-2 text-blue-400" />
            Model Configuration
          </h3>
          
          <div className="flex items-center space-x-3">
            <label className="text-xs text-gray-300">Size:</label>
            <select
              value={modelSize}
              onChange={(e) => setModelSize(e.target.value as 'small' | 'medium' | 'large')}
              disabled={isTraining || loadingStates.isTrainingStarting}
              className={`
                bg-gray-700 text-white rounded-lg px-2 py-1 text-xs
                focus:outline-none focus:ring-2 focus:ring-blue-500
                ${isTraining || loadingStates.isTrainingStarting ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-600'}
              `}
            >
              <option value="small">Small</option>
              <option value="medium">Medium</option>
              <option value="large">Large</option>
            </select>
            
            {(isTraining || loadingStates.isTrainingStarting) && (
              <div className="flex items-center space-x-1 text-xs text-yellow-400">
                <Activity className="w-3 h-3" />
                <span>Training...</span>
              </div>
            )}
          </div>
        </div>
        
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="text-center p-2 bg-gray-800/50 rounded-lg">
            <div className="text-blue-400 font-medium">Small</div>
            <div className="text-gray-400">4L, 4E, 256d</div>
          </div>
          <div className="text-center p-2 bg-gray-800/50 rounded-lg">
            <div className="text-green-400 font-medium">Medium</div>
            <div className="text-gray-400">6L, 6E, 384d</div>
          </div>
          <div className="text-center p-2 bg-gray-800/50 rounded-lg">
            <div className="text-purple-400 font-medium">Large</div>
            <div className="text-gray-400">8L, 8E, 512d</div>
          </div>
        </div>
      </motion.div>

      {/* Primary Metrics Dashboard */}
      <motion.div
        className="card-glass p-3 rounded-xl flex-shrink-0"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <div className={`${isMobile ? 'grid grid-cols-2 gap-3' : 'grid grid-cols-3 gap-4'}`}>
          {metrics.map((metric, index) => {
            const IconComponent = metric.icon
            return (
              <motion.div
                key={metric.title}
                className="flex items-center space-x-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.1 + index * 0.05 }}
              >
                <div className={`p-2 rounded-lg ${metric.bgColor}`}>
                  <IconComponent className={`w-4 h-4 ${metric.color}`} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-gray-400 font-medium truncate">{metric.title}</div>
                  <div className={`font-bold ${metric.color} text-sm truncate`}>{metric.value}</div>
                </div>
                {(isTraining || loadingStates.isTrainingStarting) && (
                  <div className={`w-2 h-2 rounded-full ${
                    loadingStates.isTrainingStarting 
                      ? 'bg-blue-400 animate-pulse' 
                      : 'bg-green-400 animate-pulse'
                  }`} />
                )}
              </motion.div>
            )
          })}
        </div>
      </motion.div>

      {/* Enhanced Metrics Dashboard */}
      <motion.div
        className="card-glass p-3 rounded-xl flex-shrink-0"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <div className={`${isMobile ? 'grid grid-cols-2 gap-3' : 'grid grid-cols-4 gap-4'}`}>
          {enhancedMetrics.map((metric, index) => {
            return (
              <motion.div
                key={metric.title}
                className="flex items-center space-x-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.2 + index * 0.05 }}
              >
                <div className={`p-2 rounded-lg ${metric.bgColor}`}>
                  {metric.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-gray-400 font-medium truncate">{metric.title}</div>
                  <div className={`font-bold ${metric.color} text-sm truncate`}>{metric.value}</div>
                </div>
              </motion.div>
            )
          })}
        </div>
      </motion.div>

      {/* Compact Charts Table */}
      <motion.div
        className="card-glass p-3 rounded-xl flex-shrink-0"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <h3 className="text-sm font-semibold mb-3 flex items-center">
          <BarChart3 className="w-4 h-4 mr-2 text-blue-400" />
          Training Analytics
        </h3>
        
        <div className="grid grid-cols-4 gap-3">
          {/* Training Loss Chart */}
          <div className="flex flex-col items-center space-y-2">
            <div className="text-xs text-gray-400 font-medium text-center">Loss</div>
            <div className="w-16 h-12 bg-gray-800/50 rounded-lg p-1">
              <Line data={lossChartData} options={compactChartOptions} />
            </div>
          </div>

          {/* Game Score Chart */}
          <div className="flex flex-col items-center space-y-2">
            <div className="text-xs text-gray-400 font-medium text-center">Score</div>
            <div className="w-16 h-12 bg-gray-800/50 rounded-lg p-1">
              <Line data={scoreChartData} options={compactChartOptions} />
            </div>
          </div>

          {/* Action Distribution Chart */}
          <div className="flex flex-col items-center space-y-2">
            <div className="text-xs text-gray-400 font-medium text-center">Actions</div>
            <div className="w-16 h-12 bg-gray-800/50 rounded-lg p-1 flex items-center justify-center">
              <Doughnut data={actionDistributionData} options={compactDoughnutOptions} />
            </div>
          </div>

          {/* Expert Usage Chart */}
          <div className="flex flex-col items-center space-y-2">
            <div className="text-xs text-gray-400 font-medium text-center">Experts</div>
            <div className="w-16 h-12 bg-gray-800/50 rounded-lg p-1">
              <Bar data={expertUsageData} options={compactBarOptions} />
            </div>
          </div>
        </div>

        {/* Chart Legend */}
        <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-red-400 rounded-sm"></div>
            <span className="text-gray-400">Training Loss</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-400 rounded-sm"></div>
            <span className="text-gray-400">Game Score</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-blue-400 rounded-sm"></div>
            <span className="text-gray-400">Action Distribution</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-purple-400 rounded-sm"></div>
            <span className="text-gray-400">Expert Usage</span>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default TrainingDashboard 