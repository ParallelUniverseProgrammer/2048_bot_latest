import React, { useMemo } from 'react'
import { motion } from 'framer-motion'
import { Line, Doughnut, Bar } from 'react-chartjs-2'
import { TrendingUp, TrendingDown, Brain, Zap, Target, Activity, Settings, Loader2 } from 'lucide-react'
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

  // Intelligent device detection
  const { displayMode } = useDeviceDetection()
  const isMobile = displayMode === 'mobile'

  // Chart data preparation
  const lossChartData = useMemo(() => {
    if (!lossHistory.episodes.length || !lossHistory.values.length) return null

    // For mobile, show only last 10 data points for cleaner display
    const dataPoints = isMobile ? Math.min(10, lossHistory.episodes.length) : lossHistory.episodes.length
    const startIndex = Math.max(0, lossHistory.episodes.length - dataPoints)

    return {
      labels: lossHistory.episodes.slice(startIndex),
      datasets: [
        {
          label: 'Loss',
          data: lossHistory.values.slice(startIndex),
          borderColor: 'rgb(239, 68, 68)',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          borderWidth: isMobile ? 1 : 2,
          fill: !isMobile,
          tension: 0.4,
          pointRadius: isMobile ? 0 : 2,
          pointHoverRadius: isMobile ? 3 : 4,
        },
      ],
    }
  }, [lossHistory, isMobile])

  const scoreChartData = useMemo(() => {
    if (!scoreHistory.episodes.length || !scoreHistory.values.length) return null

    // For mobile, show only last 10 data points for cleaner display
    const dataPoints = isMobile ? Math.min(10, scoreHistory.episodes.length) : scoreHistory.episodes.length
    const startIndex = Math.max(0, scoreHistory.episodes.length - dataPoints)

    return {
      labels: scoreHistory.episodes.slice(startIndex),
      datasets: [
        {
          label: 'Score',
          data: scoreHistory.values.slice(startIndex),
          borderColor: 'rgb(34, 197, 94)',
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          borderWidth: isMobile ? 1 : 2,
          fill: !isMobile,
          tension: 0.4,
          pointRadius: isMobile ? 0 : 2,
          pointHoverRadius: isMobile ? 3 : 4,
        },
      ],
    }
  }, [scoreHistory, isMobile])

  const actionDistributionData = useMemo(() => {
    if (!trainingData?.actions) return null

    return {
      labels: isMobile ? ['↑', '↓', '←', '→'] : ['Up', 'Down', 'Left', 'Right'],
      datasets: [
        {
          data: trainingData.actions.map(action => action * 100),
          backgroundColor: [
            'rgba(59, 130, 246, 0.8)',
            'rgba(16, 185, 129, 0.8)',
            'rgba(245, 158, 11, 0.8)',
            'rgba(239, 68, 68, 0.8)',
          ],
          borderColor: [
            'rgb(59, 130, 246)',
            'rgb(16, 185, 129)',
            'rgb(245, 158, 11)',
            'rgb(239, 68, 68)',
          ],
          borderWidth: isMobile ? 1 : 2,
        },
      ],
    }
  }, [trainingData?.actions, isMobile])

  const expertUsageData = useMemo(() => {
    if (!trainingData?.expert_usage) return null

    return {
      labels: trainingData.expert_usage.map((_, index) => isMobile ? `E${index + 1}` : `Expert ${index + 1}`),
      datasets: [
        {
          label: 'Usage %',
          data: trainingData.expert_usage.map(usage => usage * 100),
          backgroundColor: 'rgba(147, 51, 234, 0.8)',
          borderColor: 'rgb(147, 51, 234)',
          borderWidth: isMobile ? 1 : 2,
        },
      ],
    }
  }, [trainingData?.expert_usage, isMobile])

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: !isMobile,
        labels: {
          color: '#cbd5e1',
          font: {
            size: isMobile ? 10 : 12
          }
        },
      },
      tooltip: {
        enabled: !isMobile,
      },
    },
    scales: {
      x: {
        display: !isMobile,
        ticks: {
          color: '#94a3b8',
          font: {
            size: isMobile ? 8 : 10
          },
          maxTicksLimit: isMobile ? 5 : 10,
        },
        grid: {
          display: !isMobile,
          color: 'rgba(148, 163, 184, 0.1)',
        },
      },
      y: {
        display: !isMobile,
        ticks: {
          color: '#94a3b8',
          font: {
            size: isMobile ? 8 : 10
          },
          maxTicksLimit: isMobile ? 3 : 6,
        },
        grid: {
          display: !isMobile,
          color: 'rgba(148, 163, 184, 0.1)',
        },
      },
    },
    elements: {
      point: {
        radius: isMobile ? 0 : 2,
        hoverRadius: isMobile ? 2 : 4,
      },
    },
  }

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: isMobile ? 'right' as const : 'bottom' as const,
        labels: {
          color: '#cbd5e1',
          padding: isMobile ? 10 : 20,
          font: {
            size: isMobile ? 10 : 12
          },
          usePointStyle: isMobile,
        },
      },
      tooltip: {
        enabled: !isMobile,
      },
    },
    cutout: isMobile ? '60%' : '50%',
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
        // Use persisted values if available
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
      title: 'Entropy',
      value: trainingData?.entropy?.toFixed(3) || '0.000',
      icon: Brain,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/20',
    },
  ]

  return (
    <div className="space-y-6">
      {/* Loading Indicator for Training Start */}
      {loadingStates.isTrainingStarting && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="card-glass p-4 rounded-xl border border-blue-500/30 bg-blue-500/10"
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

      {/* Model Configuration */}
      <motion.div
        className="card-glass p-6 rounded-xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <Settings className="w-5 h-5 mr-2 text-blue-400" />
          Model Configuration
        </h3>
        
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-3">
            <label className="text-sm font-medium text-gray-300">Model Size:</label>
            <select
              value={modelSize}
              onChange={(e) => setModelSize(e.target.value as 'small' | 'medium' | 'large')}
              disabled={isTraining || loadingStates.isTrainingStarting}
              className={`
                bg-gray-700 text-white rounded-lg px-3 py-2 text-sm
                focus:outline-none focus:ring-2 focus:ring-blue-500
                ${isTraining || loadingStates.isTrainingStarting ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-600'}
              `}
            >
              <option value="small">Small (2GB VRAM)</option>
              <option value="medium">Medium (4GB VRAM)</option>
              <option value="large">Large (6GB VRAM)</option>
            </select>
          </div>
          
          {(isTraining || loadingStates.isTrainingStarting) && (
            <div className="flex items-center space-x-2 text-sm text-yellow-400">
              <Activity className="w-4 h-4" />
              <span>
                {loadingStates.isTrainingStarting 
                  ? 'Starting training session...' 
                  : 'Stop training to change model size'
                }
              </span>
            </div>
          )}
        </div>
        
        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="text-center p-3 bg-gray-800/50 rounded-lg">
            <div className="text-blue-400 font-medium">Small</div>
            <div className="text-gray-400">4 layers, 4 experts</div>
            <div className="text-gray-400">256 dims, ~10M params</div>
          </div>
          <div className="text-center p-3 bg-gray-800/50 rounded-lg">
            <div className="text-green-400 font-medium">Medium</div>
            <div className="text-gray-400">6 layers, 6 experts</div>
            <div className="text-gray-400">384 dims, ~45M params</div>
          </div>
          <div className="text-center p-3 bg-gray-800/50 rounded-lg">
            <div className="text-purple-400 font-medium">Large</div>
            <div className="text-gray-400">8 layers, 8 experts</div>
            <div className="text-gray-400">512 dims, ~100M params</div>
          </div>
        </div>
      </motion.div>

      {/* Metrics Cards */}
      <div className={`${isMobile ? 'mobile-grid-2' : 'grid grid-cols-2 lg:grid-cols-4 gap-4'}`}>
        {metrics.map((metric, index) => {
          const IconComponent = metric.icon
          return (
            <motion.div
              key={metric.title}
              className={`card-glass rounded-xl ${isMobile ? 'metrics-card' : 'p-6'}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ scale: 1.02 }}
            >
              <div className="flex items-center justify-between mb-3">
                <div className={`p-2 rounded-lg ${metric.bgColor}`}>
                  <IconComponent className={`w-5 h-5 ${metric.color}`} />
                </div>
                {(isTraining || loadingStates.isTrainingStarting) && (
                  <div className={`w-2 h-2 rounded-full ${
                    loadingStates.isTrainingStarting 
                      ? 'bg-blue-400 animate-pulse' 
                      : 'bg-green-400 animate-pulse'
                  }`} />
                )}
              </div>
              <h3 className={`font-medium text-gray-400 mb-1 ${isMobile ? 'metrics-card h3' : 'text-sm'}`}>{metric.title}</h3>
              <p className={`font-bold ${metric.color} ${isMobile ? 'metrics-card p' : 'text-2xl'}`}>{metric.value}</p>
            </motion.div>
          )
        })}
      </div>

      {/* Charts Grid */}
      <div className={`${isMobile ? 'mobile-grid-1' : 'grid grid-cols-1 lg:grid-cols-2 gap-6'}`}>
        {/* Loss Chart */}
        <motion.div
          className="card-glass p-6 rounded-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <h3 className={`font-semibold mb-4 flex items-center ${isMobile ? 'text-sm' : 'text-lg'}`}>
            <TrendingDown className="w-5 h-5 mr-2 text-red-400" />
            {isMobile ? 'Loss' : 'Training Loss'}
          </h3>
          <div className="chart-container">
            {lossChartData ? (
              <Line data={lossChartData} options={chartOptions} />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                No data available
              </div>
            )}
          </div>
        </motion.div>

        {/* Score Chart */}
        <motion.div
          className="card-glass p-6 rounded-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <h3 className={`font-semibold mb-4 flex items-center ${isMobile ? 'text-sm' : 'text-lg'}`}>
            <TrendingUp className="w-5 h-5 mr-2 text-green-400" />
            {isMobile ? 'Score' : 'Game Score'}
          </h3>
          <div className="chart-container">
            {scoreChartData ? (
              <Line data={scoreChartData} options={chartOptions} />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                No data available
              </div>
            )}
          </div>
        </motion.div>

        {/* Action Distribution */}
        <motion.div
          className="card-glass p-6 rounded-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
        >
          <h3 className={`font-semibold mb-4 flex items-center ${isMobile ? 'text-sm' : 'text-lg'}`}>
            <Activity className="w-5 h-5 mr-2 text-blue-400" />
            {isMobile ? 'Actions' : 'Action Distribution'}
          </h3>
          <div className="chart-container">
            {actionDistributionData ? (
              <Doughnut data={actionDistributionData} options={doughnutOptions} />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                No data available
              </div>
            )}
          </div>
        </motion.div>

        {/* Expert Usage */}
        <motion.div
          className="card-glass p-6 rounded-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.7 }}
        >
          <h3 className={`font-semibold mb-4 flex items-center ${isMobile ? 'text-sm' : 'text-lg'}`}>
            <Brain className="w-5 h-5 mr-2 text-purple-400" />
            {isMobile ? 'Experts' : 'Expert Usage'}
          </h3>
          <div className="chart-container">
            {expertUsageData ? (
              <Bar data={expertUsageData} options={chartOptions} />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                No data available
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default TrainingDashboard 