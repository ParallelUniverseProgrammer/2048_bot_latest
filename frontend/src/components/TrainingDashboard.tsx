import React, { useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Line, Doughnut, Bar } from 'react-chartjs-2'
import { 
  TrendingDown, Brain, Zap, Target, Activity,
  Clock, Gauge, BarChart3, TrendingUpIcon, TrendingDownIcon, 
  CheckCircle, AlertTriangle, Info, Star, Scale, GitBranch,
  Play, Pause, AlertTriangle as StopIcon, Save, X, Maximize2
} from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'
import { useDeviceDetection } from '../utils/deviceDetection'
import config from '../utils/config'

const TrainingDashboard: React.FC = () => {
  const readCssVar = (name: string, fallback: string) => {
    if (typeof window === 'undefined') return fallback
    const val = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
    return val || fallback
  }

  const hexToRgba = (hex: string, alpha: number) => {
    const normalized = hex.replace('#', '')
    const bigint = parseInt(normalized.length === 3
      ? normalized.split('').map((c) => c + c).join('')
      : normalized, 16)
    const r = (bigint >> 16) & 255
    const g = (bigint >> 8) & 255
    const b = bigint & 255
    return `rgba(${r}, ${g}, ${b}, ${alpha})`
  }

  const ui = useMemo(() => ({
    textPrimary: readCssVar('--ui-text-primary', '#ffffff'),
    textSecondary: readCssVar('--ui-text-secondary', '#9ca3af'),
    brand: readCssVar('--ui-brand-primary', '#60a5fa'),
    success: readCssVar('--ui-success', '#22c55e'),
    warning: readCssVar('--ui-warning', '#f59e0b'),
    danger: readCssVar('--ui-danger', '#ef4444'),
    info: readCssVar('--ui-info', '#a855f7'),
    borderMuted: readCssVar('--ui-border-muted', '#374151'),
    overlay: readCssVar('--ui-overlay', 'rgba(0,0,0,0.8)'),
    white: '#ffffff'
  }), [])

  const { 
    lossHistory, 
    scoreHistory, 
    isTraining, 
    isPaused,
    isConnected,
    lastPolicyLoss, 
    lastValueLoss, 
    modelSize, 
    setModelSize, 
    loadingStates,
    isWaitingForFirstData,
    getCurrentTrainingData,
    startTraining,
    pauseTraining,
    resumeTraining,
    stopTraining
  } = useTrainingStore()

  const { displayMode } = useDeviceDetection()
  const isMobile = displayMode === 'mobile'
  const [activeTab, setActiveTab] = useState<'core' | 'trends' | 'stats'>('core')
  
  // NEW: Manual checkpoint state
  const [isCreatingCheckpoint, setIsCreatingCheckpoint] = useState(false)

  // NEW: Chart expansion state
  const [expandedChart, setExpandedChart] = useState<{
    type: 'loss' | 'score' | 'actions' | 'experts'
    title: string
  } | null>(null)

  // Use isWaitingForFirstData from store
  const showWaitingForFirstData = isWaitingForFirstData

  // Get current training data with fallback
  const currentTrainingData = getCurrentTrainingData()

  // NEW: Chart expansion handlers
  const handleChartDoubleTap = (chartType: 'loss' | 'score' | 'actions' | 'experts', title: string) => {
    setExpandedChart({ type: chartType, title })
  }

  const handleCloseExpandedChart = () => {
    setExpandedChart(null)
  }

  // NEW: Touch-aware close handler for mobile - only close when tapping the backdrop
  const handleTouchClose = (e: React.TouchEvent) => {
    // Only close if tapping the backdrop (not the panel itself)
    if (e.currentTarget === e.target) {
      e.preventDefault()
      e.stopPropagation()
      handleCloseExpandedChart()
    }
  }

  // Training control functions
  const handleTrainingControl = async (action: 'start' | 'pause' | 'resume' | 'stop') => {
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
      }
    } catch (error) {
      console.error('Training control error:', error)
    }
  }

  // NEW: Manual checkpoint creation
  const handleManualCheckpoint = async () => {
    if (!isTraining) {
      console.warn('Cannot create checkpoint: training not active')
      return
    }
    
    try {
      setIsCreatingCheckpoint(true)
      const res = await fetch(`${config.api.baseUrl}/training/checkpoint/manual`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      
      if (!res.ok) {
        throw new Error('Failed to create manual checkpoint')
      }
      
      const result = await res.json()
      console.log('Manual checkpoint created:', result.checkpoint_id)
      
      // Show success feedback (could be enhanced with a toast notification)
      alert(`Checkpoint created successfully: ${result.checkpoint_id}`)
      
    } catch (error) {
      console.error('Manual checkpoint error:', error)
      alert(`Failed to create checkpoint: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setIsCreatingCheckpoint(false)
    }
  }

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
          borderColor: ui.danger,
          backgroundColor: hexToRgba(ui.danger, 0.1),
          borderWidth: isMobile ? 2 : 3,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: isMobile ? 4 : 6,
          pointHoverBorderWidth: 2,
          pointHoverBorderColor: ui.white,
          pointHoverBackgroundColor: ui.danger,
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
          borderColor: ui.success,
          backgroundColor: hexToRgba(ui.success, 0.1),
          borderWidth: isMobile ? 2 : 3,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: isMobile ? 4 : 6,
          pointHoverBorderWidth: 2,
          pointHoverBorderColor: ui.white,
          pointHoverBackgroundColor: ui.success,
        },
      ],
    }
  }, [scoreHistory, isMobile])

  const actionDistributionData = useMemo(() => {
    const actions = currentTrainingData?.actions || [0.25, 0.25, 0.25, 0.25]

    return {
      labels: isMobile ? ['↑', '↓', '←', '→'] : ['Up', 'Down', 'Left', 'Right'],
      datasets: [
        {
          data: actions.map(action => action * 100),
          backgroundColor: [ui.brand, ui.info, ui.success, ui.warning],
          borderColor: ui.white,
          borderWidth: 2,
          hoverBorderWidth: 3,
          hoverBorderColor: ui.white,
        },
      ],
    }
  }, [currentTrainingData?.actions, isMobile])

  const expertUsageData = useMemo(() => {
    const expertUsage = currentTrainingData?.expert_usage || [0.2, 0.2, 0.2, 0.2, 0.2]

    return {
      labels: expertUsage.map((_, index) => isMobile ? `E${index + 1}` : `Expert ${index + 1}`),
      datasets: [
        {
          label: 'Expert Usage %',
          data: expertUsage.map(usage => usage * 100),
          backgroundColor: ui.info,
          borderColor: ui.white,
          borderWidth: 2,
          borderRadius: 4,
          hoverBackgroundColor: ui.info,
          hoverBorderWidth: 3,
        },
      ],
    }
  }, [currentTrainingData?.expert_usage, isMobile])

  // Enhanced chart options for larger charts
  const enhancedChartOptions = {
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
        backgroundColor: ui.overlay,
        titleColor: ui.textPrimary,
        bodyColor: ui.textPrimary,
        borderColor: ui.borderMuted,
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: false,
        padding: 8,
        titleFont: { size: 12 },
        bodyFont: { size: 11 },
      },
    },
    scales: {
      x: {
        display: false,
        grid: {
          display: false,
        },
        ticks: {
          color: ui.textSecondary,
          maxRotation: 0,
        },
      },
      y: {
        display: false,
        grid: {
          display: false,
        },
        ticks: {
          color: ui.textSecondary,
        },
      },
    },
    elements: {
      point: {
        radius: 0,
        hoverRadius: 4,
      },
      line: {
        borderJoinStyle: 'round' as const,
        borderCapStyle: 'round' as const,
        borderWidth: 2,
      },
    },
  }

  // NEW: Enhanced chart options for expanded view
  const expandedChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'nearest' as const,
      intersect: false,
    },
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: {
           color: ui.textPrimary,
          font: { size: 14 },
          padding: 20,
        },
      },
      tooltip: {
        enabled: true,
        backgroundColor: ui.overlay,
        titleColor: ui.textPrimary,
        bodyColor: ui.textPrimary,
        borderColor: ui.borderMuted,
        borderWidth: 1,
        cornerRadius: 12,
        displayColors: true,
        padding: 12,
        titleFont: { size: 14 },
        bodyFont: { size: 13 },
      },
    },
    scales: {
      x: {
        display: true,
        grid: {
          display: true,
          color: 'rgba(255, 255, 255, 0.1)',
        },
        ticks: {
           color: ui.textSecondary,
          font: { size: 12 },
        },
      },
      y: {
        display: true,
        grid: {
          display: true,
          color: 'rgba(255, 255, 255, 0.1)',
        },
        ticks: {
           color: ui.textSecondary,
          font: { size: 12 },
        },
      },
    },
    elements: {
      point: {
        radius: 2,
        hoverRadius: 6,
      },
      line: {
        borderJoinStyle: 'round' as const,
        borderCapStyle: 'round' as const,
        borderWidth: 3,
      },
    },
  }

  const enhancedDoughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: true,
        backgroundColor: ui.overlay,
        titleColor: ui.textPrimary,
        bodyColor: ui.textPrimary,
        borderColor: ui.borderMuted,
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: false,
        padding: 8,
        titleFont: { size: 12 },
        bodyFont: { size: 11 },
        callbacks: {
          label: function(context: any) {
            return `${context.label}: ${context.parsed.toFixed(1)}%`;
          }
        }
      },
    },
    cutout: '60%',
  }

  // NEW: Enhanced doughnut options for expanded view
  const expandedDoughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'right' as const,
        labels: {
          color: '#ffffff',
          font: { size: 14 },
          padding: 20,
        },
      },
      tooltip: {
        enabled: true,
        backgroundColor: ui.overlay,
        titleColor: ui.textPrimary,
        bodyColor: ui.textPrimary,
        borderColor: ui.borderMuted,
        borderWidth: 1,
        cornerRadius: 12,
        displayColors: true,
        padding: 12,
        titleFont: { size: 14 },
        bodyFont: { size: 13 },
        callbacks: {
          label: function(context: any) {
            return `${context.label}: ${context.parsed.toFixed(1)}%`;
          }
        }
      },
    },
    cutout: '50%',
  }

  const enhancedBarOptions = {
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
        titleFont: { size: 12 },
        bodyFont: { size: 11 },
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

  // NEW: Enhanced bar options for expanded view
  const expandedBarOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: {
          color: '#ffffff',
          font: { size: 14 },
          padding: 20,
        },
      },
      tooltip: {
        enabled: true,
        backgroundColor: 'rgba(0, 0, 0, 0.9)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: '#374151',
        borderWidth: 1,
        cornerRadius: 12,
        displayColors: true,
        padding: 12,
        titleFont: { size: 14 },
        bodyFont: { size: 13 },
        callbacks: {
          label: function(context: any) {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
          }
        }
      },
    },
    scales: {
      x: {
        display: true,
        grid: {
          display: true,
          color: 'rgba(255, 255, 255, 0.1)',
        },
        ticks: {
          color: '#9ca3af',
          font: { size: 12 },
        },
      },
      y: {
        display: true,
        grid: {
          display: true,
          color: 'rgba(255, 255, 255, 0.1)',
        },
        ticks: {
          color: '#9ca3af',
          font: { size: 12 },
        },
      },
    },
  }

  // Enhanced metrics calculations
  const getScoreTrendIcon = () => {
    const trend = currentTrainingData?.score_trend || 0
    if (trend > 100) return <TrendingUpIcon className="w-4 h-4 text-green-400" />
    if (trend < -100) return <TrendingDownIcon className="w-4 h-4 text-red-400" />
    return <BarChart3 className="w-4 h-4 text-yellow-400" />
  }

  const getLossTrendIcon = () => {
    const trend = currentTrainingData?.loss_trend || 0
    if (trend < -0.1) return <TrendingDownIcon className="w-4 h-4 text-green-400" />
    if (trend > 0.1) return <TrendingUpIcon className="w-4 h-4 text-red-400" />
    return <BarChart3 className="w-4 h-4 text-yellow-400" />
  }

  const getEfficiencyStatus = () => {
    const efficiency = currentTrainingData?.training_efficiency
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
      value: currentTrainingData?.score?.toLocaleString() || '0',
      icon: Target,
      color: 'text-green-400',
      bgColor: 'bg-green-500/20',
    },
    {
      title: 'Training Loss',
      value: (() => {
        if (currentTrainingData?.loss != null) {
          return currentTrainingData.loss.toFixed(4)
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
      value: currentTrainingData?.learning_rate?.toFixed(6) || '0.000000',
      icon: Zap,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/20',
    },
    {
      title: 'Training Speed',
      value: `${currentTrainingData?.training_speed?.toFixed(1) || '0.0'} ep/min`,
      icon: Gauge,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/20',
    },
    {
      title: 'Avg Game Length',
      value: currentTrainingData?.avg_game_length?.toFixed(0) || '0',
      icon: Clock,
      color: 'text-orange-400',
      bgColor: 'bg-orange-500/20',
    },
    {
      title: 'GPU Memory',
      value: `${currentTrainingData?.gpu_memory?.toFixed(1) || '0.0'}GB`,
      icon: Brain,
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-500/20',
    },
  ]

  const enhancedMetrics = [
    {
      title: 'Score Trend',
      value: currentTrainingData?.score_trend ? `${currentTrainingData.score_trend > 0 ? '+' : ''}${currentTrainingData.score_trend.toFixed(0)}` : 'N/A',
      icon: getScoreTrendIcon(),
      color: currentTrainingData?.score_trend ? (currentTrainingData.score_trend > 100 ? 'text-green-400' : currentTrainingData.score_trend < -100 ? 'text-red-400' : 'text-yellow-400') : 'text-gray-400',
      bgColor: 'bg-gray-500/20',
    },
    {
      title: 'Loss Trend',
      value: currentTrainingData?.loss_trend ? `${currentTrainingData.loss_trend > 0 ? '+' : ''}${currentTrainingData.loss_trend.toFixed(3)}` : 'N/A',
      icon: getLossTrendIcon(),
      color: currentTrainingData?.loss_trend ? (currentTrainingData.loss_trend < -0.1 ? 'text-green-400' : currentTrainingData.loss_trend > 0.1 ? 'text-red-400' : 'text-yellow-400') : 'text-gray-400',
      bgColor: 'bg-gray-500/20',
    },
    {
      title: 'Training Efficiency',
      value: currentTrainingData?.training_efficiency ? `${((currentTrainingData.training_efficiency.score_consistency + currentTrainingData.training_efficiency.loss_stability + currentTrainingData.training_efficiency.improvement_rate + currentTrainingData.training_efficiency.plateau_detection) / 4 * 100).toFixed(0)}%` : 'N/A',
      icon: getEfficiencyStatus().icon,
      color: getEfficiencyStatus().color,
      bgColor: 'bg-gray-500/20',
    },
    {
      title: 'Best Max Tile',
      value: (() => {
        const frequency = currentTrainingData?.max_tile_frequency
        if (!frequency || Object.keys(frequency).length === 0) return 'N/A'
        const maxTile = Math.max(...Object.keys(frequency).map(Number))
        return maxTile.toString()
      })(),
      icon: <Star className="w-4 h-4 text-yellow-400" />,
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-500/20',
    },
    {
      title: 'Load Balance',
      value: currentTrainingData?.load_balancing_reward ? `${currentTrainingData.load_balancing_reward.toFixed(3)}` : 'N/A',
      icon: <Scale className="w-4 h-4 text-pink-400" />,
      color: 'text-pink-400',
      bgColor: 'bg-pink-500/20',
    },
    {
      title: 'Expert Starvation',
      value: currentTrainingData?.expert_starvation_rate ? `${(currentTrainingData.expert_starvation_rate * 100).toFixed(1)}%` : 'N/A',
      icon: <AlertTriangle className="w-4 h-4 text-red-400" />,
      color: currentTrainingData?.expert_starvation_rate ? (currentTrainingData.expert_starvation_rate > 0.1 ? 'text-red-400' : currentTrainingData.expert_starvation_rate > 0.05 ? 'text-yellow-400' : 'text-green-400') : 'text-gray-400',
      bgColor: 'bg-red-500/20',
    },
    {
      title: 'Sparsity Score',
      value: currentTrainingData?.avg_sparsity_score ? `${(currentTrainingData.avg_sparsity_score * 100).toFixed(0)}%` : 'N/A',
      icon: <GitBranch className="w-4 h-4 text-blue-400" />,
      color: currentTrainingData?.avg_sparsity_score ? (currentTrainingData.avg_sparsity_score > 0.75 ? 'text-green-400' : currentTrainingData.avg_sparsity_score > 0.5 ? 'text-yellow-400' : 'text-red-400') : 'text-gray-400',
      bgColor: 'bg-blue-500/20',
    },
    {
      title: 'Balance Quality',
      value: currentTrainingData?.avg_balance_quality ? `${(currentTrainingData.avg_balance_quality * 100).toFixed(0)}%` : 'N/A',
      icon: <BarChart3 className="w-4 h-4 text-green-400" />,
      color: currentTrainingData?.avg_balance_quality ? (currentTrainingData.avg_balance_quality > 0.8 ? 'text-green-400' : currentTrainingData.avg_balance_quality > 0.6 ? 'text-yellow-400' : 'text-red-400') : 'text-gray-400',
      bgColor: 'bg-green-500/20',
    },
  ]

  // determine which metrics to show based on selected tab
  const displayedMetrics = activeTab === 'core'
    ? metrics
    : activeTab === 'trends'
      ? enhancedMetrics.slice(0, 4)
      : enhancedMetrics.slice(4)

  // NEW: Render expanded chart overlay
  const renderExpandedChart = () => {
    if (!expandedChart) return null

    return (
      <AnimatePresence>
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          onTouchEnd={handleTouchClose}
          onClick={handleCloseExpandedChart}
        >
          <motion.div
            className="relative w-full h-full max-w-2xl max-h-[60vh] m-4 card-glass rounded-3xl overflow-hidden"
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.8, opacity: 0 }}
            transition={{ duration: 0.3, type: "spring", damping: 25, stiffness: 300 }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-700/50">
              <div className="flex items-center space-x-3">
                <Maximize2 className="w-5 h-5 text-blue-400" />
                <h2 className="text-lg font-semibold text-white">{expandedChart.title}</h2>
              </div>
              <button
                onClick={handleCloseExpandedChart}
                className="p-2 rounded-xl bg-gray-700/50 hover:bg-gray-600/50 transition-colors group"
                aria-label="Close expanded chart"
              >
                <X className="w-5 h-5 text-gray-400 group-hover:text-white transition-colors" />
              </button>
            </div>

            {/* Chart Container */}
            <div className="flex-1 p-6">
              <div className="w-full h-full min-h-[40vh]">
                {expandedChart.type === 'loss' && (
                  <Line data={lossChartData} options={expandedChartOptions} />
                )}
                {expandedChart.type === 'score' && (
                  <Line data={scoreChartData} options={expandedChartOptions} />
                )}
                {expandedChart.type === 'actions' && (
                  <Doughnut data={actionDistributionData} options={expandedDoughnutOptions} />
                )}
                {expandedChart.type === 'experts' && (
                  <Bar data={expertUsageData} options={expandedBarOptions} />
                )}
              </div>
            </div>


          </motion.div>
        </motion.div>
      </AnimatePresence>
    )
  }

  return (
    <div className="safe-area h-full grid grid-rows-[auto_auto_1fr] gap-2 pb-4 px-4">
      {/* Redesigned Training Controls */}
      <motion.div
        className="card-glass p-3 rounded-2xl"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Header Row */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-semibold text-white">Training Controls</h3>
          </div>
          
          {/* Status Indicator */}
          <div className="flex items-center space-x-2 text-xs">
            <div className={`w-2 h-2 rounded-full ${isTraining ? 'bg-ui-state-success' : 'bg-ui-state-danger'}`} />
            <span className={`font-medium ${isTraining ? 'text-ui-state-success' : 'text-ui-state-danger'}`}>
              {isTraining ? 'Active' : 'Stopped'}
            </span>
            {isPaused && isTraining && (
              <span className="text-ui-state-warning">(Paused)</span>
            )}
            {!isConnected && (
              <span className="text-ui-state-danger">(Disconnected)</span>
            )}
          </div>
        </div>

        {/* Model Size Selector - Only show when not training */}
        {!isTraining && (
          <div className="mb-2">
            <label className="block text-xs text-ui-text-secondary mb-1">Model Size</label>
            <select
              value={modelSize}
              onChange={(e) => setModelSize(e.target.value as 'tiny' | 'small' | 'medium' | 'large')}
              disabled={isTraining || loadingStates.isTrainingStarting}
              className={`
                w-full bg-ui-surface-elevated text-ui-text-primary rounded-lg px-3 py-2 text-sm border border-ui-border-muted
                focus:outline-none focus:ring-1 focus:ring-ui-focus focus:border-ui-focus
                ${isTraining || loadingStates.isTrainingStarting ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-700/40'}
              `}
            >
              <option value="tiny">Tiny Model</option>
              <option value="small">Small Model</option>
              <option value="medium">Medium Model</option>
              <option value="large">Large Model</option>
            </select>
          </div>
        )}

        {/* Control Buttons */}
        <div className="grid grid-cols-2 gap-2">
          {/* Primary Action Button */}
          <button
            onClick={() => {
              if (!isTraining) {
                handleTrainingControl('start')
              } else if (isPaused) {
                handleTrainingControl('resume')
              } else {
                handleTrainingControl('pause')
              }
            }}
            disabled={loadingStates.isTrainingStarting || !isConnected}
            className={`
              flex items-center justify-center px-4 py-2.5 rounded-xl text-sm font-medium transition-all
              ${!isTraining 
                ? 'bg-ui-state-success/90 hover:bg-ui-state-success text-white' 
                : isPaused 
                  ? 'bg-ui-state-success/90 hover:bg-ui-state-success text-white'
                  : 'bg-ui-state-warning/90 hover:bg-ui-state-warning text-black'
              }
              ${(loadingStates.isTrainingStarting || !isConnected) ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105'}
            `}
          >
            {!isTraining ? (
              <>
                <Play className="w-4 h-4 mr-2" />
                Start Training
              </>
            ) : isPaused ? (
              <>
                <Play className="w-4 h-4 mr-2" />
                Resume
              </>
            ) : (
              <>
                <Pause className="w-4 h-4 mr-2" />
                Pause
              </>
            )}
          </button>

          {/* Secondary Action Button */}
          <button
            onClick={() => {
              if (isTraining) {
                handleTrainingControl('stop')
              } else {
                // If not training, this becomes a manual checkpoint button
                handleManualCheckpoint()
              }
            }}
            disabled={loadingStates.isTrainingStarting || !isConnected || isCreatingCheckpoint}
            className={`
              flex items-center justify-center px-4 py-2.5 rounded-xl text-sm font-medium transition-all
              ${isTraining 
                ? 'bg-ui-state-danger/90 hover:bg-ui-state-danger text-white' 
                : 'bg-ui-state-info/90 hover:bg-ui-state-info text-white'
              }
              ${(loadingStates.isTrainingStarting || !isConnected || isCreatingCheckpoint) ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105'}
            `}
          >
            {isTraining ? (
              <>
                <StopIcon className="w-4 h-4 mr-2" />
                Stop Training
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                {isCreatingCheckpoint ? 'Creating...' : 'Manual Checkpoint'}
              </>
            )}
          </button>
        </div>

        {/* Manual Checkpoint Button - Only show when training */}
        {isTraining && (
          <div className="mt-2">
            <button
              onClick={handleManualCheckpoint}
              disabled={loadingStates.isTrainingStarting || !isConnected || isCreatingCheckpoint}
            className={`
                w-full flex items-center justify-center px-4 py-2 rounded-lg text-sm font-medium transition-all
                bg-ui-state-info/20 hover:bg-ui-state-info/30 text-ui-state-info border border-ui-state-info/30
                ${(loadingStates.isTrainingStarting || !isConnected || isCreatingCheckpoint) ? 'opacity-50 cursor-not-allowed' : 'hover:scale-102'}
              `}
            >
              <Save className="w-4 h-4 mr-2" />
              {isCreatingCheckpoint ? 'Creating Checkpoint...' : 'Create Manual Checkpoint'}
            </button>
          </div>
        )}

        {/* Waiting for Data Indicator */}
        {showWaitingForFirstData && (
          <div className="mt-2 p-2 rounded-lg border border-ui-state-info/30 bg-ui-state-info/10">
            <div className="flex items-center space-x-2">
              <Brain className="w-4 h-4 text-ui-state-info animate-pulse" />
              <div className="flex-1">
                <div className="text-xs font-medium text-ui-text-primary">Initializing Training</div>
                <div className="text-xs text-ui-text-secondary">
                  Waiting for first episode data...
                </div>
              </div>
              {loadingStates.estimatedTimeRemaining !== null && (
                <div className="text-xs text-ui-text-secondary numeric">
                  ~{Math.ceil(loadingStates.estimatedTimeRemaining)}s
                </div>
              )}
            </div>
          </div>
        )}
      </motion.div>

      {/* Combined Charts and Metrics Section */}
      <motion.div
        className="card-glass p-3 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <h3 className="text-sm font-semibold mb-2 flex items-center">
          <BarChart3 className="w-4 h-4 mr-2 text-blue-400" />
          Training Analytics
        </h3>
        
        {/* Compact Charts Grid */}
        <div className="grid grid-cols-2 gap-2 mb-3">
          {/* Training Loss Chart - Compact */}
          <div className="flex flex-col items-center space-y-1">
            <div className="text-xs text-ui-text-secondary font-medium text-center">Training Loss</div>
            <div 
              className="w-full h-14 bg-gray-800/50 rounded-xl p-2 cursor-pointer hover:bg-gray-700/50 transition-colors relative group"
              onDoubleClick={() => handleChartDoubleTap('loss', 'Training Loss')}
              onTouchEnd={(e) => {
                const now = Date.now()
                const lastTap = (e.currentTarget as any).lastTap || 0
                if (now - lastTap < 300) {
                  handleChartDoubleTap('loss', 'Training Loss')
                }
                (e.currentTarget as any).lastTap = now
              }}
            >
              <Line data={lossChartData} options={enhancedChartOptions} />
              {isMobile && (
                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="text-xs text-ui-text-primary" style={{ background: 'var(--ui-overlay)', padding: '4px 8px', borderRadius: 8 }}>
                    Double-tap to expand
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Game Score Chart - Compact */}
          <div className="flex flex-col items-center space-y-1">
            <div className="text-xs text-ui-text-secondary font-medium text-center">Game Score</div>
            <div 
              className="w-full h-14 bg-gray-800/50 rounded-xl p-2 cursor-pointer hover:bg-gray-700/50 transition-colors relative group"
              onDoubleClick={() => handleChartDoubleTap('score', 'Game Score')}
              onTouchEnd={(e) => {
                const now = Date.now()
                const lastTap = (e.currentTarget as any).lastTap || 0
                if (now - lastTap < 300) {
                  handleChartDoubleTap('score', 'Game Score')
                }
                (e.currentTarget as any).lastTap = now
              }}
            >
              <Line data={scoreChartData} options={enhancedChartOptions} />
              {isMobile && (
                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="text-xs text-ui-text-primary" style={{ background: 'var(--ui-overlay)', padding: '4px 8px', borderRadius: 8 }}>
                    Double-tap to expand
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Action Distribution Chart - Compact */}
          <div className="flex flex-col items-center space-y-1">
            <div className="text-xs text-ui-text-secondary font-medium text-center">Action Distribution</div>
            <div 
              className="w-full h-14 bg-gray-800/50 rounded-xl p-2 flex items-center justify-center cursor-pointer hover:bg-gray-700/50 transition-colors relative group"
              onDoubleClick={() => handleChartDoubleTap('actions', 'Action Distribution')}
              onTouchEnd={(e) => {
                const now = Date.now()
                const lastTap = (e.currentTarget as any).lastTap || 0
                if (now - lastTap < 300) {
                  handleChartDoubleTap('actions', 'Action Distribution')
                }
                (e.currentTarget as any).lastTap = now
              }}
            >
              <Doughnut data={actionDistributionData} options={enhancedDoughnutOptions} />
              {isMobile && (
                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="text-xs text-ui-text-primary" style={{ background: 'var(--ui-overlay)', padding: '4px 8px', borderRadius: 8 }}>
                    Double-tap to expand
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Expert Usage Chart - Compact */}
          <div className="flex flex-col items-center space-y-1">
            <div className="text-xs text-ui-text-secondary font-medium text-center">Expert Usage</div>
            <div 
              className="w-full h-14 bg-gray-800/50 rounded-xl p-2 cursor-pointer hover:bg-gray-700/50 transition-colors relative group"
              onDoubleClick={() => handleChartDoubleTap('experts', 'Expert Usage')}
              onTouchEnd={(e) => {
                const now = Date.now()
                const lastTap = (e.currentTarget as any).lastTap || 0
                if (now - lastTap < 300) {
                  handleChartDoubleTap('experts', 'Expert Usage')
                }
                (e.currentTarget as any).lastTap = now
              }}
            >
              <Bar data={expertUsageData} options={enhancedBarOptions} />
              {isMobile && (
                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="text-xs text-ui-text-primary" style={{ background: 'var(--ui-overlay)', padding: '4px 8px', borderRadius: 8 }}>
                    Double-tap to expand
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-2 mb-2">
          <button
            onClick={() => setActiveTab('core')}
            className={`px-2 py-1 text-xs font-medium rounded ${activeTab === 'core' ? 'bg-ui-brand-primary text-white' : 'bg-ui-surface-elevated text-ui-text-secondary'}`}
          >Core</button>
          <button
            onClick={() => setActiveTab('trends')}
            className={`px-2 py-1 text-xs font-medium rounded ${activeTab === 'trends' ? 'bg-ui-brand-primary text-white' : 'bg-ui-surface-elevated text-ui-text-secondary'}`}
          >Trends</button>
          <button
            onClick={() => setActiveTab('stats')}
            className={`px-2 py-1 text-xs font-medium rounded ${activeTab === 'stats' ? 'bg-ui-brand-primary text-white' : 'bg-ui-surface-elevated text-ui-text-secondary'}`}
          >Stats</button>
        </div>

        {/* Compact Metrics Grid */}
        <div className={`${isMobile ? 'grid grid-cols-2 gap-2' : 'grid grid-cols-4 gap-2'}`}>  
          {displayedMetrics.map((metric, index) => (
            <motion.div
              key={metric.title}
              className="flex items-center space-x-2 p-2 bg-gray-800/30 rounded-xl"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 + index * 0.03 }}
            >
              <div className={`p-1 rounded ${metric.bgColor}`}>  
                {React.isValidElement(metric.icon)
                  ? metric.icon
                  : React.createElement(metric.icon as React.ElementType, { className: `w-3 h-3 ${metric.color}` })
                }
              </div>
              <div className="min-w-0">
                <div className="text-xs text-ui-text-secondary font-medium truncate">{metric.title}</div>
                <div className={`font-bold ${metric.color} text-sm truncate numeric`}>{metric.value}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Expanded Chart Overlay */}
      {renderExpandedChart()}
    </div>
  )
}

export default TrainingDashboard 