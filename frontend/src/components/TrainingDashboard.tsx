import React, { useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Line, Doughnut, Bar } from 'react-chartjs-2'
import { 
  TrendingDown, Brain, Zap, Target,
  Clock, Gauge, BarChart3, TrendingUpIcon, TrendingDownIcon, 
  CheckCircle, AlertTriangle, Info,
  X, Maximize2,
  Compass
} from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'
import { useDeviceDetection } from '../utils/deviceDetection'
// config no longer needed on metrics tab

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
    lastPolicyLoss, 
    lastValueLoss, 
    getCurrentTrainingData,
    evaluationHistory,
    lastEvaluation,
    routerHistory,
  } = useTrainingStore()

  const { displayMode } = useDeviceDetection()
  const isMobile = displayMode === 'mobile'
  // Tab state removed in compact redesign
  
  // Controls removed from Metrics tab; no local checkpoint state needed

  // NEW: Chart expansion state
  const [expandedChart, setExpandedChart] = useState<{
    type: 'loss' | 'score' | 'actions' | 'experts'
    title: string
  } | null>(null)

  // Use isWaitingForFirstData from store (directly where needed)

  // Get current training data with fallback
  const currentTrainingData = getCurrentTrainingData()

  // Derived dominance metrics (fallback if not provided by backend)
  // Removed old dominance calc (superseded by routerHistory-based health)

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

  // Controls removed from Metrics tab; actions relocated to Controls tab

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
    // Prefer score episodes as the canonical x-axis to keep charts aligned
    const baseEpisodes = (scoreHistory.episodes.length > 0
      ? scoreHistory.episodes
      : (lossHistory.episodes.length > 0
          ? lossHistory.episodes
          : Array.from({ length: 20 }, (_, i) => i + 1)))

    const dataPoints = isMobile ? Math.min(15, baseEpisodes.length) : baseEpisodes.length
    const startIndex = Math.max(0, baseEpisodes.length - dataPoints)
    const selectedEpisodes = baseEpisodes.slice(startIndex)

    const values = lossHistory.values.length > 0
      ? selectedEpisodes.map((ep) => {
          const idx = lossHistory.episodes.indexOf(ep)
          return idx >= 0 ? lossHistory.values[idx] : Number.NaN
        })
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
  }, [lossHistory, scoreHistory, isMobile])

  const scoreChartData = useMemo(() => {
    // Use the same base episode axis selection as the loss chart
    const baseEpisodes = (scoreHistory.episodes.length > 0
      ? scoreHistory.episodes
      : (lossHistory.episodes.length > 0
          ? lossHistory.episodes
          : Array.from({ length: 20 }, (_, i) => i + 1)))

    const dataPoints = isMobile ? Math.min(15, baseEpisodes.length) : baseEpisodes.length
    const startIndex = Math.max(0, baseEpisodes.length - dataPoints)
    const selectedEpisodes = baseEpisodes.slice(startIndex)

    const values = scoreHistory.values.length > 0
      ? selectedEpisodes.map((ep) => {
          const idx = scoreHistory.episodes.indexOf(ep)
          return idx >= 0 ? scoreHistory.values[idx] : Number.NaN
        })
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
  }, [scoreHistory, lossHistory, isMobile])

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

  const expertUsageTrendData = useMemo(() => {
    const labels = routerHistory.map(e => new Date(e.timestamp).toLocaleTimeString())
    const last = routerHistory.length > 0 ? routerHistory[routerHistory.length - 1] : undefined
    const seriesCount = last?.usage.length || 0
    const colors = [ui.info, ui.brand, ui.success, ui.warning, ui.danger, '#34d399', '#60a5fa', '#f472b6']
    const datasets = Array.from({ length: seriesCount }).map((_, idx) => ({
      label: `E${idx + 1}`,
      data: routerHistory.map(e => (e.usage[idx] || 0) * 100),
      borderColor: colors[idx % colors.length],
      backgroundColor: hexToRgba(colors[idx % colors.length], 0.15),
      borderWidth: 2,
      fill: true,
      tension: 0.35,
      pointRadius: 0,
    }))
    return { labels, datasets }
  }, [routerHistory, isMobile])

  const routerHealth = useMemo(() => {
    const last = routerHistory.length > 0 ? routerHistory[routerHistory.length - 1] : undefined
    if (!last) return { entropy: 0, hhi: 0, activeExperts: 0 }
    return last
  }, [routerHistory])

  // Evaluation charts
  const evalScoreLineData = useMemo(() => {
    const labels = evaluationHistory.map(e => new Date(e.timestamp).toLocaleTimeString())
    const medians = evaluationHistory.map(e => e.median_score)
    return {
      labels,
      datasets: [
        {
          label: 'Eval Median',
          data: medians,
          borderColor: ui.brand,
          backgroundColor: hexToRgba(ui.brand, 0.1),
          borderWidth: isMobile ? 2 : 3,
          fill: true,
          tension: 0.35,
          pointRadius: 0,
        },
      ],
    }
  }, [evaluationHistory, isMobile])

  const evalTileBarData = useMemo(() => {
    const freq = lastEvaluation?.max_tile_frequency || {}
    const tiles = Object.keys(freq).map(Number).sort((a, b) => a - b)
    const values = tiles.map(t => (freq as any)[t] || 0)
    return {
      labels: tiles.map(t => `${t}`),
      datasets: [
        {
          label: 'Max Tile Frequency',
          data: values,
          backgroundColor: ui.info,
          borderColor: ui.white,
          borderWidth: 1,
        },
      ],
    }
  }, [lastEvaluation])

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

  // Removed unused expandedBarOptions (replaced expert bar with trend line)

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

  const evalColor = (value: number, goodHigher = true) => {
    if (Number.isNaN(value)) return 'text-gray-400'
    const v = value
    if (goodHigher) return v >= 0.9 ? 'text-green-400' : v >= 0.6 ? 'text-yellow-400' : 'text-red-400'
    return v <= 0.1 ? 'text-green-400' : v <= 0.3 ? 'text-yellow-400' : 'text-red-400'
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
    // Eval KPIs
    {
      title: 'Eval Median',
      value: lastEvaluation ? lastEvaluation.median_score.toFixed(0) : '—',
      icon: Target,
      color: lastEvaluation ? (lastEvaluation.median_score > 1500 ? 'text-green-400' : lastEvaluation.median_score > 800 ? 'text-yellow-400' : 'text-red-400') : 'text-gray-400',
      bgColor: 'bg-green-500/10',
    },
    {
      title: 'Eval 2048 Solve',
      value: lastEvaluation ? `${(lastEvaluation.solve_rate_2048 * 100).toFixed(0)}%` : '—',
      icon: CheckCircle,
      color: lastEvaluation ? evalColor(lastEvaluation.solve_rate_2048, true) : 'text-gray-400',
      bgColor: 'bg-green-500/10',
    },
    // NEW: Novelty signal
    {
      title: 'Novelty',
      value: (() => {
        const td: any = currentTrainingData as any
        const v = td?.avg_novelty ?? td?.novelty_rate ?? td?.novelty_bonus
        return v != null ? `${(Number(v) * 100).toFixed(0)}%` : 'N/A'
      })(),
      icon: Compass,
      color: 'text-indigo-400',
      bgColor: 'bg-indigo-500/20',
    },
    // Removed Active Params tile from metrics tab (moved to Controls)
  ]

  /*
   * Retain enhanced metrics template for future expansion
   * (disabled in compact view to keep layout minimal)
   */
  // Temporarily comment out to avoid unused warnings in compact view
  /* const _enhancedMetricsTemplate = [
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
      value: currentTrainingData?.load_balancing_reward != null ? `${currentTrainingData.load_balancing_reward.toFixed(3)}` : 'N/A',
      icon: <Scale className="w-4 h-4 text-pink-400" />,
      color: 'text-pink-400',
      bgColor: 'bg-pink-500/20',
    },
    {
      title: 'Expert Starvation',
      value: currentTrainingData?.expert_starvation_rate != null ? `${(currentTrainingData.expert_starvation_rate * 100).toFixed(1)}%` : 'N/A',
      icon: <AlertTriangle className="w-4 h-4 text-red-400" />,
      color: currentTrainingData?.expert_starvation_rate ? (currentTrainingData.expert_starvation_rate > 0.1 ? 'text-red-400' : currentTrainingData.expert_starvation_rate > 0.05 ? 'text-yellow-400' : 'text-green-400') : 'text-gray-400',
      bgColor: 'bg-red-500/20',
    },
    // Replaced sparsity with dominance visuals
    {
      title: 'Dominance (HHI)',
      value: `${(dominance.hhi * 100).toFixed(0)}%`,
      icon: <Flame className="w-4 h-4" />,
      color: dominance.hhi > 0.6 ? 'text-red-400' : dominance.hhi > 0.3 ? 'text-yellow-400' : 'text-green-400',
      bgColor: 'bg-red-500/20',
    },
    {
      title: 'Top Expert Share',
      value: `${(dominance.top1 * 100).toFixed(0)}%`,
      icon: <GitBranch className="w-4 h-4 text-blue-400" />,
      color: dominance.top1 > 0.5 ? 'text-red-400' : dominance.top1 > 0.35 ? 'text-yellow-400' : 'text-green-400',
      bgColor: 'bg-blue-500/20',
    },
    {
      title: 'Exploration',
      value: (() => {
        const td: any = currentTrainingData as any
        const t = td?.exploration?.temperature
        const e = td?.exploration?.epsilon
        if (t != null && e != null) return `T ${Number(t).toFixed(2)} • ε ${Number(e).toFixed(2)}`
        return 'Adaptive'
      })(),
      icon: <Thermometer className="w-4 h-4 text-cyan-400" />,
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-500/20',
    },
    {
      title: 'Balance Quality',
      value: currentTrainingData?.avg_balance_quality ? `${(currentTrainingData.avg_balance_quality * 100).toFixed(0)}%` : 'N/A',
      icon: <BarChart3 className="w-4 h-4 text-green-400" />,
      color: currentTrainingData?.avg_balance_quality ? (currentTrainingData.avg_balance_quality > 0.8 ? 'text-green-400' : currentTrainingData.avg_balance_quality > 0.6 ? 'text-yellow-400' : 'text-red-400') : 'text-gray-400',
      bgColor: 'bg-green-500/20',
    },
  ] */

  // determine which metrics to show based on selected tab
  // Compact redesign uses a curated subset inline

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
                  <Line data={expertUsageTrendData} options={expandedChartOptions} />
                )}
              </div>
            </div>


          </motion.div>
        </motion.div>
      </AnimatePresence>
    )
  }

  return (
    <div className="safe-area h-full min-h-0 grid grid-rows-[1fr] gap-2 px-4 overflow-hidden">

      {/* Combined Charts and Metrics Section (fills remaining space, internal scroll) */}
      <motion.div
        className="card-glass p-3 rounded-2xl h-full min-h-0 overflow-auto"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        style={{ paddingBottom: 'max(16px, env(safe-area-inset-bottom))' }}
      >
        <h3 className="text-sm font-semibold mb-2 flex items-center justify-between">
          <BarChart3 className="w-4 h-4 mr-2 text-blue-400" />
          Training Insights
          <a
            href="#controls"
            onClick={(e) => { e.preventDefault(); (window as any).scrollTo && (window as any).scrollTo(0, 0); }}
            className="text-[11px] text-ui-text-secondary hover:text-ui-brand-primary"
            title="Go to Controls"
          >
            Controls »
          </a>
        </h3>
        
        {/* KPI Tiles */}
        <div className="grid grid-cols-2 gap-2 mb-2">
          {metrics.slice(0, 4).map((metric, index) => (
            <motion.div
              key={metric.title}
              className="flex items-center space-x-2 p-2 bg-gray-800/30 rounded-xl"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.05 * index }}
            >
              <div className={`p-1 rounded ${metric.bgColor}`}>
                {React.isValidElement(metric.icon)
                  ? metric.icon
                  : React.createElement(metric.icon as React.ElementType, { className: `w-3 h-3 ${metric.color}` })}
              </div>
              <div className="min-w-0">
                <div className="text-xs text-ui-text-secondary font-medium truncate">{metric.title}</div>
                <div className={`font-bold ${metric.color} text-sm truncate numeric`}>{metric.value}</div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Micro Charts Row */}
        <div className="grid grid-cols-2 gap-2 mb-2">
          <div className="flex flex-col space-y-1">
            <div className="text-xs text-ui-text-secondary font-medium">Training Loss</div>
            <div
              className="w-full h-16 bg-gray-800/50 rounded-xl p-2 cursor-pointer hover:bg-gray-700/50 transition-colors"
              onDoubleClick={() => handleChartDoubleTap('loss', 'Training Loss')}
            >
              <Line data={lossChartData} options={enhancedChartOptions} />
            </div>
          </div>
          <div className="flex flex-col space-y-1">
            <div className="text-xs text-ui-text-secondary font-medium">Game Score</div>
            <div
              className="w-full h-16 bg-gray-800/50 rounded-xl p-2 cursor-pointer hover:bg-gray-700/50 transition-colors"
              onDoubleClick={() => handleChartDoubleTap('score', 'Game Score')}
            >
              <Line data={scoreChartData} options={enhancedChartOptions} />
            </div>
          </div>
        </div>

        {/* Distribution Row */}
        <div className="grid grid-cols-2 gap-2 mb-2">
          <div className="flex flex-col space-y-1">
            <div className="text-xs text-ui-text-secondary font-medium">Action Distribution</div>
            <div
              className="w-full h-20 bg-gray-800/50 rounded-xl p-2 flex items-center justify-center cursor-pointer hover:bg-gray-700/50 transition-colors"
              onDoubleClick={() => handleChartDoubleTap('actions', 'Action Distribution')}
            >
              <Doughnut data={actionDistributionData} options={enhancedDoughnutOptions} />
            </div>
          </div>
          <div className="flex flex-col space-y-1">
            <div className="text-xs text-ui-text-secondary font-medium">Expert Usage (trend)</div>
            <div className="w-full h-20 bg-gray-800/50 rounded-xl p-2 cursor-pointer hover:bg-gray-700/50 transition-colors">
              <Line data={expertUsageTrendData} options={enhancedChartOptions} />
            </div>
          </div>
        </div>

        {/* Evaluation Row */}
        <div className="grid grid-cols-2 gap-2 mb-2">
          <div className="flex flex-col space-y-1">
            <div className="text-xs text-ui-text-secondary font-medium">Eval Median (trend)</div>
            <div className="w-full h-16 bg-gray-800/50 rounded-xl p-2">
              <Line data={evalScoreLineData} options={enhancedChartOptions} />
            </div>
          </div>
          <div className="flex flex-col space-y-1">
            <div className="text-xs text-ui-text-secondary font-medium">Eval Max Tile Histogram</div>
            <div className="w-full h-16 bg-gray-800/50 rounded-xl p-2">
              <Bar data={evalTileBarData} options={enhancedBarOptions} />
            </div>
          </div>
        </div>

        {/* Health Strip */}
        <div className="grid grid-cols-2 gap-2">
          <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-gray-800/40">
            <div className="flex items-center gap-2">
              {getScoreTrendIcon()}
              <span className="text-xs text-ui-text-secondary">Score Trend</span>
            </div>
            <span className="text-xs numeric text-ui-text-primary">{currentTrainingData?.score_trend ? `${currentTrainingData.score_trend > 0 ? '+' : ''}${currentTrainingData.score_trend.toFixed(0)}` : 'N/A'}</span>
          </div>
          <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-gray-800/40">
            <div className="flex items-center gap-2">
              {getLossTrendIcon()}
              <span className="text-xs text-ui-text-secondary">Loss Trend</span>
            </div>
            <span className="text-xs numeric text-ui-text-primary">{currentTrainingData?.loss_trend ? `${currentTrainingData.loss_trend > 0 ? '+' : ''}${currentTrainingData.loss_trend.toFixed(3)}` : 'N/A'}</span>
          </div>
          <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-gray-800/40">
            <div className="flex items-center gap-2">
              {getEfficiencyStatus().icon}
              <span className="text-xs text-ui-text-secondary">Efficiency</span>
            </div>
            <span className={`text-xs ${getEfficiencyStatus().color}`}>{currentTrainingData?.training_efficiency ? `${((currentTrainingData.training_efficiency.score_consistency + currentTrainingData.training_efficiency.loss_stability + currentTrainingData.training_efficiency.improvement_rate + currentTrainingData.training_efficiency.plateau_detection)/4*100).toFixed(0)}%` : 'N/A'}</span>
          </div>
          <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-gray-800/40">
            <span className="text-xs text-ui-text-secondary">Active Experts</span>
            <span className={`text-xs numeric ${routerHealth.activeExperts >= 4 ? 'text-green-400' : routerHealth.activeExperts >= 3 ? 'text-yellow-400' : 'text-red-400'}`}>{routerHealth.activeExperts}</span>
          </div>
          <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-gray-800/40">
            <span className="text-xs text-ui-text-secondary">Router Entropy</span>
            <span className={`text-xs numeric ${routerHealth.entropy >= 0.7 ? 'text-green-400' : routerHealth.entropy >= 0.5 ? 'text-yellow-400' : 'text-red-400'}`}>{(routerHealth.entropy * 100).toFixed(0)}%</span>
          </div>
        </div>
      </motion.div>

      {/* Expanded Chart Overlay */}
      {renderExpandedChart()}
    </div>
  )
}

export default TrainingDashboard 