import React, { useState, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import { 
  Clock, Target, TrendingUp, Timer, Zap, Scale, Activity, GitBranch, Compass, Flame, Thermometer,
  Gauge, Brain, CheckCircle, AlertTriangle
} from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'

const TrainingMetrics: React.FC = () => {
  const { 
    isTraining, 
    getCurrentTrainingData,
    lastPolicyLoss,
    lastValueLoss,
    lastEvaluation,
    routerHistory,
    currentEpisode,
  } = useTrainingStore()
  const [currentElapsedTime, setCurrentElapsedTime] = useState(0)
  const [progressAnimation, setProgressAnimation] = useState(0)

  // Get current training data with fallback
  const currentTrainingData = getCurrentTrainingData()

  // Update elapsed time in real-time
  useEffect(() => {
    if (!isTraining || !currentTrainingData?.wall_clock_elapsed) {
      setCurrentElapsedTime(currentTrainingData?.wall_clock_elapsed || 0)
      return
    }

    // Calculate the base time when we received the last update
    const lastUpdateTime = Date.now() / 1000 - currentTrainingData.wall_clock_elapsed
    
    // Update elapsed time every second
    const interval = setInterval(() => {
      const now = Date.now() / 1000
      setCurrentElapsedTime(now - lastUpdateTime)
    }, 1000)

    return () => clearInterval(interval)
  }, [isTraining, currentTrainingData?.wall_clock_elapsed])

  // Animated progress indicator for training activity
  useEffect(() => {
    if (!isTraining || !currentTrainingData?.is_training_active) {
      setProgressAnimation(0)
      return
    }

    // Animate progress bar to show training is active
    const interval = setInterval(() => {
      setProgressAnimation(prev => {
        if (prev >= 100) return 0
        return prev + 2
      })
    }, 100)

    return () => clearInterval(interval)
  }, [isTraining, currentTrainingData?.is_training_active])

  // Helper function to format time duration (robust)
  const formatTime = (seconds?: number | null): string => {
    const s = Number(seconds)
    if (!Number.isFinite(s) || s <= 0) return '—'
    if (s < 60) return `${Math.round(s)}s`
    if (s < 3600) {
      const minutes = Math.floor(s / 60)
      const remainingSeconds = Math.round(s % 60)
      return `${minutes}m ${remainingSeconds}s`
    }
    const hours = Math.floor(s / 3600)
    const minutes = Math.floor((s % 3600) / 60)
    return `${hours}h ${minutes}m`
  }

  // Helper function to format training speed
  const formatSpeed = (speed?: number | null): string => {
    const v = Number(speed)
    if (!Number.isFinite(v) || v <= 0) return '—'
    return `${v.toFixed(1)} ep/min`
  }

  // Calculate estimated time to next episode
  const getNextEpisodeEstimate = (): string => {
    if (!currentTrainingData?.next_episode_estimate || currentTrainingData.next_episode_estimate <= 0) {
      return '—'
    }
    return formatTime(currentTrainingData.next_episode_estimate)
  }

  // Router health based on history (stable dominance/entropy)
  const routerHealth = useMemo(() => {
    const last = routerHistory.length > 0 ? routerHistory[routerHistory.length - 1] : undefined
    if (!last) return { entropy: 0, hhiNorm: 0, top1: 0, activeExperts: 0 }
    const n = (last.usage?.length || 0) || 1
    const minHhi = 1 / n
    const hhiNorm = Math.max(0, Math.min(1, (last.hhi - minHhi) / (1 - minHhi)))
    const top1 = Array.isArray(last.usage) && last.usage.length > 0 ? Math.max(...last.usage) : 0
    return { entropy: last.entropy || 0, hhiNorm, top1, activeExperts: last.activeExperts || 0 }
  }, [routerHistory])

  // Robust training loss with fallback to last known values
  const trainingLoss = useMemo(() => {
    const lossField = (currentTrainingData?.loss ?? null)
    if (typeof lossField === 'number' && Number.isFinite(lossField)) return lossField
    if (typeof lastPolicyLoss === 'number' && typeof lastValueLoss === 'number') {
      return lastPolicyLoss + lastValueLoss
    }
    return null
  }, [currentTrainingData?.loss, lastPolicyLoss, lastValueLoss])

  // Note: exploration controls removed from this compact metrics view

  const metrics = [
    {
      icon: Target,
      label: 'Episode',
      value: currentEpisode > 0 ? `${currentEpisode}` : '—',
      color: 'text-green-400',
      bgColor: 'bg-green-500/20',
    },
    {
      icon: Gauge,
      label: 'Speed',
      value: formatSpeed(currentTrainingData?.training_speed),
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/20',
    },
    {
      icon: TrendingUp,
      label: 'Avg Length',
      value: currentTrainingData?.avg_game_length != null ? `${Math.round(currentTrainingData.avg_game_length)} moves` : '—',
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/20',
    },
    {
      icon: Clock,
      label: 'Elapsed',
      value: formatTime(currentElapsedTime),
      color: 'text-orange-400',
      bgColor: 'bg-orange-500/20',
    },
    {
      icon: Timer,
      label: 'Next Checkpoint',
      value: formatTime(currentTrainingData?.estimated_time_to_checkpoint),
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-500/20',
    },
    {
      icon: Zap,
      label: 'Learning Rate',
      value: currentTrainingData?.learning_rate != null ? Number(currentTrainingData.learning_rate).toFixed(6) : '—',
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/20',
    },
    {
      icon: Brain,
      label: 'GPU Memory',
      value: currentTrainingData?.gpu_memory != null ? `${Number(currentTrainingData.gpu_memory).toFixed(1)} GB` : '—',
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-500/20',
    },
    {
      icon: Scale,
      label: 'Load Balance',
      value: currentTrainingData?.load_balancing_reward != null ? `${currentTrainingData.load_balancing_reward.toFixed(3)}` : '—',
      color: 'text-pink-400',
      bgColor: 'bg-pink-500/20',
    },
    {
      icon: Compass,
      label: 'Novelty',
      value: (() => {
        const td: any = currentTrainingData as any
        const v = td?.avg_novelty ?? td?.novelty_rate ?? td?.novelty_bonus
        return v != null ? `${(Number(v) * 100).toFixed(0)}%` : '—'
      })(),
      color: 'text-indigo-400',
      bgColor: 'bg-indigo-500/20',
    },
    {
      icon: Flame,
      label: 'Dominance (HHI)',
      value: routerHealth.hhiNorm > 0 ? `${(routerHealth.hhiNorm * 100).toFixed(0)}%` : '—',
      color: routerHealth.hhiNorm > 0.6 ? 'text-red-400' : routerHealth.hhiNorm > 0.3 ? 'text-yellow-400' : 'text-green-400',
      bgColor: 'bg-red-500/20',
    },
    {
      icon: Activity,
      label: 'Starvation Severity',
      value: currentTrainingData?.avg_starvation_severity != null ? `${(currentTrainingData.avg_starvation_severity * 100).toFixed(1)}%` : '—',
      color: currentTrainingData?.avg_starvation_severity != null ? (currentTrainingData.avg_starvation_severity > 0.5 ? 'text-red-400' : currentTrainingData.avg_starvation_severity > 0.2 ? 'text-yellow-400' : 'text-green-400') : 'text-gray-400',
      bgColor: 'bg-red-500/20',
    },
    {
      icon: GitBranch,
      label: 'Recovering Experts',
      value: (() => {
        const er = currentTrainingData?.expert_recovery_rates
        if (!er) return '—'
        const recovering = Object.values(er).filter(v => Number(v) > 0).length
        return recovering > 0 ? `${recovering}` : '—'
      })(),
      color: currentTrainingData?.expert_recovery_rates && Object.values(currentTrainingData.expert_recovery_rates).some(v => Number(v) > 0) ? 'text-green-400' : 'text-gray-400',
      bgColor: 'bg-green-500/20',
    },
    {
      icon: TrendingUp,
      label: 'Training Loss',
      value: trainingLoss != null ? `${trainingLoss.toFixed(4)}` : '—',
      color: 'text-red-400',
      bgColor: 'bg-red-500/20',
    },
    {
      icon: CheckCircle,
      label: 'Eval Median',
      value: lastEvaluation ? `${Math.round(lastEvaluation.median_score)}` : '—',
      color: lastEvaluation ? (lastEvaluation.median_score > 1500 ? 'text-green-400' : lastEvaluation.median_score > 800 ? 'text-yellow-400' : 'text-red-400') : 'text-gray-400',
      bgColor: 'bg-green-500/10',
    },
    {
      icon: AlertTriangle,
      label: 'Eval 2048 Solve',
      value: lastEvaluation ? `${(lastEvaluation.solve_rate_2048 * 100).toFixed(0)}%` : '—',
      color: lastEvaluation ? (lastEvaluation.solve_rate_2048 >= 0.9 ? 'text-green-400' : lastEvaluation.solve_rate_2048 >= 0.6 ? 'text-yellow-400' : 'text-red-400') : 'text-gray-400',
      bgColor: 'bg-green-500/10',
    },
    {
      icon: Thermometer,
      label: 'Router Entropy',
      value: routerHealth.entropy > 0 ? `${(routerHealth.entropy * 100).toFixed(0)}%` : '—',
      color: routerHealth.entropy >= 0.7 ? 'text-green-400' : routerHealth.entropy >= 0.5 ? 'text-yellow-400' : 'text-red-400',
      bgColor: 'bg-cyan-500/20',
    },
  ]

  return (
    <div className="safe-area space-y-4 px-4">
      {/* Animated Training Progress Indicator */}
      {isTraining && currentTrainingData?.is_training_active && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="card-glass p-4 rounded-2xl border border-ui-state-success/30 bg-ui-state-success/10"
        >
          <div className="flex items-center space-x-3 mb-3">
            <Activity className="w-5 h-5 text-ui-state-success animate-pulse" />
            <div className="flex-1">
              <div className="text-sm font-medium text-ui-text-primary">Training Active</div>
              <div className="text-xs text-ui-text-secondary">
                Next episode in ~{getNextEpisodeEstimate()}
              </div>
            </div>
          </div>
          
          {/* Animated Progress Bar */}
          <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
            <motion.div
              className="bg-gradient-to-r from-ui-state-success to-green-300 h-2 rounded-full"
              animate={{ x: `${progressAnimation - 100}%` }}
              transition={{ duration: 0.5, ease: "linear" }}
              style={{ width: '100%' }}
            />
          </div>
        </motion.div>
      )}

      {/* Metrics Grid */}
      <div
        className="grid gap-4"
        style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))' }}
      >
        {metrics.map((metric, index) => (
          <motion.div
            key={metric.label}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
            className={`card-glass p-4 rounded-xl bg-gray-800/70 border-l-4 ${metric.color.replace('text-', 'border-')}`}
          >
            <div className="flex items-center space-x-2 mb-2">
              <metric.icon className={`w-4 h-4 ${metric.color}`} />
              <span className="text-xs font-medium text-ui-text-secondary">{metric.label}</span>
            </div>
            <div className={`text-xl font-extrabold ${metric.color} numeric`}>
              {metric.value}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

export default TrainingMetrics 