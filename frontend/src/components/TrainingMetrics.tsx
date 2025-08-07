import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Clock, Target, TrendingUp, Timer, Zap, Scale, Activity, GitBranch } from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'

const TrainingMetrics: React.FC = () => {
  const { isTraining, getCurrentTrainingData } = useTrainingStore()
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

  // Helper function to format time duration
  const formatTime = (seconds: number): string => {
    if (seconds < 60) {
      return `${Math.round(seconds)}s`
    } else if (seconds < 3600) {
      const minutes = Math.floor(seconds / 60)
      const remainingSeconds = Math.round(seconds % 60)
      return `${minutes}m ${remainingSeconds}s`
    } else {
      const hours = Math.floor(seconds / 3600)
      const minutes = Math.floor((seconds % 3600) / 60)
      return `${hours}h ${minutes}m`
    }
  }

  // Helper function to format training speed
  const formatSpeed = (speed: number): string => {
    if (speed >= 1) {
      return `${speed.toFixed(1)} eps/min`
    } else {
      return `${(speed * 60).toFixed(1)} eps/sec`
    }
  }

  // Calculate estimated time to next episode
  const getNextEpisodeEstimate = (): string => {
    if (!currentTrainingData?.next_episode_estimate || currentTrainingData.next_episode_estimate <= 0) {
      return 'Calculating...'
    }
    return formatTime(currentTrainingData.next_episode_estimate)
  }

  const metrics = [
    {
      icon: Zap,
      label: 'Speed',
      value: currentTrainingData?.training_speed ? formatSpeed(currentTrainingData.training_speed) : '0.0 eps/min',
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/20',
    },
    {
      icon: Target,
      label: 'Avg Game Length',
      value: currentTrainingData?.avg_game_length ? `${Math.round(currentTrainingData.avg_game_length)} moves` : '0 moves',
      color: 'text-green-400',
      bgColor: 'bg-green-500/20',
    },
    {
      icon: TrendingUp,
      label: 'Min/Max Length',
      value: currentTrainingData?.min_game_length && currentTrainingData?.max_game_length 
        ? `${currentTrainingData.min_game_length}/${currentTrainingData.max_game_length}` 
        : '0/0',
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/20',
    },
    {
      icon: Clock,
      label: 'Elapsed Time',
      value: formatTime(currentElapsedTime),
      color: 'text-orange-400',
      bgColor: 'bg-orange-500/20',
    },
    {
      icon: Timer,
      label: 'Next Checkpoint',
      value: currentTrainingData?.estimated_time_to_checkpoint ? formatTime(currentTrainingData.estimated_time_to_checkpoint) : '0s',
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-500/20',
    },
    {
      icon: Scale,
      label: 'Load Balance',
      value: currentTrainingData?.load_balancing_reward ? `${currentTrainingData.load_balancing_reward.toFixed(3)}` : '0.000',
      color: 'text-pink-400',
      bgColor: 'bg-pink-500/20',
    },
    // NEW: Enhanced expert starvation metrics
    {
      icon: Activity,
      label: 'Starvation Severity',
      value: currentTrainingData?.avg_starvation_severity ? `${(currentTrainingData.avg_starvation_severity * 100).toFixed(1)}%` : '0.0%',
      color: currentTrainingData?.avg_starvation_severity ? (currentTrainingData.avg_starvation_severity > 0.5 ? 'text-red-400' : currentTrainingData.avg_starvation_severity > 0.2 ? 'text-yellow-400' : 'text-green-400') : 'text-gray-400',
      bgColor: 'bg-red-500/20',
    },
    {
      icon: GitBranch,
      label: 'Recovery Rate',
      value: currentTrainingData?.expert_recovery_rates ? `${Object.keys(currentTrainingData.expert_recovery_rates).length} experts` : '0 experts',
      color: currentTrainingData?.expert_recovery_rates && Object.keys(currentTrainingData.expert_recovery_rates).length > 0 ? 'text-green-400' : 'text-gray-400',
      bgColor: 'bg-green-500/20',
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