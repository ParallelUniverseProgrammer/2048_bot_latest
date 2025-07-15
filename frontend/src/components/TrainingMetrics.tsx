import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Clock, Target, TrendingUp, Timer, Zap } from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'

const TrainingMetrics: React.FC = () => {
  const { trainingData, isTraining } = useTrainingStore()
  const [currentElapsedTime, setCurrentElapsedTime] = useState(0)

  // Update elapsed time in real-time
  useEffect(() => {
    if (!isTraining || !trainingData?.wall_clock_elapsed) {
      setCurrentElapsedTime(trainingData?.wall_clock_elapsed || 0)
      return
    }

    // Calculate the base time when we received the last update
    const lastUpdateTime = Date.now() / 1000 - trainingData.wall_clock_elapsed
    
    // Update elapsed time every second
    const interval = setInterval(() => {
      const now = Date.now() / 1000
      setCurrentElapsedTime(now - lastUpdateTime)
    }, 1000)

    return () => clearInterval(interval)
  }, [isTraining, trainingData?.wall_clock_elapsed])

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

  const metrics = [
    {
      icon: Zap,
      label: 'Speed',
      value: trainingData?.training_speed ? formatSpeed(trainingData.training_speed) : '0.0 eps/min',
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/20',
    },
    {
      icon: Target,
      label: 'Avg Game Length',
      value: trainingData?.avg_game_length ? `${Math.round(trainingData.avg_game_length)} moves` : '0 moves',
      color: 'text-green-400',
      bgColor: 'bg-green-500/20',
    },
    {
      icon: TrendingUp,
      label: 'Min/Max Length',
      value: trainingData?.min_game_length && trainingData?.max_game_length 
        ? `${trainingData.min_game_length}/${trainingData.max_game_length}` 
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
      value: trainingData?.estimated_time_to_checkpoint ? formatTime(trainingData.estimated_time_to_checkpoint) : '0s',
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-500/20',
    },
  ]

  return (
    <div className="flex flex-wrap items-center gap-3 text-sm">
      {metrics.map((metric, index) => {
        const IconComponent = metric.icon
        return (
          <motion.div
            key={metric.label}
            className="flex items-center space-x-2 px-3 py-2 rounded-lg bg-gray-800/50 border border-gray-700/50"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
            whileHover={{ scale: 1.05 }}
          >
            <div className={`p-1 rounded ${metric.bgColor}`}>
              <IconComponent className={`w-3 h-3 ${metric.color}`} />
            </div>
            <div className="flex flex-col">
              <span className="text-xs text-gray-400">{metric.label}</span>
              <span className={`font-medium ${metric.color}`}>
                {metric.value}
              </span>
            </div>
            {isTraining && index === 0 && (
              <div className="w-1 h-1 bg-green-400 rounded-full animate-pulse" />
            )}
          </motion.div>
        )
      })}
    </div>
  )
}

export default TrainingMetrics 