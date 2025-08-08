import React, { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, Cpu, Gauge, HardDrive, Pause, Play, Power, RotateCw, Wifi, Settings, Zap, Clock } from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'
import config from '../utils/config'

type HealthReport = {
  status: string
  degradation_mode: string
  metrics: {
    cpu_percent: number
    memory_percent: number
    memory_available_gb: number
    gpu_memory_percent: number
    gpu_utilization: number
    disk_usage_percent: number
    load_average: number
    model_inference_time: number
    broadcast_success_rate: number
    error_rate: number
  }
  degradation_settings: {
    max_fps: number
    enable_visualization: boolean
    enable_attention_weights: boolean
    model_timeout: number
    board_update_interval: number
  }
  emergency_mode: boolean
  last_updated: string
  history_size: number
}

const formatPct = (v: number | undefined) =>
  typeof v === 'number' && isFinite(v) ? `${Math.round(v)}%` : '—'

const ControlsDashboard: React.FC = () => {
  const {
    isTraining,
    isPaused,
    startTraining,
    pauseTraining,
    resumeTraining,
    stopTraining,
    resetTraining,
    setModelSize,
    modelSize,
    isConnected,
    loadingStates,
    trainingData,
  } = useTrainingStore()

  const [health, setHealth] = useState<HealthReport | null>(null)

  // Poll system health lightly
  useEffect(() => {
    let mounted = true
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${config.api.baseUrl}${config.api.endpoints.system.health}`)
        if (!res.ok) return
        const data: HealthReport = await res.json()
        if (mounted) setHealth(data)
      } catch {}
    }
    fetchHealth()
    const id = setInterval(fetchHealth, 3000)
    return () => {
      mounted = false
      clearInterval(id)
    }
  }, [])

  const statusPill = useMemo(() => {
    const label = isTraining ? (isPaused ? 'Paused' : 'Active') : 'Stopped'
    const color = isTraining ? (isPaused ? 'bg-ui-warning' : 'bg-ui-success') : 'bg-ui-text-secondary/20'
    return (
      <span className={`px-2 py-0.5 rounded-full text-[11px] border border-ui-border-muted ${color} text-ui-text-primary/90`}>{label}</span>
    )
  }, [isTraining, isPaused])

  const formatTime = (seconds?: number) => {
    if (!seconds || !isFinite(seconds)) return '—'
    const s = Math.max(0, Math.floor(seconds))
    const h = Math.floor(s / 3600)
    const m = Math.floor((s % 3600) / 60)
    const sec = s % 60
    const pad = (n: number) => n.toString().padStart(2, '0')
    return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`
  }

  const gpuMemPercent = health?.metrics.gpu_memory_percent
  const gpuUsedGb = trainingData?.gpu_memory
  const approxTotalVram = gpuMemPercent && gpuUsedGb && gpuMemPercent > 0 ? (gpuUsedGb / (gpuMemPercent / 100)) : undefined
  const vramAvailableGb = approxTotalVram && gpuUsedGb ? Math.max(0, approxTotalVram - gpuUsedGb) : undefined
  const cudaEnabled = (gpuMemPercent ?? 0) > 0 || (gpuUsedGb ?? 0) > 0
  const gpuUtil = health?.metrics.gpu_utilization
  const modelParams = trainingData?.model_params

  const formatGb = (v?: number) => (typeof v === 'number' && isFinite(v) ? `${v.toFixed(1)}GB` : '—')
  const formatLarge = (v?: number) => {
    if (typeof v !== 'number' || !isFinite(v)) return '—'
    if (v >= 1e9) return `${(v / 1e9).toFixed(1)}B`
    if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`
    if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`
    return `${v}`
  }

  return (
    <div className="safe-area h-full min-h-0 grid grid-rows-[auto_auto] px-4 overflow-hidden gap-2" id="controls">
      {/* Header & Controls row */}
      <motion.div className="card-glass p-3 rounded-2xl flex-shrink-0">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between mb-2">
          <h2 className="text-sm font-semibold flex items-center gap-2">
            <Settings className="w-4 h-4 text-ui-brand-primary" /> Control Center
          </h2>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2">
            <div className={isTraining && !isPaused ? 'status-pulse rounded-full p-0.5' : ''}>{statusPill}</div>
            <span className="text-[12px] text-ui-text-secondary flex items-center gap-1">
              <Wifi className={`w-3.5 h-3.5 ${isConnected ? 'text-ui-success' : 'text-ui-danger'}`} />
              {isConnected ? 'Online' : 'Offline'}
            </span>
          </div>
          {/* Primary controls */}
          <div className="grid grid-cols-3 gap-2 w-full items-stretch justify-items-stretch sm:max-w-[640px] sm:mx-auto">
            {!isTraining && (
              <button className={`h-11 px-4 rounded-xl btn-solid-success font-medium flex items-center justify-center w-full shadow-glass transition-all hover:shadow-glass-lg hover:translate-y-[1px] ring-1 ring-ui-border-muted hover:ring-ui-brand-primary/40 active:scale-[0.99] ${loadingStates.isTrainingStarting ? 'shimmer' : ''}`} onClick={() => startTraining()}>
                <Play className="w-4 h-4 mr-2" /> Start
              </button>
            )}
            {isTraining && !isPaused && (
              <button className="h-11 px-4 rounded-xl btn-solid-warning font-medium flex items-center justify-center w-full shadow-glass transition-all hover:shadow-glass-lg hover:translate-y-[1px] ring-1 ring-ui-border-muted hover:ring-ui-state-warning/40 active:scale-[0.99]" onClick={() => pauseTraining()}>
                <Pause className="w-4 h-4 mr-2" /> Pause
              </button>
            )}
            {isTraining && isPaused && (
              <button className="h-11 px-4 rounded-xl btn-solid-success font-medium flex items-center justify-center w-full shadow-glass transition-all hover:shadow-glass-lg hover:translate-y-[1px] ring-1 ring-ui-border-muted hover:ring-ui-state-success/40 active:scale-[0.99]" onClick={() => resumeTraining()}>
                <Play className="w-4 h-4 mr-2" /> Resume
              </button>
            )}
            <button className="h-11 px-4 rounded-xl btn-solid-danger font-medium flex items-center justify-center w-full shadow-glass transition-all hover:shadow-glass-lg hover:translate-y-[1px] ring-1 ring-ui-border-muted hover:ring-ui-state-danger/40 active:scale-[0.99]" onClick={() => stopTraining()}>
              <Power className="w-4 h-4 mr-2" /> Stop
            </button>
            <button className="h-11 px-4 rounded-xl btn-outline-neutral font-medium flex items-center justify-center w-full transition-all hover:translate-y-[1px] ring-1 ring-ui-border-muted hover:ring-ui-brand-primary/30 active:scale-[0.99]" onClick={() => resetTraining()}>
              <RotateCw className="w-4 h-4 mr-2" /> Reset
            </button>
          </div>
        </div>

        {/* Model profile segmented (enhanced) */}
        <div className="mt-2 grid grid-cols-3 gap-2 w-full items-stretch justify-items-stretch sm:max-w-[640px] sm:mx-auto" role="tablist" aria-label="Model profile">
          {(['lightning', 'base', 'expert'] as const).map(p => (
            <button
              key={p}
              role="tab"
              aria-selected={modelSize === p}
              className={`h-11 rounded-xl text-sm font-medium w-full border transition-all shadow-glass hover:shadow-glass-lg hover:translate-y-[1px] ring-1 ${
                modelSize === p
                  ? 'bg-token-brand-20 text-ui-text-primary border-token-brand-50 ring-ui-brand-primary ring-pulse-brand'
                  : 'bg-token-surface-50 text-ui-text-secondary border-ui-border-muted hover:ring-ui-brand-primary/30'
              }`}
              onClick={() => setModelSize(p)}
            >
              {p.charAt(0).toUpperCase() + p.slice(1)}
            </button>
          ))}
        </div>
      </motion.div>

      {/* Compact system health */}
      <motion.div
        className="card-glass p-3 rounded-2xl flex-shrink-0"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
      >
        <div className="flex items-center mb-2">
          <h3 className="text-sm font-semibold flex items-center gap-2">
            <Activity className="w-4 h-4 text-ui-info" /> Live System Health
          </h3>
        </div>
        {/* KPI row (system) */}
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-xl border border-ui-border-muted p-2 bg-token-surface-70">
            <div className="text-[11px] text-ui-text-secondary mb-1 flex items-center gap-1"><Cpu className="w-3.5 h-3.5"/> CPU</div>
            <div className="text-lg font-semibold numeric">{formatPct(health?.metrics.cpu_percent)}</div>
          </div>
          <div className="rounded-xl border border-ui-border-muted p-2 bg-token-surface-70">
            <div className="text-[11px] text-ui-text-secondary mb-1 flex items-center gap-1"><HardDrive className="w-3.5 h-3.5"/> Memory</div>
            <div className="text-lg font-semibold numeric">{formatPct(health?.metrics.memory_percent)}</div>
          </div>
        </div>

        {/* GPU metrics (enhanced) */}
        <div className="mt-2 grid grid-cols-2 sm:grid-cols-3 gap-2">
          <div className="rounded-xl border border-ui-border-muted p-2 bg-token-surface-70">
            <div className="text-[11px] text-ui-text-secondary mb-1 flex items-center gap-1"><Activity className="w-3.5 h-3.5"/> GPU Util</div>
            <div className="text-lg font-semibold numeric">{formatPct(gpuUtil)}</div>
          </div>
          <div className="rounded-xl border border-ui-border-muted p-2 bg-token-surface-70">
            <div className="text-[11px] text-ui-text-secondary mb-1 flex items-center gap-1"><Gauge className="w-3.5 h-3.5"/> VRAM %</div>
            <div className="text-lg font-semibold numeric">{formatPct(gpuMemPercent)}</div>
          </div>
          <div className="rounded-xl border border-ui-border-muted p-2 bg-token-surface-70">
            <div className="text-[11px] text-ui-text-secondary mb-1 flex items-center gap-1"><Gauge className="w-3.5 h-3.5"/> VRAM Used</div>
            <div className="text-lg font-semibold numeric">{formatGb(gpuUsedGb)}</div>
          </div>
          <div className="rounded-xl border border-ui-border-muted p-2 bg-token-surface-70">
            <div className="text-[11px] text-ui-text-secondary mb-1 flex items-center gap-1"><Gauge className="w-3.5 h-3.5"/> VRAM Free</div>
            <div className="text-lg font-semibold numeric">{formatGb(vramAvailableGb)}</div>
          </div>
          <div className="rounded-xl border border-ui-border-muted p-2 bg-token-surface-70">
            <div className="text-[11px] text-ui-text-secondary mb-1 flex items-center gap-1"><Zap className="w-3.5 h-3.5"/> CUDA</div>
            <div className={`text-lg font-semibold ${cudaEnabled ? 'text-ui-success' : 'text-ui-text-secondary'}`}>{cudaEnabled ? 'On' : 'Off'}</div>
          </div>
          <div className="rounded-xl border border-ui-border-muted p-2 bg-token-surface-70">
            <div className="text-[11px] text-ui-text-secondary mb-1">Params</div>
            <div className="text-lg font-semibold numeric">{formatLarge(modelParams)}</div>
          </div>
        </div>

        {/* Mini strip */}
        <div className="mt-2 rounded-xl border border-ui-border-muted p-2 bg-token-surface-50 text-[12px] text-ui-text-secondary flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="flex items-center gap-1" title="Inference time">
            <Clock className="w-3.5 h-3.5" />
            <span className="numeric text-ui-text-primary">{health ? `${Math.round(health.metrics.model_inference_time * 1000)}ms` : '—'}</span>
          </span>
          <span className="flex items-center gap-1" title="Broadcast success">
            <span>OK</span>
            <span className="numeric text-ui-text-primary">{health ? `${Math.round(health.metrics.broadcast_success_rate)}%` : '—'}</span>
          </span>
          <span className="flex items-center gap-1" title="Error rate">
            <span>Err</span>
            <span className="numeric text-ui-text-primary">{health ? `${Math.round(health.metrics.error_rate)}%` : '—'}</span>
          </span>
          <span className="flex items-center gap-1 ml-auto" title="Training runtime">
            <span className="numeric text-ui-text-primary">{formatTime(trainingData?.wall_clock_elapsed)}</span>
          </span>
        </div>
      </motion.div>
    </div>
  )
}

export default ControlsDashboard

