import { create } from 'zustand'
import config from '../utils/config'

export interface TrainingData {
  type: string
  timestamp: number
  episode: number
  score: number
  reward: number
  loss: number | null
  policy_loss: number | null
  value_loss: number | null
  entropy: number
  learning_rate: number
  actions: number[]
  board_state: number[][]
  attention_weights: number[][]
  expert_usage: number[]
  gpu_memory: number
  model_params: number
  loss_history: {episodes: number[], values: number[]}
  score_history: {episodes: number[], values: number[]}
  training_speed: number
  avg_game_length: number
  min_game_length: number
  max_game_length: number
  wall_clock_elapsed: number
  estimated_time_to_checkpoint: number
}

export interface CheckpointPlaybackData {
  type: 'checkpoint_playback'
  checkpoint_id: string
  step_data: {
    step: number
    board_state: number[][]
    score: number
    action: number
    action_probs: number[]
    legal_actions: number[]
    attention_weights: number[][] | null
    timestamp: number
    reward: number
    done: boolean
  }
  game_summary: {
    final_score: number
    total_steps: number
    max_tile: number
  }
}

export interface LoadingStates {
  isTrainingStarting: boolean
  isPlaybackStarting: boolean
  isNewGameStarting: boolean
  isTrainingStopping: boolean
  isTrainingResetting: boolean
  loadingMessage: string | null
}

export interface TrainingState {
  // Connection state
  isConnected: boolean
  connectionError: string | null
  
  // Training state
  isTraining: boolean
  isPaused: boolean
  currentEpisode: number
  totalEpisodes: number
  
  // Model configuration
  modelSize: 'small' | 'medium' | 'large'
  
  // Training data
  trainingData: TrainingData | null
  lossHistory: {episodes: number[], values: number[]}
  scoreHistory: {episodes: number[], values: number[]}
  
  // Checkpoint playback data
  checkpointPlaybackData: CheckpointPlaybackData | null
  isPlayingCheckpoint: boolean
  
  // Loading states
  loadingStates: LoadingStates
  
  // Persisted values
  lastPolicyLoss: number | null
  lastValueLoss: number | null
  
  // Actions
  setConnected: (connected: boolean) => void
  setConnectionError: (error: string | null) => void
  setTrainingStatus: (training: boolean, paused?: boolean) => void
  setEpisode: (episode: number) => void
  updateTrainingData: (data: TrainingData) => void
  updateCheckpointPlaybackData: (data: CheckpointPlaybackData) => void
  setPlayingCheckpoint: (playing: boolean) => void
  setModelSize: (size: 'small' | 'medium' | 'large') => void
  setLoadingState: (key: keyof LoadingStates, value: boolean | string | null) => void
  startTraining: () => Promise<void>
  pauseTraining: () => Promise<void>
  resumeTraining: () => Promise<void>
  stopTraining: () => Promise<void>
  resetTraining: () => void
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  // Initial state
  isConnected: false,
  connectionError: null,
  isTraining: false,
  isPaused: false,
  currentEpisode: 0,
  totalEpisodes: 10000,
  modelSize: 'medium',
  trainingData: null,
  lossHistory: {episodes: [], values: []},
  scoreHistory: {episodes: [], values: []},
  
  // Checkpoint playback data
  checkpointPlaybackData: null,
  isPlayingCheckpoint: false,
  
  // Loading states
  loadingStates: {
    isTrainingStarting: false,
    isPlaybackStarting: false,
    isNewGameStarting: false,
    isTrainingStopping: false,
    isTrainingResetting: false,
    loadingMessage: null,
  },
  
  // Persisted values
  lastPolicyLoss: null,
  lastValueLoss: null,
  
  // Actions
  setConnected: (connected) => set({ isConnected: connected }),
  
  setConnectionError: (error) => set({ connectionError: error }),
  
  setTrainingStatus: (training, paused = false) => set({ 
    isTraining: training, 
    isPaused: paused 
  }),
  
  setEpisode: (episode) => set({ currentEpisode: episode }),
  
  updateTrainingData: (data) => {
    const state = get()
    set({
      trainingData: data,
      currentEpisode: data.episode,
      lossHistory: data.loss_history || state.lossHistory,
      scoreHistory: data.score_history || state.scoreHistory,
      // Persist policy and value loss values when they're not null
      lastPolicyLoss: data.policy_loss !== null ? data.policy_loss : state.lastPolicyLoss,
      lastValueLoss: data.value_loss !== null ? data.value_loss : state.lastValueLoss,
      // Clear training start loading state when first data arrives
      loadingStates: {
        ...state.loadingStates,
        isTrainingStarting: false,
        loadingMessage: null
      }
    })
  },
  
  updateCheckpointPlaybackData: (data) => {
    const state = get()
    set({
      checkpointPlaybackData: data,
      isPlayingCheckpoint: true,
      // Clear playback loading state when first data arrives
      loadingStates: {
        ...state.loadingStates,
        isPlaybackStarting: false,
        isNewGameStarting: false,
        loadingMessage: null
      }
    })
  },
  
  setPlayingCheckpoint: (playing) => {
    set({ isPlayingCheckpoint: playing })
  },
  
  setModelSize: (size) => {
    set({ modelSize: size })
  },

  setLoadingState: (key, value) => {
    const currentStates = get().loadingStates
    set({
      loadingStates: {
        ...currentStates,
        [key]: value
      }
    })
  },
  
  startTraining: async () => {
    // Set loading state for training start
    set(prev => ({ 
      loadingStates: { 
        ...prev.loadingStates, 
        isTrainingStarting: true, 
        loadingMessage: 'Initializing model and training environment...' 
      } 
    }))
    
    // Optimistic update for immediate UI feedback
    set({ isTraining: true, isPaused: false })
    
    try {
      const response = await fetch(`${config.api.baseUrl}${config.api.endpoints.training.start}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          model_size: get().modelSize 
        })
      })
      
      if (!response.ok) {
        // Revert on error
        set({ 
          isTraining: false, 
          loadingStates: { 
            ...get().loadingStates, 
            isTrainingStarting: false, 
            loadingMessage: null 
          } 
        })
        throw new Error('Failed to start training')
      }
      
      // Keep loading state active until first training data arrives
      
    } catch (error) {
      console.error('Error starting training:', error)
      set({ 
        connectionError: error instanceof Error ? error.message : 'Failed to start training',
        isTraining: false, // Revert the optimistic update
        loadingStates: { 
          ...get().loadingStates, 
          isTrainingStarting: false, 
          loadingMessage: null 
        }
      })
    }
  },
  
  pauseTraining: async () => {
    // Optimistic update for immediate UI feedback
    set({ isTraining: true, isPaused: true })
    
    try {
      const response = await fetch(`${config.api.baseUrl}${config.api.endpoints.training.pause}`, {
        method: 'POST',
      })
      
      if (!response.ok) {
        // Revert on error
        set({ isPaused: false })
        throw new Error('Failed to pause training')
      }
    } catch (error) {
      console.error('Error pausing training:', error)
      set({ 
        connectionError: error instanceof Error ? error.message : 'Failed to pause training',
        isPaused: false // Revert the optimistic update
      })
    }
  },

  resumeTraining: async () => {
    // Optimistic update for immediate UI feedback
    set({ isTraining: true, isPaused: false })
    
    try {
      const response = await fetch(`${config.api.baseUrl}${config.api.endpoints.training.resume}`, {
        method: 'POST',
      })
      
      if (!response.ok) {
        // Revert on error
        set({ isPaused: true })
        throw new Error('Failed to resume training')
      }
    } catch (error) {
      console.error('Error resuming training:', error)
      set({ 
        connectionError: error instanceof Error ? error.message : 'Failed to resume training',
        isPaused: true // Revert the optimistic update
      })
    }
  },
  
  stopTraining: async () => {
    const previousState = { isTraining: true, isPaused: false }
    
    // Set stopping state for UI feedback
    set(prev => ({ 
      loadingStates: { 
        ...prev.loadingStates, 
        isTrainingStopping: true 
      } 
    }))
    
    // Optimistic update for immediate UI feedback
    set({ isTraining: false, isPaused: false })
    
    try {
      const response = await fetch(`${config.api.baseUrl}${config.api.endpoints.training.stop}`, {
        method: 'POST',
      })
      
      if (!response.ok) {
        // Revert on error
        set(previousState)
        throw new Error('Failed to stop training')
      }
    } catch (error) {
      console.error('Error stopping training:', error)
      set({ 
        connectionError: error instanceof Error ? error.message : 'Failed to stop training',
        ...previousState // Revert the optimistic update
      })
    } finally {
      // Clear stopping state
      set(prev => ({ 
        loadingStates: { 
          ...prev.loadingStates, 
          isTrainingStopping: false 
        } 
      }))
    }
  },
  
  resetTraining: async () => {
    try {
      // Set loading state
      set(state => ({
        loadingStates: {
          ...state.loadingStates,
          isTrainingResetting: true,
          loadingMessage: 'Resetting to fresh model...'
        }
      }))
      
      const response = await fetch(`${config.api.baseUrl}${config.api.endpoints.training.reset}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const result = await response.json()
      console.log('Training reset:', result.message)
      
      // Reset store state
      set({
        isTraining: false,
        isPaused: false,
        currentEpisode: 0,
        trainingData: null,
        lossHistory: {episodes: [], values: []},
        scoreHistory: {episodes: [], values: []},
        lastPolicyLoss: null,
        lastValueLoss: null,
        checkpointPlaybackData: null,
        isPlayingCheckpoint: false,
        loadingStates: {
          isTrainingStarting: false,
          isPlaybackStarting: false,
          isNewGameStarting: false,
          isTrainingStopping: false,
          isTrainingResetting: false,
          loadingMessage: null,
        },
      })
      
      return result
    } catch (error) {
      console.error('Failed to reset training:', error)
      // Clear loading state on error
      set(state => ({
        loadingStates: {
          ...state.loadingStates,
          isTrainingResetting: false,
          loadingMessage: null
        }
      }))
      throw error
    }
  },
})) 