import { create } from 'zustand'
import { persist } from 'zustand/middleware'
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
  // Enhanced metrics
  score_trend: number
  loss_trend: number
  max_tile_frequency: Record<number, number>
  training_efficiency: {
    score_consistency: number
    loss_stability: number
    improvement_rate: number
    plateau_detection: number
    load_balancing_efficiency: number
  }
  load_balancing_reward: number
  // Enhanced load balancing metrics
  expert_starvation_rate: number
  avg_sparsity_score: number
  avg_balance_quality: number
  expert_usage_trend: number
  // NEW: Enhanced expert starvation tracking
  starvation_by_model_size: Record<string, number>
  avg_starvation_severity: number
  expert_recovery_rates: Record<number, number>
  // Animated progress feedback
  is_training_active?: boolean
  next_episode_estimate?: number
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

export interface GameCompletionData {
  final_score: number
  total_steps: number
  max_tile: number
  final_board_state: number[][]
  checkpoint_id: string
  game_number: number
}

export interface LoadingStates {
  isTrainingStarting: boolean
  isPlaybackStarting: boolean
  isNewGameStarting: boolean
  isTrainingStopping: boolean
  isTrainingResetting: boolean
  loadingMessage: string | null
  // Enhanced loading feedback
  loadingProgress: number // 0-100
  loadingStep: string | null // Current step description
  loadingSteps: string[] // All steps for this operation
  currentStepIndex: number // Current step index
  estimatedTimeRemaining: number | null // Estimated seconds remaining
  startTime: number | null // When the operation started
  // Add timeout tracking
  playbackStartTimeout: number | null
  newGameStartTimeout: number | null
  progressInterval: number | null // Interval ID for progress simulation
  // Checkpoint loading states
  isCheckpointLoading: boolean
  checkpointId: string | null
  checkpointLoadingMessage: string | null
  checkpointLoadingProgress: number
  checkpointLoadingError: string | null
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
  modelSize: 'tiny' | 'small' | 'medium' | 'large'
  
  // Training data
  trainingData: TrainingData | null
  lossHistory: {episodes: number[], values: number[]}
  scoreHistory: {episodes: number[], values: number[]}
  
  // Checkpoint playback data
  checkpointPlaybackData: CheckpointPlaybackData | null
  isPlayingCheckpoint: boolean
  
  // Game completion state
  gameCompletionData: GameCompletionData | null
  isShowingGameOver: boolean
  
  // Loading states
  loadingStates: LoadingStates
  
  // Persisted values
  lastPolicyLoss: number | null
  lastValueLoss: number | null
  
  // NEW: Enhanced persistence for training metrics
  lastTrainingData: TrainingData | null  // Store last complete training data for persistence
  lastTrainingTimestamp: number | null   // When the last data was received
  // NEW: Waiting for first data flag
  isWaitingForFirstData: boolean
  // Actions
  setConnected: (connected: boolean) => void
  setConnectionError: (error: string | null) => void
  setTrainingStatus: (training: boolean, paused?: boolean) => void
  setEpisode: (episode: number) => void
  updateTrainingData: (data: TrainingData) => void
  updateCheckpointPlaybackData: (data: CheckpointPlaybackData) => void
  setPlayingCheckpoint: (playing: boolean) => void
  setGameCompletionData: (data: GameCompletionData | null) => void
  setShowingGameOver: (showing: boolean) => void
  setModelSize: (size: 'tiny' | 'small' | 'medium' | 'large') => void
  setLoadingState: (key: keyof LoadingStates, value: boolean | string | null) => void
  startLoadingOperation: (operationType: 'training' | 'playback' | 'newGame' | 'reset', steps: string[]) => void
  updateLoadingProgress: (progress: number, step?: string, estimatedTime?: number) => void
  completeLoadingOperation: () => void
  // Checkpoint loading methods
  setCheckpointLoadingState: (state: { isCheckpointLoading: boolean; checkpointId: string | null; loadingMessage: string | null; loadingProgress: number }) => void
  updateCheckpointLoadingProgress: (progress: number, message?: string) => void
  completeCheckpointLoading: (message?: string) => void
  setCheckpointLoadingError: (error: string) => void
  startTraining: () => Promise<void>
  pauseTraining: () => Promise<void>
  resumeTraining: () => Promise<void>
  stopTraining: () => Promise<void>
  resetTraining: () => void
  // NEW: Helper function to get current training data with fallback
  getCurrentTrainingData: () => TrainingData | null
}

export const useTrainingStore = create<TrainingState>()(
  persist(
    (set, get) => ({
      // Initial state
      isConnected: false,
      connectionError: null,
      isTraining: false,
      isPaused: false,
      currentEpisode: 0,
      totalEpisodes: 10000,
      modelSize: 'tiny',
      trainingData: null,
      lossHistory: {episodes: [], values: []},
      scoreHistory: {episodes: [], values: []},
      
      // Checkpoint playback data
      checkpointPlaybackData: null,
      isPlayingCheckpoint: false,
      
      // Game completion state
      gameCompletionData: null,
      isShowingGameOver: false,
      
      // Loading states
      loadingStates: {
        isTrainingStarting: false,
        isPlaybackStarting: false,
        isNewGameStarting: false,
        isTrainingStopping: false,
        isTrainingResetting: false,
        loadingMessage: null,
        loadingProgress: 0,
        loadingStep: null,
        loadingSteps: [],
        currentStepIndex: 0,
        estimatedTimeRemaining: null,
        startTime: null,
        playbackStartTimeout: null,
        newGameStartTimeout: null,
        progressInterval: null,
        // Checkpoint loading states
        isCheckpointLoading: false,
        checkpointId: null,
        checkpointLoadingMessage: null,
        checkpointLoadingProgress: 0,
        checkpointLoadingError: null,
      },
      
      // Persisted values
      lastPolicyLoss: null,
      lastValueLoss: null,
      
      // NEW: Enhanced persistence
      lastTrainingData: null,
      lastTrainingTimestamp: null,
      // NEW: Waiting for first data
      isWaitingForFirstData: false,
      
      // Actions
      setConnected: (connected) => set({ isConnected: connected }),
      
      setConnectionError: (error) => set({ connectionError: error }),
      
      setTrainingStatus: (training, paused = false) => set({ 
        isTraining: training, 
        isPaused: paused 
      }),
      
      setEpisode: (episode) => set({ currentEpisode: episode }),
      
      // NEW: Helper function to get current training data with fallback
      getCurrentTrainingData: () => {
        const state = get()
        return state.trainingData || state.lastTrainingData
      },

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
          // NEW: Store last complete training data for persistence
          lastTrainingData: data,
          lastTrainingTimestamp: Date.now(),
          // Clear training start loading state when first data arrives
          loadingStates: {
            ...state.loadingStates,
            isTrainingStarting: false,
            loadingMessage: null
          },
          // NEW: Mark that we've received the first data
          isWaitingForFirstData: false
        })
      },
      
      updateCheckpointPlaybackData: (data) => {
        const state = get()
        
        // Clear any existing timeouts
        if (state.loadingStates.playbackStartTimeout) {
          clearTimeout(state.loadingStates.playbackStartTimeout)
        }
        if (state.loadingStates.newGameStartTimeout) {
          clearTimeout(state.loadingStates.newGameStartTimeout)
        }
        
        set({
          checkpointPlaybackData: data,
          isPlayingCheckpoint: true,
          // Clear playback loading state when first data arrives
          loadingStates: {
            ...state.loadingStates,
            isPlaybackStarting: false,
            isNewGameStarting: false,
            loadingMessage: null,
            playbackStartTimeout: null,
            newGameStartTimeout: null,
          }
        })
      },
      
      setPlayingCheckpoint: (playing) => {
        const state = get()
        
        // If stopping playback, clear loading states and timeouts
        if (!playing) {
          if (state.loadingStates.playbackStartTimeout) {
            clearTimeout(state.loadingStates.playbackStartTimeout)
          }
          if (state.loadingStates.newGameStartTimeout) {
            clearTimeout(state.loadingStates.newGameStartTimeout)
          }
          
          set({ 
            isPlayingCheckpoint: playing,
            loadingStates: {
              ...state.loadingStates,
              isPlaybackStarting: false,
              isNewGameStarting: false,
              loadingMessage: null,
              playbackStartTimeout: null,
              newGameStartTimeout: null,
            }
          })
        } else {
          set({ isPlayingCheckpoint: playing })
        }
      },
      
      setGameCompletionData: (data) => set({ gameCompletionData: data }),
      
      setShowingGameOver: (showing) => set({ isShowingGameOver: showing }),
      
      setModelSize: (size) => {
        set({ modelSize: size })
      },

      setLoadingState: (key, value) => {
        const currentStates = get().loadingStates
        const newStates = {
          ...currentStates,
          [key]: value
        }
        
        // Add timeout for playback starting
        if (key === 'isPlaybackStarting' && value === true) {
          // Clear any existing timeout
          if (currentStates.playbackStartTimeout) {
            clearTimeout(currentStates.playbackStartTimeout)
          }
          
          // Set timeout to clear loading state after 30 seconds
          const timeout = setTimeout(() => {
            console.warn('Playback start timeout - clearing loading state and setting error')
            const currentState = get()
            set({
              ...currentState,
              connectionError: 'Playback loading timed out after 30 seconds. This may indicate a server issue or slow connection. Please try again, or check if the backend is running.',
              loadingStates: {
                ...currentState.loadingStates,
                isPlaybackStarting: false,
                loadingMessage: null,
                loadingProgress: 0,
                loadingStep: null,
                loadingSteps: [],
                currentStepIndex: 0,
                estimatedTimeRemaining: null,
                startTime: null,
                playbackStartTimeout: null
              }
            })
          }, 30000)
          
          newStates.playbackStartTimeout = timeout
        }
        
        // Add timeout for new game starting
        if (key === 'isNewGameStarting' && value === true) {
          // Clear any existing timeout
          if (currentStates.newGameStartTimeout) {
            clearTimeout(currentStates.newGameStartTimeout)
          }
          
          // Set timeout to clear loading state after 15 seconds
          const timeout = setTimeout(() => {
            console.warn('New game start timeout - clearing loading state')
            get().setLoadingState('isNewGameStarting', false)
            get().setLoadingState('loadingMessage', null)
            get().setLoadingState('newGameStartTimeout', null)
          }, 15000)
          
          newStates.newGameStartTimeout = timeout
        }
        
        // Clear timeout if manually setting to false
        if (key === 'isPlaybackStarting' && value === false && currentStates.playbackStartTimeout) {
          clearTimeout(currentStates.playbackStartTimeout)
          newStates.playbackStartTimeout = null
        }
        
        if (key === 'isNewGameStarting' && value === false && currentStates.newGameStartTimeout) {
          clearTimeout(currentStates.newGameStartTimeout)
          newStates.newGameStartTimeout = null
        }
        
        set({
          loadingStates: newStates
        })
      },

      // Enhanced loading state management
      startLoadingOperation: (operationType: 'training' | 'playback' | 'newGame' | 'reset', steps: string[]) => {
        const startTime = Date.now()
        const newStates = {
          ...get().loadingStates,
          loadingProgress: 0,
          loadingStep: steps[0] || null,
          loadingSteps: steps,
          currentStepIndex: 0,
          estimatedTimeRemaining: null,
          startTime: startTime,
          [`is${operationType.charAt(0).toUpperCase() + operationType.slice(1)}Starting`]: true,
          loadingMessage: `Starting ${operationType}...`
        }
        
        set({ loadingStates: newStates })
        
        // NEW: More realistic progress simulation for training
        if (operationType === 'training') {
          const progressInterval = setInterval(() => {
            const currentState = get()
            const elapsed = (Date.now() - (currentState.loadingStates.startTime || startTime)) / 1000
            
            // NEW: More realistic progress distribution
            // First 4 steps complete quickly (0-60%), last step takes most time (60-95%)
            let progress = 0
            let estimatedTime = 0
            
            if (elapsed < 2) {
              // Steps 1-4: Complete quickly (0-60% in 2 seconds)
              progress = Math.min(60, (elapsed / 2) * 60)
              estimatedTime = Math.max(0, 8 - elapsed) // 8 seconds remaining for last step
            } else {
              // Step 5: Waiting for first data (60-95% over 8+ seconds)
              const step5Elapsed = elapsed - 2
              progress = Math.min(95, 60 + (step5Elapsed / 8) * 35)
              estimatedTime = Math.max(0, 10 - elapsed)
            }
            
            if (progress < 95) {
              set({
                loadingStates: {
                  ...currentState.loadingStates,
                  loadingProgress: progress,
                  estimatedTimeRemaining: estimatedTime,
                  // Update step based on progress
                  currentStepIndex: progress < 60 ? Math.floor(progress / 15) : 4,
                  loadingStep: progress < 60 ? steps[Math.floor(progress / 15)] : steps[4]
                }
              })
            } else {
              clearInterval(progressInterval)
            }
          }, 200) // Update more frequently for smoother progress
          
          // Store interval ID for cleanup
          newStates.progressInterval = progressInterval
        } else if (operationType === 'playback') {
          // Keep existing playback progress simulation
          const progressInterval = setInterval(() => {
            const currentState = get()
            const elapsed = (Date.now() - (currentState.loadingStates.startTime || startTime)) / 1000
            const progress = Math.min(95, (elapsed / 10) * 100) // Simulate progress over 10 seconds
            
            if (progress < 95) {
              set({
                loadingStates: {
                  ...currentState.loadingStates,
                  loadingProgress: progress,
                  estimatedTimeRemaining: Math.max(0, 10 - elapsed)
                }
              })
            } else {
              clearInterval(progressInterval)
            }
          }, 500)
          
          // Store interval ID for cleanup
          newStates.progressInterval = progressInterval
        }
      },

      updateLoadingProgress: (progress: number, step?: string, estimatedTime?: number) => {
        const currentState = get()
        const newStates = {
          ...currentState.loadingStates,
          loadingProgress: Math.min(100, Math.max(0, progress))
        }
        
        if (step) {
          newStates.loadingStep = step
          const stepIndex = newStates.loadingSteps.indexOf(step)
          if (stepIndex !== -1) {
            newStates.currentStepIndex = stepIndex
          }
        }
        
        if (estimatedTime !== undefined) {
          newStates.estimatedTimeRemaining = estimatedTime
        }
        
        set({ loadingStates: newStates })
      },

      completeLoadingOperation: () => {
        const currentState = get()
        set({
          loadingStates: {
            ...currentState.loadingStates,
            loadingProgress: 100,
            loadingStep: 'Complete',
            estimatedTimeRemaining: 0,
            isTrainingStarting: false,
            isPlaybackStarting: false,
            isNewGameStarting: false,
            isTrainingResetting: false,
            loadingMessage: null
          }
        })
        
        // Clear progress after a short delay
        setTimeout(() => {
          const state = get()
          set({
            loadingStates: {
              ...state.loadingStates,
              loadingProgress: 0,
              loadingStep: null,
              loadingSteps: [],
              currentStepIndex: 0,
              estimatedTimeRemaining: null,
              startTime: null
            }
          })
        }, 1000)
      },

      // Checkpoint loading methods
      setCheckpointLoadingState: (state) => {
        set(prev => ({
          loadingStates: {
            ...prev.loadingStates,
            isCheckpointLoading: state.isCheckpointLoading,
            checkpointId: state.checkpointId,
            checkpointLoadingMessage: state.loadingMessage,
            checkpointLoadingProgress: state.loadingProgress,
            checkpointLoadingError: null,
          }
        }))
      },

      updateCheckpointLoadingProgress: (progress, message) => {
        set(prev => ({
          loadingStates: {
            ...prev.loadingStates,
            checkpointLoadingProgress: progress,
            checkpointLoadingMessage: message || prev.loadingStates.checkpointLoadingMessage,
          }
        }))
      },

      completeCheckpointLoading: (message) => {
        set(prev => ({
          loadingStates: {
            ...prev.loadingStates,
            isCheckpointLoading: false,
            checkpointLoadingMessage: message || 'Checkpoint loaded successfully',
            checkpointLoadingProgress: 100,
            checkpointLoadingError: null,
          }
        }))
        
        // Auto-clear success message after 3 seconds
        setTimeout(() => {
          set(prev => ({
            loadingStates: {
              ...prev.loadingStates,
              checkpointLoadingMessage: null,
              checkpointLoadingProgress: 0,
            }
          }))
        }, 3000)
      },

      setCheckpointLoadingError: (error) => {
        set(prev => ({
          loadingStates: {
            ...prev.loadingStates,
            isCheckpointLoading: false,
            checkpointLoadingError: error,
            checkpointLoadingMessage: null,
            checkpointLoadingProgress: 0,
          }
        }))
        
        // Auto-clear error message after 5 seconds
        setTimeout(() => {
          set(prev => ({
            loadingStates: {
              ...prev.loadingStates,
              checkpointLoadingError: null,
            }
          }))
        }, 5000)
      },
      
      startTraining: async () => {
        // NEW: More realistic training steps with better timing
        const trainingSteps = [
          'Initializing model configuration...',
          'Loading training environment...',
          'Starting GPU/CPU optimization...',
          'Establishing training loop...',
          'Waiting for first training data...'
        ]
        
        get().startLoadingOperation('training', trainingSteps)
        
        // Optimistic update for immediate UI feedback
        set({ isTraining: true, isPaused: false, isWaitingForFirstData: true })
        
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
                loadingMessage: null,
                loadingProgress: 0,
                loadingStep: null,
                loadingSteps: [],
                currentStepIndex: 0,
                estimatedTimeRemaining: null,
                startTime: null
              },
              isWaitingForFirstData: false
            })
            throw new Error('Failed to start training')
          }
          
          // NEW: Progress simulation is now handled by startLoadingOperation
          // The last step (waiting for first data) will continue until data arrives
          
        } catch (error) {
          console.error('Error starting training:', error)
          set({ 
            connectionError: error instanceof Error ? error.message : 'Failed to start training',
            isTraining: false, // Revert the optimistic update
            loadingStates: { 
              ...get().loadingStates, 
              isTrainingStarting: false, 
              loadingMessage: null,
              loadingProgress: 0,
              loadingStep: null,
              loadingSteps: [],
              currentStepIndex: 0,
              estimatedTimeRemaining: null,
              startTime: null
            },
            isWaitingForFirstData: false
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
            // NEW: Also reset persistence data
            lastTrainingData: null,
            lastTrainingTimestamp: null,
            isWaitingForFirstData: false,
            checkpointPlaybackData: null,
            isPlayingCheckpoint: false,
            loadingStates: {
              isTrainingStarting: false,
              isPlaybackStarting: false,
              isNewGameStarting: false,
              isTrainingStopping: false,
              isTrainingResetting: false,
              loadingMessage: null,
              loadingProgress: 0,
              loadingStep: null,
              loadingSteps: [],
              currentStepIndex: 0,
              estimatedTimeRemaining: null,
              startTime: null,
              playbackStartTimeout: null,
              newGameStartTimeout: null,
              progressInterval: null,
              // Checkpoint loading states
              isCheckpointLoading: false,
              checkpointId: null,
              checkpointLoadingMessage: null,
              checkpointLoadingProgress: 0,
              checkpointLoadingError: null,
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
            },
            isWaitingForFirstData: false
          }))
          throw error
        }
      },
    }),
    {
      name: 'training-store',
      partialize: (state) => ({
        // Persist only important state across page refreshes
        isTraining: state.isTraining,
        isPaused: state.isPaused,
        currentEpisode: state.currentEpisode,
        totalEpisodes: state.totalEpisodes,
        modelSize: state.modelSize,
        isPlayingCheckpoint: state.isPlayingCheckpoint,
        lastPolicyLoss: state.lastPolicyLoss,
        lastValueLoss: state.lastValueLoss,
        // NEW: Persist current training data directly
        trainingData: state.trainingData,
        // NEW: Persist last training data for reconnection
        lastTrainingData: state.lastTrainingData,
        lastTrainingTimestamp: state.lastTrainingTimestamp,
        isWaitingForFirstData: state.isWaitingForFirstData,
        // Persist recent history (last 100 points)
        lossHistory: {
          episodes: state.lossHistory.episodes.slice(-100),
          values: state.lossHistory.values.slice(-100)
        },
        scoreHistory: {
          episodes: state.scoreHistory.episodes.slice(-100),
          values: state.scoreHistory.values.slice(-100)
        }
      }),
      // NEW: Add onRehydrateStorage to restore trainingData from lastTrainingData if needed
      onRehydrateStorage: () => (state) => {
        if (state) {
          // If trainingData is null but we have lastTrainingData, restore it
          if (!state.trainingData && state.lastTrainingData) {
            state.trainingData = state.lastTrainingData
          }
          
          // If we have training data but no episode info, set it from the data
          if (state.trainingData && state.currentEpisode === 0) {
            state.currentEpisode = state.trainingData.episode
          }
        }
      },
      // Don't persist connection state and loading states as they should be reset on page load
      skipHydration: false,
    }
  )
) 