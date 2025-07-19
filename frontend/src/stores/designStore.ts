import { create } from 'zustand'

export interface ModelComponent {
  id: string
  type: string
  props: Record<string, number | string | boolean>
  position?: { x: number; y: number }
}

export interface ModelArchitecture {
  id: string
  name: string
  components: ModelComponent[]
  edges: Array<[string, string]>
  meta?: {
    d_model?: number
    n_heads?: number
    n_experts?: number
  }
}

export interface ValidationResult {
  valid: boolean
  paramCount: number // millions
  estimatedMemory: number // MB
  errors: string[]
}

export interface CompileResult {
  import_path: string
  paramCount: number
  config: Record<string, any>
}

export interface DesignStore {
  // State
  currentDesign: ModelArchitecture | null
  validation: ValidationResult | null
  paramCount: number
  estimatedMemory: number
  isCompiling: boolean
  isTraining: boolean
  compileResult: CompileResult | null
  
  // Actions
  setDesign: (design: ModelArchitecture | null) => void
  setValidation: (validation: ValidationResult | null) => void
  setCompileResult: (result: CompileResult | null) => void
  setIsCompiling: (compiling: boolean) => void
  setIsTraining: (training: boolean) => void
  updateComponent: (id: string, updates: Partial<ModelComponent>) => void
  addComponent: (component: ModelComponent) => void
  removeComponent: (id: string) => void
  addEdge: (from: string, to: string) => void
  removeEdge: (from: string, to: string) => void
  resetDesign: () => void
}

export const useDesignStore = create<DesignStore>((set, get) => ({
  // Initial state
  currentDesign: null,
  validation: null,
  paramCount: 0,
  estimatedMemory: 0,
  isCompiling: false,
  isTraining: false,
  compileResult: null,
  
  // Actions
  setDesign: (design) => set({ currentDesign: design }),
  
  setValidation: (validation) => set({ 
    validation,
    paramCount: validation?.paramCount || 0,
    estimatedMemory: validation?.estimatedMemory || 0
  }),
  
  setCompileResult: (result) => set({ compileResult: result }),
  
  setIsCompiling: (compiling) => set({ isCompiling: compiling }),
  
  setIsTraining: (training) => set({ isTraining: training }),
  
  updateComponent: (id, updates) => {
    const { currentDesign } = get()
    if (!currentDesign) return
    
    set({
      currentDesign: {
        ...currentDesign,
        components: currentDesign.components.map(comp =>
          comp.id === id ? { ...comp, ...updates } : comp
        )
      }
    })
  },
  
  addComponent: (component) => {
    const { currentDesign } = get()
    if (!currentDesign) return
    
    set({
      currentDesign: {
        ...currentDesign,
        components: [...currentDesign.components, component]
      }
    })
  },
  
  removeComponent: (id) => {
    const { currentDesign } = get()
    if (!currentDesign) return
    
    set({
      currentDesign: {
        ...currentDesign,
        components: currentDesign.components.filter(comp => comp.id !== id),
        edges: currentDesign.edges.filter(([from, to]) => from !== id && to !== id)
      }
    })
  },
  
  addEdge: (from, to) => {
    const { currentDesign } = get()
    if (!currentDesign) return
    
    const edgeExists = currentDesign.edges.some(([f, t]) => f === from && t === to)
    if (edgeExists) return
    
    set({
      currentDesign: {
        ...currentDesign,
        edges: [...currentDesign.edges, [from, to]]
      }
    })
  },
  
  removeEdge: (from, to) => {
    const { currentDesign } = get()
    if (!currentDesign) return
    
    set({
      currentDesign: {
        ...currentDesign,
        edges: currentDesign.edges.filter(([f, t]) => !(f === from && t === to))
      }
    })
  },
  
  resetDesign: () => set({
    currentDesign: null,
    validation: null,
    paramCount: 0,
    estimatedMemory: 0,
    compileResult: null,
    isCompiling: false,
    isTraining: false
  })
}))

// Subscribe to design changes for validation
useDesignStore.subscribe(
  (state) => {
    const design = state.currentDesign
    if (design) {
      // Trigger validation in web worker
      if ((window as any).designWorker) {
        (window as any).designWorker.postMessage({ type: 'validate', design })
      }
    }
  }
) 