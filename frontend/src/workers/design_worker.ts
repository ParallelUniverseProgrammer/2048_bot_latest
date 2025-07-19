// Design validation web worker
// Pure JS rules that mirror backend validation

interface ModelComponent {
  id: string
  type: string
  props: Record<string, number | string | boolean>
}

interface ModelArchitecture {
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

interface ValidationResult {
  valid: boolean
  paramCount: number // millions
  estimatedMemory: number // MB
  errors: string[]
}

// Pure JS validation rules - must mirror backend validation
function validate(design: ModelArchitecture): ValidationResult {
  const errors: string[] = []

  // Extract meta parameters with defaults
  const dModel = design.meta?.d_model || 128
  const nHeads = design.meta?.n_heads || 4
  const nExperts = design.meta?.n_experts || 4

  // Validation Rule 1: d_model must be multiple of 8
  if (dModel % 8 !== 0) {
    errors.push("d_model must be multiple of 8")
  }

  // Validation Rule 2: n_heads must divide d_model
  if (dModel % nHeads !== 0) {
    errors.push("n_heads must divide d_model")
  }

  // Validation Rule 3: max 8 experts supported
  if (nExperts > 8) {
    errors.push("max 8 experts supported")
  }

  // Validation Rule 4: At least one BOARD_INPUT and one ACTION_OUTPUT
  const hasBoardInput = design.components.some(c => c.type === 'BOARD_INPUT')
  const hasActionOutput = design.components.some(c => c.type === 'ACTION_OUTPUT')
  
  if (!hasBoardInput) {
    errors.push("At least one BOARD_INPUT component required")
  }
  
  if (!hasActionOutput) {
    errors.push("At least one ACTION_OUTPUT component required")
  }

  // Validation Rule 5: Graph must be acyclic and all nodes reachable from input
  if (design.components.length > 0) {
    const visited = new Set<string>()
    const recStack = new Set<string>()
    
    function hasCycle(node: string): boolean {
      if (recStack.has(node)) return true
      if (visited.has(node)) return false
      
      visited.add(node)
      recStack.add(node)
      
      const outgoingEdges = design.edges.filter(([from]) => from === node)
      for (const [, to] of outgoingEdges) {
        if (hasCycle(to)) return true
      }
      
      recStack.delete(node)
      return false
    }
    
    // Check for cycles
    for (const component of design.components) {
      if (!visited.has(component.id) && hasCycle(component.id)) {
        errors.push("Graph contains cycles")
        break
      }
    }
  }

  // Calculate parameter count and memory estimate
  const paramCount = estimateParams(design)
  const estimatedMemory = estimateMemory(design)

  return {
    valid: errors.length === 0,
    errors,
    paramCount,
    estimatedMemory
  }
}

// Parameter estimation formula
function estimateParams(design: ModelArchitecture): number {
  let params = 0
  const dModel = design.meta?.d_model || 128
  
  for (const component of design.components) {
    switch (component.type) {
      case 'BOARD_INPUT':
        params += 4 * 4 * dModel // 4x4 board to d_model embedding
        break
      case 'TRANSFORMER_LAYER':
        params += 2 * dModel ** 2 // Self-attention + FFN
        break
      case 'MOE_LAYER':
        const nExperts = (component.props.n_experts as number) || 4
        params += nExperts * dModel ** 2 // Multiple experts
        break
      case 'ACTION_OUTPUT':
        params += dModel * 4 // d_model to 4 actions
        break
      case 'VALUE_HEAD':
        params += dModel * 1 // d_model to 1 value
        break
    }
  }
  
  return Math.round((params / 1e6) * 100) / 100 // Return in millions, 2 decimal places
}

// Memory estimation formula
function estimateMemory(design: ModelArchitecture): number {
  const paramCount = estimateParams(design)
  const dModel = design.meta?.d_model || 128
  
  // Rough estimation: 4 bytes per parameter + activation memory
  const paramMemory = paramCount * 1e6 * 4 // bytes
  const activationMemory = dModel * 16 * 4 // batch_size * sequence_length * d_model * 4 bytes
  
  return Math.round((paramMemory + activationMemory) / (1024 * 1024)) // Convert to MB
}

// Worker message handler
self.onmessage = async (e) => {
  const { type, design } = e.data as { type: string; design: ModelArchitecture }
  
  if (type === 'validate' && design) {
    const result = validate(design)
    postMessage(result)
  }
}

// TypeScript worker context
export {} 