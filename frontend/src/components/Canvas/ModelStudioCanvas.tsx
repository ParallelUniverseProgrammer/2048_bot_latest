import React, { useRef, useState, useCallback, useEffect, useMemo } from 'react'
import { Stage, Layer, Line } from 'react-konva'
import { snap, clamp, GRID, ZOOM_MIN, ZOOM_MAX } from '../../utils/geometry'
import ModelStudioBlock from './ModelStudioBlock'
import ConnectionLayer from './ConnectionLayer'
import { useDesignStore } from '../../stores/designStore'

interface InnerStageProps {
  stageRef: React.RefObject<any>
}

const InnerStage: React.FC<InnerStageProps> = ({ stageRef }) => {
  const [stageSize, setStageSize] = useState({ w: 800, h: 600 })
  const [selectedBlock, setSelectedBlock] = useState<string | null>(null)
  const [isDraggingBlock, setIsDraggingBlock] = useState(false)
  const [viewState, setViewState] = useState({
    scale: 0.5, // More zoomed out default
    pos: { x: 0, y: 0 }
  })
  
  const raf = useRef<number>()
  const containerRef = useRef<HTMLDivElement>(null)
  
  const {
    currentDesign,
    updateComponent
  } = useDesignStore()

  // Update stage size based on container
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        // Use container size with some padding
        setStageSize({
          w: Math.max(rect.width - 20, 400), // Min width 400px
          h: Math.max(rect.height - 20, 300)  // Min height 300px
        })
      }
    }
    
    updateSize()
    window.addEventListener('resize', updateSize)
    
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  // Handle block movement with grid snapping
  const handleBlockMove = useCallback((id: string, x: number, y: number) => {
    updateComponent(id, { position: { x: snap(x), y: snap(y) } })
  }, [updateComponent])

  // Handle block selection
  const handleBlockSelect = useCallback((id: string) => {
    setSelectedBlock(id === selectedBlock ? null : id)
  }, [selectedBlock])

  // Handle block drag start
  const handleBlockDragStart = useCallback(() => {
    setIsDraggingBlock(true)
  }, [])

  // Handle block drag end
  const handleBlockDragEnd = useCallback(() => {
    setIsDraggingBlock(false)
  }, [])

  // Auto-position new components at center of current view
  useEffect(() => {
    if (currentDesign?.components) {
      const newComponents = currentDesign.components.filter(comp => 
        comp.position?.x === 0 && comp.position?.y === 0
      )
      
      if (newComponents.length > 0) {
        // Calculate center of current view in world coordinates
        // In Konva: stage position is the offset of world origin from viewport top-left
        // So to get the world coordinates at the center of the viewport:
        // 1. Start at the world origin (0,0) 
        // 2. Add the stage position (which moves the world origin)
        // 3. Add half the viewport size (scaled to world coordinates)
        const visibleCenterX = viewState.pos.x + (stageSize.w / 2) / viewState.scale
        const visibleCenterY = viewState.pos.y + (stageSize.h / 2) / viewState.scale
        
        // Snap to grid
        const snappedX = snap(visibleCenterX)
        const snappedY = snap(visibleCenterY)
        
        // Update all new components to center position
        newComponents.forEach(comp => {
          updateComponent(comp.id, { 
            position: { x: snappedX, y: snappedY } 
          })
        })
      }
    }
  }, [currentDesign?.components, viewState, stageSize, updateComponent])

  // Handle wheel zoom with throttling
  const handleWheel = useCallback((e: any) => {
    if (isDraggingBlock) return // Don't zoom when dragging blocks
    e.evt.preventDefault()
    const by = 1.03
    const next = e.evt.deltaY > 0 ? viewState.scale / by : viewState.scale * by
    setViewState(v => ({ ...v, scale: clamp(next, ZOOM_MIN, ZOOM_MAX) }))
  }, [viewState.scale, isDraggingBlock])

  // Handle stage drag with throttling and bounds checking
  const handleDragMove = useCallback((e: any) => {
    if (isDraggingBlock) return // Don't move stage when dragging blocks
    
    cancelAnimationFrame(raf.current!)
    raf.current = requestAnimationFrame(() => {
      const newX = e.target.x()
      const newY = e.target.y()
      
      // Constrain stage position to prevent scrolling outside grid bounds
      const maxX = 0
      const minX = -(stageSize.w * viewState.scale - stageSize.w)
      const maxY = 0
      const minY = -(stageSize.h * viewState.scale - stageSize.h)
      
      const constrainedX = Math.max(minX, Math.min(maxX, newX))
      const constrainedY = Math.max(minY, Math.min(maxY, newY))
      
      setViewState(v => ({
        ...v,
        pos: { x: constrainedX, y: constrainedY }
      }))
    })
  }, [isDraggingBlock, stageSize, viewState.scale])

  // Handle stage drag start - prevent when dragging blocks
  const handleDragStart = useCallback((e: any) => {
    if (isDraggingBlock) {
      e.evt.preventDefault()
      return
    }
  }, [isDraggingBlock])

  // Handle stage drag end
  const handleDragEnd = useCallback((e: any) => {
    if (isDraggingBlock) {
      e.evt.preventDefault()
      return
    }
  }, [isDraggingBlock])

  // Memoize grid lines to prevent re-rendering
  const gridLines = useMemo(() => {
    const verticalLines = Array.from({ length: Math.ceil(stageSize.w / GRID) }).map((_, i) => (
      <Line
        key={`v-${i}`}
        points={[i * GRID, 0, i * GRID, stageSize.h]}
        stroke="#5555"
        strokeWidth={i % 4 === 0 ? 1 : 0.5}
        listening={false}
      />
    ))
    
    const horizontalLines = Array.from({ length: Math.ceil(stageSize.h / GRID) }).map((_, i) => (
      <Line
        key={`h-${i}`}
        points={[0, i * GRID, stageSize.w, i * GRID]}
        stroke="#5555"
        strokeWidth={i % 4 === 0 ? 1 : 0.5}
        listening={false}
      />
    ))
    
    return [...verticalLines, ...horizontalLines]
  }, [stageSize.w, stageSize.h])

  // Memoize components to prevent re-rendering during drag
  const components = useMemo(() => {
    if (!currentDesign?.components) return []
    
    return currentDesign.components.map(component => (
      <ModelStudioBlock
        key={component.id}
        id={component.id}
        type={component.type}
        x={component.position?.x || 0}
        y={component.position?.y || 0}
        selected={selectedBlock === component.id}
        onMove={handleBlockMove}
        onSelect={handleBlockSelect}
        onDragStart={handleBlockDragStart}
        onDragEnd={handleBlockDragEnd}
      />
    ))
  }, [currentDesign?.components, selectedBlock, handleBlockMove, handleBlockSelect, handleBlockDragStart, handleBlockDragEnd])

  if (!currentDesign) {
    return (
      <div className="h-full flex items-center justify-center text-gray-400">
        <p>Create a design to start building</p>
      </div>
    )
  }

  return (
    <div className="h-full w-full">
      <Stage
        width={stageSize.w}
        height={stageSize.h}
        ref={stageRef}
        draggable={!isDraggingBlock} // Disable stage dragging when dragging blocks
        x={viewState.pos.x}
        y={viewState.pos.y}
        scaleX={viewState.scale}
        scaleY={viewState.scale}
        className="touch-none"
        onDragMove={handleDragMove}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
        onWheel={handleWheel}
        listening={!isDraggingBlock} // Disable all stage interactions when dragging blocks
      >
        {/* Grid overlay */}
        <Layer listening={false}>
          {gridLines}
        </Layer>

        {/* Connection layer */}
        <ConnectionLayer 
          edges={currentDesign.edges}
          components={currentDesign.components}
        />

        {/* Blocks layer */}
        <Layer listening={true}>
          {components}
        </Layer>
      </Stage>
    </div>
  )
}

const ModelStudioCanvas: React.FC = () => {
  const stageRef = useRef<any>(null)

  return (
    <div className="h-full w-full flex items-center justify-center">
      <InnerStage stageRef={stageRef} />
    </div>
  )
}

export default ModelStudioCanvas 