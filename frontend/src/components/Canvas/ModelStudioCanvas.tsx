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
        // Since viewState.pos is (0,0), the world origin is at the viewport top-left
        // The center of the viewport in world coordinates is:
        const visibleCenterX = (stageSize.w / 2) / viewState.scale
        const visibleCenterY = (stageSize.h / 2) / viewState.scale
        
        // Snap to grid
        const snappedX = snap(visibleCenterX)
        const snappedY = snap(visibleCenterY)
        
        // Update all new components to center position with small offsets to prevent stacking
        newComponents.forEach((comp, index) => {
          // Add a small offset based on index to prevent components from stacking
          const offsetX = index * GRID * 2 // 2 grid units apart
          const offsetY = index * GRID * 2
          
          updateComponent(comp.id, { 
            position: { 
              x: snap(snappedX + offsetX), 
              y: snap(snappedY + offsetY) 
            } 
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
        draggable={false} // Disable stage dragging completely
        x={viewState.pos.x}
        y={viewState.pos.y}
        scaleX={viewState.scale}
        scaleY={viewState.scale}
        className="touch-none"
        onWheel={handleWheel}
        listening={true} // Keep stage listening for wheel events
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