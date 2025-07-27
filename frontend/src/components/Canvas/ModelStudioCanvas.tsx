import React, { useRef, useState, useCallback } from 'react'
import { Stage, Layer, Line } from 'react-konva'
import { DndProvider, useDrop } from 'react-dnd'
import { TouchBackend } from 'react-dnd-touch-backend'
import { useStageSize } from '../../hooks/useStageSize'
import { snap, clamp, GRID, ZOOM_MIN, ZOOM_MAX } from '../../utils/geometry'
import ModelStudioBlock from './ModelStudioBlock'
import ConnectionLayer from './ConnectionLayer'
import { useDesignStore, ModelComponent } from '../../stores/designStore'

interface InnerStageProps {
  stageRef: React.RefObject<any>
}

const InnerStage: React.FC<InnerStageProps> = ({ stageRef }) => {
  const { w, h } = useStageSize()
  const [selectedBlock, setSelectedBlock] = useState<string | null>(null)
  const [viewState, setViewState] = useState({
    scale: 1,
    pos: { x: 0, y: 0 }
  })
  
  const lastDist = useRef(0)
  const raf = useRef<number>()
  
  const {
    currentDesign,
    addComponent,
    updateComponent
  } = useDesignStore()

  // DnD drop handler
  useDrop(
    () => ({
      accept: 'BLOCK',
      drop: (item: { type: string }) => {
        if (!stageRef.current) return
        
        const pointerPos = stageRef.current.getPointerPosition()
        if (!pointerPos) return
        
        const transform = stageRef.current.getAbsoluteTransform().invert()
        const pos = transform.point(pointerPos)
        
        const newComponent: ModelComponent = {
          id: `${item.type}_${Date.now()}`,
          type: item.type,
          props: getDefaultProps(item.type),
          position: { x: snap(pos.x), y: snap(pos.y) }
        }
        
        addComponent(newComponent)
      }
    }),
    [addComponent]
  )

  // Get default props for component type
  const getDefaultProps = (type: string): Record<string, number | string | boolean> => {
    switch (type) {
      case 'TRANSFORMER_LAYER':
        return { d_model: 128, n_heads: 4 }
      case 'MOE_LAYER':
        return { d_model: 128, n_experts: 4 }
      case 'BOARD_INPUT':
        return { d_model: 128 }
      case 'ACTION_OUTPUT':
        return { d_model: 128 }
      case 'VALUE_HEAD':
        return { d_model: 128 }
      default:
        return {}
    }
  }

  // Handle block movement with grid snapping
  const handleBlockMove = useCallback((id: string, x: number, y: number) => {
    updateComponent(id, { position: { x: snap(x), y: snap(y) } })
  }, [updateComponent])

  // Handle block selection
  const handleBlockSelect = useCallback((id: string) => {
    setSelectedBlock(id === selectedBlock ? null : id)
  }, [selectedBlock])

  // Handle wheel zoom
  const handleWheel = useCallback((e: any) => {
    e.evt.preventDefault()
    const by = 1.03
    const next = e.evt.deltaY > 0 ? viewState.scale / by : viewState.scale * by
    setViewState(v => ({ ...v, scale: clamp(next, ZOOM_MIN, ZOOM_MAX) }))
  }, [viewState.scale])

  // Handle touch move for pinch zoom
  const handleTouchMove = useCallback((e: any) => {
    e.evt.preventDefault()
    if (e.evt.touches.length === 2) {
      const [a, b] = e.evt.touches
      const dist = Math.hypot(
        a.clientX - b.clientX,
        a.clientY - b.clientY
      )
      if (lastDist.current) {
        const delta = dist / lastDist.current
        cancelAnimationFrame(raf.current!)
        raf.current = requestAnimationFrame(() =>
          setViewState(v => ({
            ...v,
            scale: clamp(v.scale * delta, ZOOM_MIN, ZOOM_MAX)
          }))
        )
      }
      lastDist.current = dist
    }
  }, [])

  // Handle touch end
  const handleTouchEnd = useCallback(() => {
    lastDist.current = 0
  }, [])

  // Handle stage drag
  const handleDragMove = useCallback((e: any) => {
    setViewState(v => ({
      ...v,
      pos: { x: e.target.x(), y: e.target.y() }
    }))
  }, [])

  if (!currentDesign) {
    return (
      <div className="h-full flex items-center justify-center text-gray-400">
        <p>Create a design to start building</p>
      </div>
    )
  }

  return (
    <Stage
      width={w}
      height={h}
      ref={stageRef}
      draggable
      x={viewState.pos.x}
      y={viewState.pos.y}
      scaleX={viewState.scale}
      scaleY={viewState.scale}
      className="touch-none"
      onDragMove={handleDragMove}
      onWheel={handleWheel}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      {/* Grid overlay */}
      <Layer listening={false}>
        {Array.from({ length: Math.ceil(w / GRID) }).map((_, i) => (
          <Line
            key={`v-${i}`}
            points={[i * GRID, 0, i * GRID, h]}
            stroke="#5555"
            strokeWidth={i % 4 === 0 ? 1 : 0.5}
          />
        ))}
        {Array.from({ length: Math.ceil(h / GRID) }).map((_, i) => (
          <Line
            key={`h-${i}`}
            points={[0, i * GRID, w, i * GRID]}
            stroke="#5555"
            strokeWidth={i % 4 === 0 ? 1 : 0.5}
          />
        ))}
      </Layer>

      {/* Connection layer */}
      <ConnectionLayer 
        edges={currentDesign.edges}
        components={currentDesign.components}
      />

      {/* Blocks layer */}
      <Layer>
        {currentDesign.components.map(component => (
          <ModelStudioBlock
            key={component.id}
            id={component.id}
            type={component.type}
            x={component.position?.x || 0}
            y={component.position?.y || 0}
            selected={selectedBlock === component.id}
            onMove={handleBlockMove}
            onSelect={handleBlockSelect}
          />
        ))}
      </Layer>
    </Stage>
  )
}

const ModelStudioCanvas: React.FC = () => {
  const stageRef = useRef<any>(null)

  return (
    <DndProvider
      backend={TouchBackend}
      options={{ enableMouseEvents: true, delayTouchStart: 0 }}
    >
      <div className="h-full w-full">
        <InnerStage stageRef={stageRef} />
      </div>
    </DndProvider>
  )
}

export default ModelStudioCanvas 