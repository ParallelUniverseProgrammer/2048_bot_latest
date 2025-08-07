import React from 'react'
import { Group, Rect, Text } from 'react-konva'
import { BLOCK_SIZE } from '../../utils/geometry'

const radius = 8
const colors: Record<string, string> = {
  BOARD_INPUT: 'var(--ui-warning)',      // Orange
  TRANSFORMER_LAYER: 'var(--ui-brand-primary)', // Blue
  MOE_LAYER: 'var(--ui-success)',        // Green
  ACTION_OUTPUT: 'var(--ui-danger)',    // Red
  VALUE_HEAD: 'var(--ui-info)'        // Purple
}

const labels: Record<string, string> = {
  BOARD_INPUT: 'IN',
  TRANSFORMER_LAYER: 'T',
  MOE_LAYER: 'M',
  ACTION_OUTPUT: 'A',
  VALUE_HEAD: 'V'
}

interface BlockProps {
  id: string
  type: string
  x: number
  y: number
  selected?: boolean
  onMove: (id: string, x: number, y: number) => void
  onSelect?: (id: string) => void
  onDragStart?: () => void
  onDragEnd?: () => void
}

const ModelStudioBlock: React.FC<BlockProps> = ({ 
  id, 
  type, 
  x, 
  y, 
  selected = false, 
  onMove, 
  onSelect,
  onDragStart,
  onDragEnd
}) => {
  const color = colors[type] || 'var(--ui-text-secondary)'
  const label = labels[type] || type[0]

  return (
    <Group
      x={x}
      y={y}
      draggable
      onDragStart={(e) => {
        e.evt.preventDefault()
        // Don't change position on drag start - maintain current position
        onDragStart?.()
      }}
      onDragMove={(e) => {
        // Block follows touch exactly - no position resetting
        e.evt.preventDefault()
        // Don't call onMove here - only at the end
      }}
      onDragEnd={(e) => {
        e.evt.preventDefault()
        // Only update position at the end of drag
        const { x: newX, y: newY } = e.target.position()
        onMove(id, newX, newY)
        onDragEnd?.()
      }}
      onClick={() => onSelect?.(id)}
      onTap={() => onSelect?.(id)}
    >
      <Rect
        width={BLOCK_SIZE}
        height={BLOCK_SIZE}
        cornerRadius={radius}
        fill={color}
        stroke={selected ? '#FFFFFF' : undefined}
        strokeWidth={selected ? 2 : 0}
        shadowBlur={4}
        shadowColor="rgba(0,0,0,0.35)"
        shadowOffset={{ x: 0, y: 2 }}
      />
      <Text
        text={label}
        fontSize={18}
        fontStyle="bold"
        fill="#FFFFFF"
        width={BLOCK_SIZE}
        height={BLOCK_SIZE}
        align="center"
        verticalAlign="middle"
      />
    </Group>
  )
}

export default ModelStudioBlock 