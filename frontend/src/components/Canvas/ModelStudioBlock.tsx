import React from 'react'
import { Group, Rect, Text } from 'react-konva'
import { BLOCK_SIZE } from '../../utils/geometry'

const radius = 8
const colors: Record<string, string> = {
  BOARD_INPUT: '#f97316',      // Orange
  TRANSFORMER_LAYER: '#3b82f6', // Blue
  MOE_LAYER: '#22c55e',        // Green
  ACTION_OUTPUT: '#ef4444',    // Red
  VALUE_HEAD: '#a855f7'        // Purple
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
}

const ModelStudioBlock: React.FC<BlockProps> = ({ 
  id, 
  type, 
  x, 
  y, 
  selected = false, 
  onMove, 
  onSelect 
}) => {
  const color = colors[type] || '#6b7280'
  const label = labels[type] || type[0]

  return (
    <Group
      x={x}
      y={y}
      draggable
      onDragEnd={(e) => {
        const { x: newX, y: newY } = e.target.position()
        onMove(id, newX, newY)
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