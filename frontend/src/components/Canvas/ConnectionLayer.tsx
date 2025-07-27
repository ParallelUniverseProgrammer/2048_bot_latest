import React from 'react'
import { Layer, Arrow } from 'react-konva'
import { BLOCK_SIZE } from '../../utils/geometry'
import { ModelComponent } from '../../stores/designStore'

interface ConnectionLayerProps {
  edges: [string, string][]
  components: ModelComponent[]
}

const ConnectionLayer: React.FC<ConnectionLayerProps> = ({ edges, components }) => {
  const componentMap = Object.fromEntries(
    components.map(c => [c.id, c])
  )

  return (
    <Layer listening={false}>
      {edges.map(([from, to]) => {
        const fromComponent = componentMap[from]
        const toComponent = componentMap[to]
        
        if (!fromComponent || !toComponent || !fromComponent.position || !toComponent.position) return null
        
        const fromCenter = {
          x: fromComponent.position.x + BLOCK_SIZE / 2,
          y: fromComponent.position.y + BLOCK_SIZE / 2
        }
        
        const toCenter = {
          x: toComponent.position.x + BLOCK_SIZE / 2,
          y: toComponent.position.y + BLOCK_SIZE / 2
        }
        
        return (
          <Arrow
            key={`${from}-${to}`}
            points={[
              fromCenter.x,
              fromCenter.y,
              toCenter.x,
              toCenter.y
            ]}
            stroke="#06b6d4"
            strokeWidth={1.5}
            pointerLength={4}
            pointerWidth={4}
            opacity={0.8}
          />
        )
      })}
    </Layer>
  )
}

export default ConnectionLayer 