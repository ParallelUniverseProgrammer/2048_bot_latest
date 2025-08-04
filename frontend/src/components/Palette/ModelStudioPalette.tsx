import React, { useState, useCallback, useRef, useEffect } from 'react'
import { useDrag } from 'react-dnd'
import { motion } from 'framer-motion'
import { useDesignStore } from '../../stores/designStore'

export interface PaletteEntry {
  type: string
  label: string
  icon: string
  color: string
}

const paletteEntries: PaletteEntry[] = [
  { type: 'BOARD_INPUT', label: 'Input', icon: '📥', color: 'text-orange-400' },
  { type: 'TRANSFORMER_LAYER', label: 'Transformer', icon: '⚡', color: 'text-blue-400' },
  { type: 'MOE_LAYER', label: 'MoE', icon: '🧠', color: 'text-green-400' },
  { type: 'ACTION_OUTPUT', label: 'Actions', icon: '🎯', color: 'text-red-400' },
  { type: 'VALUE_HEAD', label: 'Value', icon: '💰', color: 'text-purple-400' }
]

// Custom hook for double-tap detection
const useDoubleTap = (callback: () => void, delay: number = 300) => {
  const lastTapRef = useRef(0)
  const timeoutRef = useRef<number | null>(null)

  const handleTap = useCallback(() => {
    const now = Date.now()
    const timeSinceLastTap = now - lastTapRef.current

    if (timeSinceLastTap < delay) {
      // Double tap detected
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }
      callback()
      lastTapRef.current = 0 // Reset to prevent triple-tap
    } else {
      // First tap - set timeout for potential double tap
      lastTapRef.current = now
      timeoutRef.current = setTimeout(() => {
        lastTapRef.current = 0 // Reset if no double tap
        timeoutRef.current = null
      }, delay)
    }
  }, [callback, delay])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  return handleTap
}

const PaletteItem: React.FC<{ entry: PaletteEntry }> = ({ entry }) => {
  const [dragError, setDragError] = useState<string | null>(null)
  const { addComponent } = useDesignStore()
  
  const [{ isDragging }, drag] = useDrag(() => ({
    type: 'BLOCK',
    item: { type: entry.type },
    collect: (monitor) => ({
      isDragging: monitor.isDragging()
    }),
    end: (_item, monitor) => {
      if (monitor.didDrop()) {
        setDragError(null)
      } else {
        setDragError('Drop failed - try again')
        // Clear error after 3 seconds
        setTimeout(() => setDragError(null), 3000)
      }
    }
  }), [entry.type])

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

  // Handle double-tap to place block at center
  const handleDoubleTapPlacement = useCallback(() => {
    const newComponent = {
      id: `${entry.type}_${Date.now()}`,
      type: entry.type,
      props: getDefaultProps(entry.type),
      position: { x: 0, y: 0 } // Will be updated by canvas to center
    }
    
    addComponent(newComponent)
    setDragError(null)
  }, [entry.type, addComponent])

  // Use custom double-tap hook
  const handleTap = useDoubleTap(handleDoubleTapPlacement, 300)

  return (
    <motion.button
      ref={drag}
      aria-label={entry.type}
      className={`card-glass flex flex-col items-center justify-center space-y-1 px-3 py-2 rounded-lg 
        ${entry.color} hover:bg-gray-700/50 transition-colors flex-shrink-0 relative cursor-grab active:cursor-grabbing`}
      style={{ 
        height: 60,
        width: 80,
        opacity: isDragging ? 0.5 : 1
      }}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onTouchStart={(e) => {
        // Prevent horizontal scroll when starting drag
        e.stopPropagation()
      }}
      onClick={handleTap}
    >
      <span className="text-xl">{entry.icon}</span>
      <span className="text-xs font-medium capitalize text-center">
        {entry.label}
      </span>
      {dragError && (
        <div className="absolute -bottom-6 left-0 right-0 text-center">
          <span className="text-xs text-red-400 bg-red-500/10 px-1 rounded">
            {dragError}
          </span>
        </div>
      )}
    </motion.button>
  )
}

const ModelStudioPalette: React.FC = () => {
  const [paletteError] = useState<string | null>(null)

  // Error boundary for palette
  if (paletteError) {
    return (
      <div className="flex items-center justify-center text-red-400 p-2">
        <p className="text-sm">{paletteError}</p>
      </div>
    )
  }

  return (
    <div className="overflow-x-auto overflow-y-hidden scrollbar-hide" style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
      <motion.div
        className="flex space-x-2 py-1"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        style={{ minWidth: 'max-content' }}
      >
        {paletteEntries.map((entry, index) => (
          <motion.div
            key={entry.type}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
          >
            <PaletteItem entry={entry} />
          </motion.div>
        ))}
      </motion.div>
    </div>
  )
}

export default ModelStudioPalette 