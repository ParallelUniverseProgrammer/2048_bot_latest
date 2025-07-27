import React from 'react'
import { useDrag } from 'react-dnd'
import { motion } from 'framer-motion'

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

const PaletteItem: React.FC<{ entry: PaletteEntry }> = ({ entry }) => {
  const [{ isDragging }, drag] = useDrag(() => ({
    type: 'BLOCK',
    item: { type: entry.type },
    collect: (monitor) => ({
      isDragging: monitor.isDragging()
    })
  }))

  return (
    <motion.button
      ref={drag}
      aria-label={entry.type}
      className={`card-glass flex items-center space-x-2 w-full px-3 py-2 rounded-xl mb-2 
        ${entry.color} hover:bg-gray-700/50 transition-colors`}
      style={{ 
        minHeight: 44,
        opacity: isDragging ? 0.5 : 1
      }}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <span className="text-lg">{entry.icon}</span>
      <span className="text-sm font-medium capitalize truncate">
        {entry.label}
      </span>
    </motion.button>
  )
}

const ModelStudioPalette: React.FC = () => {
  return (
    <motion.div
      className="h-full card-glass p-4 rounded-2xl"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
    >
      <h3 className="text-sm font-semibold text-gray-300 mb-4">Block Palette</h3>
      <div className="space-y-2">
        {paletteEntries.map((entry) => (
          <PaletteItem key={entry.type} entry={entry} />
        ))}
      </div>
    </motion.div>
  )
}

export default ModelStudioPalette 