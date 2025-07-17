import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Loader2 } from 'lucide-react'
import { useTrainingStore } from '../stores/trainingStore'
import { createPortal } from 'react-dom'

const TrainingLoadingIndicator: React.FC = () => {
  const { loadingStates } = useTrainingStore()
  const { isTrainingStarting } = loadingStates

  if (!isTrainingStarting) {
    return null
  }

  // Create portal content
  const portalContent = (
    <AnimatePresence>
      <motion.div
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 20, opacity: 1 }}
        exit={{ y: -100, opacity: 0 }}
        transition={{ 
          type: "spring", 
          stiffness: 300, 
          damping: 30,
          duration: 0.5
        }}
        style={{
          position: 'fixed',
          top: 0,
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 99999,
          pointerEvents: 'none'
        }}
      >
        <motion.div
          style={{
            backgroundColor: 'rgba(30, 41, 59, 0.95)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            borderRadius: '12px',
            padding: '12px 16px',
            boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.3)',
            backdropFilter: 'blur(8px)',
            WebkitBackdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            minWidth: '200px',
            maxWidth: '300px'
          }}
        >
          <Loader2 
            className="animate-spin" 
            style={{ 
              width: '16px', 
              height: '16px', 
              color: '#60a5fa',
              flexShrink: 0
            }} 
          />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ 
              fontSize: '13px', 
              fontWeight: 500, 
              color: '#93c5fd',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis'
            }}>
              Starting Training
            </div>
            <div style={{ 
              fontSize: '11px', 
              color: 'rgba(147, 197, 253, 0.7)',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis'
            }}>
              {loadingStates.loadingStep || 'Initializing...'}
            </div>
          </div>
          <div style={{ 
            fontSize: '11px', 
            color: '#60a5fa',
            flexShrink: 0,
            fontWeight: 500
          }}>
            {loadingStates.estimatedTimeRemaining !== null 
              ? `${Math.ceil(loadingStates.estimatedTimeRemaining)}s`
              : ''
            }
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )

  // Render to document.body using portal
  return createPortal(portalContent, document.body)
}

export default TrainingLoadingIndicator 