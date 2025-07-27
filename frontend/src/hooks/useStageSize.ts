import { useEffect, useState } from 'react'

export const useStageSize = () => {
  const [size, setSize] = useState({ w: 0, h: 0 })
  
  useEffect(() => {
    const updateSize = () => {
      // Use 70% of viewport height for canvas as specified in Week 1 plan
      const canvasHeight = Math.floor(window.innerHeight * 0.7)
      setSize({ 
        w: window.innerWidth, 
        h: canvasHeight 
      })
    }
    
    updateSize()
    window.addEventListener('resize', updateSize)
    
    return () => window.removeEventListener('resize', updateSize)
  }, [])
  
  return size
} 