/**
 * Mobile Safari detection and handling utilities
 */

export const isMobile = (): boolean => {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
}

export const isSafari = (): boolean => {
  return /Safari/.test(navigator.userAgent) && !/Chrome/.test(navigator.userAgent)
}

export const isMobileSafari = (): boolean => {
  return isMobile() && isSafari()
}

export const isLocalNetwork = (): boolean => {
  const hostname = window.location.hostname
  return hostname.startsWith('192.168.') || 
         hostname.startsWith('10.') || 
         hostname.startsWith('172.') ||
         hostname === 'localhost' ||
         hostname === '127.0.0.1'
}

// Keep backend host/port – overriding to the frontend port breaks the connection.
// We still allow protocol upgrade (ws / wss) but never change host or port automatically.
export const getMobileOptimizedWebSocketURL = (baseUrl: string): string => {
  // Ensure we respect the current page protocol to avoid mixed-content issues.
  const useSecure = window.location.protocol === 'https:'
  if (useSecure && baseUrl.startsWith('ws://')) {
    return baseUrl.replace('ws://', 'wss://')
  }
  if (!useSecure && baseUrl.startsWith('wss://')) {
    return baseUrl.replace('wss://', 'ws://')
  }
  return baseUrl
}

export const getConnectionRetryDelay = (): number => {
  // Mobile Safari needs longer delays between connection attempts
  return isMobileSafari() ? 3000 : 2000 // Reduced from 5000/3000
}

export const getMaxReconnectAttempts = (): number => {
  // Mobile devices should try more times before giving up
  return isMobile() ? 5 : 10 // Increased from 3 for mobile
}

export const shouldUsePolling = (): boolean => {
  // Fallback to polling if WebSocket consistently fails on mobile
  return isMobileSafari() && isLocalNetwork()
}

/**
 * Calculate total time for all WebSocket retry attempts
 */
export const getTotalRetryTime = (): number => {
  const baseDelay = getConnectionRetryDelay()
  const maxAttempts = getMaxReconnectAttempts()
  let totalTime = 0
  
  for (let i = 1; i <= maxAttempts; i++) {
    // Exponential backoff capped at 30s
    const delay = Math.min(baseDelay * Math.pow(2, i - 1), 30000)
    totalTime += delay
  }
  
  return totalTime
}

/**
 * Get appropriate loading fallback timeout (should be after all retries)
 */
export const getLoadingFallbackTimeout = (): number => {
  if (isMobileSafari()) {
    // Add 5 second buffer after all retry attempts
    return getTotalRetryTime() + 5000
  }
  return 10000 // Default for other platforms
}

/**
 * Network quality detection utilities
 */
export interface NetworkQuality {
  level: 'poor' | 'fair' | 'good' | 'excellent'
  latency: number
  bandwidth: number
  isOnline: boolean
}

export const detectNetworkQuality = (): Promise<NetworkQuality> => {
  return new Promise((resolve) => {
    const startTime = Date.now()
    
    // Test with a small image to measure latency and basic connectivity
    const testImage = new Image()
    testImage.onload = () => {
      const latency = Date.now() - startTime
      
      // Get connection info if available
      const connection = (navigator as any).connection || 
                        (navigator as any).mozConnection || 
                        (navigator as any).webkitConnection
      
      const effectiveType = connection?.effectiveType || 'unknown'
      const downlink = connection?.downlink || 0
      
      let level: NetworkQuality['level'] = 'fair'
      
      if (latency < 100 && (downlink > 10 || effectiveType === '4g')) {
        level = 'excellent'
      } else if (latency < 300 && (downlink > 1 || effectiveType === '3g')) {
        level = 'good'
      } else if (latency < 1000) {
        level = 'fair'
      } else {
        level = 'poor'
      }
      
      resolve({
        level,
        latency,
        bandwidth: downlink,
        isOnline: navigator.onLine
      })
    }
    
    testImage.onerror = () => {
      resolve({
        level: 'poor',
        latency: 9999,
        bandwidth: 0,
        isOnline: navigator.onLine
      })
    }
    
    // Use a small favicon or similar lightweight asset
    testImage.src = '/favicon.ico?' + Date.now()
  })
}

/**
 * Get adaptive polling interval based on network quality
 */
export const getAdaptivePollingInterval = (networkQuality: NetworkQuality): number => {
  switch (networkQuality.level) {
    case 'excellent':
      return 1000 // 1 second
    case 'good':
      return 2000 // 2 seconds  
    case 'fair':
      return 3000 // 3 seconds
    case 'poor':
      return 5000 // 5 seconds
    default:
      return 2000 // Default fallback
  }
}

/**
 * Get adaptive retry strategy based on network quality
 */
export const getAdaptiveRetryStrategy = (networkQuality: NetworkQuality) => {
  if (networkQuality.level === 'poor') {
    return {
      maxAttempts: isMobileSafari() ? 3 : 5,
      baseDelay: 10000, // 10 seconds for poor connections
      maxDelay: 60000   // 1 minute max
    }
  } else if (networkQuality.level === 'fair') {
    return {
      maxAttempts: isMobileSafari() ? 4 : 7,
      baseDelay: 7000,  // 7 seconds
      maxDelay: 45000   // 45 seconds max
    }
  } else {
    return {
      maxAttempts: getMaxReconnectAttempts(),
      baseDelay: getConnectionRetryDelay(),
      maxDelay: 30000   // 30 seconds max
    }
  }
} 