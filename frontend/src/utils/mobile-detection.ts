/**
 * Mobile Safari detection and handling utilities
 */

export const isMobile = (): boolean => {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
}

export const isSafari = (): boolean => {
  return /^((?!chrome|android).)*safari/i.test(navigator.userAgent)
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
  return isMobileSafari() ? 5000 : 3000
}

export const getMaxReconnectAttempts = (): number => {
  // Mobile Safari should try fewer times to avoid blocking
  return isMobileSafari() ? 5 : 10
}

export const shouldUsePolling = (): boolean => {
  // Fallback to polling if WebSocket consistently fails on mobile
  return isMobileSafari() && isLocalNetwork()
} 