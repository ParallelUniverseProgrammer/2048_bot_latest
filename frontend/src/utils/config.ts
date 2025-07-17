/**
 * Configuration utility for the 2048 Bot Training app
 */

// Get backend URL from environment variable or default to localhost
const getBackendUrl = (): string => {
  // Check if we have a backend URL defined in the window object (injected by launcher)
  if (typeof window !== 'undefined' && (window as any).__BACKEND_URL__) {
    return (window as any).__BACKEND_URL__
  }
  
  // Check if we're running on a tunnel domain (*.trycloudflare.com or *.cfargotunnel.com)
  const { hostname, protocol } = window.location
  if (hostname.includes('.trycloudflare.com') || hostname.includes('.cfargotunnel.com')) {
    // We're accessing via tunnel - use the same origin for API calls
    return `${protocol}//${hostname}`
  }
  
  // Fallback: use current hostname so production preview on LAN works
  const backendProtocol = protocol === 'https:' ? 'https:' : 'http:'
  return `${backendProtocol}//${hostname}:8000`
}

export const API_BASE_URL = getBackendUrl()
export const WS_BASE_URL = API_BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://')

export const config = {
  api: {
    baseUrl: API_BASE_URL,
    endpoints: {
      training: {
        start: '/training/start',
        pause: '/training/pause',
        resume: '/training/resume',
        stop: '/training/stop',
        reset: '/training/reset',
        config: '/training/config'
      },
      checkpoints: {
        list: '/checkpoints',
        playback: '/checkpoints/playback'
      },
      websocket: '/ws'
    }
  },
  websocket: {
    url: `${WS_BASE_URL}/ws`,
    reconnectAttempts: 10,
    reconnectDelay: 3000
  }
}

export default config 