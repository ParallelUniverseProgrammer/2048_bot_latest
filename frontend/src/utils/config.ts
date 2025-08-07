/**
 * Configuration utility for the 2048 Bot Training app
 */

// Get backend URL from environment or defaults
const getBackendUrl = (): string => {
  const envBackendUrl = (import.meta as any).env?.VITE_BACKEND_URL as string | undefined
  const envBackendPort = (import.meta as any).env?.VITE_BACKEND_PORT as string | undefined
  if (envBackendUrl && typeof envBackendUrl === 'string' && envBackendUrl.startsWith('http')) {
    return envBackendUrl
  }
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
  const port = envBackendPort && /^\d+$/.test(envBackendPort) ? envBackendPort : '8000'
  return `${backendProtocol}//${hostname}:${port}`
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