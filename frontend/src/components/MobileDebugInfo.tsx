import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useTrainingStore } from '../stores/trainingStore'
import { 
  isMobile, 
  isSafari, 
  isMobileSafari, 
  isLocalNetwork 
} from '../utils/mobile-detection'
import config from '../utils/config'

const MobileDebugInfo: React.FC = () => {
  const [debugInfo, setDebugInfo] = useState<any>({})
  const [backendReachable, setBackendReachable] = useState<boolean | null>(null)
  const [websocketTest, setWebsocketTest] = useState<string>('Not tested')
  const { isConnected, connectionError } = useTrainingStore()

  useEffect(() => {
    const info = {
      userAgent: navigator.userAgent,
      isMobile: isMobile(),
      isSafari: isSafari(),
      isMobileSafari: isMobileSafari(),
      isLocalNetwork: isLocalNetwork(),
      location: {
        hostname: window.location.hostname,
        port: window.location.port,
        protocol: window.location.protocol,
        href: window.location.href
      },
      config: {
        apiBaseUrl: config.api.baseUrl,
        websocketUrl: config.websocket.url
      },
      network: {
        onLine: navigator.onLine,
        connection: (navigator as any).connection || (navigator as any).mozConnection || (navigator as any).webkitConnection
      }
    }
    setDebugInfo(info)
    
    // Test backend reachability
    testBackendConnection()
    
    // Test WebSocket connection
    testWebSocketConnection()
  }, [])

  const testBackendConnection = async () => {
    try {
      const response = await fetch(`${config.api.baseUrl}/training/status`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        }
      })
      
      if (response.ok) {
        const data = await response.json()
        setBackendReachable(true)
        console.log('Backend test successful:', data)
      } else {
        setBackendReachable(false)
        console.log('Backend test failed:', response.status, response.statusText)
      }
    } catch (error) {
      setBackendReachable(false)
      console.error('Backend test error:', error)
    }
  }

  const testWebSocketConnection = () => {
    try {
      const ws = new WebSocket(config.websocket.url)
      
      ws.onopen = () => {
        setWebsocketTest('✅ Connected')
        ws.close()
      }
      
      ws.onerror = (error) => {
        setWebsocketTest('❌ Failed to connect')
        console.error('WebSocket test error:', error)
      }
      
      ws.onclose = (event) => {
        if (event.code === 1000) {
          setWebsocketTest('✅ Connected (then closed)')
        } else {
          setWebsocketTest(`❌ Closed with code: ${event.code}`)
        }
      }
      
      // Timeout after 5 seconds
      setTimeout(() => {
        if (ws.readyState === WebSocket.CONNECTING) {
          ws.close()
          setWebsocketTest('⏱️ Connection timeout')
        }
      }, 5000)
    } catch (error) {
      setWebsocketTest('❌ Exception during connection')
      console.error('WebSocket test exception:', error)
    }
  }

  if (!isMobileSafari()) {
    return null // Only show on mobile Safari
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="fixed bottom-4 left-4 right-4 z-50 bg-gray-900 text-white text-xs p-4 rounded-lg shadow-lg max-h-60 overflow-y-auto"
    >
      <div className="mb-2 font-bold text-yellow-400">Mobile Safari Debug Info</div>
      
      <div className="mb-2">
        <div className="font-semibold">Connection Status:</div>
        <div>WebSocket: {isConnected ? '✅ Connected' : '❌ Disconnected'}</div>
        <div>Backend API: {backendReachable === null ? '⏳ Testing...' : backendReachable ? '✅ Reachable' : '❌ Not reachable'}</div>
        <div>WebSocket Test: {websocketTest}</div>
        {connectionError && <div className="text-red-400">Error: {connectionError}</div>}
      </div>

      <div className="mb-2">
        <div className="font-semibold">Device Info:</div>
        <div>Mobile: {debugInfo.isMobile ? '✅' : '❌'}</div>
        <div>Safari: {debugInfo.isSafari ? '✅' : '❌'}</div>
        <div>Local Network: {debugInfo.isLocalNetwork ? '✅' : '❌'}</div>
        <div>Online: {debugInfo.network?.onLine ? '✅' : '❌'}</div>
      </div>

      <div className="mb-2">
        <div className="font-semibold">Network:</div>
        <div>Host: {debugInfo.location?.hostname}</div>
        <div>Port: {debugInfo.location?.port}</div>
        <div>Protocol: {debugInfo.location?.protocol}</div>
      </div>

      <div className="mb-2">
        <div className="font-semibold">Config:</div>
        <div>API: {debugInfo.config?.apiBaseUrl}</div>
        <div>WS: {debugInfo.config?.websocketUrl}</div>
      </div>

      <div className="text-xs text-gray-400 mt-2">
        User Agent: {debugInfo.userAgent}
      </div>
    </motion.div>
  )
}

export default MobileDebugInfo 