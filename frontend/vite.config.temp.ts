
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react({
      // Disable fast refresh for mobile compatibility
      refresh: false
    }),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'favicon-16x16.png'],
      manifest: {
            name: '2048 Bot Training',
    short_name: '2048 AI',
    description: 'Real-time visualization for 2048 bot training',
        theme_color: '#3b82f6',
        background_color: '#0f172a',
        display: 'standalone',
        orientation: 'portrait',
        scope: '/',
        start_url: '/',
        icons: [
          {
            src: 'pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any maskable'
          },
          {
            src: 'pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable'
          }
        ]
      }
    })
  ],
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    // Mobile-friendly server settings
    hmr: {
      port: 5174,
      host: '0.0.0.0'
    },
    // Longer timeout for mobile connections
    timeout: 30000,
    // CORS settings for mobile
    cors: {
      origin: ['http://192.168.1.254:5173', 'http://localhost:5173'],
      credentials: true
    }
  },
  define: {
    __BACKEND_URL__: JSON.stringify('http://192.168.1.254:8000')
  },
  // Build optimizations for mobile
  build: {
    target: 'es2015',
    minify: false,
    sourcemap: true
  }
})
