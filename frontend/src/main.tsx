import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import LoadingFallback from './components/LoadingFallback.tsx'
import './index.css'

// Chart.js registration
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  BarElement,
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  BarElement
)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <LoadingFallback>
      <App />
    </LoadingFallback>
  </React.StrictMode>,
) 