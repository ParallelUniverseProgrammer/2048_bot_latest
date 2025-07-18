# 🎨 2048 Bot Style Guide

> Design principles, coding standards, and development guidelines for the 2048 Bot project.

---

## 📋 Table of Contents
1. [Design Principles](#design-principles)
2. [UI/UX Guidelines](#uiux-guidelines)
3. [Component Patterns](#component-patterns)
4. [Frontend Development](#frontend-development)
5. [Backend Development](#backend-development)
6. [Mobile-First Approach](#mobile-first-approach)
7. [Performance Guidelines](#performance-guidelines)
8. [Testing Standards](#testing-standards)
9. [Documentation](#documentation)

---

## Design Principles

### 🎯 Core Values
- **Mobile-First**: Design for mobile devices first, then enhance for larger screens
- **Performance**: Optimize for speed and responsiveness across all devices
- **Accessibility**: Ensure the interface is usable by everyone
- **Simplicity**: Reduce cognitive load while maintaining functionality
- **Consistency**: Maintain visual and behavioral consistency throughout

### 🎨 Visual Identity
- **Dark Theme**: Primary dark background with glass-morphism effects
- **Color Palette**: 
  - Primary: Blue (#3b82f6) - `text-blue-400`
  - Success: Green (#22c55e) - `text-green-400`
  - Warning: Yellow (#f59e0b) - `text-yellow-400`
  - Error: Red (#ef4444) - `text-red-400`
  - Purple: (#a855f7) - `text-purple-400` for special features
  - Orange: (#f97316) - `text-orange-400` for time/metrics
  - Cyan: (#06b6d4) - `text-cyan-400` for technical data
- **Typography**: Clean, readable fonts with proper hierarchy
- **Spacing**: Consistent 4px grid system (0.25rem increments)

---

## UI/UX Guidelines

### 📱 Mobile Optimization
- **Touch Targets**: Minimum 44px (2.75rem) for interactive elements
- **Spacing**: Use compact spacing on mobile, expand for larger screens
- **Charts**: Optimize for small screens with expandable full-view options
- **Navigation**: Clear, accessible navigation with visual feedback

### 🎛️ Component Design
- **Cards**: Use `card-glass` class for consistent glass-morphism styling
- **Buttons**: Context-aware styling with hover states and loading indicators
- **Forms**: Clear labels, validation feedback, and mobile-friendly inputs
- **Charts**: Responsive design with touch interaction support

### 📊 Data Visualization
- **Charts**: Use Chart.js with mobile-optimized configurations
- **Colors**: Consistent color coding for different data types
- **Interactions**: Double-tap to expand, hover for details
- **Loading States**: Smooth transitions and progress indicators

---

## Component Patterns

### 🏗️ Layout Structure
All main components should follow this consistent layout pattern:

```tsx
// Standard component layout
<div className="h-full flex flex-col space-y-2 pb-6">
  {/* Error Display */}
  {error && (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="card-glass p-4 rounded-2xl border border-red-500/30 bg-red-500/5 flex-shrink-0"
    >
      {/* Error content */}
    </motion.div>
  )}

  {/* Stats/Overview Section */}
  <motion.div
    className="card-glass p-4 rounded-2xl flex-shrink-0"
    initial={{ opacity: 0, y: -10 }}
    animate={{ opacity: 1, y: 0 }}
  >
    {/* Stats content */}
  </motion.div>

  {/* Main Content Section */}
  <motion.div
    className="card-glass p-4 rounded-2xl flex-shrink-0"
    initial={{ opacity: 0, y: -10 }}
    animate={{ opacity: 1, y: 0 }}
  >
    {/* Main content */}
  </motion.div>

  {/* Scrollable List/Content */}
  <div className="flex-1 overflow-y-auto space-y-2">
    {/* List items */}
  </div>
</div>
```

### 🎨 Animation System
Use simple, consistent animations across all components:

```tsx
// Standard animation patterns
const standardAnimations = {
  // Card entrance
  card: {
    initial: { opacity: 0, y: -10 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.3 }
  },
  
  // Error display
  error: {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1 },
    transition: { duration: 0.2 }
  },
  
  // List items
  item: {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1 },
    transition: { delay: index * 0.05 },
    whileHover: { scale: 1.02 },
    whileTap: { scale: 0.98 }
  },
  
  // Expand/collapse
  expand: {
    initial: { opacity: 0, height: 0 },
    animate: { opacity: 1, height: 'auto' },
    exit: { opacity: 0, height: 0 },
    transition: { duration: 0.3, ease: "easeInOut" }
  }
}
```

### 🎯 Button Patterns
Consistent button styling across components:

```tsx
// Primary action buttons
<button className="flex-1 flex items-center justify-center space-x-2 bg-green-500/20 text-green-400 rounded-xl py-2.5 text-sm font-medium hover:bg-green-500/30 transition-colors">
  <Icon className="w-4 h-4" />
  <span>Action</span>
</button>

// Secondary action buttons
<button className="flex items-center justify-center bg-gray-700 text-gray-400 rounded-lg px-3 py-1.5 hover:bg-gray-600 transition-colors text-sm">
  <Icon className="w-4 h-4" />
  <span>Action</span>
</button>

// Danger action buttons
<button className="flex items-center justify-center bg-red-500/20 text-red-400 rounded-xl py-2.5 px-3 text-sm font-medium hover:bg-red-500/30 transition-colors">
  <Icon className="w-4 h-4" />
</button>
```

### 📊 Stats Display Pattern
Consistent stats/metrics display:

```tsx
// Stats grid pattern
<div className={`grid ${isMobile ? 'grid-cols-2' : 'grid-cols-4'} gap-3`}>
  <div className="text-center">
    <div className="text-lg font-bold text-blue-400">{value}</div>
    <div className="text-xs text-gray-400">{label}</div>
  </div>
</div>

// Compact metrics pattern
<div className={`${isMobile ? 'grid grid-cols-2 gap-2' : 'grid grid-cols-4 gap-2'}`}>
  {metrics.map((metric, index) => (
    <motion.div
      key={metric.title}
      className="flex items-center space-x-2 p-2 bg-gray-800/30 rounded-xl"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay: 0.2 + index * 0.03 }}
    >
      <div className={`p-1 rounded ${metric.bgColor}`}>
        <metric.icon className={`w-3 h-3 ${metric.color}`} />
      </div>
      <div className="min-w-0">
        <div className="text-xs text-gray-400 font-medium truncate">{metric.title}</div>
        <div className={`font-bold ${metric.color} text-sm truncate`}>{metric.value}</div>
      </div>
    </motion.div>
  ))}
</div>
```

### 🔍 Search and Filter Pattern
Consistent search and filter interface:

```tsx
// Search bar pattern
<div className="relative mb-4">
  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
  <input
    type="text"
    placeholder="Search..."
    className="w-full pl-10 pr-4 py-2.5 bg-gray-700 text-white rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
    inputMode="text"
    autoComplete="off"
    autoCorrect="off"
    autoCapitalize="off"
    spellCheck="false"
    style={{ fontSize: '16px' }}
  />
</div>

// Filter buttons pattern
<div className="flex space-x-1">
  {filters.map(({ key, label }) => (
    <button
      key={key}
      onClick={() => setFilterBy(key)}
      className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
        filterBy === key 
          ? 'bg-blue-500 text-white' 
          : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
      }`}
    >
      {label}
    </button>
  ))}
</div>
```

---

## Frontend Development

### 🏗️ Architecture
- **Framework**: React 18+ with TypeScript
- **State Management**: Zustand for global state
- **Styling**: Tailwind CSS with custom utilities
- **Animations**: Framer Motion for smooth transitions
- **Build Tool**: Vite for fast development and optimized builds

### 📝 Code Standards
```typescript
// Component Structure
interface ComponentProps {
  // Props interface
}

const Component: React.FC<ComponentProps> = ({ prop1, prop2 }) => {
  // Hooks first
  const [state, setState] = useState()
  
  // Effects
  useEffect(() => {
    // Effect logic
  }, [])
  
  // Event handlers
  const handleEvent = useCallback(() => {
    // Handler logic
  }, [])
  
  // Render
  return (
    <div className="component-class">
      {/* JSX */}
    </div>
  )
}
```

### 🎨 Styling Guidelines
```css
/* Use Tailwind classes with custom utilities */
.card-glass {
  @apply bg-gray-800/50 backdrop-blur-sm border border-gray-700/50;
}

/* Mobile-first responsive design */
.responsive-grid {
  @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-2 md:gap-3;
}

/* Consistent spacing */
.standard-padding {
  @apply p-3 md:p-4;
}
```

### 📱 Mobile Considerations
- **Viewport**: Always include proper viewport meta tag
- **Touch Events**: Use `onTouchEnd` for mobile interactions
- **Keyboard**: Ensure proper focus management
- **Performance**: Optimize bundle size and loading times

---

## Backend Development

### 🐍 Python Standards
```python
# Use type hints and docstrings
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException

def process_training_data(
    data: Dict[str, any],
    model_size: str = "medium"
) -> Dict[str, float]:
    """
    Process training data and return metrics.
    
    Args:
        data: Training data dictionary
        model_size: Size of the model to use
        
    Returns:
        Dictionary containing processed metrics
    """
    # Implementation
    pass
```

### 🚀 FastAPI Guidelines
- **Endpoints**: Use descriptive route names
- **Validation**: Use Pydantic models for request/response validation
- **Error Handling**: Consistent error responses with proper HTTP codes
- **Documentation**: Auto-generated API docs with examples

### 🧪 Testing
```python
# Use pytest with async support
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_training_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/training/start")
        assert response.status_code == 200
```

---

## Mobile-First Approach

### 📱 Design Strategy
1. **Start with Mobile**: Design for the smallest screen first
2. **Progressive Enhancement**: Add features for larger screens
3. **Touch Optimization**: Ensure all interactions work with touch
4. **Performance**: Optimize for slower mobile connections

### 🎯 Implementation
```typescript
// Use device detection for conditional rendering
const { displayMode } = useDeviceDetection()
const isMobile = displayMode === 'mobile'

// Conditional styling
const chartHeight = isMobile ? 'h-14' : 'h-16'
const gridCols = isMobile ? 'grid-cols-2' : 'grid-cols-4'
```

### 📊 Chart Optimization
```typescript
// Mobile-optimized chart options
const mobileChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false }, // Hide legend on mobile
  },
  scales: {
    x: { display: false }, // Hide axes on mobile
    y: { display: false },
  },
}
```

---

## Performance Guidelines

### ⚡ Frontend Performance
- **Bundle Size**: Keep main bundle under 500KB
- **Code Splitting**: Use dynamic imports for large components
- **Image Optimization**: Use WebP format with fallbacks
- **Caching**: Implement proper cache headers and service workers

### 🚀 Backend Performance
- **Database**: Use connection pooling and query optimization
- **Caching**: Implement Redis for frequently accessed data
- **Async Operations**: Use background tasks for heavy operations
- **Monitoring**: Track response times and error rates

### 📱 Mobile Performance
- **Network**: Optimize for slow connections
- **Battery**: Minimize CPU usage and background processes
- **Memory**: Efficient memory management for large datasets
- **Offline**: Implement offline-first architecture where possible

---

## Testing Standards

### 🧪 Frontend Testing
```typescript
// Use Jest and React Testing Library
import { render, screen, fireEvent } from '@testing-library/react'
import { TrainingDashboard } from './TrainingDashboard'

test('renders training controls', () => {
  render(<TrainingDashboard />)
  expect(screen.getByText('Start Training')).toBeInTheDocument()
})

test('handles mobile interactions', () => {
  render(<TrainingDashboard />)
  const chart = screen.getByTestId('loss-chart')
  fireEvent.touchEnd(chart)
  // Test double-tap behavior
})
```

### 🐍 Backend Testing
```python
# Comprehensive API testing
class TestTrainingAPI:
    def test_start_training(self, client):
        response = client.post("/training/start")
        assert response.status_code == 200
        
    def test_invalid_model_size(self, client):
        response = client.post("/training/start", json={"model_size": "invalid"})
        assert response.status_code == 422
```

### 📱 Mobile Testing
- **Device Testing**: Test on actual mobile devices
- **Network Simulation**: Test with slow network conditions
- **Touch Testing**: Verify all touch interactions work correctly
- **Accessibility**: Test with screen readers and assistive technologies

---

## Documentation

### 📚 Code Documentation
- **Comments**: Explain complex logic, not obvious code
- **Docstrings**: Use Google-style docstrings for functions
- **README**: Keep main README updated with latest features
- **API Docs**: Auto-generate from code comments

### 🎨 Design Documentation
- **Component Library**: Document reusable components
- **Design Tokens**: Maintain consistent design system
- **User Flows**: Document key user journeys
- **Accessibility**: Document accessibility considerations

### 📖 User Documentation
- **Installation**: Clear setup instructions
- **Usage**: Step-by-step usage guides
- **Troubleshooting**: Common issues and solutions
- **FAQ**: Frequently asked questions

---

## 🎯 Quick Reference

### Color Classes
```css
.text-primary    /* Blue */
.text-success    /* Green */
.text-warning    /* Yellow */
.text-error      /* Red */
.text-purple     /* Purple */
.text-orange     /* Orange */
.text-cyan       /* Cyan */
```

### Spacing Scale
```css
.p-1  /* 0.25rem - 4px */
.p-2  /* 0.5rem  - 8px */
.p-3  /* 0.75rem - 12px */
.p-4  /* 1rem    - 16px */
.p-6  /* 1.5rem  - 24px */
```

### Breakpoints
```css
sm: 640px   /* Small devices */
md: 768px   /* Medium devices */
lg: 1024px  /* Large devices */
xl: 1280px  /* Extra large devices */
```

### Common Patterns
```typescript
// Loading states
const [isLoading, setIsLoading] = useState(false)

// Error handling
const [error, setError] = useState<string | null>(null)

// Responsive design
const isMobile = displayMode === 'mobile'

// Touch interactions
const handleTouchEnd = (e: React.TouchEvent) => {
  // Touch logic
}

// Animation patterns
<motion.div
  initial={{ opacity: 0, y: -10 }}
  animate={{ opacity: 1, y: 0 }}
  className="card-glass p-4 rounded-2xl flex-shrink-0"
>
  {/* Content */}
</motion.div>
```

---

## 🚀 Contributing

When contributing to the project:

1. **Follow the Style Guide**: Adhere to all guidelines in this document
2. **Use Component Patterns**: Follow established patterns for consistency
3. **Test Thoroughly**: Ensure all changes work on mobile and desktop
4. **Document Changes**: Update documentation for any new features
5. **Performance**: Consider the impact on bundle size and performance
6. **Accessibility**: Ensure changes maintain accessibility standards

Remember: **Mobile-first, performance-focused, and user-centered design** should guide all decisions! 🎯 