# 🎨 2048 Bot Style Guide

> Design principles, coding standards, and development guidelines for the 2048 Bot project.

---

## 📋 Table of Contents
1. [Design Principles](#design-principles)
2. [UI/UX Guidelines](#uiux-guidelines)
3. [Frontend Development](#frontend-development)
4. [Backend Development](#backend-development)
5. [Mobile-First Approach](#mobile-first-approach)
6. [Performance Guidelines](#performance-guidelines)
7. [Testing Standards](#testing-standards)
8. [Documentation](#documentation)

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
  - Primary: Blue (#3b82f6)
  - Success: Green (#22c55e)
  - Warning: Yellow (#f59e0b)
  - Error: Red (#ef4444)
  - Purple: (#a855f7) for special features
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
```

---

## 🚀 Contributing

When contributing to the project:

1. **Follow the Style Guide**: Adhere to all guidelines in this document
2. **Test Thoroughly**: Ensure all changes work on mobile and desktop
3. **Document Changes**: Update documentation for any new features
4. **Performance**: Consider the impact on bundle size and performance
5. **Accessibility**: Ensure changes maintain accessibility standards

Remember: **Mobile-first, performance-focused, and user-centered design** should guide all decisions! 🎯 