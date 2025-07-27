// Geometry utilities for canvas operations
export const GRID = 32
export const BLOCK_SIZE = 56 // matches 44 px touch min + 12 px halo
export const ZOOM_MIN = 0.25
export const ZOOM_MAX = 2

export const snap = (v: number, g = GRID) => Math.round(v / g) * g

export const clamp = (v: number, min: number, max: number) =>
  Math.min(Math.max(v, min), max)

export const getDistance = (x1: number, y1: number, x2: number, y2: number) =>
  Math.hypot(x2 - x1, y2 - y1)

export const getCenter = (x: number, y: number, width: number, height: number) => ({
  x: x + width / 2,
  y: y + height / 2
}) 