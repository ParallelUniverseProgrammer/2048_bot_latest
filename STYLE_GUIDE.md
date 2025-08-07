## 2048 Bot Design System (Authoritative)

This guide is the single source of truth for UI/UX decisions. When existing code differs from this guide, this guide wins. It defines a mobile‑first, native app‑like experience that is consistent across Training, Game, Checkpoints, and Model Studio.

### Scope
- **Platform**: PWA (mobile‑first), responsive up to desktop
- **Audience**: Designers, frontend engineers, and contributors
- **Tenets**: Performance at 60fps, touch‑first interactions, consistent patterns, accessibility, and clarity

## Core principles
- **Mobile‑first by default**: Design for small screens first. Add enhancements as screens grow.
- **Native feel**: Instant feedback, fluid motion, no unexpected scroll, large touch targets, predictable gestures.
- **Consistency over novelty**: Reuse patterns. One way to do a thing.
- **Performance is a feature**: Prefer inexpensive layouts, avoid layout thrash, and cap animation costs.
- **Accessible from the start**: Color contrast, focus order, semantics, and keyboard support on desktop.

## Design tokens (do not hardcode values)
All styles must consume tokens. Implement via CSS variables or Tailwind theme mappings. Example variables are provided; adjust implementation details as needed.

### Color roles (dark theme first)
- **surface/base**: app background
- **surface/elevated**: cards, sheets
- **text/primary**: main text
- **text/secondary**: secondary text, labels
- **border/muted**: hairlines, dividers
- **brand/primary**: interactive accents, links
- **state/success**: positive actions, good status
- **state/warning**: caution, paused
- **state/danger**: destructive, errors
- **state/info**: neutral technical highlights
- **overlay**: scrim for modals/menus
- **focus**: keyboard focus ring

Suggested defaults (may be tuned during visual QA):
```css
:root {
  /* Surfaces */
  --ui-surface-base: #0b0f14;
  --ui-surface-elevated: #111827;
  /* Text */
  --ui-text-primary: #f3f4f6;
  --ui-text-secondary: #9ca3af;
  /* Lines */
  --ui-border-muted: #1f2937;
  /* Brand & states */
  --ui-brand-primary: #60a5fa;
  --ui-success: #34d399;
  --ui-warning: #fbbf24;
  --ui-danger: #f87171;
  --ui-info: #a78bfa;
  /* Overlays & focus */
  --ui-overlay: rgba(0,0,0,0.55);
  --ui-focus: #93c5fd;
}
```

### Spacing scale (4px base)
- `0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64`
- Default vertical rhythm between stacked blocks: **8–12px on mobile**, **12–16px on desktop**.

### Radii
- `xs: 8px`, `sm: 12px`, `md: 16px`, `lg: 20px`
- Cards use `sm` by default; modals use `md`.

### Elevation
- `e0` none
- `e1` subtle (cards)
- `e2` raised (menus, popovers)
- `e3` modal surfaces
- Avoid harsh shadows; prefer low‑alpha large‑blur shadows.

### Typography
- **Family**: Inter (UI), system fallbacks; JetBrains Mono (numeric & code) optional.
- **Sizes**: `12, 14, 16, 18, 20, 24` (base 14 on mobile, 16 on desktop).
- **Weights**: 400, 500, 600. Use 500 for action labels, 600 for headings.
- **Numeric**: Use `font-variant-numeric: tabular-nums` for metrics and tiles.

### Motion tokens
- **Durations**: `fast 120ms`, `base 200ms`, `slow 300ms`
- **Easing**: `standard: cubic-bezier(0.2, 0, 0, 1)`, `enter: (0, 0, 0.2, 1)`, `exit: (0.4, 0, 1, 1)`
- **Stagger**: 20–40ms between list items
- Never block interactions with long animations.

## Layout system

### Safe‑area & viewport
- Always include `viewport-fit=cover` and respect `env(safe-area-inset-*)` paddings.
- Core screens must fit height on mobile without scrolling; content inside cards may scroll.

### Page structure (canonical)
```tsx
<div className="h-full flex flex-col gap-2 pb-6 px-4">
  {/* 1) Error/status banner (collapsible) */}
  {/* 2) Overview stats row/cards */}
  {/* 3) Primary content card (fills available space) */}
  {/* 4) Secondary list or controls footer */}
  {/* Tab switch preserves internal scroll positions */}
  {/* Mobile: avoid vertical overflow of the whole page */}
  {/* Desktop: increase container padding to px-6/8/10 */}
  {/* Use css vars/tokens for colors and radii */}
</div>
```

### Breakpoints (mobile first)
- `xs` ≤ 360
- `sm` 361–479
- `md` 480–767
- `lg` 768–1023
- `xl` ≥ 1024

### Density
- Mobile hit targets ≥ 40px height; compact icon buttons ≥ 36px.
- List rows 44–52px. Input fields ≥ 44px height.

## Components

### App chrome
- **Tabs**: Four tabs (Training, Game, Checkpoints, Model Studio). Label + icon. Keep labels short.
- **Bars**: Avoid persistent bottom bars on mobile unless essential. Prefer in‑content controls.

### Cards
- Use elevated surface with subtle border; padding `16–20px`.
- Title row with optional action buttons on the right.
- Never nest more than two card levels.

### Buttons
- Sizes: `sm (36h)`, `md (40h)`, `lg (44h)`.
- Variants: `primary`, `secondary`, `ghost`, `danger`.
- Content: icon‑leading 16px, label medium weight.
- States: hover (desktop), pressed, focused, disabled. Maintain contrast in all states.

### Inputs
- Single‑line text, select, segmented controls.
- Include clear affordances, placeholder as hint (not label).
- Validation states: success/warning/danger; inline message under field.

### Lists
- Row: leading icon/thumbnail, primary text, secondary text, trailing meta/action.
- Dividers use muted borders; prefer grouped sections with sticky headers when long.

### Modals & sheets
- Mobile: full‑width bottom sheets for transient tasks; modals for confirmation.
- Desktop: centered modal with `md` radius; lock background scroll with overlay.

### Toasts & banners
- Use to confirm background actions or surface non‑blocking errors.
- Auto‑dismiss after 3–5s with manual close.

### Progress & status
- Inline progress bars for long operations; never rely only on spinners.
- Use determinate progress when possible; otherwise show staged text updates.

### Charts (metrics)
- Compact spark/mini charts on mobile (56–72px height).
- Double‑tap to expand into an overlay with legends and tooltips.
- Limit datasets for performance; prefer last N points.

### Game board
- Square tiles with consistent scale. Numbers use tabular numerals.
- Color steps map to difficulty; ensure 4.5:1 contrast minimum against tile surface.
- Animate merges with scale+fade `200ms`, easing `standard`.

### Model Studio (canvas)
- Blocks snap to grid (8px). Hit targets ≥ 40px for ports.
- Connections use smooth curves; highlight on hover/selection.
- Zoom 0.5–2.0 with pinch; pan with two‑finger drag; avoid single‑finger conflict with scroll.

## Motion & interaction
- Page/card enter: fade+translate‑y small, `200ms`.
- List appearance: stagger 20–40ms per item.
- Button press: scale 0.98 with shadow reduction.
- Reduce motion: respect `prefers-reduced-motion`; fall back to opacity only.

## Accessibility
- Contrast: text 4.5:1, icons 3:1 minimum.
- Focus: visible ring using `--ui-focus`; never remove outlines without replacement.
- Semantics: proper roles for buttons, lists, tabs, and dialogs.
- Keyboard: all interactive elements reachable in a logical order.

## Content guidelines
- Labels are concise and action‑oriented.
- Avoid jargon in user‑facing text; technical details live in secondary text.
- Numbers: use short format (e.g., 12.3k) and consistent precision.

## Responsive behavior
- Column rules: metrics 2 cols on mobile, 4 on desktop.
- Tables/lists collapse secondary columns on mobile; expose details in row sheet.
- Images/charts scale to container, never overflow viewport width.

## Error, empty, loading states
- **Loading**: skeletons that match final layout; avoid blank screens.
- **Empty**: friendly illustration/icon, one‑line description, primary action.
- **Error**: inline banner describing the issue and next step; destructive actions gated by confirmation.

## Performance budget
- Aim for < 100ms interaction latency on mid‑range mobile.
- Avoid heavy box‑shadows and large blurs on scrolling lists.
- Virtualize large lists; throttle resize/scroll handlers.
- Batch updates; memoize expensive renders.

## PWA & platform specifics
- iOS: prevent rubber‑band scrolling where harmful; manage 100vh with safe‑areas.
- Offline: cache shell + core assets; degrade charts gracefully.
- Install: provide clear “Add to Home Screen” affordance; ensure icon and name are crisp.

## Frontend conventions (engineering)
- Components are presentational; state and side‑effects live in dedicated hooks/stores.
- Order: hooks → derived values → handlers → JSX.
- Avoid ad‑hoc WebSocket instances; use a centralized connection and events.
- Prefer composition over deep prop drilling; keep components small and focused.
- Use TypeScript with strict props and meaningful names.

## Theming & implementation notes
- Map tokens to Tailwind theme and/or CSS variables. Do not inline hex values in components.
- Create utility classes for repeated patterns (e.g., `card`, `btn-primary`, `btn-secondary`).
- Do not introduce new hues without updating the token set.

## Do & don’t
- Do reuse existing patterns; don’t invent one‑offs for a single screen.
- Do prioritize mobile; don’t optimize desktop at the expense of touch.
- Do keep animations short; don’t block interactions with motion.
- Do write accessible markup; don’t rely solely on color for meaning.

## Review checklist (every PR)
- Uses page structure and spacing rhythm defined above
- Touch targets ≥ 40px; no page‑level vertical overflow on mobile
- Tokens used for color, spacing, radii; no hardcoded values
- Buttons, cards, and lists match variants and states
- Charts support compact + expanded modes
- Focus styles visible and accessible
- No ad‑hoc sockets; state flows through stores/hooks

## References
- Canonical examples: `Training`, `Game`, `Checkpoints`, `Model Studio` screens
- When code and guide conflict, this guide is authoritative

