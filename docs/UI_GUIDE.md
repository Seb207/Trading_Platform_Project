# UI Design Guide — Consulting Dashboard

Reverse-engineered from the actual, already-shipped components — this
documents the convention in use, it doesn't propose a new one. When adding
a component, match what's here rather than introducing a new pattern.

## Design Principles

1. Looks like a terminal/trading desk tool, not a SaaS landing page —
   dense, monospace, small type, low chrome.
2. Color is functional, not decorative — an accent color means something
   (a provider, a status, a severity), it isn't there to look nice.
3. Motion is a loading/status signal only (a pulsing dot, a staggered
   cursor) — never decorative.

## AI Slop Anti-Patterns — Do Not Use

| Banned | Why |
|---|---|
| `backdrop-filter: blur()` / glassmorphism | the single most common tell of an AI-templated UI |
| Gradient text | generic AI-SaaS landing signature |
| "Powered by AI" badges | decoration, not information |
| Glow-animation box-shadows on cards/buttons | neon glow reads as AI slop — the one exception is a small solid-color **status dot** (e.g. connection indicator), which is functional, not decorative |
| Purple/indigo as the primary brand color | the reflexive "AI = purple" cliché |
| Uniform `rounded-2xl` on every surface | this project uses `rounded-sm` (barely rounded) everywhere; `rounded-full` only for pills/dots |
| Blurred gradient "orb" background decoration | not used anywhere in this codebase — don't introduce it |

## Color

Background and border are tiered (`bg`/`bg2`/`bg3`, `border`/`border2`),
darkest at the page root, lighter for nested surfaces.

### Accents (semantic, not decorative)

| Token | Hex | Used for |
|---|---|---|
| `accent` | `#00ff88` (green) | primary/positive — Claude provider, "pass"/"verified", positive values |
| `accent2` | `#00cfff` (cyan) | secondary — Ollama provider, informational badges |
| `accent3` | pink | tertiary — OpenRouter provider, "revised" state |
| `accent4` | gold | quaternary — warnings, "revise" verdicts |
| `neg` | red | errors, negative values |

### Text tiers

`text` (primary) → `text-mid` (secondary) → `text-dim` (tertiary/disabled).
No pure white/black — always through these tokens.

## Components

### Badge (`src/components/ui/Badge.tsx`)

```
inline-flex items-center px-2 py-0.5 rounded-sm
font-mono text-[10px] tracking-wide border
```
Variant = `bg-{accent}/10 text-{accent} border-{accent}/30` for whichever
accent token applies (`green`/`cyan`/`pink`/`gold`/`dim`). This
translucent-fill + matching-border pattern is the standard way to render
any status/category pill in this codebase — don't invent a new pill style.

### Section labels

`font-mono text-[9px]–[11px] text-text-dim uppercase tracking-wide` (or
`tracking-widest` for the most prominent ones, e.g. tab headers). Every
panel/section header in the dashboard follows this, not a heading font.

### Status dot

Small (`w-1.5 h-1.5`) `rounded-full`, solid background matching status
color, with a *tight* matching `box-shadow` glow (`0 0 4px <color>`) — the
one sanctioned use of glow, because it's a functional live-status
indicator (e.g. "API key configured", "Ollama connected"), not decoration.

### Loading/streaming indicator

Three small dots (`w-1 h-1 rounded-full bg-accent`), `animate-pulse`, each
with a staggered `animationDelay` (`i * 150ms`). This is the only
"animation" pattern in the codebase — don't add spinners, skeleton
shimmer, or other motion beyond this and the plain `animate-pulse`
skeleton blocks used for loading states.

## Typography

Monospace (`font-mono`) for nearly everything — labels, badges, buttons,
metadata. Sizes run small: `text-[9px]` to `text-[13px]` for most UI;
body/paragraph copy in chat bubbles is the largest text on screen at
`text-[12px]`. No large display headings anywhere in the app shell.

## Buttons

`border` + `text-{accent}` + `hover:bg-{accent}/10`, `rounded-sm`,
`font-mono text-[10px]–[11px] uppercase tracking-wide`. No filled/solid
primary buttons — everything is an outlined, accent-colored control that
fills faintly on hover.

## Layout

- Split-panel layouts (`SplitPanel`) for most pages — a list/table on the
  left, detail/chat on the right.
- Left-aligned by default; nothing centered.
- Tight spacing: `gap-1.5`–`gap-4`, `px-3.5`–`px-6`, `py-1`–`py-4`. No large
  whitespace sections.

## Transitions

`transition-colors duration-75`–`duration-150` only — color/background
changes on hover/active. No transform, scale, or layout-shift transitions.
