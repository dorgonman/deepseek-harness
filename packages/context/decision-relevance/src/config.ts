/** Configuration resolution for opt-in background Decision relevance filtering. */

import { MAX_TIMER_DELAY_MS } from '@deepseek-ai/dsh-timeout'
import { deepFreeze } from '@deepseek-ai/dsh-util-values'
import type { DecisionRelevanceConfig, ResolvedConfig } from './types.ts'

/** Replacement text for an elided tool-result body. */
export const DEFAULT_ELISION_TEXT = '[tool result omitted for this request: judged irrelevant to the current goal]'

/** Defaults sized for a small local Decision model and a bounded background pass. */
export const DEFAULTS = deepFreeze({
  mode: 'off' as const,
  retainThreshold: 0.6,
  uncertaintyMargin: 0.1,
  maxCandidatesPerRequest: 4,
  minCandidateChars: 512,
  maxInputChars: 4_096,
  timeoutMs: 15_000,
  elisionText: DEFAULT_ELISION_TEXT,
})

const CONFIG_KEYS: ReadonlySet<string> = new Set([
  'mode',
  'provider',
  'model',
  'retainThreshold',
  'uncertaintyMargin',
  'maxCandidatesPerRequest',
  'minCandidateChars',
  'maxInputChars',
  'timeoutMs',
  'elisionText',
])

/**
 * Resolve and validate relevance-filter controls.
 *
 * An absent route disables the feature even under `mode: 'on'`, so a partially
 * migrated profile never sends background traffic to an unintended route and
 * never falls back to the conversation's chat model.
 * @param config - raw plugin configuration.
 * @returns a detached deeply immutable configuration.
 */
export function resolveConfig(config: DecisionRelevanceConfig = {}): ResolvedConfig {
  for (const key of Object.keys(config)) {
    if (!CONFIG_KEYS.has(key)) {
      throw new Error(
        `DecisionRelevanceConfig: unknown key "${key}" `
        + '(allowed: mode, provider, model, retainThreshold, uncertaintyMargin, '
        + 'maxCandidatesPerRequest, minCandidateChars, maxInputChars, timeoutMs, elisionText)',
      )
    }
  }

  const sparse: Record<string, unknown> = { ...config }
  const mode = sparse['mode'] ?? DEFAULTS.mode
  if (mode !== 'off' && mode !== 'on') {
    throw new Error(`DecisionRelevanceConfig: mode (${describeValue(mode)}) must be "off" or "on"`)
  }
  const provider = optionalText('provider', sparse['provider'])
  const model = optionalText('model', sparse['model'])
  if ((provider === undefined) !== (model === undefined)) {
    throw new Error('DecisionRelevanceConfig: provider and model must be configured together')
  }

  const retainThreshold = numberInRange('retainThreshold', sparse['retainThreshold'], DEFAULTS.retainThreshold, 0, 1)
  const uncertaintyMargin = numberInRange('uncertaintyMargin', sparse['uncertaintyMargin'], DEFAULTS.uncertaintyMargin, 0, 1)
  if (retainThreshold - uncertaintyMargin < 0 || retainThreshold + uncertaintyMargin > 1) {
    throw new Error(
      `DecisionRelevanceConfig: retainThreshold ${retainThreshold} +/- uncertaintyMargin ${uncertaintyMargin} `
      + 'must stay within [0, 1]',
    )
  }

  const resolved: ResolvedConfig = {
    mode,
    ...provider === undefined ? {} : { provider },
    ...model === undefined ? {} : { model },
    retainThreshold,
    uncertaintyMargin,
    maxCandidatesPerRequest: positiveInteger('maxCandidatesPerRequest', sparse['maxCandidatesPerRequest'], DEFAULTS.maxCandidatesPerRequest),
    minCandidateChars: nonNegativeInteger('minCandidateChars', sparse['minCandidateChars'], DEFAULTS.minCandidateChars),
    maxInputChars: positiveInteger('maxInputChars', sparse['maxInputChars'], DEFAULTS.maxInputChars),
    timeoutMs: positiveInteger('timeoutMs', sparse['timeoutMs'], DEFAULTS.timeoutMs),
    elisionText: elisionText(sparse['elisionText']),
  }
  return deepFreeze(structuredClone(resolved))
}

/**
 * Whether this configuration can actually issue background Decision calls.
 * @param config - a resolved configuration.
 * @returns whether the feature is enabled and has an explicit route.
 */
export function isEffective(config: ResolvedConfig): config is ResolvedConfig & {
  readonly mode: 'on'
  readonly provider: string
  readonly model: string
} {
  return config.mode === 'on' && config.provider !== undefined && config.model !== undefined
}

/**
 * Count Unicode code points without splitting surrogate pairs.
 * @param text - text to measure.
 * @returns the Unicode code-point count.
 */
export function codePointLength(text: string): number {
  return Array.from(text).length
}

/**
 * Bound text to a Unicode code-point budget without splitting a surrogate pair.
 * @param text - text to bound.
 * @param maxChars - maximum code points to retain.
 * @returns the original text when already within budget, otherwise its leading code points.
 */
export function truncateCodePoints(text: string, maxChars: number): string {
  const points = Array.from(text)
  return points.length <= maxChars ? text : points.slice(0, maxChars).join('')
}

function optionalText(name: string, value: unknown): string | undefined {
  if (value === undefined) return undefined
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`DecisionRelevanceConfig: ${name} must be a non-empty string`)
  }
  return value
}

function elisionText(value: unknown): string {
  if (value === undefined) return DEFAULTS.elisionText
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error('DecisionRelevanceConfig: elisionText must be a non-empty string')
  }
  return value
}

/** Render one rejected config value without leaking an object's default form. */
function describeValue(value: unknown): string {
  if (value === null) return 'null'
  if (typeof value === 'string') return JSON.stringify(value)
  if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'bigint') return `${value}`
  return typeof value
}

function numberInRange(name: string, value: unknown, fallback: number, min: number, max: number): number {
  if (value === undefined) return fallback
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
    throw new Error(`DecisionRelevanceConfig: ${name} (${describeValue(value)}) must be a number in [${min}, ${max}]`)
  }
  return value
}

function positiveInteger(name: string, value: unknown, fallback: number): number {
  if (value === undefined) return fallback
  if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
    throw new Error(`DecisionRelevanceConfig: ${name} (${describeValue(value)}) must be a positive integer`)
  }
  if (name === 'timeoutMs' && value > MAX_TIMER_DELAY_MS) {
    throw new Error(`DecisionRelevanceConfig: ${name} (${value}) exceeds the maximum timer delay ${MAX_TIMER_DELAY_MS}`)
  }
  return value
}

function nonNegativeInteger(name: string, value: unknown, fallback: number): number {
  if (value === undefined) return fallback
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new Error(`DecisionRelevanceConfig: ${name} (${describeValue(value)}) must be a non-negative integer`)
  }
  return value
}
