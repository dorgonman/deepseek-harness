// Assistant timing readings used by session statistics.

import type {
  AssistantMessageNode,
} from '@deepseek-ai/dsh-client-ui-conversation/client'

/** One assistant step's derivable latency facts; null marks an unrecorded part. */
export interface StepReading {
  /** step/start → first token delta, in ms. */
  ttftMs: number | null
  /** Positive model-phase wall time (step/start → final message), including TTFT, used for output rate. */
  decodeMs: number | null
  /** Provider-reported completion tokens. */
  outputTokens: number | null
}

interface UsageLike {
  outputTokens?: number
}

type AssistantNode = AssistantMessageNode

function usageOutputTokens(usage: unknown): number | null {
  if (typeof usage !== 'object' || usage === null) return null
  const value = (usage as UsageLike).outputTokens
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
}

/**
 * Read one assistant node's TTFT, model-phase wall time, and output tokens.
 * @param node - A settled assistant node.
 * @returns Per-part readings with `null` for unrecorded values or unusable model-phase durations.
 */
export function assistantStepReading(node: AssistantNode): StepReading {
  const timing = node.timing
  const ttftMs = timing !== undefined && timing.stepStartTime !== null && timing.firstTokenTime !== null
    ? Math.max(0, timing.firstTokenTime - timing.stepStartTime)
    : null
  const modelMs = timing !== undefined && timing.stepStartTime !== null
    ? timing.completedTime - timing.stepStartTime
    : null
  const decodeMs = modelMs !== null && Number.isFinite(modelMs) && modelMs > 0 ? modelMs : null
  return { ttftMs, decodeMs, outputTokens: usageOutputTokens(node.usage) }
}
