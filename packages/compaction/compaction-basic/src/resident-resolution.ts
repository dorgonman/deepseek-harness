/**
 * Product resolution seam for resident-preferred summarization: turn one
 * projected summary envelope into the exact frozen provider/model target and
 * receipt, or a typed refusal. Fixed mode stays the default and never reaches
 * this path.
 *
 * @module @deepseek-ai/dsh-compaction-basic/resident-resolution
 */

import { createHash } from 'node:crypto'
import type { RequestMessage } from '@deepseek-ai/dsh-llm'
import {
  selectResidentFromSource,
  type ResidentObservationSource,
  type ResidentSelectionReceipt,
} from './resident-observation.ts'

/** Failure classes for resident-preferred resolution. */
export class ResidentResolutionError extends Error {
  constructor(
    /** Machine-readable cause. */
    readonly code: 'mode-fixed' | 'source-unavailable' | 'not-configured' | 'scope-required' | 'no-eligible',
    message: string,
  ) {
    super(message)
    this.name = 'ResidentResolutionError'
  }
}

/** Backend configuration relevant to resident resolution. */
export interface ResidentResolutionConfig {
  /** Selection mode; only `resident-preferred` resolves here. */
  readonly mode: 'fixed' | 'resident-preferred'
  /** Authorized gateway identity. */
  readonly gatewayId?: string
  /** Conversation-pinned node used when no allowlist is supplied. */
  readonly pinnedNodeId?: string
  /** Explicit compaction allowlist; replaces the pin. */
  readonly allowedNodeIds?: readonly string[]
  /** Configured quality preference, most preferred first. */
  readonly preferredVariantIds?: readonly string[]
  /** DSH LLM provider route that fronts the gateway. */
  readonly provider?: string
  /** Maximum accepted observation age. */
  readonly maxAgeMs: number
}

/** Exact summary envelope the selector measures per candidate. */
export interface ResidentProjection {
  /** Exact messages the summarizer call will send. */
  readonly messages: readonly RequestMessage[]
  /** Output tokens reserved on the call. */
  readonly maxTokens: number
  /** Additional tokens reserved beyond the measured projection. */
  readonly reserveTokens: number
}

/** Frozen target returned to the compaction backend. */
export interface ResidentResolvedTarget {
  /** DSH LLM provider route. */
  readonly provider: string
  /** Strict model alias, including the registered-node suffix. */
  readonly model: string
  /** Receipt identifying the accepted deployment. */
  readonly receipt: ResidentSelectionReceipt
}

/** JSON with recursively sorted object keys, so equal content hashes equally. */
function canonicalJson(value: unknown): string {
  if (value === null) return 'null'
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return JSON.stringify(value)
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>
    const entries = Object.keys(record)
      .filter(key => record[key] !== undefined)
      .sort()
      .map(key => `${JSON.stringify(key)}:${canonicalJson(record[key])}`)
    return `{${entries.join(',')}}`
  }
  return 'null'
}

function digestCanonical(value: unknown): string {
  return createHash('sha256').update(canonicalJson(value)).digest('hex')
}
/**
 * Stable identity of a projected summary envelope.
 * @param projection - exact messages and output reservation.
 * @returns lowercase hex digest binding the projection content.
 */
export function residentProjectionId(projection: ResidentProjection): string {
  return digestCanonical({ messages: projection.messages, maxTokens: projection.maxTokens })
}

/**
 * Resolve one compaction's resident summarizer target.
 *
 * The source sees only the pinned/allowlisted nodes and the exact projection.
 * A missing source, scope or eligible route refuses with a typed error; there
 * is no catalog fallback, cold load or silent fixed-route substitution.
 * @param source - mounted trusted observation provider, when available.
 * @param config - mode, scope, gateway and provider configuration.
 * @param projection - exact summary envelope and output reservation.
 * @param options - cancellation and an injectable clock for tests.
 * @returns the frozen provider/model target with its receipt.
 */
export async function resolveResidentSummary(
  source: ResidentObservationSource | undefined,
  config: ResidentResolutionConfig,
  projection: ResidentProjection,
  options: { readonly signal?: AbortSignal; readonly now?: () => number } = {},
): Promise<ResidentResolvedTarget> {
  if (config.mode !== 'resident-preferred') {
    throw new ResidentResolutionError('mode-fixed', 'compaction: resident resolution requires resident-preferred mode')
  }
  if (source === undefined) {
    throw new ResidentResolutionError('source-unavailable', 'compaction: no resident observation source is mounted')
  }
  if (config.gatewayId === undefined || config.provider === undefined) {
    throw new ResidentResolutionError('not-configured', 'compaction: resident gateway and provider must be configured')
  }
  const selection = await selectResidentFromSource({
    gatewayId: config.gatewayId,
    ...config.pinnedNodeId === undefined ? {} : { pinnedNodeId: config.pinnedNodeId },
    ...config.allowedNodeIds === undefined ? {} : { allowedNodeIds: config.allowedNodeIds },
    ...config.preferredVariantIds === undefined ? {} : { preferredVariantIds: config.preferredVariantIds },
    projectionId: residentProjectionId(projection),
    outputTokens: projection.maxTokens,
    reserveTokens: projection.reserveTokens,
    maxAgeMs: config.maxAgeMs,
    projectionMessages: projection.messages,
  }, source, options)
  if (!selection.ok) {
    if (selection.code === 'resident_summary_scope_required') {
      throw new ResidentResolutionError('scope-required', 'compaction: resident selection requires a node scope')
    }
    throw new ResidentResolutionError('no-eligible', 'compaction: no eligible resident summarizer is available')
  }
  return { provider: config.provider, model: selection.target.alias, receipt: selection.receipt }
}
