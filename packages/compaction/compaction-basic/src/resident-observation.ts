/**
 * Trusted observation seam between an authorized resident source and the pure
 * summary selector. The source supplies only already-normalized candidates for
 * the declared node scope; this module never queries endpoints or widens scope.
 *
 * @module @deepseek-ai/dsh-compaction-basic/resident-observation
 */

import {
  selectResidentSummary,
  type Candidate,
  type ResidentPrecondition,
  type SelectionResult,
} from './resident-selector.ts'

/** One authorized, registered-node observation request. */
export interface ResidentObservationRequest {
  /** Gateway identity shared by every route in the snapshot. */
  readonly gatewayId: string
  /** Exactly the registered nodes the caller authorized. */
  readonly nodeIds: readonly string[]
  /** Cooperative cancellation for the transport call. */
  readonly signal?: AbortSignal
}

/**
 * Provider of normalized resident candidates. The implementation owns
 * transport, authorization and normalization; this module owns only scope.
 */
export interface ResidentObservationSource {
  /**
   * Read one bounded snapshot for the declared nodes.
   * @param request - gateway, authorized node ids and cancellation.
   * @returns normalized candidates; rows may be ineligible and are filtered by the selector.
   */
  observe(request: ResidentObservationRequest): Promise<readonly Candidate[]>
}

/** One selection request before observations are joined. */
export interface ResidentSelectionRequest {
  /** Gateway identity used for route binding. */
  readonly gatewayId: string
  /** Conversation-pinned node; used as the whole scope when no allowlist is supplied. */
  readonly pinnedNodeId?: string
  /** Explicit compaction allowlist; replaces, never widens, the pin. */
  readonly allowedNodeIds?: readonly string[]
  /** Configurable quality preference, most preferred first. */
  readonly preferredVariantIds?: readonly string[]
  /** Identity of the already projected summary envelope. */
  readonly projectionId: string
  /** Output tokens reserved on the summarizer call. */
  readonly outputTokens: number
  /** Additional tokens reserved beyond the measured projection. */
  readonly reserveTokens: number
  /** Maximum observation age accepted by the selector. */
  readonly maxAgeMs: number
}

/** Selection evidence identifying the exact accepted deployment. */
export interface ResidentSelectionReceipt {
  readonly alias: string
  readonly gatewayId: string
  readonly routeId: string
  readonly nodeId: string
  readonly nodeName: string
  readonly instance: string
  readonly servingVariantId: string
  readonly servingVariantRevision: string
  readonly modelArtifact: string
  readonly artifactRevision: string
  readonly targetConfigRevision: string
  readonly residentGeneration: string
  /** Authority behind the capacity numbers in the accepted observation. */
  readonly budgetSource: 'resident-only-observation'
  readonly projectionId: string
  readonly reason: Extract<SelectionResult, { ok: true }>['reason']
  readonly precondition: ResidentPrecondition
}

/** Exact selection or a typed refusal. */
export type ResidentSourceSelection =
  | {
    readonly ok: true
    readonly target: Extract<SelectionResult, { ok: true }>['target']
    readonly receipt: ResidentSelectionReceipt
  }
  | { readonly ok: false; readonly code: Extract<SelectionResult, { ok: false }>['code'] }

/**
 * Freeze one accepted candidate into the selection receipt.
 * @param winner - the exact candidate row the selector chose.
 * @param reason - selector ranking reason.
 * @param projectionId - the projected summary envelope identity.
 * @returns an immutable receipt with the complete v1 identity.
 */
function receiptFor(
  winner: Candidate,
  reason: ResidentSelectionReceipt['reason'],
  projectionId: string,
): ResidentSelectionReceipt {
  const identity = winner.host.identity
  return Object.freeze({
    alias: winner.route.alias,
    gatewayId: winner.route.gatewayId,
    routeId: winner.route.id,
    nodeId: winner.node.id,
    nodeName: winner.node.name,
    instance: identity.instance,
    servingVariantId: identity.serving_variant_id,
    servingVariantRevision: identity.serving_variant_revision,
    modelArtifact: identity.model_artifact,
    artifactRevision: identity.artifact_revision,
    targetConfigRevision: identity.target_config_revision,
    residentGeneration: identity.resident_generation,
    budgetSource: 'resident-only-observation',
    projectionId,
    reason,
    precondition: winner.host.identity,
  })
}

/**
 * Observe the authorized scope once and select a resident summarizer.
 *
 * A missing scope refuses before any observation. The source sees exactly the
 * pinned node or the explicit allowlist, never the whole network.
 * @param request - scope, projection identity and budget accounting.
 * @param source - trusted observation provider.
 * @param options - cooperative cancellation and an injectable clock for tests.
 * @returns the exact selection with its receipt, or a typed refusal.
 */
export async function selectResidentFromSource(
  request: ResidentSelectionRequest,
  source: ResidentObservationSource,
  options: { readonly signal?: AbortSignal; readonly now?: () => number } = {},
): Promise<ResidentSourceSelection> {
  const scope = request.allowedNodeIds ?? (request.pinnedNodeId === undefined ? undefined : [request.pinnedNodeId])
  if (scope === undefined) return { ok: false, code: 'resident_summary_scope_required' }
  const nodeIds = [...new Set(scope)]
  const candidates = await source.observe({
    gatewayId: request.gatewayId,
    nodeIds,
    ...options.signal === undefined ? {} : { signal: options.signal },
  })
  const result = selectResidentSummary({
    nowMs: (options.now ?? Date.now)(),
    maxAgeMs: request.maxAgeMs,
    gatewayId: request.gatewayId,
    ...request.pinnedNodeId === undefined ? {} : { pinnedNodeId: request.pinnedNodeId },
    ...request.allowedNodeIds === undefined ? {} : { allowedNodeIds: request.allowedNodeIds },
    ...request.preferredVariantIds === undefined ? {} : { preferredVariantIds: request.preferredVariantIds },
    projectionId: request.projectionId,
    outputTokens: request.outputTokens,
    reserveTokens: request.reserveTokens,
    candidates: [...candidates],
  })
  if (!result.ok) return result
  const winner = candidates.find(candidate => candidate.route.id === result.target.routeId)
  if (winner === undefined) return { ok: false, code: 'resident_summary_no_eligible' }
  return { ok: true, target: result.target, receipt: receiptFor(winner, result.reason, request.projectionId) }
}
