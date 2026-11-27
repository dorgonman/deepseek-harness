/**
 * Heuristic composition of the current retained surface, independent of route
 * image pricing and provider usage. The state keeps the surviving system
 * prompts in surface order and one running surface total; every other node is
 * priced only as a delta, so state size follows the system prompts, not the
 * message count.
 */

import { z } from 'zod'
import { canonicalHeader, deriveEventMessage, isSurfaceEvent, SessionSeq } from '@deepseek-ai/dsh-session'
import type { SurfaceEvent } from '@deepseek-ai/dsh-session'
import type { ProjectionDefinition } from '@deepseek-ai/dsh-session-projection'
// Type-only: the `compaction/*` SessionEventMap merges (shadow-price events).
import type {} from '@deepseek-ai/dsh-compaction'
import { estimateMessage, estimateToolsTokens } from './estimate.ts'
import { checkpointPriceTransition, meteredCheckpointSchema, type MeteredCheckpoint } from './surface-projection.ts'
// Type-only: the client projection key and value declaration.
import type {} from './projection.ts'

const tokenCount = z.number().int().nonnegative()
const sessionSeq = z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER).transform(SessionSeq)

const breakdownSchema = z.object({
  systemTokens: tokenCount,
  toolsTokens: tokenCount,
  messageTokens: tokenCount,
}).strict()

/** One surviving `system/message` node and its heuristic price. */
const systemNodeSchema = z.object({
  seq: sessionSeq,
  tokens: tokenCount,
}).strict()

/**
 * Shadow price armed by the `compaction/summary` or `compaction/prune` event
 * immediately before a replacement: the replaced range, its heuristic price,
 * and the surviving system prompts inside it.
 */
const claimSchema = z.object({
  start: sessionSeq,
  end: sessionSeq,
  tokens: tokenCount,
  systemSeqs: z.array(sessionSeq),
}).strict()

type SystemNode = z.infer<typeof systemNodeSchema>
type BreakdownClaim = z.infer<typeof claimSchema>

interface ContextBreakdownState {
  systems: SystemNode[]
  surfaceTokens: number
  claim?: BreakdownClaim | undefined
  checkpoint?: MeteredCheckpoint<{ deltaTokens: number; systemSeqs: SystemNode['seq'][] }> | undefined
  breakdown: z.infer<typeof breakdownSchema>
}

/** Plain-JSON checkpoint: surviving system prompts, the surface total, and any pending price. */
const contextBreakdownStateSchema: z.ZodType<ContextBreakdownState> = z.object({
  systems: z.array(systemNodeSchema),
  surfaceTokens: tokenCount,
  claim: claimSchema.optional(),
  checkpoint: meteredCheckpointSchema(z.object({
    deltaTokens: z.number().int(),
    systemSeqs: z.array(sessionSeq),
  }).strict()).optional(),
  breakdown: breakdownSchema,
}).strict()

/** The system prompts and surface total after one surface event. */
interface SurfaceTotals {
  readonly systems: SystemNode[]
  readonly surfaceTokens: number
}

/**
 * Place one surface event's priced node.
 * @param state - state before the event, without its armed claim.
 * @param claim - shadow price armed by the immediately preceding event, if any.
 * @param event - the surface event.
 * @param tokens - the event message's heuristic price.
 * @returns the system prompts and surface total after the event.
 * @throws when a metered replacement names another range than its claim, or
 *   puts a system message in place of a metered range.
 */
function placeSurfaceEvent(
  state: ContextBreakdownState,
  claim: BreakdownClaim | undefined,
  event: SurfaceEvent,
  tokens: number,
): SurfaceTotals {
  const op = event.surfaceOp
  const node: SystemNode = { seq: event.seq, tokens }
  const isSystem = event.type === 'system/message'
  if (op === 'append') {
    return { systems: isSystem ? [...state.systems, node] : state.systems, surfaceTokens: state.surfaceTokens + tokens }
  }
  if (claim !== undefined) {
    if (claim.start !== op.startSeq || claim.end !== op.endSeq) {
      throw new Error(
        `context breakdown: replace at seq ${event.seq} over range ${op.startSeq}-${op.endSeq} has no adjacent shadow price`
        + ` (armed claim covers ${claim.start}-${claim.end})`,
      )
    }
    if (isSystem) {
      throw new Error(
        `context breakdown: replace at seq ${event.seq} puts a system message in place of a metered range;`
        + ' its order among the surviving system prompts is unknown',
      )
    }
    const shadowed = new Set<number>(claim.systemSeqs)
    return {
      systems: state.systems.filter(system => !shadowed.has(system.seq)),
      surfaceTokens: state.surfaceTokens + tokens - claim.tokens,
    }
  }
  const index = op.startSeq === op.endSeq ? state.systems.findIndex(system => system.seq === op.startSeq) : -1
  const replaced = state.systems[index]
  if (replaced === undefined) {
    // Bounded state cannot price an unmetered non-system range, so it folds at
    // zero delta; current producers meter every such replacement.
    return { systems: state.systems, surfaceTokens: state.surfaceTokens }
  }
  return {
    systems: isSystem ? state.systems.with(index, node) : state.systems.toSpliced(index, 1),
    surfaceTokens: state.surfaceTokens + tokens - replaced.tokens,
  }
}

/** Publish totals, keeping the previous breakdown object when its figures are unchanged. */
function withTotals(state: ContextBreakdownState, totals: SurfaceTotals): ContextBreakdownState {
  const surfaceTokens = Math.max(0, totals.surfaceTokens)
  const systemTokens = totals.systems.findLast(system => system.tokens > 0)?.tokens ?? 0
  const messageTokens = Math.max(0, surfaceTokens - systemTokens)
  const breakdown = systemTokens === state.breakdown.systemTokens && messageTokens === state.breakdown.messageTokens
    ? state.breakdown
    : { systemTokens, toolsTokens: state.breakdown.toolsTokens, messageTokens }
  return { ...state, systems: totals.systems, surfaceTokens, breakdown }
}

/** Drop an armed claim; the state is returned unchanged when none is armed. */
function withoutClaim(state: ContextBreakdownState): ContextBreakdownState {
  if (state.claim === undefined) return state
  const { claim: _expired, ...unclaimed } = state
  return unclaimed
}

/**
 * Context composition with the last nonempty surviving system in surface
 * order classified as system tokens; all other visible prices are messages.
 * Appends add their price; a replacement subtracts the shadow price of the
 * adjacent `compaction/summary` or `compaction/prune` event, whose
 * `shadowedSeqs` also name the system prompts it removes, and a system-prompt
 * rewrite subtracts the retained price of the system node it replaces. Tools
 * are priced from the latest request header. No route pricing applies.
 */
export const contextBreakdownProjectionDefinition = {
  key: 'contextBreakdown',
  stateVersion: 7,
  stateSchema: contextBreakdownStateSchema,
  init: (): ContextBreakdownState => ({
    systems: [],
    surfaceTokens: 0,
    breakdown: { systemTokens: 0, toolsTokens: 0, messageTokens: 0 },
  }),
  apply: (state, event) => {
    const transition = checkpointPriceTransition(state.checkpoint, event, () => {
      if (!isSurfaceEvent(event)) throw new Error('pending compaction checkpoint has no surface operation')
      const message = deriveEventMessage(event)
      const tokens = message === null ? 0 : estimateMessage(message)
      const totals = placeSurfaceEvent(state, state.claim, event, tokens)
      return { deltaTokens: totals.surfaceTokens - state.surfaceTokens, systemSeqs: state.claim?.systemSeqs ?? [] }
    })
    if (transition !== undefined) {
      const { checkpoint: _previous, ...rest } = state
      state = transition.checkpoint === undefined ? rest : { ...rest, checkpoint: transition.checkpoint }
    }
    if (transition?.suppress === true) return withoutClaim(state)
    if (transition?.publish !== undefined) {
      const removed = new Set(transition.publish.systemSeqs)
      return withTotals(withoutClaim(state), {
        systems: state.systems.filter(system => !removed.has(system.seq)),
        surfaceTokens: state.surfaceTokens + transition.publish.deltaTokens,
      })
    }
    if (event.type === 'compaction/summary' || event.type === 'compaction/prune') {
      const shadowed = new Set<number>(event.data.shadowedSeqs)
      return {
        ...state,
        claim: {
          start: SessionSeq(event.data.shadowedRange.start),
          end: SessionSeq(event.data.shadowedRange.end),
          tokens: event.data.shadowedTokenCount,
          systemSeqs: state.systems.filter(system => shadowed.has(system.seq)).map(system => system.seq),
        },
      }
    }
    // A claim prices only the event immediately after its metering event.
    const unclaimed = withoutClaim(state)
    if (event.type === 'request/header') {
      const toolsTokens = estimateToolsTokens(canonicalHeader(event.data.header))
      return toolsTokens === unclaimed.breakdown.toolsTokens
        ? unclaimed
        : { ...unclaimed, breakdown: { ...unclaimed.breakdown, toolsTokens } }
    }
    if (!isSurfaceEvent(event)) return unclaimed
    const message = deriveEventMessage(event)
    const tokens = message === null ? 0 : estimateMessage(message)
    return withTotals(unclaimed, placeSurfaceEvent(unclaimed, state.claim, event, tokens))
  },
  wire: {
    viewSchema: breakdownSchema,
    view: state => state.breakdown,
    summary: false,
  },
} satisfies ProjectionDefinition<'contextBreakdown', ContextBreakdownState>
