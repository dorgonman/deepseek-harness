/**
 * Basic replay-aware compaction backend.
 *
 * @module @deepseek-ai/dsh-compaction-basic
 */

import { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { CompactionEngine, ManualCompactionError } from '@deepseek-ai/dsh-compaction'
import type { CompactionResult, CompactionTrigger } from '@deepseek-ai/dsh-compaction'
import type { EpochHeader, Session, SessionSeq } from '@deepseek-ai/dsh-session'
import { CONTEXT_WINDOW_EXCEEDED_CODE, LlmError } from '@deepseek-ai/dsh-llm'
import type { LlmCallConfig, LlmResolvedModelInfo } from '@deepseek-ai/dsh-llm'
import { assertNever } from '@deepseek-ai/dsh-util-values'
import type { Agent, PendingAgentRequest, PreStepDecision } from '@deepseek-ai/dsh-agent'
import type { CommandId } from '@deepseek-ai/dsh-commands/brand'
// Type-only: makes the optional sibling service available to `ctx.get()`.
import type {} from '@deepseek-ai/dsh-compaction-tool-result-pruner'
import {
  resolveCompactSpec,
  resolveConfig,
  resolveSummaryMaxTokens,
  resolveTargetPolicy,
  TargetPressureConfigError,
} from './config.ts'
import {
  assertNoActiveCompaction,
  compactSurfaceRegion,
  selectCompactableRange,
} from './region.ts'
import type { RegionSummary, SurfaceSelection } from './region.ts'
import { summarizeWithLlm } from './summarizer.ts'
import type { SummarizationInput, SummaryResult } from './summarizer.ts'
import type {
  BasicCompactionConfig,
  ModelCompactPolicyConfig,
  ResolvedConfig,
} from './types.ts'

export { CompactionPolicyControl } from './policy-control.ts'

export type {
  BasicCompactionConfig,
  CompactionPolicyConfig,
  ModelCompactPolicyConfig,
  ResolvedCompactSpec,
  ResolvedConfig,
  ResolvedRetention,
  ResolvedTargetPolicy,
} from './types.ts'

/** Resolve the exact provider/model durably routed for the latest request. */
function routedTarget(
  session: Session,
): Pick<LlmCallConfig, 'provider' | 'model'> | undefined {
  const config = session.requestHeader()?.config
  if (config === undefined || config.provider.length === 0 || config.model.length === 0) {
    return undefined
  }
  return { provider: config.provider, model: config.model }
}

/**
 * Output tokens the routed request reserves, which the provider charges to the
 * same window as the prompt. The effective envelope's own cap wins; otherwise
 * the adapter's request default or the transport's own fallback, which the
 * adapter materializes when that envelope omits one. No declared cap means no
 * reservation.
 */
function reservedCompletionTokens(header: EpochHeader, fallbackMaxTokens: number | undefined): number {
  return header.config.maxTokens ?? fallbackMaxTokens ?? 0
}

/** Resolve the conversation target used to select an optional policy override. */
function conversationTarget(
  agent: Agent,
): Pick<LlmCallConfig, 'provider' | 'model'> | undefined {
  const routed = routedTarget(agent.session)
  if (routed !== undefined) return routed
  if (agent.options.provider === undefined || agent.options.provider.length === 0
    || agent.options.model === undefined || agent.options.model.length === 0) return undefined
  return { provider: agent.options.provider, model: agent.options.model }
}

/** Per-failure phase tag for the model-switch receipt, without changing error identity. */
const compactionStages = new WeakMap<object, string>()

/** Attach a receipt stage to a thrown error and return it unchanged for rethrowing. */
function tagStage(error: unknown, stage: string): unknown {
  if (error !== null && typeof error === 'object') compactionStages.set(error, stage)
  return error
}

/** Read the receipt stage tagged on a thrown error. */
function stageOf(error: unknown): string | undefined {
  return error !== null && typeof error === 'object' ? compactionStages.get(error) : undefined
}

const thresholdRatioSchema = z.number()
const headroomTokensSchema = z.number().step(1).min(0)
const retainRatioSchema = z.number()
const retainTokensSchema = z.number().step(1).min(0)
const summarizationProviderSchema = z.string()
const summarizationModelSchema = z.string()
const summarizationFallbackSchema = z.union([z.const('latest-route'), z.const('error')])
const maxTokensSchema = z.number().step(1).min(1)
const compactionRetriesSchema = z.number().step(1).min(0)
const maxOverflowRetriesSchema = z.number().step(1).min(0)

const modelPolicy: z<ModelCompactPolicyConfig> = z.object({
  provider: z.string().required(),
  model: z.string().required(),
  thresholdRatio: thresholdRatioSchema,
  headroomTokens: headroomTokensSchema,
  retainRatio: retainRatioSchema,
  retainTokens: retainTokensSchema,
  summarizationProvider: summarizationProviderSchema,
  summarizationModel: summarizationModelSchema,
  summarizationFallback: summarizationFallbackSchema,
  maxTokens: maxTokensSchema,
  compactionRetries: compactionRetriesSchema,
  maxOverflowRetries: maxOverflowRetriesSchema,
})

/**
 * Dependency-light compaction backend using `ctx.tokenMeter` for pressure,
 * retention, cited source events, and summary-convergence pricing.
 *
 * The region transaction owns replay, admission, and durable mutation.
 */
export class BasicCompactionEngine extends CompactionEngine {
  static inject = ['llm', 'tokenMeter', 'sessions']

  static Config: z<BasicCompactionConfig> = z.object({
    thresholdRatio: thresholdRatioSchema,
    headroomTokens: headroomTokensSchema,
    retainRatio: retainRatioSchema,
    retainTokens: retainTokensSchema,
    summarizationProvider: summarizationProviderSchema,
    summarizationModel: summarizationModelSchema,
    summarizationFallback: summarizationFallbackSchema,
    maxTokens: maxTokensSchema,
    compactionRetries: compactionRetriesSchema,
    maxOverflowRetries: maxOverflowRetriesSchema,
    modelPolicies: z.array(modelPolicy),
    auto: z.boolean(),
  })

  /** Resolved and validated compaction configuration. */
  readonly config: ResolvedConfig

  private readonly warnedPressureConfigTargets = new Set<string>()
  private readonly overflowRetries = new WeakMap<Agent, number>()
  private readonly overflowAgents = new WeakMap<Session, Agent>()

  constructor(ctx: Context, config: BasicCompactionConfig = {}) {
    super(ctx)
    this.config = resolveConfig(config)
    if (this.config.auto) this._registerAutomaticCompaction()
  }

  /**
   * Register between-step pressure, model-switch pressure, and request-overflow
   * recovery. Pre-step and overflow listeners dispatch through `compactIfNeeded`;
   * model switches price the proposed header directly. All summaries use `summarize`.
   */
  private _registerAutomaticCompaction(): void {
    const { ctx } = this
    const logResult = (result: CompactionResult, trigger: string): void => {
      ctx.logger.info(
        `compaction (${trigger}): shadowed ${result.shadowedSeqs.length} surface nodes `
        + `(seqs ${result.shadowedRange.start}-${result.shadowedRange.end}, `
        + `~${result.shadowedTokenCount} tokens)`,
      )
    }

    ctx.on('agent/pre-step', async (
      { agent, signal },
      next,
    ): Promise<PreStepDecision> => {
      if (!signal.aborted) {
        try {
          const result = await this.compactIfNeeded(agent, 'pressure', signal)
          if (result !== null) logResult(result, 'step pressure')
        } catch (error: unknown) {
          if (this.stopAfterWriteFailure(error)) throw error
          if (error instanceof TargetPressureConfigError) {
            if (this.warnedPressureConfigTargets.has(error.targetKey)) return next()
            this.warnedPressureConfigTargets.add(error.targetKey)
          }
          const message = error instanceof Error ? error.message : String(error)
          ctx.logger.warn(`step compaction failed: ${message}; continuing the turn`)
        }
      }
      return next()
    })

    // Last gate: price the exact envelope — pending input, system/runtime
    // context, tools, retained history and the bound completion reservation.
    // `preview` may commit a checkpoint (the loop rebuilds and re-admits);
    // `verify` sees the frozen request and may only refuse.
    ctx.on('agent/request-envelope', async ({ agent, header, view, contextWindow, completionMaxTokens, completionSource, phase, signal }) => {
      if (signal.aborted || contextWindow === undefined) return
      const meter = this.ctx.tokenMeter
      const prompt = view.deriveMessages().reduce((total, message) => total + meter.estimateMessage(message), 0)
        + meter.estimateTools(header.tools)
      const reserve = completionMaxTokens ?? 0
      const total = prompt + reserve
      if (total <= contextWindow) return
      const detail = `~${total} tokens (${prompt} request + ${reserve} reserved, ${completionSource}) `
        + `exceed the ${contextWindow}-token window`
      if (phase === 'verify') {
        throw new LlmError(
          `compaction: request refused before dispatch; ${detail}`,
          'CONTEXT_ADMISSION_REFUSED',
        )
      }
      // Prefer trimming oversized tool output over shadowing history: a trim
      // changes the Surface, and the loop rebuilds and prices it again.
      const pruner = this.ctx.get('toolResultPruner')
      if (pruner !== undefined) {
        const generation = agent.session.surface.contentGeneration
        pruner.pruneSession(agent.session)
        if (agent.session.surface.contentGeneration !== generation) return
      }
      let compacted: CompactionResult | null
      try {
        compacted = await this.compactIfNeeded(agent, 'pressure', signal)
        // The pressure threshold can sit above the exact envelope: the durable
        // surface is still "cheap enough" while pending input pushes this
        // request over. Fall back to a range selected for the envelope itself.
        if (compacted === null) compacted = await this.compactEnvelopeOverflow(agent, header, signal)
      } catch (error: unknown) {
        signal.throwIfAborted()
        const reason = error instanceof Error ? error.message : String(error)
        throw new LlmError(
          `compaction: request refused before dispatch; ${detail} and compaction failed: ${reason}`,
          'CONTEXT_ADMISSION_REFUSED',
        )
      }
      if (compacted === null) {
        throw new LlmError(
          `compaction: request refused before dispatch; ${detail} and no safe range remains`,
          'CONTEXT_ADMISSION_REFUSED',
        )
      }
      logResult(compacted, 'request envelope')
    })

    ctx.on('agent/status', ({ agent, status }) => {
      if (status === 'idle') this.overflowRetries.delete(agent)
    })

    // Prepend wraps routing listeners: await their final selection while the
    // committed header still identifies the model that can summarize the history.
    ctx.on('agent/request', async ({ agent, signal }, next) => {
      const config = await next()
      if (!this.usesLegacySwitchCompaction()) return config
      const previous = agent.session.requestHeader()
      if (signal.aborted || previous === undefined || config.provider === '' || config.model === ''
        || (previous.config.provider === config.provider && previous.config.model === config.model)) return config
      try {
        const header = { ...previous, config }
        const result = await this.compactPressure(agent, header, signal, true)
        if (result !== null) logResult(result, 'model switch')
      } catch (error: unknown) {
        signal.throwIfAborted()
        const message = error instanceof Error ? error.message : String(error)
        // A switch receipt names both routes and the summarizer that actually
        // governed the failed step, so an operator can tell a destination
        // capacity problem from a summarizer service or policy problem.
        const from = `${previous.config.provider}/${previous.config.model}`
        const to = `${config.provider}/${config.model}`
        const summaryPolicy = resolveTargetPolicy(this.config, previous.config)
        const summarizer = summaryPolicy.summarizationProvider === ''
          ? summaryPolicy.summarizationFallback === 'error'
            ? 'unconfigured (fallback disabled)'
            : 'latest routed request'
          : `${summaryPolicy.summarizationProvider}/${summaryPolicy.summarizationModel}`
        const code: unknown = (error as { code?: unknown }).code
        const stage = error instanceof TargetPressureConfigError
          ? 'destination capacity preflight'
          : typeof code === 'string' && code.startsWith('SUMMARY_')
            ? 'summarizer preflight'
            : stageOf(error) ?? 'summary generation or apply'
        throw new LlmError(
          `Cannot switch from ${from} to ${to}: compaction failed during ${stage} `
          + `(summarizer ${summarizer}): ${message}. `
          + 'The previous request model is preserved; retry or choose a different model.',
          'MODEL_SWITCH_COMPACTION_FAILED',
          { cause: error },
        )
      }
      return config
    }, { prepend: true })

    // A successful response starts a fresh overflow-recovery sequence even
    // when tool calls continue the same turn into another request.
    ctx.on('session/event', (session, event) => {
      if (event.type !== 'assistant/message') return
      const agent = this.overflowAgents.get(session)
      if (agent !== undefined) this.overflowRetries.delete(agent)
    })

    ctx.on('agent/request-error', async (
      { agent, failure, signal },
      next,
    ) => {
      const isCancelled = (): boolean => signal.aborted
      if (failure.code !== CONTEXT_WINDOW_EXCEEDED_CODE || isCancelled()) return next()
      this.overflowAgents.set(agent.session, agent)
      const target = routedTarget(agent.session)
      if (target === undefined) return next()
      const policy = resolveTargetPolicy(this.config, target)
      const retries = this.overflowRetries.get(agent) ?? 0
      if (retries >= policy.maxOverflowRetries) return next()

      const generation = agent.session.surface.replaceGeneration
      let result: CompactionResult | null
      try {
        result = await this.compactIfNeeded(agent, 'context-overflow', signal)
      } catch (recoveryError: unknown) {
        const message = recoveryError instanceof Error ? recoveryError.message : String(recoveryError)
        if (this.stopAfterWriteFailure(recoveryError)) {
          ctx.logger.warn(`context-overflow compaction write was not confirmed: ${message}; refusing retry`)
          return undefined
        }
        // A model-free prune can land before later summary work fails. That
        // durable reduction is sufficient retry proof; do not discard it just
        // because the optional second phase threw. Cancellation still wins.
        if (!isCancelled() && agent.session.surface.replaceGeneration > generation) {
          ctx.logger.warn(
            `context-overflow compaction failed after durable surface progress: ${message}; `
            + 'retrying from the replacement surface',
          )
          this.overflowRetries.set(agent, retries + 1)
          return { kind: 'retry' }
        }
        ctx.logger.warn(
          `context-overflow compaction failed: ${message}; ${isCancelled()
            ? 'cancellation prevents retry'
            : 'preserving the original request error'}`,
        )
        return next()
      }
      if (isCancelled()
        || agent.session.surface.replaceGeneration <= generation) return next()
      if (result !== null) logResult(result, 'context overflow recovery')
      this.overflowRetries.set(agent, retries + 1)
      return { kind: 'retry' }
    })
  }

  /**
   * Summarize the selected region through one auxiliary LLM call.
   * Ordinary Chat targets with nativeTools:false receive quoted text
   * history; other routes retain the conversation prefix.
   * Override this sole hook for a template or remote summarizer.
   * @param input - replayed conversation prefix (system, tools, and leading messages) to condense.
   * @param agent - supplies routed-model history, fallback model, and session id.
   * @param signal - optional cancellation forwarded to the adapter.
   * @returns safe text summary blocks and the exact auxiliary call envelope and output.
   */
  protected async summarize(
    input: SummarizationInput,
    agent: Agent,
    signal?: AbortSignal,
  ): Promise<SummaryResult> {
    if (this.config.summarizationMode === 'resident-preferred') {
      throw new Error('compaction: resident-preferred summarization requires the Hybrid backend')
    }
    const target = conversationTarget(agent)
    if (target === undefined) return summarizeWithLlm(this.ctx, this.config, input, agent, signal)
    const policy = resolveTargetPolicy(this.config, target)
    if (policy.adaptiveHeadroom !== 'pressure-and-summary') {
      return summarizeWithLlm(this.ctx, policy, input, agent, signal)
    }
    const info = await this.ctx.llm.resolveModelInfo(target.provider, target.model, signal).catch(() => {
      // The stream owns route redirection and dispatch failures; sizing is optional.
      signal?.throwIfAborted()
      return undefined
    })
    if (info?.context === undefined) return summarizeWithLlm(this.ctx, policy, input, agent, signal)
    const reserve = agent.session.requestHeader()?.config.maxTokens
      ?? info.defaultMaxTokens ?? info.transportMaxTokens ?? 0
    const maxTokens = resolveSummaryMaxTokens(policy, info.context.contextWindow - reserve)
    return summarizeWithLlm(this.ctx, { ...policy, maxTokens }, input, agent, signal)
  }

  /** Summarize one verified Surface selection using the ordinary Basic backend. */
  protected summarizeSelection(
    input: SummarizationInput,
    agent: Agent,
    _selection: SurfaceSelection,
    _destination: EpochHeader | undefined,
    signal?: AbortSignal,
    _admission?: PendingAgentRequest,
  ): Promise<RegionSummary> {
    return this.summarize(input, agent, signal)
  }

  /** Dispatch pressure compaction while retaining the proposed request header. */
  protected compactPressureRegion(
    start: SessionSeq, end: SessionSeq, agent: Agent, _destination: EpochHeader, signal: AbortSignal,
  ): Promise<CompactionResult> {
    return this.compactRegion(start, end, agent, signal)
  }

  /**
   * Compact one range selected for an exact-envelope overflow rather than for
   * the pressure threshold. Retains nothing: the loop rebuilds the envelope and
   * prices it again, so an unnecessary second reduction is bounded by the
   * caller's rebuild budget.
   * @param agent - agent whose Surface is reduced.
   * @param header - proposed envelope the reduction is priced against.
   * @param signal - current turn cancellation.
   * @returns the committed result, or null when no safe range remains.
   */
  private async compactEnvelopeOverflow(
    agent: Agent,
    header: EpochHeader,
    signal: AbortSignal,
  ): Promise<CompactionResult | null> {
    signal.throwIfAborted()
    const measurement = this.ctx.tokenMeter.measure(agent.session, header)
    const range = selectCompactableRange(agent.session, measurement, 0)
    if (range === null) return null
    return this.compactPressureRegion(range.start, range.end, agent, header, signal)
  }
  /** Supply a durability checkpoint for automatic and explicit transactions. */
  protected regionFlush(_agent: Agent): (() => Promise<void>) | undefined {
    return undefined
  }

  /** Decide whether a failed compaction write precludes further automatic dispatch. */
  protected stopAfterWriteFailure(_error: unknown): boolean {
    return false
  }

  /** Basic sizes a model switch before adapter and prompt admission. */
  protected usesLegacySwitchCompaction(): boolean {
    return true
  }

  /**
   * Compact for replayed step-boundary pressure or one provider-confirmed context
   * overflow. Both triggers price the latest durable routed request envelope;
   * overflow bypasses the normal threshold and retained-tail policy so it can
   * force one useful balanced reduction.
   * @param agent - agent whose latest durable routed request is measured.
   * @param trigger - normal step-boundary pressure or context-overflow recovery.
   * @param signal - live turn cancellation signal forwarded to summarization.
   * @returns the latest summary compaction result, or `null` when no summary ran.
   */
  override async compactIfNeeded(
    agent: Agent,
    trigger: CompactionTrigger,
    signal: AbortSignal,
  ): Promise<CompactionResult | null> {
    const header = agent.session.requestHeader()
    if (header === undefined || routedTarget(agent.session) === undefined) return null
    switch (trigger) {
      case 'pressure':
        return this.compactPressure(agent, header, signal, false)
      case 'context-overflow': {
        this.ctx.get('toolResultPruner')?.pruneSession(agent.session)
        const measurement = this.ctx.tokenMeter.measure(agent.session)
        const range = selectCompactableRange(agent.session, measurement, 0)
        if (range === null) return null
        return this.compactRegion(range.start, range.end, agent, signal)
      }
      /* v8 ignore next -- closed-union exhaustiveness guard */
      default:
        assertNever(trigger, 'compaction trigger')
    }
  }

  /** Price the proposed header; switch checks require history below its threshold. */
  private async compactPressure(
    agent: Agent,
    header: EpochHeader,
    signal: AbortSignal,
    requireBelowThreshold: boolean,
  ): Promise<CompactionResult | null> {
    const target = header.config
    const policy = resolveTargetPolicy(this.config, target)
    const meter = this.ctx.tokenMeter
    let measurement = meter.measure(agent.session, header)
    const prune = this.ctx.get('toolResultPruner')
    let info: LlmResolvedModelInfo
    try {
      info = await this.ctx.llm.resolveModelInfo(target.provider, target.model, signal)
    } catch (error: unknown) {
      throw tagStage(error, 'destination model lookup')
    }
    try {
      assertNoActiveCompaction(agent.session, 'automatic pressure compaction')
    } catch (error: unknown) {
      throw tagStage(error, 'active compaction conflict')
    }
    const targetKey = `${target.provider}/${target.model}`
    if (info.context === undefined) {
      throw new TargetPressureConfigError(
        targetKey,
        `compaction-basic: no context capacity for ${targetKey}; `
        + 'configure contextWindow on that adapter model',
      )
    }
    const spec = resolveCompactSpec(
      policy,
      info.context.contextWindow,
      reservedCompletionTokens(header, info.defaultMaxTokens ?? info.transportMaxTokens),
    )
    if (measurement.totalTokens < spec.thresholdTokens) return null

    // Once pressure qualifies, land the model-free pass before choosing a
    // summary range, then remeasure through the singleton replay fold.
    if (prune !== undefined) {
      prune.pruneSession(agent.session)
      measurement = meter.measure(agent.session, header)
    }
    if (measurement.totalTokens < spec.thresholdTokens) return null

    let result: CompactionResult | null = null
    for (let attempt = 0; attempt <= spec.compactionRetries; attempt += 1) {
      const range = selectCompactableRange(agent.session, measurement, spec.retainTokens)
      if (range === null) {
        if (result === null && !requireBelowThreshold) return null
        throw new Error(
          'no safe compaction range can reduce context below threshold '
          + `(${measurement.totalTokens} estimated tokens >= threshold ${spec.thresholdTokens})`,
        )
      }
      result = await this.compactPressureRegion(range.start, range.end, agent, header, signal)
      measurement = meter.measure(agent.session, header)
      if (measurement.totalTokens < spec.thresholdTokens) return result
    }

    throw new Error(
      `compaction still above threshold after ${spec.compactionRetries + 1} compaction attempts `
      + `(${measurement.totalTokens} estimated tokens >= threshold ${spec.thresholdTokens})`,
    )
  }

  /**
   * Compact one inclusive positional range from the agent-owned surface using
   * the effective token meter for all retention and shrink pricing.
   * @param start - inclusive first surface-node seq.
   * @param end - inclusive last surface-node seq.
   * @param agent - owner of the target session, used by the summarizer.
   * @param signal - optional summarization cancellation signal.
   * @param destination - proposed request header when compaction targets another route.
   * @param admission - pending switch request used to price the replacement before commit.
   * @returns the successful durable compaction result.
   */
  override async compactRegion(
    start: SessionSeq,
    end: SessionSeq,
    agent: Agent,
    signal?: AbortSignal,
    destination?: EpochHeader,
    admission?: PendingAgentRequest,
  ): Promise<CompactionResult> {
    const flush = this.regionFlush(agent)
    return compactSurfaceRegion(
      this.regionDependencies(destination, admission),
      agent.session,
      start,
      end,
      agent,
      { owner: 'current-turn', stability: 'whole-surface', ...flush === undefined ? {} : { flush } },
      signal,
    )
  }

  /**
   * Force one useful idle-session compaction below the pressure threshold, and
   * resolve only after its standalone marker pair is durably checkpointed.
   * @param agent - idle agent whose next-turn admission this call reserves.
   * @param signal - cancellation scoped to this compaction request.
   * @param sourceCommandId - initiating command identity for presentation correlation.
   * @returns the committed result, or `null` when no safe useful range exists.
   */
  override compactNow(
    agent: Agent,
    signal: AbortSignal,
    sourceCommandId?: CommandId,
  ): Promise<CompactionResult | null> {
    signal.throwIfAborted()
    try {
      return agent.runMaintenance(async (agentSignal) => {
        const operationSignal = AbortSignal.any([agentSignal, signal])
        try {
          operationSignal.throwIfAborted()
          const range = selectCompactableRange(
            agent.session,
            this.ctx.tokenMeter.measure(agent.session),
            0,
          )
          if (range === null) return null
          return await compactSurfaceRegion(
            this.regionDependencies(),
            agent.session,
            range.start,
            range.end,
            agent,
            {
              owner: null,
              stability: 'selected-span',
              ...sourceCommandId === undefined ? {} : { sourceCommandId },
              flush: async () => {
                await this.ctx.sessions.flush(agent.session)
              },
            },
            operationSignal,
          )
        } catch (error: unknown) {
          if (agentSignal.aborted && operationSignal.reason === agentSignal.reason) {
            throw new ManualCompactionError(
              'cancelled',
              'manual compaction was cancelled',
              { cause: error },
            )
          }
          operationSignal.throwIfAborted()
          throw error
        }
      })
    } catch (error: unknown) {
      throw new ManualCompactionError(
        'busy',
        'manual compaction requires an idle agent with no waking queued work',
        { cause: error },
      )
    }
  }

  /** Bind the effective token meter and dynamically dispatched summarizer hook. */
  private regionDependencies(
    destination?: EpochHeader, admission?: PendingAgentRequest,
  ): Parameters<typeof compactSurfaceRegion>[0] {
    return {
      meter: this.ctx.tokenMeter,
      summarize: (input, owner, selection, abort) => this.summarizeSelection(
        input, owner, selection, destination ?? owner.session.requestHeader(), abort, admission,
      ),
      recover: (error, agent, sourceEventSeqs, signal) => this.ctx.waterfall('compaction/summary-error', {
        session: agent.session,
        sourceEventSeqs,
        error,
        ...signal === undefined ? {} : { signal },
      }, () => false),
    }
  }
}

export default BasicCompactionEngine
