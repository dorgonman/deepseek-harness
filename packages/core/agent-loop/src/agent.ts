/**
 * Default Agent driver over queued turns and step-boundary input. Every request
 * is derived from the session log.
 * @module dsh-agent-loop/agent
 */

import type {
  Agent,
  AgentCancelCause,
  AgentEventDispatch,
  AgentOptions,
  AgentStatus,
  CancelOptions,
  ModelSelection,
  ModelSelectionProjectionState,
  InboxTarget,
  InboxState,
  PendingAgentRequest,
  PreStepDecision,
  RequestErrorAction,
} from '@deepseek-ai/dsh-agent'
import { AgentModel, agentEvents, assembleContextFor } from '@deepseek-ai/dsh-agent'
import type { GenerateOptions, LlmCallConfig, Message, PreparedLlmCall } from '@deepseek-ai/dsh-llm'
import {
  LlmError,
  createAssistantMessage,
  createDeveloperMessage,
  errorChain,
  markAgentLoopRequest,
} from '@deepseek-ai/dsh-llm'
import { assertNever, deepFreeze } from '@deepseek-ai/dsh-util-values'
import type { Scope } from '@deepseek-ai/dsh-scope'
import { createScope } from '@deepseek-ai/dsh-scope'
import type { EpochHeader, RequestContext, SessionEvent, SessionId, SessionSeq, TurnEndReason, UserMessage } from '@deepseek-ai/dsh-session'
import { canonicalHeader, headerEquals, Session, ToolCallRecovery } from '@deepseek-ai/dsh-session'
import type { SessionHandle } from '@deepseek-ai/dsh-session-persistence'
import { renderContextSections, renderPrompt } from '@deepseek-ai/dsh-system-prompt'
import type { PromptAssembly } from '@deepseek-ai/dsh-system-prompt'
import { ToolActivationError, ToolBatchExecutionError } from '@deepseek-ai/dsh-tools'
import type { ResolvedToolSet } from '@deepseek-ai/dsh-tools'
import type { Context } from '@deepseek-ai/cordis'
import { ReactLoopInbox } from './inbox.ts'
import { RuntimeContextProjection } from './runtime-context.ts'
import { AssistantStreamAttempt } from './assistant-stream.ts'
import { REPEATED_MODEL_OUTPUT, StreamRepetition } from './stream-repetition.ts'
import { SystemPromptProjection } from './runtime-context.ts'
import type { SystemPromptCommit } from './runtime-context.ts'

/** Projected facts consumed by this driver; registration stays with the factory. */
interface LoopProjections {
  /** Current model selection, including pending changes. */
  readonly modelSelection: () => ModelSelectionProjectionState
  /** Current queued and claimed input. */
  readonly inbox: () => InboxState
}

type Phase =
  | { kind: 'idle'; lastTurn: number }
  | {
    kind: 'maintenance'
    abort: AbortController
    lastTurn: number
    wakeRequested: boolean
  }
  | { kind: 'running'; abort: AbortController; turn: number; step: number; wakeRequested: boolean }

type StepEndReason = Extract<TurnEndReason, { kind: 'completed' | 'max-tokens' }>

/** Flatten the bound reservation into the envelope-admission payload. */
function completionFields(prepared: PreparedLlmCall | undefined): {
  completionMaxTokens?: number
  completionSource: 'explicit' | 'adapter-default' | 'transport-fallback' | 'unknown'
} {
  const completion = prepared?.completion
  return {
    ...completion?.maxTokens === undefined ? {} : { completionMaxTokens: completion.maxTokens },
    completionSource: completion?.source ?? 'unknown',
  }
}

/** Bounded preview rebuilds after a committed compaction inside one admission. */
const MAX_ENVELOPE_REBUILDS = 2

type PreparedStep =
  | { kind: 'reject' }
  | {
    kind: 'enter'
    messages: UserMessage[]
    startsRequestSeries?: true
    assembly: PromptAssembly
    selection: Readonly<ModelSelection>
    /** Admission shared by the step's declarations and dispatch; the turn disposes it after the step. */
    toolSet: ResolvedToolSet
  }

type EnteredStep = Extract<PreparedStep, { kind: 'enter' }>
type StepPosition = { turn: number; step: number }

interface StepAdmission {
  system: SystemPromptCommit[]
  users: UserMessage[]
}

/** Remove adapter-derived values before plugins propose the next request config. */
function requestProposal(header: EpochHeader): LlmCallConfig {
  if (header.adapterDefaults === undefined) return header.config
  const proposal = { ...header.config }
  if (header.adapterDefaults.reasoningEffort === true) delete proposal.reasoningEffort
  if (header.adapterDefaults.maxTokens === true) delete proposal.maxTokens
  return proposal
}

/**
 * Read the cause `cancel()` passed when aborting a loop-owned signal, copying
 * only the fields `turn/end` records. The live reason stays the caller's
 * object, and Node's fetch assigns a `stack` onto it that `Session.append`
 * would either log or reject as data JSON cannot hold.
 * @param signal - a turn or maintenance signal this loop owns.
 * @returns the copied cause, or undefined while the signal is still live.
 */
function abortedCancelCause(signal: AbortSignal): AgentCancelCause | undefined {
  if (!signal.aborted) return undefined
  // `cancel()` is the only aborter of the signals this loop owns.
  const cause = signal.reason as AgentCancelCause
  switch (cause.kind) {
    case 'user':
    case 'parent':
    case 'disposed':
      return { kind: cause.kind }
    case 'hook':
      return { kind: 'hook', reason: cause.reason }
    /* v8 ignore next -- cancel accepts the closed AgentCancelCause union */
    default:
      return assertNever(cause)
  }
}

/** Drives one session through turn and step boundaries. */
export class ReactLoopAgent implements Agent {
  readonly model: AgentModel
  readonly inbox: ReactLoopInbox
  private phase: Phase
  private activityDone: Promise<void> = Promise.resolve()

  /** The agent-scoped registration boundary; the lifecycle owner unwinds it after the driver exits. */
  readonly scope: Scope
  readonly ctx: Context

  /** Fused dispatcher, built once in the constructor so hot-path dispatches never allocate. */
  private readonly dispatch: AgentEventDispatch

  /** Whether this loop instance has appended its initial/resume request anchor. */
  private requestHeaderLogged = false
  /** Surface generation at attachment or the preceding built request. */
  private requestSurfaceGeneration: number
  private readonly runtimeContext: RuntimeContextProjection
  /** Process-local revision of assistant frames for this attached Session. */
  private assistantStreamRevision = 0
  private assistantAttemptCounter = 0
  private readonly systemPrompt: SystemPromptProjection
  /** Identities fully frozen by this loop; weak references do not retain replaced history. */
  private readonly frozenMessages = new WeakSet<Message>()

  constructor(
    private loopCtx: Context,
    public readonly id: SessionId,
    public readonly options: AgentOptions,
    public readonly session: Session,
    projections: LoopProjections,
    lastTurn: number,
    private readonly lifetimeSignal: AbortSignal,
    private readonly sessionWriter?: SessionHandle,
  ) {
    this.requestSurfaceGeneration = session.surface.contentGeneration
    this.dispatch = agentEvents(loopCtx, this)
    this.scope = createScope(loopCtx, this)
    this.ctx = this.scope.ctx
    this.model = new AgentModel(projections.modelSelection, session, options)
    this.inbox = new ReactLoopInbox(projections.inbox, session, this.dispatch)
    this.phase = { kind: 'idle', lastTurn }
    this.runtimeContext = new RuntimeContextProjection(this.ctx, session)
    this.systemPrompt = new SystemPromptProjection(session)
  }

  get status(): AgentStatus {
    return this.phase.kind === 'idle' || this.phase.kind === 'maintenance' ? 'idle' : 'running'
  }

  /** Commit a phase and publish its externally visible status transition. */
  private setPhase(next: Phase): void {
    const previousStatus = this.status
    this.phase = next
    const status = this.status
    if (status !== previousStatus) {
      this.dispatch.emit('agent/status', { status })
    }
  }

  send(message: UserMessage, target: InboxTarget, wakeup: boolean): void {
    // Waking input cannot join an aborted activity, so it starts the next turn.
    // Captured before the insertion so a reentrant cancel from a splice observer cannot reclassify it.
    const wakingAfterAbort = wakeup && this.phase.kind !== 'idle' && this.phase.abort.signal.aborted
    const resolvedTarget = wakingAfterAbort ? 'next-turn' : target
    this.inbox.splice(resolvedTarget, Infinity, 0, [message])
    if (wakeup) this.wakeDriver(wakingAfterAbort)
  }

  followup(input: UserMessage): void {
    this.send(input, 'next-turn', true)
  }

  steer(input: UserMessage): void {
    this.send(input, 'next-step', true)
  }

  inject(input: UserMessage): void {
    this.send(input, 'next-step', false)
  }

  cancel(cause: AgentCancelCause, options: CancelOptions = {}): void {
    try {
      if (!options.keepInbox) this.inbox.clear()
    } finally {
      // Queue persistence must not prevent cancellation of admitted work.
      if (this.phase.kind !== 'idle') {
        if (!options.keepInbox) this.phase.wakeRequested = false
        this.phase.abort.abort(cause)
      }
    }
  }

  runMaintenance<T>(job: (signal: AbortSignal) => Promise<T>): Promise<T> {
    if (this.phase.kind !== 'idle') throw new Error(`agent "${this.id}" already has active work`)
    const done = Promise.withResolvers<void>()
    const maintenance: Phase = {
      kind: 'maintenance',
      abort: new AbortController(),
      lastTurn: this.phase.lastTurn,
      wakeRequested: false,
    }
    this.setPhase(maintenance)
    this.activityDone = done.promise
    return (async () => {
      try {
        return await job(maintenance.abort.signal)
      } finally {
        this.setPhase({ kind: 'idle', lastTurn: maintenance.lastTurn })
        const cause = abortedCancelCause(maintenance.abort.signal)
        if (cause?.kind !== 'disposed' && maintenance.wakeRequested && this.inbox.hasPending) this.wakeDriver()
        done.resolve()
      }
    })()
  }

  /**
   * Start one driver, or latch its wake behind maintenance or an aborted
   * activity. A wake sent while idle always opens its turn boundary, even
   * when its message was cleared; only a latched replay is suppressed when
   * the queue no longer holds the wake.
   * @param wakeAfterAbort - the {@link send} classification, captured before
   *   the inbox insertion so a reentrant cancel cannot reclassify it.
   */
  private wakeDriver(wakeAfterAbort = false): void {
    if (this.lifetimeSignal.aborted) return
    if (this.phase.kind !== 'idle') {
      // Maintenance and aborted drivers cannot deliver the wake: latch it for
      // replay at convergence. Live drivers claim queued work themselves;
      // disposal never latches, so teardown waits on no model turn.
      const reason = abortedCancelCause(this.phase.abort.signal)
      if (reason?.kind !== 'disposed' && (this.phase.kind === 'maintenance' || wakeAfterAbort)) {
        this.phase.wakeRequested = true
      }
      return
    }
    const driver = Promise.withResolvers<void>()
    this.activityDone = driver.promise
    this.setPhase({
      kind: 'running',
      abort: new AbortController(),
      turn: this.phase.lastTurn,
      step: 0,
      wakeRequested: false,
    })
    this.loopCtx.agents.withInitiator(this, () => this.kick()).then(driver.resolve, driver.reject)
  }

  async whenIdle(): Promise<void> {
    let activity: Promise<void>
    do {
      await (activity = this.activityDone)
    } while (activity !== this.activityDone)
  }

  /** Report one failure at its live boundary, then preserve it for driver containment. */
  private throwError(error: unknown): never {
    const turn = this.phase.kind === 'running' ? this.phase.turn : this.phase.lastTurn
    const step = this.phase.kind === 'running' ? this.phase.step : 0
    this.dispatch.emit('agent/error', { turn, step, error })
    throw error
  }

  private async kick(): Promise<void> {
    try {
      while (await this.turn()) {}
    } catch (_error) {
      // Reported failures and cancellation are contained at the driver boundary.
    } finally {
      /* v8 ignore next -- kick owns a running phase until this driver boundary */
      if (this.phase.kind === 'running') {
        const { turn, wakeRequested } = this.phase
        this.setPhase({ kind: 'idle', lastTurn: turn })
        if (wakeRequested && this.inbox.hasPending) this.wakeDriver()
      }
    }
  }

  private async preStep(target: InboxTarget, position: { turn: number; step: number }): Promise<PreparedStep> {
    /* v8 ignore next -- private callers establish the running phase before proposing a step */
    if (this.phase.kind !== 'running') throw new Error(`agent "${this.id}": pre-step outside running phase`)
    const signal = this.phase.abort.signal
    const claimed = this.inbox.claim(target, position.turn)
    const selection = this.model.current()
    if (selection === undefined) {
      throw new Error(`agent "${this.id}" has no provider/model: set AgentOptions.provider and AgentOptions.model or record a model selection`)
    }
    const toolSet = await this.loopCtx.tools.resolve({ agent: this, modelSelection: selection, signal })
    try {
      const assembly = await this.loopCtx.systemPrompt.assemble({ ...assembleContextFor(this, signal), modelSelection: selection, toolSet })
      signal.throwIfAborted()
      const decision = await this.dispatch.waterfall(
        'agent/pre-step', { messages: claimed, ...position, signal, modelSelection: selection },
        (): Promise<PreStepDecision> => {
          const sections = renderContextSections(assembly)
          const context = this.runtimeContext.project(sections)
          return Promise.resolve<PreStepDecision>({
            kind: 'enter',
            messages: context === undefined ? claimed : [...claimed, context],
          })
        },
      )
      signal.throwIfAborted()
      if (decision.kind === 'reject') {
        toolSet.dispose()
        return decision
      }
      return { ...decision, assembly, selection, toolSet }
    } catch (error: unknown) {
      toolSet.dispose()
      throw error
    }
  }

  /** Whether the assembled tool schemas differ from the logged request header's. */
  private toolsChanged(tools: PromptAssembly['tools']): boolean {
    const baseline = this.session.requestHeader()
    if (baseline === undefined) return false
    return !headerEquals(baseline, canonicalHeader({ ...baseline, tools: [...tools] }))
  }

  /** Open one turn before claiming its first proposed step. */
  private async turn(): Promise<boolean> {
    if (this.phase.kind !== 'running') {
      this.throwError(new Error(`agent "${this.id}": turn without driver reservation`))
    }
    const phase = this.phase
    const { signal } = phase.abort
    signal.throwIfAborted()
    const turn = phase.turn + 1
    try {
      this.session.append('turn/start', { turn })
    } catch (error: unknown) {
      this.throwError(error)
    }
    phase.turn = turn
    let turnEnds: TurnEndReason | null = null
    let target: InboxTarget = 'next-turn'
    try {
      while (true) {
        signal.throwIfAborted()
        const step = phase.step + 1
        const decision = await this.preStep(target, { turn, step })
        if (decision.kind === 'reject') {
          turnEnds = { kind: 'blocked' }
          return false
        }
        try {
          if (turnEnds && decision.messages.length === 0) break
          // A removed waking message or an enter decision rewritten to empty
          // still owns the initial turn boundary, but it spends no model call.
          if (phase.step === 0 && decision.messages.length === 0) {
            turnEnds = { kind: 'completed' }
            return false
          }
          signal.throwIfAborted()
          this.session.append('step/start', { turn, step })
          phase.step = step
          const toolRecovery = new ToolCallRecovery()
          const stopRecovery = this.ctx.on('session/event', (session, event) => {
            if (session === this.session) toolRecovery.observe(event)
          })
          try {
            // max-tokens is sticky: once any step hits the ceiling, later steps
            // that complete normally must not downgrade the turn outcome.
            const stepEnd = await this.step(decision)
            // max-tokens stays sticky: a later completed step must not
            // downgrade the turn outcome.
            if (turnEnds === null || turnEnds.kind !== 'max-tokens') turnEnds = stepEnd
          } catch (error: unknown) {
            try {
              for (const event of toolRecovery.results()) {
                this.session.append('tool/result', event.data, {
                  surfaceOp: 'append',
                  ...event.sourceEventSeqs === undefined ? {} : { sourceEventSeqs: event.sourceEventSeqs },
                })
              }
            } catch (recoveryError: unknown) {
              throw new AggregateError([error, recoveryError], 'Step failed and its pending tool results could not be recorded', { cause: error })
            }
            throw error
          } finally {
            stopRecovery()
            this.session.append('step/end', { turn, step })
          }
        } finally {
          // Admission closes once the step's tool batch settles or no step opens.
          decision.toolSet.dispose()
        }
        signal.throwIfAborted()
        if (turnEnds && this.inbox.nextStep.length === 0) {
          await this.dispatch.serial('agent/turn-stopping', { turn, signal })
          signal.throwIfAborted()
        }
        if (turnEnds && this.inbox.nextStep.length === 0) break
        target = 'next-step'
      }
    } catch (error: unknown) {
      // A cause is present exactly while the signal is aborted.
      const cause = abortedCancelCause(signal)
      if (cause !== undefined) {
        turnEnds = { kind: 'aborted', reason: cause }
        throw error
      }
      // Every failure is structured: an `LlmError` keeps its facts, a tool
      // activation failure keeps its code, and anything else flattens to
      // `errorChain` text under the `UNKNOWN` code.
      turnEnds = {
        kind: 'error',
        error: error instanceof LlmError
          ? error.failure
          : { message: errorChain(error), code: error instanceof ToolActivationError ? error.code : 'UNKNOWN' },
      }
      this.throwError(error)
    } finally {
      try {
        // oxlint-disable-next-line typescript/no-non-null-assertion -- every exit assigns a turn ending
        this.session.append('turn/end', { turn, reason: turnEnds! })
      } catch (error: unknown) {
        this.throwError(error)
      }
    }
    if (!this.inbox.hasPending) return false
    phase.abort = new AbortController()
    // A fresh controller makes a latch set on the old one stale: the live driver claims the queue itself.
    phase.wakeRequested = false
    phase.step = 0
    return true
  }

  private async step(decision: EnteredStep): Promise<StepEndReason | null> {
    /* v8 ignore next -- private callers establish the running phase before executing a step */
    if (this.phase.kind !== 'running') throw new Error(`agent "${this.id}": step outside running phase`)
    const { turn, step, abort: { signal } } = this.phase
    signal.throwIfAborted()

    const { assembly } = decision
    const renderedPrompt = renderPrompt(assembly)
    let firstAttempt = true
    while (true) {
      const { config, preparedCall } = await this.prepareRequest(turn, step, decision.selection, signal)
      const startsRequestSeries = firstAttempt && decision.startsRequestSeries === true
      const header = canonicalHeader({
        config,
        ...preparedCall === undefined ? {} : { adapterDefaults: preparedCall.adapterDefaults },
        ...assembly.tools.length > 0 ? { tools: assembly.tools } : {},
      })
      const position = { turn, step }
      const plan = (view: Session, shadowed?: readonly SessionSeq[]) => this.planStepAdmission(
        view, decision, renderedPrompt, preparedCall, firstAttempt, startsRequestSeries, signal, shadowed,
      )
      const baseline = this.session.requestHeader()
      const switching = baseline !== undefined
        && (baseline.config.provider !== config.provider || baseline.config.model !== config.model)
      let admitted: StepAdmission
      try {
        const admission = switching
          ? await this.admitSwitch(baseline, header, preparedCall, plan, position, signal)
          : undefined
        signal.throwIfAborted()
        // A switch commits its own priced replacement; every other path prices
        // the pending envelope here, before the step commits it, so compaction
        // cannot strand the runtime-context or system-prompt projection behind a
        // shadow.
        admitted = switching
          ? admission ?? plan(this.session)
          : await this.admitEnvelope(header, preparedCall, plan, position, startsRequestSeries, signal)
      } catch (error: unknown) {
        // A refused admission must not consume the input it refused to price.
        // Only the first attempt owns that claim: a retry already committed it.
        try {
          if (firstAttempt) this.restoreClaimedInput(decision.messages)
        } catch {
          // A failed restore must not mask why the request was refused.
        }
        throw error
      }
      this.appendStepAdmission(this.session, admitted, position)
      firstAttempt = false
      const request = this.buildRequest(header, preparedCall, position, startsRequestSeries, signal)
      const requestGeneration = this.session.surface.contentGeneration
      const requestNodeCount = this.session.surface.nodes.length
      const requestHeader = this.session.requestHeader()
      const requestContext = this.session.requestContext()
      const assertCurrent = () => {
        if (this.loopCtx.sessions.get(this.session.id) !== this.session
          || this.session.surface.contentGeneration !== requestGeneration
          || this.session.surface.nodes.length !== requestNodeCount
          || this.session.requestHeader() !== requestHeader
          || this.session.requestContext() !== requestContext) {
          throw new LlmError('agent: Session changed before the admitted request could be dispatched', 'CONTEXT_ADMISSION_REFUSED')
        }
      }
      const live = new AssistantStreamAttempt(
        this.session.id,
        ++this.assistantAttemptCounter,
        () => ++this.assistantStreamRevision,
        turn,
        step,
        (frame) => { this.dispatch.emit('agent/assistant-stream', { frame }) },
      )
      const repetition = this.loopCtx.agentLoop.config.stopOnRepeatedOutput.get()
        ? new StreamRepetition() : undefined
      let started = false
      try {
        const stream = preparedCall?.stream(
          request,
          async (projected) => {
            assertCurrent()
            await this.verifyEnvelope(header, preparedCall, projected, position, signal)
            return assertCurrent
          },
        ) ?? this.loopCtx.llm.stream(request)
        signal.throwIfAborted()
        live.start()
        started = true
        for await (const chunk of stream) {
          signal.throwIfAborted()
          if (chunk.type === 'text-delta' || chunk.type === 'reasoning-delta') {
            const offset = repetition?.push(chunk)
            if (offset !== undefined) {
              live.push({ ...chunk, text: chunk.text.slice(0, offset) })
              this.cancel({ kind: 'hook', reason: REPEATED_MODEL_OUTPUT }, { keepInbox: true })
              signal.throwIfAborted()
            }
          }
          live.push(chunk)
        }
        signal.throwIfAborted()
      } catch (error: unknown) {
        if (!started) throw error
        try {
          if (signal.aborted) {
            const content = live.interruptedBlocks()
            if (content.length > 0) {
              live.settle('assistant/message', () => this.session.append('assistant/message', {
                turn,
                step,
                message: createAssistantMessage({
                  content,
                  source: {
                    provider: request.provider,
                    model: request.model,
                    ...live.replayState === undefined ? {} : { replayState: live.replayState },
                  },
                }),
                interrupted: true,
                ...live.usage === undefined ? {} : { usage: live.usage },
                stream: live.stream,
              }, { surfaceOp: 'append' }).seq)
            } else {
              live.settle(
                'assistant/attempt',
                () => this.session.append('assistant/attempt', { turn, step, stream: live.stream }).seq,
              )
            }
          } else {
            live.settle(
              'assistant/attempt',
              () => this.session.append('assistant/attempt', { turn, step, stream: live.stream }).seq,
            )
          }
        } catch (settlementError: unknown) {
          throw new AggregateError(
            [error, settlementError],
            'Assistant stream failed and its durable settlement was rejected',
            { cause: error },
          )
        }
        throw error
      }
      try {
        const finish = live.finish
        if (finish.kind === 'error' || finish.kind === 'aborted') {
          live.settle(
            'assistant/attempt',
            () => this.session.append('assistant/attempt', { turn, step, stream: live.stream }).seq,
          )
          const action = await this.dispatch.waterfall(
            'agent/request-error', {
              turn,
              step,
              provider: request.provider,
              failure: finish.failure,
              retryPolicy: preparedCall?.retryPolicy,
              signal,
            },
            () => Promise.resolve<RequestErrorAction>(undefined),
          )
          signal.throwIfAborted()
          if (action?.kind !== 'retry') {
            throw LlmError.fromFailure(finish.failure)
          }
          continue
        }

        const message = createAssistantMessage({
          content: live.blocks(),
          source: {
            provider: request.provider,
            model: request.model,
            ...live.replayState !== undefined ? { replayState: live.replayState } : {},
          },
        })
        live.settle(
          'assistant/message',
          () => this.session.append('assistant/message', {
            turn,
            step,
            message,
            ...live.usage === undefined ? {} : { usage: live.usage },
            stream: live.stream,
          }, { surfaceOp: 'append' }).seq,
        )
        if (finish.kind === 'max-tokens') return { kind: 'max-tokens' }

        const toolCalls = message.content.filter(block => block.type === 'tool-call')
        if (toolCalls.length === 0) return { kind: 'completed' }
        const maxParallelToolCalls = this.loopCtx.agentLoop.config.maxParallelToolCalls.get()
        try {
          const result = await this.loopCtx.tools.execute({
            calls: toolCalls, signal, agent: this, log: { turn, step },
            maxParallelToolCalls, toolSet: decision.toolSet,
          })
          for (const context of result.additionalContexts) {
            this.inbox.splice('next-step', this.inbox.nextStep.length, 0, [context])
          }
          return result.concluded ? { kind: 'completed' } : null
        } catch (error: unknown) {
          if (!(error instanceof ToolBatchExecutionError)) throw error
          for (const context of error.partial.additionalContexts) {
            this.inbox.splice('next-step', this.inbox.nextStep.length, 0, [context])
          }
          throw error.cause
        }
      } catch (error: unknown) {
        if (!live.ended) live.abandon()
        throw error
      }
    }
  }

  private planStepAdmission(
    view: Session,
    decision: EnteredStep,
    renderedPrompt: string,
    preparedCall: PreparedLlmCall | undefined,
    firstAttempt: boolean,
    startsRequestSeries: boolean,
    signal: AbortSignal,
    shadowed?: readonly SessionSeq[],
  ): StepAdmission {
    const { assembly } = decision
    const currentContext = this.loopCtx.systemPrompt.refreshContext(assembly, {
      ...assembleContextFor(this, signal), modelSelection: decision.selection,
    })
    const context = this.runtimeContext.project(renderContextSections(currentContext), shadowed)
    signal.throwIfAborted()
    const projection = view === this.session ? this.systemPrompt : new SystemPromptProjection(view)
    const system = projection.project(renderedPrompt, {
      inHistory: preparedCall?.systemPromptUpdate === 'in-history',
      startsSeries: startsRequestSeries
        || this.requestSurfaceGeneration !== view.surface.contentGeneration
        || (preparedCall?.toolUpdate === undefined && this.toolsChanged(assembly.tools)),
    })
    const users: UserMessage[] = []
    let contextAdmitted = false
    for (const message of firstAttempt ? decision.messages : []) {
      if (message.source.kind !== 'runtime-context') users.push(message)
      else if (context !== undefined && !contextAdmitted) {
        users.push(context)
        contextAdmitted = true
      }
    }
    if (context !== undefined && !contextAdmitted) users.push(context)
    return { system, users }
  }

  private appendStepAdmission(view: Session, admission: StepAdmission, position: StepPosition): void {
    for (const { message, intent } of admission.system) {
      view.append('system/message', { ...position, message }, intent)
    }
    for (const message of admission.users) view.append('user/message', message, { surfaceOp: 'append' })
  }

  /** Price a switch against the same pending admission the live loop will use. */
  private async admitSwitch(
    baseline: EpochHeader,
    header: EpochHeader,
    preparedCall: PreparedLlmCall | undefined,
    plan: (view: Session, shadowed?: readonly SessionSeq[]) => StepAdmission,
    position: StepPosition,
    signal: AbortSignal,
  ): Promise<StepAdmission | undefined> {
    const { session } = this
    let checked: { admission: StepAdmission; nodes: readonly SessionSeq[]; generation: number } | undefined
    const contextWindow = preparedCall?.context?.contextWindow
    const request: PendingAgentRequest = {
      header,
      ...contextWindow === undefined ? {} : { contextWindow },
      completionSource: preparedCall?.completion.source ?? 'unknown',
      ...preparedCall === undefined ? {} : { projectRequest: preparedCall.projectRequest.bind(preparedCall) },
      ...preparedCall?.imagePricing === undefined ? {} : { imagePricing: preparedCall.imagePricing },
      preview: (events, replacement) => {
        signal.throwIfAborted()
        if (events.length !== session.seq || session.requestHeader() !== baseline) {
          throw new Error('agent: Session changed before request admission could be priced')
        }
        const view = Session.fromRestore(session.id, events, session.header, session.inheritedEventCount,
          'shared-frozen', this.loopCtx.sessions.messageProjections)
        const nodes = [...session.surface.nodes]
        if (replacement !== undefined) {
          const { start, end, shadowedSeqs, replacementSeq, message } = replacement
          const index = nodes.indexOf(start)
          if (index < 0 || nodes[index + shadowedSeqs.length - 1] !== end
            || shadowedSeqs.some((seq, offset) => nodes[index + offset] !== seq)) {
            throw new Error('agent: proposed replacement does not match the current Surface')
          }
          nodes.splice(index, shadowedSeqs.length, replacementSeq)
          view.append('user/message', message, {
            surfaceOp: { op: 'replace', startSeq: start, endSeq: end },
            sourceEventSeqs: [...shadowedSeqs],
          })
        }
        const admission = plan(view, replacement?.shadowedSeqs)
        this.appendStepAdmission(view, admission, position)
        const { seq } = view.append('request/header', { header, reason: 'change' })
        this.appendToolDelta(view, baseline, header.tools ?? [], position, seq)
        checked = {
          admission, nodes,
          generation: session.surface.contentGeneration + (replacement === undefined ? 0 : 1),
        }
        return view
      },
    }
    await this.dispatch.serial('agent/request-admission', { request, signal })
    if (checked === undefined) return undefined
    const { admission, nodes, generation } = checked
    if (session.requestHeader() !== baseline || session.surface.contentGeneration !== generation
      || session.surface.nodes.length !== nodes.length || nodes.some((seq, i) => session.surface.nodes[i] !== seq)) {
      throw new Error('agent: the priced request no longer matches the Session Surface')
    }
    return admission
  }

  /** Resolve request config for the step's selection and bind its adapter before admitting model-visible input. */
  private async prepareRequest(
    turn: number,
    step: number,
    selection: Readonly<ModelSelection>,
    signal: AbortSignal,
  ): Promise<{ config: LlmCallConfig; preparedCall?: PreparedLlmCall }> {
    const { session } = this

    // A fresh loop uses its configured cap or inherits the Session's last explicit cap.
    // Marked adapter defaults are resolved again for the active profile and route.
    const previousHeader = session.requestHeader()
    const persistedCap = previousHeader?.adapterDefaults?.maxTokens === true
      ? undefined : previousHeader?.config.maxTokens
    const maxTokens = this.options.maxTokens ?? persistedCap
    const persisted: Partial<LlmCallConfig> = this.requestHeaderLogged
      // oxlint-disable-next-line typescript/no-non-null-assertion -- the instance logged the header it now folds
      ? requestProposal(previousHeader!)
      : maxTokens === undefined ? {} : { maxTokens }
    const { provider: _provider, model: _model, reasoningEffort: _effort, ...controls } = persisted
    const seedConfig = deepFreeze(structuredClone({ ...controls, ...selection }))
    const proposedConfig = await this.dispatch.waterfall(
      'agent/request', { turn, step, signal },
      () => Promise.resolve(seedConfig),
    )
    signal.throwIfAborted()
    if (proposedConfig.provider !== selection.provider || proposedConfig.model !== selection.model) {
      throw new Error(`agent "${this.id}": agent/request proposed "${proposedConfig.provider}/${proposedConfig.model}" after step preparation fixed "${selection.provider}/${selection.model}"; use agent.model.select() before step preparation`)
    }
    if (proposedConfig.reasoningEffort !== selection.reasoningEffort) {
      throw new Error(`agent "${this.id}": agent/request proposed reasoning effort "${proposedConfig.reasoningEffort ?? 'adapter default'}" after step preparation fixed "${selection.reasoningEffort ?? 'adapter default'}"; use agent.model.select() before step preparation`)
    }
    let config: LlmCallConfig
    let preparedCall: PreparedLlmCall | undefined
    try {
      preparedCall = await this.loopCtx.llm.prepareCall(proposedConfig, signal)
      config = preparedCall.config
    } catch (error: unknown) {
      // Middleware may serve an unregistered route; terminal dispatch still requires an adapter.
      if (!(error instanceof LlmError) || error.code !== 'NO_ADAPTER') throw error
      config = proposedConfig
    }
    signal.throwIfAborted()
    return { config, ...preparedCall === undefined ? {} : { preparedCall } }
  }

  /**
   * Price the pending envelope before the step commits its admission. A listener
   * may compact the Surface, in which case the admission is recomputed against
   * the post-compaction Surface and priced again, bounded by
   * {@link MAX_ENVELOPE_REBUILDS}. A listener that cannot get the preview under
   * budget throws, so the refused request is never committed or dispatched.
   * @param header - the resolved envelope for this attempt.
   * @param preparedCall - registration-bound call supplying window and reservation.
   * @param plan - recomputes the step admission against the current Surface.
   * @param position - turn and step owning the admission.
   * @param startsRequestSeries - whether the step explicitly starts a new series.
   * @param signal - the current turn's cancellation signal.
   * @returns the admission to commit for this attempt.
   */
  private async admitEnvelope(
    header: EpochHeader,
    preparedCall: PreparedLlmCall | undefined,
    plan: (view: Session, shadowed?: readonly SessionSeq[]) => StepAdmission,
    position: StepPosition,
    startsRequestSeries: boolean,
    signal: AbortSignal,
  ): Promise<StepAdmission> {
    let admission = plan(this.session)
    for (let attempt = 0; ; attempt += 1) {
      const generation = this.session.surface.contentGeneration
      // Capture the exact pre-dispatch log once: replaying it is what prices the
      // preview, and the replay must observe this attempt's events whatever the
      // listener does to the live Session in between.
      // oxlint-disable-next-line typescript/no-deprecated -- full-sequence read pending a projection (2026-09-09 note).
      const events = this.session.snapshotEvents()
      let view: Session | undefined
      await this.dispatch.serial('agent/request-envelope', {
        header,
        view: () => (view ??= this.previewView(events, admission, header, position, startsRequestSeries)),
        ...preparedCall === undefined ? {} : { projectRequest: preparedCall.projectRequest.bind(preparedCall) },
        ...preparedCall?.imagePricing === undefined ? {} : { imagePricing: preparedCall.imagePricing },
        ...preparedCall?.context === undefined ? {} : { contextWindow: preparedCall.context.contextWindow },
        ...completionFields(preparedCall),
        phase: 'preview',
        signal,
      })
      signal.throwIfAborted()
      if (this.session.surface.contentGeneration === generation) return admission
      if (attempt >= MAX_ENVELOPE_REBUILDS) {
        throw new LlmError(
          'agent: request admission still exceeds the destination budget after bounded compaction',
          'CONTEXT_ADMISSION_REFUSED',
        )
      }
      admission = plan(this.session)
    }
  }

  /** Price the frozen request dispatch is about to send; refusal only. */
  private async verifyEnvelope(
    header: EpochHeader,
    preparedCall: PreparedLlmCall,
    request: GenerateOptions,
    position: StepPosition,
    signal: AbortSignal,
  ): Promise<void> {
    await this.dispatch.serial('agent/request-envelope', {
      header,
      request,
      flushReceipt: async () => {
        const acknowledged = await this.loopCtx.sessions.flush(this.session, this.sessionWriter)
        if (this.sessionWriter !== undefined && !acknowledged) {
          throw new LlmError(
            'agent: Session writer did not acknowledge the admission receipt; reopen the session before sending another request',
            'CONTEXT_ADMISSION_REFUSED',
          )
        }
      },
      ...position,
      ...preparedCall.context === undefined ? {} : { contextWindow: preparedCall.context.contextWindow },
      ...preparedCall.imagePricing === undefined ? {} : { imagePricing: preparedCall.imagePricing },
      ...completionFields(preparedCall),
      phase: 'verify',
      signal,
    })
    signal.throwIfAborted()
  }

  /** Return a step's claimed user input to the pending Inbox after an admission failure. */
  private restoreClaimedInput(messages: readonly UserMessage[]): void {
    const input = messages.filter(message => message.source.kind !== 'runtime-context')
    if (input.length > 0) this.inbox.splice('next-step', 0, 0, [...input])
  }
  /**
   * Clone the Session with the pending admission committed, for pricing only.
   * @param events - the exact pre-dispatch log observation to replay.
   * @param admission - the pending admission to append to the clone.
   * @param header - the prepared header and tool declarations.
   * @param position - turn and step owning the admission.
   * @param startsRequestSeries - whether the step explicitly starts a new series.
   * @returns the clone a listener prices.
   */
  private previewView(
    events: readonly SessionEvent[],
    admission: StepAdmission,
    header: EpochHeader,
    position: StepPosition,
    startsRequestSeries: boolean,
  ): Session {
    const view = Session.fromRestore(this.session.id, events, this.session.header,
      this.session.inheritedEventCount, 'shared-frozen', this.loopCtx.sessions.messageProjections)
    this.appendStepAdmission(view, admission, position)
    this.appendRequestHeader(view, header, position, startsRequestSeries)
    return view
  }

  /** Apply the same request header and tool deltas to preview and live Sessions. */
  private appendRequestHeader(
    view: Session,
    header: EpochHeader,
    position: StepPosition,
    startsRequestSeries: boolean,
  ): void {
    const baseline = view.requestHeader()
    const startsSeries = startsRequestSeries || this.requestSurfaceGeneration !== view.surface.contentGeneration
    const reason = !this.requestHeaderLogged ? (baseline === undefined ? 'initial' : 'resume')
      : baseline === undefined || !headerEquals(baseline, header) ? 'change'
        : startsSeries ? 'series' : undefined
    if (reason === undefined) return
    const { seq } = view.append('request/header', {
      header,
      reason,
      ...startsSeries && reason !== 'series' ? { startsSeries: true } : {},
    })
    if (baseline !== undefined && reason !== 'series') {
      this.appendToolDelta(view, baseline, header.tools ?? [], position, seq)
    }
  }

  /** Log the resolved envelope and derive a frozen request from the admitted surface. */
  private buildRequest(
    header: EpochHeader,
    preparedCall: PreparedLlmCall | undefined,
    position: StepPosition,
    startsRequestSeries: boolean,
    signal: AbortSignal,
  ): GenerateOptions {
    const { session } = this
    const { config } = header
    const surfaceGeneration = session.surface.contentGeneration
    this.appendRequestHeader(session, header, position, startsRequestSeries)
    this.requestHeaderLogged = true
    this.requestSurfaceGeneration = surfaceGeneration

    const contextWindow = preparedCall?.context?.contextWindow
    const systemPromptUpdate = preparedCall?.systemPromptUpdate
    const requestContext: RequestContext = {
      provider: config.provider,
      model: config.model,
      ...contextWindow === undefined ? {} : { contextWindow },
      ...systemPromptUpdate === undefined ? {} : { systemPromptUpdate },
    }
    const previousContext = session.requestContext()
    if (previousContext?.provider !== requestContext.provider
      || previousContext.model !== requestContext.model
      || previousContext.contextWindow !== requestContext.contextWindow
      || previousContext.systemPromptUpdate !== requestContext.systemPromptUpdate) {
      session.append('request/context', requestContext)
    }
    signal.throwIfAborted()

    // canonicalHeader is shallow; append logs a detached snapshot, not these local values.
    deepFreeze(header)
    const boundaryMessages = session.deriveMessages()
    for (const message of boundaryMessages) {
      if (this.frozenMessages.has(message)) continue
      deepFreeze(message)
      this.frozenMessages.add(message)
    }
    Object.freeze(boundaryMessages)
    const request = markAgentLoopRequest(Object.freeze({
      ...header.config,
      messages: boundaryMessages,
      toolHistory: session.toolHistory(),
      ...header.tools !== undefined ? { tools: header.tools } : {},
      sessionId: this.session.id,
      signal,
    }))
    return request
  }

  private appendToolDelta(
    view: Session,
    baseline: EpochHeader,
    tools: NonNullable<EpochHeader['tools']>,
    position: StepPosition,
    headerSeq: SessionSeq,
  ): void {
    const previousNames = new Set(baseline.tools?.map(tool => tool.name))
    const currentNames = new Set(tools.map(tool => tool.name))
    const additions = tools.filter(tool => !previousNames.has(tool.name))
      .map(tool => ({ type: 'tool-addition' as const, toolName: tool.name }))
    const removals = (baseline.tools ?? []).filter(tool => !currentNames.has(tool.name))
      .map(tool => ({ type: 'tool-removal' as const, toolName: tool.name }))
    if (additions.length > 0 || removals.length > 0) {
      view.append('developer/message', {
        ...position,
        message: createDeveloperMessage({ source: { kind: 'tool-registry' }, content: [...additions, ...removals] }),
        ...additions.length > 0 ? { headerSeq } : {},
      }, { surfaceOp: 'append' })
    }
  }
}
