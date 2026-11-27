/**
 * Opt-in background KINF Decision relevance filtering for outbound tool results.
 *
 * The feature asks a separately configured Decision model whether each eligible
 * tool-result body is relevant to the current user goal, and applies the
 * validated verdicts to the OUTBOUND request projection only. Durable session
 * history, tool-call/tool-result correlation and message identity are never
 * rewritten, so the loop's reconstructability contract still holds.
 *
 * @module @deepseek-ai/dsh-context-decision-relevance
 */

declare module '@deepseek-ai/dsh-llm/types' {
  interface LlmRequestPurposeMap {
    'decision-relevance': true
  }
}

declare module '@deepseek-ai/dsh-llm' {
  interface MessageSourceMap {
    'dsh-context-decision-relevance': { kind: 'dsh-context-decision-relevance' }
  }
}

import { Context, Service } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { BlockAssembler, createUserMessage } from '@deepseek-ai/dsh-llm'
import type {
  ContentBlock,
  GenerateOptions,
  Message,
  MessageId,
  RequestMessage,
  ToolResultMessage,
} from '@deepseek-ai/dsh-llm'
import { Session, SessionId } from '@deepseek-ai/dsh-session'
import { deadline } from '@deepseek-ai/dsh-timeout'
import { deepFreeze } from '@deepseek-ai/dsh-util-values'
import { codePointLength, DEFAULTS, isEffective, resolveConfig, truncateCodePoints } from './config.ts'
import type { DecisionRelevanceConfig, ProjectionResult, RelevanceDisposition, RelevanceJudgment, ResolvedConfig } from './types.ts'

declare module '@deepseek-ai/cordis' {
  interface Context {
    decisionRelevance: DecisionRelevanceFilter
  }
}

declare module '@deepseek-ai/dsh-session/types' {
  interface SessionEventMap {
    /** Log-only record of one bounded background Decision relevance pass. */
    'context/decision-relevance': DecisionRelevanceEventData
  }
}

/** Durable bounded evidence for one outbound relevance pass. */
export interface DecisionRelevanceEventData {
  /** Conversation route whose outbound request this pass shaped. */
  readonly conversation: { readonly provider: string; readonly model: string }
  /** Whether any content was actually elided. */
  readonly applied: boolean
  /** Bounded per-candidate judgments; carries no credentials or history dump. */
  readonly judgments: readonly RelevanceJudgment[]
}

/** Capability-owned timeout reason code for background relevance requests. */
const RELEVANCE_TIMEOUT_CODE = 'DECISION_RELEVANCE_TIMEOUT'

/**
 * Structural view of the session store.
 *
 * The filter needs only a live-session lookup, so it declares `sessions` as an
 * injection and reads it through this narrow shape rather than depending on the
 * store's full surface.
 */
interface SessionLookup {
  /** Look up one live session by identity. */
  get(id: SessionId): Session | undefined
}

/** One eligible tool result found in the outbound message list. */
interface Candidate {
  readonly messageIndex: number
  readonly message: ToolResultMessage
  readonly chars: number
}

/**
 * Bounded background relevance filter.
 *
 * The filter never selects or mutates the conversation model: it resolves its
 * own explicit Decision route and fails open on every uncertainty, error or
 * cancellation, so a background judgment can only ever remove content that was
 * positively judged irrelevant.
 */
export class DecisionRelevanceFilter extends Service {
  static inject = ['llm', 'sessions']

  static Config: z<DecisionRelevanceConfig> = z.object({
    mode: z.union([z.const('off'), z.const('on')]).default(DEFAULTS.mode),
    provider: z.string().default(''),
    model: z.string().default(''),
    retainThreshold: z.number().min(0).max(1).default(DEFAULTS.retainThreshold),
    uncertaintyMargin: z.number().min(0).max(1).default(DEFAULTS.uncertaintyMargin),
    maxCandidatesPerRequest: z.number().step(1).min(1).default(DEFAULTS.maxCandidatesPerRequest),
    minCandidateChars: z.number().step(1).min(0).default(DEFAULTS.minCandidateChars),
    maxInputChars: z.number().step(1).min(1).default(DEFAULTS.maxInputChars),
    timeoutMs: z.number().step(1).min(1).default(DEFAULTS.timeoutMs),
    elisionText: z.string().default(DEFAULTS.elisionText),
  })

  /** Resolved and immutable execution controls. */
  readonly config: ResolvedConfig

  /** Per-request judgment cache keyed by message identity, so one tool result is judged once. */
  protected readonly cache = new Map<MessageId, RelevanceJudgment>()

  constructor(ctx: Context, config: DecisionRelevanceConfig = {}) {
    super(ctx, 'decisionRelevance')
    this.config = resolveConfig(normalizeConfig(config))
  }

  /**
   * Whether this configuration can issue background Decision calls.
   * @returns whether the feature is enabled with an explicit route.
   */
  get enabled(): boolean {
    return isEffective(this.config)
  }

  /**
   * Shape one outbound request by eliding tool results judged irrelevant.
   *
   * Returns the original list untouched when the feature is inactive, when no
   * eligible candidate exists, when the session is not live, or when every
   * evaluation failed open. Message identity is preserved for every retained
   * message.
   * @param messages - the resolved outbound message list.
   * @param sessionId - the session identity the loop stamped on the request.
   * @param conversation - the conversation route, recorded with the evidence.
   * @param signal - caller cancellation, fused with each evaluation deadline.
   * @returns the projected list and the bounded judgments recorded.
   */
  async project(
    messages: readonly RequestMessage[],
    sessionId: string,
    conversation: { provider: string; model: string },
    signal: AbortSignal | undefined,
  ): Promise<{ messages: readonly RequestMessage[] } & ProjectionResult> {
    const unchanged = { messages, applied: false, judgments: [] as readonly RelevanceJudgment[] }
    const config = this.config
    if (!isEffective(config) || messages.length === 0) return unchanged

    const session = this.resolveSession(sessionId)
    if (session === undefined) return unchanged

    const candidates = this.eligible(messages)
    if (candidates.length === 0) return unchanged

    const goal = latestUserGoal(messages)
    if (goal === undefined) return unchanged

    const budget = this.config.maxCandidatesPerRequest
    const judgments: RelevanceJudgment[] = []
    const elided = new Map<number, RelevanceJudgment>()
    let evaluated = 0

    for (const candidate of candidates) {
      const cached = this.cache.get(candidate.message.id)
      if (cached !== undefined) {
        judgments.push(cached)
        if (cached.disposition === 'elided-irrelevant') elided.set(candidate.messageIndex, cached)
        continue
      }
      if (evaluated >= budget) {
        const judgment = this.judgment(candidate, 'retained-limit', 'evaluation budget exhausted for this request', conversation)
        this.remember(candidate, judgment)
        judgments.push(judgment)
        continue
      }
      evaluated += 1
      const outcome = await this.evaluate(candidate, goal, conversation, signal, config)
      this.remember(candidate, outcome)
      judgments.push(outcome)
      if (outcome.disposition === 'elided-irrelevant') elided.set(candidate.messageIndex, outcome)
    }

    if (elided.size === 0) {
      this.record(session, conversation, false, judgments)
      return { messages, applied: false, judgments }
    }

    const projected = messages.map((message, index) => {
      if (!elided.has(index)) return message
      // The message keeps its identity, role and tool-call correlation; only
      // the model-visible body is replaced, and only for this request.
      const content: readonly ContentBlock[] = [{ type: 'text', text: this.config.elisionText }]
      return { ...message, content: Object.isFrozen(message.content) ? Object.freeze(content) : content }
    })

    this.record(session, conversation, true, judgments)
    return { messages: projected, applied: true, judgments }
  }

  /** Collect eligible tool-result candidates in message order. */
  protected eligible(messages: readonly RequestMessage[]): Candidate[] {
    const candidates: Candidate[] = []
    for (const [messageIndex, message] of messages.entries()) {
      if (message.role !== 'tool') continue
      // A failed tool result stays visible: its error is part of the goal record.
      if (message.isError === true) continue
      const chars = measure(message.content)
      if (chars < this.config.minCandidateChars) continue
      candidates.push({ messageIndex, message, chars })
    }
    return candidates
  }

  /**
   * Run one bounded Decision evaluation. Every failure retains the content.
   * The caller supplies the immutable route validated at project entry.
   */
  protected async evaluate(
    candidate: Candidate,
    goal: string,
    conversation: { provider: string; model: string },
    signal: AbortSignal | undefined,
    route: { readonly provider: string; readonly model: string },
  ): Promise<RelevanceJudgment> {
    const base = this.judgment(candidate, 'retained-error', 'evaluation failed', conversation)
    const { provider, model } = route

    const input = truncateCodePoints(candidate.message.content
      .filter((block): block is ContentBlock & { type: 'text' } => block.type === 'text')
      .map(block => block.text)
      .join('\n'), this.config.maxInputChars)

    try {
      signal?.throwIfAborted()
      using callDeadline = deadline(signal, this.config.timeoutMs, RELEVANCE_TIMEOUT_CODE)
      const call = await this.ctx.llm.prepareCall({ provider, model, maxTokens: 1, temperature: 0 }, callDeadline.signal)
      const options: GenerateOptions = deepFreeze({
        ...call.config,
        messages: [createUserMessage({
          content: [{ type: 'text', text: buildPrompt(goal, input) }],
          source: { kind: 'dsh-context-decision-relevance' },
        })],
        purpose: 'decision-relevance',
        signal: callDeadline.signal,
      })
      callDeadline.signal.throwIfAborted()

      const assembler = new BlockAssembler()
      for await (const chunk of call.stream(options)) {
        callDeadline.signal.throwIfAborted()
        assembler.push(chunk)
      }
      callDeadline.signal.throwIfAborted()

      const finish = assembler.finish
      if (finish.kind === 'aborted') {
        return { ...base, disposition: 'retained-error', reason: 'evaluation cancelled' }
      }
      if (finish.kind === 'error') {
        return { ...base, disposition: 'retained-error', reason: `decision transport failed (${finish.failure.code})` }
      }

      const readout = parseReadout(assembler.blocks())
      if (readout === undefined) {
        return { ...base, disposition: 'retained-error', reason: 'malformed Decision readout' }
      }
      if (readout.model !== model) {
        return { ...base, disposition: 'retained-error', reason: 'served model differs from the requested Decision model' }
      }

      const served = readout.served_identity
      const servedFields = {
        model: readout.model,
        mode: textField(served['mode']),
        instance: textField(served['instance']),
      }
      const score = readout.decision.label_scores.yes

      if (score < this.config.retainThreshold - this.config.uncertaintyMargin) {
        return {
          ...base,
          disposition: 'elided-irrelevant',
          reason: `relevance ${formatScore(score)} is below the retain threshold`,
          served: servedFields,
          score,
          label: readout.decision.label,
        }
      }
      if (score < this.config.retainThreshold + this.config.uncertaintyMargin) {
        return {
          ...base,
          disposition: 'retained-uncertain',
          reason: `relevance ${formatScore(score)} is inside the uncertainty band`,
          served: servedFields,
          score,
          label: readout.decision.label,
        }
      }
      return {
        ...base,
        disposition: 'retained-relevant',
        reason: `relevance ${formatScore(score)} meets the retain threshold`,
        served: servedFields,
        score,
        label: readout.decision.label,
      }
    } catch (error: unknown) {
      if (signal?.aborted === true) {
        return { ...base, disposition: 'retained-error', reason: 'evaluation cancelled' }
      }
      return { ...base, disposition: 'retained-error', reason: `evaluation failed (${errorName(error)})` }
    }
  }

  protected judgment(
    candidate: Candidate,
    disposition: RelevanceDisposition,
    reason: string,
    conversation: { provider: string; model: string },
  ): RelevanceJudgment {
    return {
      messageId: candidate.message.id,
      callId: candidate.message.toolCallId,
      candidateChars: candidate.chars,
      inputChars: Math.min(candidate.chars, this.config.maxInputChars),
      disposition,
      reason,
      requested: { provider: conversation.provider, model: conversation.model },
    }
  }

  protected remember(candidate: Candidate, judgment: RelevanceJudgment): void {
    this.cache.set(candidate.message.id, judgment)
  }

  /** Resolve the live session the request was stamped with. */
  protected resolveSession(sessionId: string): Session | undefined {
    return (this.ctx.get('sessions') as SessionLookup | undefined)?.get(SessionId(sessionId))
  }

  protected record(
    session: Session,
    conversation: { provider: string; model: string },
    applied: boolean,
    judgments: readonly RelevanceJudgment[],
  ): void {
    session.append('context/decision-relevance', {
      conversation,
      applied,
      judgments: judgments.map(judgment => ({ ...judgment })),
    })
  }
}

/**
 * Turn the Loader's normalized config back into the sparse shape the resolver
 * expects, so schema defaults do not mask an absent route.
 * @param config - normalized plugin configuration.
 * @returns the same fields, with empty route strings treated as absent.
 */
function normalizeConfig(config: DecisionRelevanceConfig): DecisionRelevanceConfig {
  const normalized = { ...config }
  if (normalized.provider === '') delete normalized.provider
  if (normalized.model === '') delete normalized.model
  return normalized
}

/** Total text size of a content list in Unicode code points. */
function measure(content: readonly ContentBlock[]): number {
  let chars = 0
  for (const block of content) {
    if (block.type === 'text') chars += codePointLength(block.text)
  }
  return chars
}

/** Join the text blocks of one content list. */
function joinText(blocks: readonly ContentBlock[]): string {
  const text: string[] = []
  for (const block of blocks) {
    if (block.type === 'text') text.push(block.text)
  }
  return text.join('\n')
}

/** Read one optional JSON field as a bounded text value. */
function textField(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

/** The most recent user message text, used as the current goal. */
function latestUserGoal(messages: readonly RequestMessage[]): string | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index] as Message | undefined
    if (message === undefined || message.role !== 'user') continue
    const text = joinText(message.content).trim()
    if (text.length > 0) return text
  }
  return undefined
}

/** Build the bounded plain-text Decision prompt. */
function buildPrompt(goal: string, candidate: string): string {
  return `Goal:\n${goal}\n\nTool result:\n${candidate}\n\nIs this tool result relevant to the goal?`
}

/** Fixed finite-score bound for a durable judgment record. */
function formatScore(score: number): string {
  return score.toFixed(4)
}

function errorName(error: unknown): string {
  if (error instanceof Error && error.name.length > 0) return error.name
  return 'unknown'
}

/**
 * Parse the durable Decision readout the adapter renders as JSON text.
 * @param blocks - assembled response blocks.
 * @returns the validated readout, or `undefined` when it is malformed.
 */
function parseReadout(blocks: readonly ContentBlock[]): DecisionReadout | undefined {
  const text = joinText(blocks).trim()
  if (text.length === 0) return undefined
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch {
    return undefined
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return undefined
  const record = raw as Record<string, unknown>
  const model = record['model']
  const decision = record['decision']
  const identity = record['served_identity']
  if (typeof model !== 'string' || model.length === 0) return undefined
  if (typeof decision !== 'object' || decision === null) return undefined
  if (typeof identity !== 'object' || identity === null) return undefined
  const readout = decision as Record<string, unknown>
  if (readout['schema'] !== 'kinf.decision.v1' || readout['mode'] !== 'label_scores') return undefined
  const label = readout['label']
  if (label !== 'yes' && label !== 'no') return undefined
  const scores = readout['label_scores']
  if (typeof scores !== 'object' || scores === null) return undefined
  const pair = scores as Record<string, unknown>
  const yes = pair['yes']
  const no = pair['no']
  if (typeof yes !== 'number' || !Number.isFinite(yes) || yes < 0 || yes > 1) return undefined
  if (typeof no !== 'number' || !Number.isFinite(no) || no < 0 || no > 1) return undefined
  return {
    model,
    decision: { label, label_scores: { yes, no } },
    served_identity: identity as Record<string, unknown>,
  }
}

/** The subset of the durable Decision readout this filter consumes. */
interface DecisionReadout {
  readonly model: string
  readonly decision: {
    readonly label: 'yes' | 'no'
    readonly label_scores: { readonly yes: number; readonly no: number }
  }
  readonly served_identity: Record<string, unknown>
}

export type {
  DecisionRelevanceConfig,
  ProjectionResult,
  RelevanceDisposition,
  RelevanceJudgment,
  ResolvedConfig,
} from './types.ts'

export default DecisionRelevanceFilter
