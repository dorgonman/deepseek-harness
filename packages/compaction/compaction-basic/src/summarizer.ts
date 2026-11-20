/**
 * Default one-shot summarization and durable checkpoint framing.
 *
 * @module @deepseek-ai/dsh-compaction-basic/summarizer
 */

import type { Context } from '@deepseek-ai/cordis'
import { BlockAssembler, LlmError } from '@deepseek-ai/dsh-llm'
import type { SummaryRequestProjection } from '@deepseek-ai/dsh-compaction'
import { deepFreeze } from '@deepseek-ai/dsh-util-values'
// Type-only: makes the token meter service available to `ctx.get()`.
import type {} from '@deepseek-ai/dsh-token-meter'
import type {
  ContentBlock, FinishReason, GenerateOptions, LlmCallConfig, LlmResolvedModelInfo, Message,
  PreparedLlmCall, RequestMessage, TokenUsage, ToolSchema,
} from '@deepseek-ai/dsh-llm'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { projectSummaryTranscript } from './request-projection.ts'

interface SummaryConfig {
  readonly summarizationProvider: string
  readonly summarizationModel: string
  readonly summarizationFallback: 'latest-route' | 'error'
  readonly maxTokens: number
}

/** Tags wrapping the structured summary inside the landed checkpoint node. */
const SUMMARY_OPEN_TAG = '<compacted-summary>'
const SUMMARY_CLOSE_TAG = '</compacted-summary>'

/**
 * The summarization directive, delivered as the FINAL user message after the
 * selected history representation rather than as a separate system prompt.
 * Routes using the conversation prefix retain its cache alignment; text-only
 * transcript routes quote that history before this instruction.
 */
const COMPACTION_INSTRUCTION = [
  'You are now acting as a compaction engine for this AI coding assistant. Condense the conversation ABOVE into a structured checkpoint that lets another model resume the work with no loss of essential context.',
  '',
  'Output EXACTLY the Markdown structure below: keep every section, in order. Use terse bullets, not prose paragraphs. Write "(none)" for an empty section — never drop a section.',
  '',
  '## Primary Request and Intent',
  "- [the user's original and evolving goals; quote verbatim where the exact wording matters]",
  '',
  '## Key Technical Concepts',
  '- [technologies, frameworks, patterns, and conventions in play]',
  '',
  '## Files and Code',
  '- [exact path: why it matters, key changes or snippets]',
  '',
  '## Errors and Fixes',
  '- [error: how it was resolved, plus any related user feedback]',
  '',
  '## Pending Jobs',
  '- [explicitly requested work not yet completed]',
  '',
  '## Current Work',
  '- [precisely what was in progress at this checkpoint]',
  '',
  '## Next Step',
  '- [the single next action, directly in line with the most recent request, or "(none)"]',
  '',
  '## Critical Context',
  '- [decisions and their rationale, constraints, user preferences, open questions, data needed to continue]',
  '',
  'Rules:',
  '- Write concise English engineering prose. Preserve exact file paths, commands, error strings, identifiers, numeric values, function signatures, and syntax fragments.',
  '- Capture user feedback and explicit instructions faithfully, especially corrections.',
  '- Do NOT mention this summarization request or that the context was compacted.',
  '- Output only the checkpoint text: do not call any tool or take any other action.',
  `- If the conversation already contains a ${SUMMARY_OPEN_TAG} block, it is a PRIOR checkpoint. Do not copy it forward verbatim: preserve still-true facts, drop stale ones, and merge newer information into a single consolidated summary under the same structure.`,
].join('\n')

/** Framing that makes the replacement user message established context. */
const CHECKPOINT_PREAMBLE =
  'This is an automatically generated checkpoint condensing an earlier span of the conversation to free up context. Treat the captured context as established background and build on it without restating it. Continue the task directly from the messages that follow, without acknowledging this checkpoint.'

/**
 * The selected conversation surface before auxiliary request projection.
 * Routes with native tools retain its prefix; explicit no-tools Chat routes
 * quote it as text without changing these durable messages.
 */
export interface SummarizationInput {
  /** The conversation's tool schemas, reused for prefix-cache alignment; absent when the request carried none. */
  readonly tools?: readonly ToolSchema[]
  /** The derived system head, when present, followed by the shadowed region in surface order. */
  readonly messages: readonly Message[]
}

/** Safe summary content plus the exact auxiliary call envelope recorded with it. */
export type SummaryResult = {
  summary: ContentBlock[]
  provider: string
  model: string
  maxTokens?: number
  /** Selected input representation; absent for a custom summarizer that does not report it. */
  requestProjection?: SummaryRequestProjection
  /** Provider-reported usage for this summarization request. */
  usage?: TokenUsage
} & (
  | {
    /** Complete provider output before the text-only summary projection. */
    rawOutput: ContentBlock[]
    /** Identifies exactly one call through this context's LLM waterfall. */
    llmStreamCall: true
  }
  | {
    /** Optional complete output from an unmarked template, remote, or other summarizer. */
    rawOutput?: ContentBlock[]
    /** An unmarked result does not identify a call through this context's LLM seam. */
    llmStreamCall?: never
  }
)

/** Reject a target that explicitly uses another protocol or excludes text. */
function requireTextSummarizer(info: LlmResolvedModelInfo): void {
  if (info.requestProtocol !== undefined) {
    throw new LlmError(
      `compaction: summarizer ${info.provider}/${info.id} uses the ${info.requestProtocol} request protocol, `
      + 'not ordinary chat text generation',
      'SUMMARY_TARGET_NOT_CHAT',
    )
  }
  if (info.inputModalities?.includes('text') === false) {
    throw new LlmError(
      `compaction: summarizer ${info.provider}/${info.id} does not accept text input required for compaction`,
      'SUMMARY_TARGET_NOT_CHAT',
    )
  }
}

/**
 * Bind a verified text target and its context window to the eventual dispatch.
 * @param ctx - context providing the LLM service.
 * @param config - selected provider, model, and call controls.
 * @param signal - cancellation forwarded to call preparation.
 * @returns the prepared call, its context window, and the adapter's tool capability.
 */
export async function prepareSummaryCall(
  ctx: Context,
  config: LlmCallConfig,
  signal?: AbortSignal,
): Promise<{ call: PreparedLlmCall; window: number; nativeTools: boolean | undefined }> {
  let nativeTools: boolean | undefined
  let call: PreparedLlmCall
  const { provider, model } = config
  try {
    call = await ctx.llm.prepareCall(config, signal, (controls, info) => {
      requireTextSummarizer(info)
      nativeTools = info.nativeTools
      return { ...controls }
    })
  } catch (error: unknown) {
    signal?.throwIfAborted()
    if (error instanceof LlmError && error.code === 'SUMMARY_TARGET_NOT_CHAT') throw error
    const message = error instanceof Error ? error.message : String(error)
    throw new LlmError(
      `compaction: cannot verify summarizer ${provider}/${model}: ${message}`,
      'SUMMARY_TARGET_UNVERIFIED',
      { cause: error },
    )
  }
  signal?.throwIfAborted()
  const window = call.context?.contextWindow
  if (window === undefined || !Number.isSafeInteger(window) || window <= 0) {
    throw new LlmError(
      `compaction: cannot verify summarizer ${provider}/${model}: `
      + 'context capacity is unavailable; configure contextWindow on the adapter model',
      'SUMMARY_TARGET_UNVERIFIED',
    )
  }
  return { call, window, nativeTools }
}

/**
 * Summarize through the exact ordinary Chat route. Explicit no-tools routes
 * receive quoted text history; other routes retain the conversation prefix.
 * Both append the final compaction instruction and report their selected input.
 * Strict policies bind metadata verification and dispatch to one adapter call.
 * @param ctx - context providing the LLM service.
 * @param config - resolved backend configuration.
 * @param input - replayed conversation prefix (system, tools, and leading messages) to condense.
 * @param agent - supplies routed-model history, fallback model, and session id.
 * @param signal - optional cancellation forwarded to the adapter.
 * @returns safe text-only summary blocks and the exact call envelope and output.
 */
export async function summarizeWithLlm(
  ctx: Context,
  config: SummaryConfig,
  input: SummarizationInput,
  agent: Agent,
  signal?: AbortSignal,
): Promise<SummaryResult> {
  const latest = agent.session.requestHeader()?.config
  const configured = config.summarizationProvider.length === 0
    ? undefined
    : { provider: config.summarizationProvider, model: config.summarizationModel }
  const agentTarget = agent.options.provider !== undefined
    && agent.options.provider.length > 0
    && agent.options.model !== undefined
    && agent.options.model.length > 0
    ? { provider: agent.options.provider, model: agent.options.model }
    : undefined
  const target = configured ?? latest ?? agentTarget
  if (configured === undefined && config.summarizationFallback === 'error') {
    const candidate = latest ?? agentTarget
    const source = latest === undefined ? 'the Agent target' : 'the last routed request'
    throw new LlmError(
      'compaction: no configured summarization target and fallback is disabled'
      + (candidate === undefined ? '' : ` (refusing ${source} ${candidate.provider}/${candidate.model})`)
      + '; set both summarization fields on the compaction policy',
      'SUMMARY_TARGET_UNCONFIGURED',
    )
  }
  if (target === undefined) {
    throw new Error(
      'no provider/model available for summarization: set both BasicCompactionConfig summarization fields, route one request, or set both AgentOptions fields',
    )
  }

  const proposed = { provider: target.provider, model: target.model, maxTokens: config.maxTokens }
  const prepared = config.summarizationFallback === 'error'
    ? await prepareSummaryCall(ctx, proposed, signal)
    : undefined
  // Legacy stream middleware may reroute the call, and missing capacity is
  // allowed when the adapter cannot provide it.
  const info = prepared === undefined
    ? await ctx.llm.resolveModelInfo(target.provider, target.model, signal).catch(() => undefined)
    : undefined
  signal?.throwIfAborted()
  if (info !== undefined) requireTextSummarizer(info)
  const requestProjection: SummaryRequestProjection = (prepared?.nativeTools ?? info?.nativeTools) === false
    ? 'text-transcript-v1'
    : 'conversation-prefix-v1'
  const prefix = requestProjection === 'text-transcript-v1'
    ? projectSummaryTranscript(input.messages, ref => ctx.llm.fileRequestText(ref))
    : input.messages
  const messages: RequestMessage[] = [
    ...prefix,
    deepFreeze({
      role: 'user',
      content: [{ type: 'text', text: COMPACTION_INSTRUCTION }],
    }),
  ]
  const options: GenerateOptions = {
    ...(prepared?.call.config ?? proposed),
    messages,
    ...requestProjection === 'conversation-prefix-v1' ? {
      toolHistory: agent.session.toolHistory(),
      ...input.tools === undefined ? {} : { tools: [...input.tools] },
    } : {},
    sessionId: agent.session.id,
    purpose: 'compaction',
    ...signal === undefined ? {} : { signal },
  }
  const window = prepared?.window ?? info?.context?.contextWindow
  const meter = ctx.get('tokenMeter')
  if (window !== undefined && meter !== undefined) {
    const inputTokens = messages.reduce((total, message) => total + meter.estimateMessage(message), 0)
      + meter.estimateTools(options.tools)
    const required = inputTokens + config.maxTokens
    if (required > window) {
      throw new LlmError(
        `compaction: summarizer ${target.provider}/${target.model} estimated ${inputTokens} input + `
        + `${config.maxTokens} output tokens exceed its ${window}-token context window; `
        + 'increase the retained tail, lower maxTokens, or use a larger-context summarizer',
        'SUMMARY_CAPACITY_EXCEEDED',
      )
    }
  }

  const assembler = new BlockAssembler()
  const stream = prepared === undefined ? ctx.llm.stream(options) : prepared.call.stream(options)
  for await (const chunk of stream) assembler.push(chunk)
  const error = finishError(assembler.finish)
  if (error !== undefined) throw error

  const rawOutput = assembler.blocks()
  const summary = summaryText(rawOutput)
  if (!summary.some(block => block.text.trim().length > 0)) {
    throw new Error('summarization produced no text summary content')
  }
  return {
    summary,
    rawOutput,
    llmStreamCall: true,
    provider: options.provider,
    model: options.model,
    maxTokens: config.maxTokens,
    requestProjection,
    ...(assembler.usage === undefined ? {} : { usage: assembler.usage }),
  }
}

/**
 * Wrap raw summary blocks in the durable checkpoint framing.
 * @param summary - safe text-only model output.
 * @returns content for the synthesized replacement user message.
 */
export function frameSummary(summary: readonly ContentBlock[]): ContentBlock[] {
  return [
    { type: 'text', text: `${CHECKPOINT_PREAMBLE}\n\n${SUMMARY_OPEN_TAG}` },
    ...summary,
    { type: 'text', text: SUMMARY_CLOSE_TAG },
  ]
}

/** Map a terminal summarization finish to its fail-closed error. */
function finishError(finish: FinishReason): Error | undefined {
  switch (finish.kind) {
    case 'error':
    case 'aborted': {
      return LlmError.fromFailure(finish.failure)
    }
    case 'max-tokens': {
      const error = new Error('summarization truncated at the token cap (incomplete checkpoint)') as Error & { code?: string }
      error.code = 'MAX_TOKENS'
      return error
    }
    case 'tool-calls':
      return new LlmError('compaction summary must not call tools', 'UNSUPPORTED_CONTENT')
    default:
      return undefined
  }
}

/** Reject unsupported output and omit reasoning before synthesizing a user message. */
function summaryText(
  blocks: readonly ContentBlock[],
): Array<Extract<ContentBlock, { type: 'text' }>> {
  for (const block of blocks) {
    if (block.type !== 'text' && block.type !== 'reasoning') {
      throw new LlmError(`compaction summary cannot contain ${block.type} output`, 'UNSUPPORTED_CONTENT')
    }
  }
  return blocks.filter((block): block is Extract<ContentBlock, { type: 'text' }> => block.type === 'text')
}
