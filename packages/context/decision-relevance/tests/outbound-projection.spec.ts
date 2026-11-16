/**
 * End-to-end proof that the relevance filter shapes the OUTBOUND request.
 *
 * The conversation model is a recording adapter: it captures exactly what the
 * runtime was about to send. The Decision model is a separate scripted route.
 * The assertions therefore distinguish the two things the ticket requires —
 * the wire copy is filtered, while the session's durable history is not.
 */
import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, {
  LlmAdapter,
  ToolCallId,
  createUserMessage,
  createToolResultMessage,
} from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import { SessionId, SessionStore } from '@deepseek-ai/dsh-session'
import DecisionRelevanceFilter from '@deepseek-ai/dsh-context-decision-relevance'

const CHAT = { provider: 'chat-provider', model: 'chat-model' }
const DECISION = { provider: 'decision-provider', model: 'decision-model' }

const REPLY: StreamChunk[] = [
  { type: 'block-start', index: 0, blockType: 'text' },
  { type: 'text-delta', index: 0, text: 'the reply' },
  { type: 'block-end', index: 0, block: { type: 'text', text: 'the reply' } },
  { type: 'finish', reason: { kind: 'stop' } },
]

/** The chat adapter: records the exact options the runtime dispatches. */
class RecordingAdapter extends LlmAdapter {
  readonly dispatches: GenerateOptions[] = []

  override async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.dispatches.push(options)
    yield * REPLY
  }
}

/** The Decision adapter: answers one bounded relevance question per call. */
class DecisionAdapter extends LlmAdapter {
  readonly prompts: string[] = []

  constructor(private readonly relevant: (candidate: string) => boolean) {
    super()
  }

  override async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const prompt = (options.messages[0]?.content[0] as { text: string }).text
    this.prompts.push(prompt)
    const candidate = prompt.split('\n\nTool result:\n')[1]?.split('\n\nIs this tool result')[0] ?? ''
    const yes = this.relevant(candidate)
    const text = JSON.stringify({
      model: DECISION.model,
      decision: {
        schema: 'kinf.decision.v1',
        mode: 'label_scores',
        label: yes ? 'yes' : 'no',
        token_position: 10,
        logits: [],
        label_scores: { yes: yes ? 0.97 : 0.03, no: yes ? 0.03 : 0.97 },
        calibration: 'uncalibrated',
      },
      served_identity: { mode: 'service', instance: 'inst-1', model: DECISION.model },
      usage: { prompt_tokens: 10, completion_tokens: 1, total_tokens: 11 },
    })
    yield { type: 'block-start', index: 0, blockType: 'text' }
    yield { type: 'text-delta', index: 0, text }
    yield { type: 'block-end', index: 0, block: { type: 'text', text } }
    yield { type: 'usage', usage: { inputTokens: 10, outputTokens: 1, totalTokens: 11 } }
    yield { type: 'finish', reason: { kind: 'stop' } }
  }
}

function text(message: { content?: readonly { type: string; text?: string }[] } | undefined): string {
  return (message?.content?.[0] as { text?: string } | undefined)?.text ?? ''
}

async function collect(stream: AsyncIterable<unknown>): Promise<void> {
  for await (const _ of stream) void _
}

describe('decision relevance outbound projection', () => {
  it('filters the wire copy while the original chat model still replies', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(SessionStore)
    const chat = new RecordingAdapter()
    const decision = new DecisionAdapter(candidate => candidate.startsWith('KEEP'))
    ctx.llm.registerAdapter([CHAT.provider], chat)
    ctx.llm.registerAdapter([DECISION.provider], decision)
    const filter = new DecisionRelevanceFilter(ctx, { mode: 'on', ...DECISION, minCandidateChars: 4 })

    const session = ctx.sessions.create(SessionId('outbound-filtered'))
    // Durable history is written by the loop as events; the test appends the
    // same events so `deriveMessages()` can show what survived.
    session.append('user/message',
      createUserMessage({ content: [{ type: 'text', text: 'summarize the report' }], source: { kind: 'user' } }),
      { surfaceOp: 'append' })
    session.append('tool/result', {
      turn: 1,
      step: 1,
      message: createToolResultMessage({ callId: ToolCallId('drop'), content: [{ type: 'text', text: 'DROP '.repeat(20) }], isError: false }),
    }, { surfaceOp: 'append' })
    const messages = [
      createUserMessage({ content: [{ type: 'text', text: 'summarize the report' }], source: { kind: 'user' } }),
      createToolResultMessage({ callId: ToolCallId('drop'), content: [{ type: 'text', text: 'DROP '.repeat(20) }], isError: false }),
      createToolResultMessage({ callId: ToolCallId('keep'), content: [{ type: 'text', text: 'KEEP '.repeat(20) }], isError: false }),
    ]

    await collect(ctx.llm.stream({
      provider: CHAT.provider,
      model: CHAT.model,
      messages,
      sessionId: session.id,
    }))

    // The conversation model received the filtered copy.
    expect(chat.dispatches).toHaveLength(1)
    const sent = chat.dispatches[0]?.messages ?? []
    expect(sent).toHaveLength(3)
    expect(text(sent[1])).toBe(filter.config.elisionText)
    expect(text(sent[2])).toBe('KEEP '.repeat(20))

    // The Decision model was asked about each candidate, separately from the chat route.
    expect(decision.prompts).toHaveLength(2)
    expect(decision.prompts.every(prompt => prompt.includes('summarize the report'))).toBe(true)

    // Durable history is untouched: the original full bodies are still there.
    const durable = session.deriveMessages()
    expect(durable.some(message => text(message).startsWith('DROP '))).toBe(true)
    expect(durable.some(message => text(message) === filter.config.elisionText)).toBe(false)

    // Bounded evidence was recorded, and it names both routes.
    const evidence = session.snapshotEvents().filter(event => event.type === 'context/decision-relevance')
    expect(evidence).toHaveLength(1)
    const data = evidence[0]?.data as { applied: boolean; conversation: unknown; judgments: unknown[] }
    expect(data.applied).toBe(true)
    expect(data.conversation).toEqual({ provider: CHAT.provider, model: CHAT.model })
    expect(data.judgments).toHaveLength(2)
  })

  it('sends the unfiltered request when the capability is not installed', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(SessionStore)
    const chat = new RecordingAdapter()
    ctx.llm.registerAdapter([CHAT.provider], chat)

    const session = ctx.sessions.create(SessionId('outbound-unfiltered'))
    const messages = [
      createUserMessage({ content: [{ type: 'text', text: 'goal' }], source: { kind: 'user' } }),
      createToolResultMessage({ callId: ToolCallId('a'), content: [{ type: 'text', text: 'x'.repeat(50) }], isError: false }),
    ]
    await collect(ctx.llm.stream({ provider: CHAT.provider, model: CHAT.model, messages, sessionId: session.id }))

    expect(chat.dispatches[0]?.messages).toBe(messages)
    expect(session.snapshotEvents().filter(event => event.type === 'context/decision-relevance')).toHaveLength(0)
  })

  it('sends the unfiltered request when the capability is disabled', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(SessionStore)
    const chat = new RecordingAdapter()
    const decision = new DecisionAdapter(() => false)
    ctx.llm.registerAdapter([CHAT.provider], chat)
    ctx.llm.registerAdapter([DECISION.provider], decision)
    new DecisionRelevanceFilter(ctx, { mode: 'off', ...DECISION, minCandidateChars: 4 })

    const session = ctx.sessions.create(SessionId('outbound-disabled'))
    const messages = [
      createUserMessage({ content: [{ type: 'text', text: 'goal' }], source: { kind: 'user' } }),
      createToolResultMessage({ callId: ToolCallId('a'), content: [{ type: 'text', text: 'x'.repeat(50) }], isError: false }),
    ]
    await collect(ctx.llm.stream({ provider: CHAT.provider, model: CHAT.model, messages, sessionId: session.id }))

    expect(chat.dispatches[0]?.messages).toBe(messages)
    expect(decision.prompts).toHaveLength(0)
  })

  it('never mutates the caller request when it projects', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(SessionStore)
    const chat = new RecordingAdapter()
    ctx.llm.registerAdapter([CHAT.provider], chat)
    ctx.llm.registerAdapter([DECISION.provider], new DecisionAdapter(() => false))
    new DecisionRelevanceFilter(ctx, { mode: 'on', ...DECISION, minCandidateChars: 4 })

    const session = ctx.sessions.create(SessionId('outbound-immutable'))
    const messages = [
      createUserMessage({ content: [{ type: 'text', text: 'goal' }], source: { kind: 'user' } }),
      createToolResultMessage({ callId: ToolCallId('a'), content: [{ type: 'text', text: 'x'.repeat(50) }], isError: false }),
    ]
    const before = messages.map(message => text(message))
    await collect(ctx.llm.stream({ provider: CHAT.provider, model: CHAT.model, messages, sessionId: session.id }))

    // Projection builds a new list; the caller's messages are untouched, so the
    // loop's request stays a pure function of the session log.
    expect(messages.map(message => text(message))).toEqual(before)
    expect(chat.dispatches[0]?.messages).not.toBe(messages)
    expect(text(chat.dispatches[0]?.messages[1])).toBe(filterOf(ctx).config.elisionText)
  })
})

/** Read the filter registered on a context. */
function filterOf(ctx: Context): DecisionRelevanceFilter {
  return ctx.decisionRelevance
}
