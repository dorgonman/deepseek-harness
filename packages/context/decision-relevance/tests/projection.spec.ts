import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { ToolCallId, createUserMessage, createToolResultMessage } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, Message, RequestMessage } from '@deepseek-ai/dsh-llm'
import { SessionId, SessionStore } from '@deepseek-ai/dsh-session'
import DecisionRelevanceFilter from '@deepseek-ai/dsh-context-decision-relevance'
import type { RelevanceJudgment } from '@deepseek-ai/dsh-context-decision-relevance'

const ROUTE = { provider: 'chat-model', model: 'chat-model' }
const DECISION_ROUTE = { provider: 'decision-provider', model: 'decision-model' }

/** One valid KINF Decision readout rendered as the adapter renders it. */
function readout(label: 'yes' | 'no', yes: number, no: number, servedModel = DECISION_ROUTE.model): string {
  return JSON.stringify({
    model: servedModel,
    decision: {
      schema: 'kinf.decision.v1',
      mode: 'label_scores',
      label,
      token_position: 10,
      logits: [],
      label_scores: { yes, no },
      calibration: 'uncalibrated',
    },
    served_identity: {
      mode: 'service',
      instance: 'inst-1',
      model: servedModel,
      requested_model: DECISION_ROUTE.model,
    },
    usage: { prompt_tokens: 10, completion_tokens: 1, total_tokens: 11 },
  })
}

/** Recorded state for one fake Decision call. */
interface FakeCall {
  options: GenerateOptions
  /** The candidate text the filter actually sent, extracted from the prompt. */
  candidate: string
}

/** Extract the tool-result body from the bounded Decision prompt. */
function candidateOf(options: GenerateOptions): string {
  const prompt = (options.messages[0]?.content[0] as { text: string }).text
  const marker = '\n\nTool result:\n'
  const start = prompt.indexOf(marker)
  if (start < 0) return ''
  return prompt.slice(start + marker.length).split('\n\nIs this tool result')[0] ?? ''
}

/**
 * Fake Decision transport.
 *
 * Yields exactly the event shape the real KINF Decision adapter emits
 * (`block-start` / `text-delta` / `block-end` / `usage` / `finish`), so the
 * filter is exercised against the production stream contract. `respond` sees
 * the candidate actually sent, so one harness can script different verdicts.
 */
function fakeLlm(respond: (call: FakeCall) => string | Error, calls: FakeCall[]) {
  return {
    async prepareCall(route: { provider: string; model: string }, signal?: AbortSignal) {
      signal?.throwIfAborted()
      return {
        config: { provider: route.provider, model: route.model, maxTokens: 1, temperature: 0 },
        async * stream(options: GenerateOptions) {
          const call: FakeCall = { options, candidate: candidateOf(options) }
          calls.push(call)
          const outcome = respond(call)
          if (outcome instanceof Error) {
            yield { type: 'finish', reason: { kind: 'error', failure: { code: 'TRANSPORT', message: outcome.message } } }
            return
          }
          yield { type: 'block-start', index: 0, blockType: 'text' }
          yield { type: 'text-delta', index: 0, text: outcome }
          yield { type: 'block-end', index: 0, block: { type: 'text', text: outcome } }
          yield { type: 'usage', usage: { inputTokens: 10, outputTokens: 1, totalTokens: 11 } }
          yield { type: 'finish', reason: { kind: 'stop' } }
        },
      }
    },
  }
}

/** Build a context whose `llm.prepareCall` returns a scripted Decision response. */
function harness(
  respond: (call: FakeCall) => string | Error,
  config: Record<string, unknown> = {},
): { ctx: Context; filter: DecisionRelevanceFilter; calls: FakeCall[] } {
  const ctx = new Context()
  const calls: FakeCall[] = []
  Object.defineProperty(ctx, 'llm', {
    configurable: true,
    value: fakeLlm(respond, calls),
  })
  // The real session store, so the filter resolves sessions exactly as it does
  // at the runtime seam.
  new SessionStore(ctx)
  const filter = new DecisionRelevanceFilter(ctx, {
    mode: 'on',
    ...DECISION_ROUTE,
    minCandidateChars: 4,
    ...config,
  })
  return { ctx, filter, calls }
}

function toolMessage(text: string, callId: string, isError = false): Message {
  return createToolResultMessage({
    callId: ToolCallId(callId),
    content: [{ type: 'text', text }],
    isError,
  })
}

/** Read the tool-call correlation of a projected message. */
function callIdOf(message: RequestMessage | undefined): string | undefined {
  return message?.role === 'tool' ? message.toolCallId : undefined
}

/** Read the tool-call correlation of a locally built tool result. */
function callIdOfBuilt(message: Message): string {
  return message.role === 'tool' ? message.toolCallId : ''
}

function userMessage(text: string): Message {
  return createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } })
}

function project(
  filter: DecisionRelevanceFilter,
  messages: readonly RequestMessage[],
  sessionId = 'relevance-spec',
  signal?: AbortSignal,
) {
  // Register the session on the filter's own store, as the runtime does.
  const ctx = (filter as unknown as { ctx: Context }).ctx
  if (ctx.sessions.get(SessionId(sessionId)) === undefined) ctx.sessions.create(SessionId(sessionId))
  return filter.project(messages, sessionId, ROUTE, signal)
}

describe('decision relevance projection', () => {
  it('is inactive without an explicit route and leaves the request untouched', async () => {
    const { filter, calls } = harness(() => readout('no', 0, 1), { mode: 'off' })
    const messages = [userMessage('goal'), toolMessage('x'.repeat(50), 'a')]
    const result = await project(filter, messages)
    expect(result.messages).toBe(messages)
    expect(result.applied).toBe(false)
    expect(calls).toHaveLength(0)
  })

  it('elides an irrelevant tool result for this request only', async () => {
    const { filter, calls } = harness(call => call.candidate.startsWith('z')
      ? readout('no', 0.02, 0.98)
      : readout('yes', 0.98, 0.02))
    const candidate = toolMessage('z'.repeat(50), 'a')
    const retained = toolMessage('y'.repeat(50), 'b')
    const messages = [userMessage('fix the build'), candidate, retained]
    const result = await project(filter, messages, 'relevance-elide')

    expect(result.applied).toBe(true)
    expect(result.messages).toHaveLength(3)
    expect(result.messages[1]?.content).toEqual([{ type: 'text', text: filter.config.elisionText }])
    // Untouched messages keep their exact identity: no rewrite, no reorder.
    expect(result.messages[0]).toBe(messages[0])
    expect(result.messages[2]).toBe(retained)
    // The elided message keeps its identity, role and tool-call correlation.
    expect(result.messages[1]?.id).toBe(candidate.id)
    expect(result.messages[1]?.role).toBe('tool')
    expect(callIdOf(result.messages[1])).toBe(callIdOfBuilt(candidate))
    // The original list is never mutated.
    expect(candidate.content).toEqual([{ type: 'text', text: 'z'.repeat(50) }])

    expect(calls).toHaveLength(2)
    expect(calls[0]?.options.provider).toBe(DECISION_ROUTE.provider)
    expect(calls[0]?.options.model).toBe(DECISION_ROUTE.model)
    expect(calls[0]?.options.purpose).toBe('decision-relevance')
    expect(result.judgments.map(j => j.disposition)).toEqual(['elided-irrelevant', 'retained-relevant'])
    expect(result.judgments[0]?.callId).toBe(ToolCallId('a'))
    expect(result.judgments[1]?.callId).toBe(ToolCallId('b'))
  })

  it('retains a relevant tool result and records the served identity', async () => {
    const { filter } = harness(() => readout('yes', 0.95, 0.05))
    const candidate = toolMessage('z'.repeat(50), 'a')
    const messages = [userMessage('fix the build'), candidate]
    const result = await project(filter, messages)

    expect(result.applied).toBe(false)
    expect(result.messages).toBe(messages)
    const judgment = result.judgments[0] as RelevanceJudgment
    expect(judgment.disposition).toBe('retained-relevant')
    expect(judgment.score).toBeCloseTo(0.95, 10)
    expect(judgment.label).toBe('yes')
    expect(judgment.served).toEqual({ model: DECISION_ROUTE.model, mode: 'service', instance: 'inst-1' })
    expect(judgment.requested).toEqual(ROUTE)
  })

  it('retains a score inside the uncertainty band', async () => {
    const { filter } = harness(() => readout('yes', 0.6, 0.4))
    const messages = [userMessage('goal'), toolMessage('z'.repeat(50), 'a')]
    const result = await project(filter, messages)
    expect(result.applied).toBe(false)
    expect(result.judgments[0]?.disposition).toBe('retained-uncertain')
  })

  it('fails open on a transport error, a malformed readout, and a served-model mismatch', async () => {
    const failures: Array<() => string | Error> = [
      () => new Error('boom'),
      () => 'not json',
      () => JSON.stringify({ model: DECISION_ROUTE.model }),
      () => readout('yes', 0.9, 0.1, 'someone-else'),
    ]
    for (const respond of failures) {
      const { filter } = harness(respond)
      const messages = [userMessage('goal'), toolMessage('z'.repeat(50), 'a')]
      const result = await project(filter, messages)
      expect(result.applied).toBe(false)
      expect(result.messages).toBe(messages)
      expect(result.judgments[0]?.disposition).toBe('retained-error')
    }
  })

  it('retains content when the caller cancels', async () => {
    const controller = new AbortController()
    controller.abort(new Error('caller cancelled'))
    const { filter, calls } = harness(() => readout('no', 0, 1))
    const messages = [userMessage('goal'), toolMessage('z'.repeat(50), 'a')]
    const result = await project(filter, messages, 'relevance-cancel', controller.signal)
    expect(result.applied).toBe(false)
    expect(result.judgments[0]?.disposition).toBe('retained-error')
    expect(calls).toHaveLength(0)
  })

  it('bounds evaluations per request and retains the rest', async () => {
    const { filter, calls } = harness(() => readout('no', 0, 1), { maxCandidatesPerRequest: 1 })
    const messages = [
      userMessage('goal'),
      toolMessage('a'.repeat(50), 'a'),
      toolMessage('b'.repeat(50), 'b'),
      toolMessage('c'.repeat(50), 'c'),
    ]
    const result = await project(filter, messages)
    expect(calls).toHaveLength(1)
    expect(result.judgments.map(j => j.disposition)).toEqual([
      'elided-irrelevant',
      'retained-limit',
      'retained-limit',
    ])
    // Only the evaluated candidate is elided.
    expect(result.messages[1]?.content).toEqual([{ type: 'text', text: filter.config.elisionText }])
    expect(result.messages[2]?.content).toEqual([{ type: 'text', text: 'b'.repeat(50) }])
  })

  it('skips short results, failed results, and requests without a user goal', async () => {
    const shortHarness = harness(() => readout('no', 0, 1))
    const short = await project(shortHarness.filter, [userMessage('goal'), toolMessage('ab', 'a')], 'relevance-short')
    expect(short.applied).toBe(false)
    expect(short.judgments).toHaveLength(0)
    expect(shortHarness.calls).toHaveLength(0)

    const failedHarness = harness(() => readout('no', 0, 1))
    const failed = await project(failedHarness.filter, [userMessage('goal'), toolMessage('q'.repeat(50), 'b', true)], 'relevance-failed')
    expect(failed.applied).toBe(false)
    expect(failed.judgments).toHaveLength(0)
    expect(failedHarness.calls).toHaveLength(0)

    const noGoalHarness = harness(() => readout('no', 0, 1))
    const noGoal = await project(noGoalHarness.filter, [toolMessage('w'.repeat(50), 'c')], 'relevance-nogoal')
    expect(noGoal.applied).toBe(false)
    expect(noGoal.judgments).toHaveLength(0)
    expect(noGoalHarness.calls).toHaveLength(0)
  })

  it('judges each tool result once across requests', async () => {
    const { filter, calls } = harness(() => readout('no', 0, 1))
    const messages = [userMessage('goal'), toolMessage('z'.repeat(50), 'a')]
    await project(filter, messages)
    await project(filter, messages)
    await project(filter, messages)
    expect(calls).toHaveLength(1)
  })

  it('bounds the Decision input without splitting a surrogate pair', async () => {
    const { filter, calls } = harness(() => readout('no', 0, 1), { maxInputChars: 5 })
    const text = '😀'.repeat(20)
    await project(filter, [userMessage('goal'), toolMessage(text, 'a')])
    const prompt = calls[0]?.options.messages[0]?.content
    const sent = (prompt?.[0] as { text: string } | undefined)?.text ?? ''
    expect(sent).toContain('😀'.repeat(5))
    expect(sent).not.toContain('😀'.repeat(6))
  })

  it('records bounded durable evidence with no credentials or history dump', async () => {
    const { filter, ctx } = harness(() => readout('no', 0.01, 0.99))
    const store = ctx.sessions.create(SessionId('relevance-evidence'))
    const messages = [userMessage('the current goal'), toolMessage('z'.repeat(50), 'a')]
    await filter.project(messages, 'relevance-evidence', ROUTE, undefined)

    const events = store.snapshotEvents().filter(event => event.type === 'context/decision-relevance')
    expect(events).toHaveLength(1)
    const data = events[0]?.data as {
      applied: boolean
      conversation: { provider: string; model: string }
      judgments: RelevanceJudgment[]
    }
    expect(data.applied).toBe(true)
    expect(data.conversation).toEqual(ROUTE)
    expect(data.judgments).toHaveLength(1)
    const judgment = data.judgments[0] as unknown as Record<string, unknown>
    expect(Object.keys(judgment).sort()).toEqual([
      'callId', 'candidateChars', 'disposition', 'inputChars', 'label', 'messageId',
      'reason', 'requested', 'score', 'served',
    ])
    // No prompt, no tool-result body, no credential field.
    const serialized = JSON.stringify(data)
    expect(serialized).not.toContain('z'.repeat(50))
    expect(serialized).not.toContain('the current goal')
    expect(serialized.toLowerCase()).not.toContain('authorization')
    expect(serialized.toLowerCase()).not.toContain('bearer')
  })

  it('records evidence even when nothing was elided', async () => {
    const { filter, ctx } = harness(() => readout('yes', 0.99, 0.01))
    const store = ctx.sessions.create(SessionId('relevance-retained'))
    await filter.project([userMessage('goal'), toolMessage('z'.repeat(50), 'a')], 'relevance-retained', ROUTE, undefined)
    const events = store.snapshotEvents().filter(event => event.type === 'context/decision-relevance')
    expect(events).toHaveLength(1)
    expect((events[0]?.data as { applied: boolean }).applied).toBe(false)
  })

  it('keeps tool-call correlation intact when a result is elided', async () => {
    const { filter } = harness(() => readout('no', 0, 1))
    const candidate = toolMessage('z'.repeat(50), 'call-42')
    const result = await project(filter, [userMessage('goal'), candidate])
    const projected = result.messages[1] as { role: string; toolCallId: string; content: ContentBlock[] }
    expect(projected.role).toBe('tool')
    expect(projected.toolCallId).toBe('call-42')
    expect(projected.content).toEqual([{ type: 'text', text: filter.config.elisionText }])
  })

  it('uses the most recent user message as the goal', async () => {
    const { filter, calls } = harness(() => readout('no', 0, 1))
    await project(filter, [
      userMessage('the old goal'),
      toolMessage('z'.repeat(50), 'a'),
      userMessage('the newest goal'),
    ])
    const prompt = (calls[0]?.options.messages[0]?.content[0] as { text: string }).text
    expect(prompt).toContain('the newest goal')
    expect(prompt).not.toContain('the old goal')
  })
})
