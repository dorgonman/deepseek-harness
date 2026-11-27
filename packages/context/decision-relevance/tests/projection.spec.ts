import assert from 'node:assert/strict'
import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { ToolCallId, createUserMessage, createToolResultMessage } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, FinishReason, GenerateOptions, Message, RequestMessage, StreamChunk } from '@deepseek-ai/dsh-llm'
import { SessionId, SessionStore } from '@deepseek-ai/dsh-session'
import DecisionRelevanceFilter from '@deepseek-ai/dsh-context-decision-relevance'
import type { RelevanceJudgment } from '@deepseek-ai/dsh-context-decision-relevance'

const ROUTE = { provider: 'chat-model', model: 'chat-model' }
const DECISION_ROUTE = { provider: 'decision-provider', model: 'decision-model' }

function readoutData(label: 'yes' | 'no', yes: number, no: number, servedModel = DECISION_ROUTE.model) {
  return {
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
  }
}

/** One valid KINF Decision readout rendered as the adapter renders it. */
function readout(label: 'yes' | 'no', yes: number, no: number, servedModel = DECISION_ROUTE.model): string {
  return JSON.stringify(readoutData(label, yes, no, servedModel))
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
function fakeLlm(respond: (call: FakeCall) => string | FinishReason, calls: FakeCall[]) {
  return {
    async prepareCall(route: { provider: string; model: string }, signal?: AbortSignal) {
      signal?.throwIfAborted()
      return {
        config: { provider: route.provider, model: route.model, maxTokens: 1, temperature: 0 },
        async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
          const call: FakeCall = { options, candidate: candidateOf(options) }
          calls.push(call)
          const outcome = respond(call)
          if (typeof outcome !== 'string') {
            yield { type: 'finish', reason: outcome }
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
  respond: (call: FakeCall) => string | FinishReason,
  config: Record<string, unknown> = {},
) {
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
  const project = (
    messages: readonly RequestMessage[],
    sessionId = 'relevance-spec',
    signal?: AbortSignal,
  ) => {
    const id = SessionId(sessionId)
    if (ctx.sessions.get(id) === undefined) ctx.sessions.create(id)
    return filter.project(messages, sessionId, ROUTE, signal)
  }
  return { ctx, filter, calls, project }
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

describe('decision relevance projection', () => {
  it('is inactive without an explicit route and leaves the request untouched', async () => {
    const { project, calls } = harness(() => readout('no', 0, 1), { mode: 'off' })
    const messages = [userMessage('goal'), toolMessage('x'.repeat(50), 'a')]
    const result = await project(messages)
    expect(result.messages).toBe(messages)
    expect(result.applied).toBe(false)
    expect(calls).toHaveLength(0)
  })

  it('elides an irrelevant tool result for this request only', async () => {
    const { filter, project, calls } = harness(call => call.candidate.startsWith('z')
      ? readout('no', 0.02, 0.98)
      : readout('yes', 0.98, 0.02))
    const candidate = toolMessage('z'.repeat(50), 'a')
    const retained = toolMessage('y'.repeat(50), 'b')
    const messages = [userMessage('fix the build'), candidate, retained]
    const result = await project(messages, 'relevance-elide')

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
    const { project } = harness(() => readout('yes', 0.95, 0.05))
    const candidate = toolMessage('z'.repeat(50), 'a')
    const messages = [userMessage('fix the build'), candidate]
    const result = await project(messages)

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
    const { project } = harness(() => readout('yes', 0.6, 0.4))
    const messages = [userMessage('goal'), toolMessage('z'.repeat(50), 'a')]
    const result = await project(messages)
    expect(result.applied).toBe(false)
    expect(result.judgments[0]?.disposition).toBe('retained-uncertain')
  })

  const validReadout = readoutData('no', 0, 1)
  const malformedReadouts = [
    ['blank output', '  '],
    ['invalid JSON', 'not json'],
    ['a primitive', 'true'],
    ['null', 'null'],
    ['an array', '[]'],
    ['a missing model', JSON.stringify({ ...validReadout, model: undefined })],
    ['an empty model', JSON.stringify({ ...validReadout, model: '' })],
    ['a missing decision', JSON.stringify({ ...validReadout, decision: undefined })],
    ['a missing served identity', JSON.stringify({ ...validReadout, served_identity: undefined })],
    ['the wrong schema', JSON.stringify({ ...validReadout, decision: { ...validReadout.decision, schema: 'other' } })],
    ['the wrong mode', JSON.stringify({ ...validReadout, decision: { ...validReadout.decision, mode: 'other' } })],
    ['an invalid label', JSON.stringify({ ...validReadout, decision: { ...validReadout.decision, label: 'other' } })],
    ['missing scores', JSON.stringify({ ...validReadout, decision: { ...validReadout.decision, label_scores: undefined } })],
    ['an out-of-range yes score', JSON.stringify(readoutData('no', -0.1, 1))],
    ['an out-of-range no score', JSON.stringify(readoutData('no', 0, 1.1))],
  ] as const

  it.each(malformedReadouts)('retains the result when the readout has %s', async (_label, response) => {
    const { project } = harness(() => response)
    const messages = [userMessage('goal'), toolMessage('z'.repeat(50), 'a')]
    const result = await project(messages)
    expect(result.messages).toBe(messages)
    expect(result.applied).toBe(false)
    expect(result.judgments).toMatchObject([{ disposition: 'retained-error', reason: 'malformed Decision readout' }])
  })

  const failure = { code: 'TRANSPORT', message: 'provider detail' }
  const transportFailures: { name: string; respond: () => string | FinishReason; reason: string }[] = [
    {
      name: 'a transport error finish',
      respond: () => ({ kind: 'error', failure }),
      reason: 'decision transport failed (TRANSPORT)',
    },
    {
      name: 'a transport abort finish',
      respond: () => ({ kind: 'aborted', failure }),
      reason: 'evaluation cancelled',
    },
    {
      name: 'a thrown error',
      respond: () => { throw new TypeError(failure.message) },
      reason: 'evaluation failed (TypeError)',
    },
    {
      name: 'a thrown error with no name',
      respond: () => { throw Object.assign(new Error(failure.message), { name: '' }) },
      reason: 'evaluation failed (unknown)',
    },
    {
      name: 'a served-model mismatch',
      respond: () => readout('no', 0, 1, 'someone-else'),
      reason: 'served model differs from the requested Decision model',
    },
  ]

  it.each(transportFailures)('retains content after $name', async ({ respond, reason }) => {
    const { project } = harness(respond)
    const messages = [userMessage('goal'), toolMessage('z'.repeat(50), 'a')]
    const result = await project(messages)
    expect(result.messages).toBe(messages)
    expect(result.applied).toBe(false)
    expect(result.judgments).toMatchObject([{ disposition: 'retained-error', reason }])
  })

  it('retains messages when their session is no longer live', async () => {
    const { filter, calls } = harness(() => readout('no', 0, 1))
    const messages = [userMessage('goal'), toolMessage('z'.repeat(50), 'a')]
    const result = await filter.project(messages, 'relevance-missing', ROUTE, undefined)
    expect(result.messages).toBe(messages)
    expect(result.applied).toBe(false)
    expect(result.judgments).toHaveLength(0)
    expect(calls).toHaveLength(0)
  })

  it('retains content when the caller cancels', async () => {
    const controller = new AbortController()
    controller.abort(new Error('caller cancelled'))
    const { project, calls } = harness(() => readout('no', 0, 1))
    const messages = [userMessage('goal'), toolMessage('z'.repeat(50), 'a')]
    const result = await project(messages, 'relevance-cancel', controller.signal)
    expect(result.applied).toBe(false)
    expect(result.judgments[0]?.disposition).toBe('retained-error')
    expect(calls).toHaveLength(0)
  })

  it('bounds evaluations per request and retains the rest', async () => {
    const { filter, project, calls } = harness(() => readout('no', 0, 1), { maxCandidatesPerRequest: 1 })
    const messages = [
      userMessage('goal'),
      toolMessage('a'.repeat(50), 'a'),
      toolMessage('b'.repeat(50), 'b'),
      toolMessage('c'.repeat(50), 'c'),
    ]
    const result = await project(messages)
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
    const short = await shortHarness.project([userMessage('goal'), toolMessage('ab', 'a')], 'relevance-short')
    expect(short.applied).toBe(false)
    expect(short.judgments).toHaveLength(0)
    expect(shortHarness.calls).toHaveLength(0)

    const failedHarness = harness(() => readout('no', 0, 1))
    const failed = await failedHarness.project([userMessage('goal'), toolMessage('q'.repeat(50), 'b', true)], 'relevance-failed')
    expect(failed.applied).toBe(false)
    expect(failed.judgments).toHaveLength(0)
    expect(failedHarness.calls).toHaveLength(0)

    const noGoalHarness = harness(() => readout('no', 0, 1))
    const noGoal = await noGoalHarness.project([toolMessage('w'.repeat(50), 'c')], 'relevance-nogoal')
    expect(noGoal.applied).toBe(false)
    expect(noGoal.judgments).toHaveLength(0)
    expect(noGoalHarness.calls).toHaveLength(0)
  })

  it('reuses elision and retention judgments across requests', async () => {
    const { project, calls } = harness(call => call.candidate.startsWith('z')
      ? readout('no', 0, 1)
      : readout('yes', 1, 0))
    const retained = toolMessage('y'.repeat(50), 'b')
    const messages = [userMessage('goal'), toolMessage('z'.repeat(50), 'a'), retained]
    const first = await project(messages)
    const second = await project(messages)
    expect(second).toEqual(first)
    expect(second.messages[2]).toBe(retained)
    expect(calls).toHaveLength(2)
  })

  it('keeps the freeze state of replaced content', async () => {
    const { project } = harness(() => readout('no', 0, 1))
    const frozen = toolMessage('z'.repeat(50), 'a')
    const original = toolMessage('y'.repeat(50), 'b')
    const mutable = { ...original, content: [...original.content] }
    const result = await project([userMessage('goal'), frozen, mutable])
    expect(result.judgments.map(judgment => judgment.disposition)).toEqual(['elided-irrelevant', 'elided-irrelevant'])
    expect(Object.isFrozen(result.messages[1]?.content)).toBe(true)
    expect(Object.isFrozen(result.messages[2]?.content)).toBe(false)
  })

  it('counts and sends only text from mixed content and skips blank goals', async () => {
    const { project, calls } = harness(() => readout('no', 0, 1))
    const reasoning = { type: 'reasoning', text: 'reasoning omitted from the Decision prompt' } as const
    const candidate = createToolResultMessage({
      callId: ToolCallId('a'),
      content: [reasoning, { type: 'text', text: 'report' }],
      isError: false,
    })
    const belowMinimum = createToolResultMessage({
      callId: ToolCallId('b'),
      content: [reasoning, { type: 'text', text: 'abc' }],
      isError: false,
    })
    const goal = createUserMessage({ content: [reasoning, { type: 'text', text: 'find the report' }], source: { kind: 'user' } })
    const result = await project([goal, candidate, belowMinimum, userMessage('  ')])
    const prompt = calls[0]?.options.messages[0]?.content[0]
    expect(calls).toHaveLength(1)
    expect(prompt).toEqual({
      type: 'text',
      text: 'Goal:\nfind the report\n\nTool result:\nreport\n\nIs this tool result relevant to the goal?',
    })
    expect(result.judgments).toMatchObject([{ callId: ToolCallId('a'), candidateChars: 6, inputChars: 6 }])
    expect(result.messages[2]).toBe(belowMinimum)
  })

  it('records the served model when optional identity fields are absent', async () => {
    const { project } = harness(() => JSON.stringify({ ...validReadout, served_identity: {} }))
    const result = await project([userMessage('goal'), toolMessage('z'.repeat(50), 'a')])
    expect(result.judgments).toMatchObject([{
      disposition: 'elided-irrelevant',
      served: { model: DECISION_ROUTE.model, mode: '', instance: '' },
    }])
  })

  it('bounds the Decision input without splitting a surrogate pair', async () => {
    const { project, calls } = harness(() => readout('no', 0, 1), { maxInputChars: 5 })
    const text = '😀'.repeat(20)
    await project([userMessage('goal'), toolMessage(text, 'a')])
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
    const data = events[0]?.data
    assert.ok(data)
    expect(data.applied).toBe(true)
    expect(data.conversation).toEqual(ROUTE)
    expect(data.judgments).toHaveLength(1)
    const judgment = data.judgments[0]
    assert.ok(judgment)
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
    const { filter, project } = harness(() => readout('no', 0, 1))
    const candidate = toolMessage('z'.repeat(50), 'call-42')
    const result = await project([userMessage('goal'), candidate])
    const projected = result.messages[1] as { role: string; toolCallId: string; content: ContentBlock[] }
    expect(projected.role).toBe('tool')
    expect(projected.toolCallId).toBe('call-42')
    expect(projected.content).toEqual([{ type: 'text', text: filter.config.elisionText }])
  })

  it('uses the most recent user message as the goal', async () => {
    const { project, calls } = harness(() => readout('no', 0, 1))
    await project([
      userMessage('the old goal'),
      toolMessage('z'.repeat(50), 'a'),
      userMessage('the newest goal'),
    ])
    const prompt = (calls[0]?.options.messages[0]?.content[0] as { text: string }).text
    expect(prompt).toContain('the newest goal')
    expect(prompt).not.toContain('the old goal')
  })
})
