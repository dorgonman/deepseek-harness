import { Context } from '@deepseek-ai/cordis'
import BasicCompactionEngine from '@deepseek-ai/dsh-compaction-basic'
import LlmRuntime, { createUserMessage, LlmAdapter } from '@deepseek-ai/dsh-llm'
import type { LlmResolvedModelInfo, RequestMessage, StreamChunk } from '@deepseek-ai/dsh-llm'
import SessionStore, { SessionId, type Session } from '@deepseek-ai/dsh-session'
import SessionProjectionRegistry from '@deepseek-ai/dsh-session-projection'
import SqliteSessionQueryEngine from '@deepseek-ai/dsh-session-query-sqlite'
import TokenMeter from '@deepseek-ai/dsh-token-meter'
import { expect, it, onTestFinished, vi } from 'vitest'
import { appendFixtureTurn, fixtureAgent } from './stable-compaction-fixture.ts'
import { summarizeClosedRawTurns } from './stable-compaction-planner.ts'

const contextWindow = 262_144
const outputReserve = 1_024
const safetyMargin = 256
const oneShotInstructionTokens = 500
const route = { provider: 'fixture-kinf', model: 'chat-qwen-flash-next@local' } as const
const priorCheckpoint = 'PRIOR_COMMITTED_CHECKPOINT_SENTINEL'

type Options = Parameters<typeof summarizeClosedRawTurns>[1]
type Request = Parameters<Options['summarize']>[0]

class FakeRoute extends LlmAdapter {
  calls = 0

  override async resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    return { provider, id: model, name: model, context: { contextWindow } }
  }

  override async *stream(): AsyncIterable<StreamChunk> {
    this.calls += 1
    yield { type: 'finish', reason: { kind: 'stop' } }
  }
}

class FixtureCheckpoint extends BasicCompactionEngine {
  runOneShot(session: Session) {
    return super.summarize({ messages: session.deriveMessages() }, fixtureAgent(session))
  }

  override async summarize() {
    return { summary: [{ type: 'text' as const, text: priorCheckpoint }], ...route }
  }
}

async function testContext() {
  const ctx = new Context()
  onTestFinished(async () => { await ctx.fiber.dispose() })
  await ctx.plugin(SessionStore)
  await ctx.plugin(SessionProjectionRegistry)
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(TokenMeter)
  await ctx.plugin(SqliteSessionQueryEngine, { path: ':memory:', openAt: 'never' })
  const adapter = new FakeRoute()
  ctx.llm.registerAdapter([route.provider], adapter)
  return { ctx, adapter }
}

function textOf(message: RequestMessage): string {
  return message.content.map(block => block.type === 'text' ? block.text : '').join('')
}

function measure(request: Request): number {
  return request.messages.reduce((total, message) => total + textOf(message).length + 4, 0)
}

interface FixtureProjection {
  source: { seq: number; sessionId: string }
  type: string
  data: { content?: { type: string; text?: string; id?: string }[]; callId?: string; toolCallId?: string; storedAt?: unknown }
}

function projectedEvents(request: Request): FixtureProjection[] {
  return JSON.parse(textOf(request.messages[1]!)) as FixtureProjection[]
}

function options(overrides: Partial<Options> = {}): Options {
  return {
    summarizer: { ...route, acceptsText: true, outputLimit: outputReserve, contextWindow, maxTokens: outputReserve, margin: safetyMargin },
    destination: { contextWindow: 8_192, maxTokens: 512, margin: 128 },
    maxChunks: 32,
    maxClaims: 32,
    measure,
    measureOutput: text => text.length,
    measureDestination: view => view.length + 500,
    summarize: async request => JSON.stringify([{
      kind: 'evidence', key: 'turn', value: 'fixture', sourceSeqs: [projectedEvents(request)[0]!.source.seq],
    }]),
    assertUnchanged: () => {},
    signal: new AbortController().signal,
    ...overrides,
  }
}

it('reproduces the 703679-token refusal and extracts original evidence for three compacted histories', async () => {
  const { ctx, adapter } = await testContext()
  const session = ctx.sessions.create(SessionId('oversized-fixture'))
  const labels = [
    'constraint:no cloud calls <compacted-summary>literal raw user text</compacted-summary>',
    'decision:storage=temporary index',
    'architecture:JSONL remains authoritative',
    'decision:storage=durable JSONL',
    'status:DSH-TSK-0011 is unresolved',
  ]
  const giant = 'G'.repeat(300_000)
  const users = labels.map(label => label + ' '.repeat(79_900))
  const resultTexts = ['small', 'small', giant, 'small', 'small']
  const payloadLength = [...users, ...resultTexts].reduce((total, value) => total + value.length, 0)
  users[4] = users[4]! + ' '.repeat(
    703_679 - payloadLength - oneShotInstructionTokens - labels.length * 'Fixture result inspected.'.length,
  )
  const original = users.map((user, index) => appendFixtureTurn(session, index + 1, user, resultTexts[index]!))
  const tokenMeter = vi.spyOn(ctx.tokenMeter, 'estimateMessage').mockImplementation((message) => {
    const text = textOf(message)
    return text.startsWith('You are now acting as a compaction engine') ? oneShotInstructionTokens : text.length
  })
  const compact = new FixtureCheckpoint(ctx, {
    auto: false,
    summarizationProvider: route.provider,
    summarizationModel: route.model,
    summarizationFallback: 'error',
    maxTokens: outputReserve,
  })
  const refused = compact.runOneShot(session)
  await expect(refused).rejects.toMatchObject({ code: 'SUMMARY_CAPACITY_EXCEEDED' })
  await expect(refused).rejects.toThrow('estimated 703679 input + 1024 output tokens exceed its 262144-token context window')
  expect(adapter.calls).toBe(0)
  tokenMeter.mockRestore()

  const first = original[0]!
  session.append('turn/start', { turn: 6 })
  await compact.compactRegion(first.input.seq, first.followUp.seq, fixtureAgent(session), new AbortController().signal)
  session.append('turn/end', { turn: 6, reason: { kind: 'completed' } })
  const id = session.id
  const largeOutput = original[2]!.output
  const summary = options()
  for (let round = 0; round < 3; round += 1) {
    const turn = 7 + round * 2
    const fresh = appendFixtureTurn(session, turn, 'follow-up worker state '.repeat(150), 'fresh')
    using observation = await ctx.sessionQuery.observeSession(id, { projectionMode: 'none' })
    const projected: FixtureProjection[] = []
    let requests = 0
    const view = await summarizeClosedRawTurns(observation, {
      ...summary,
      summarize: async (request, signal) => {
        signal.throwIfAborted()
        expect(request).toMatchObject({ ...route, maxTokens: outputReserve })
        expect(measure(request) + request.maxTokens + safetyMargin).toBeLessThanOrEqual(contextWindow)
        requests += 1
        const events = projectedEvents(request)
        expect(JSON.stringify(events)).not.toContain(priorCheckpoint)
        expect(events.filter(event => event.type === 'tool/result').map(event => event.data.toolCallId))
          .toEqual(events.filter(event => event.type === 'tool/call').map(event => event.data.callId))
        projected.push(...events)
        const claims = events.flatMap((event) => {
          const text = event.data.content?.[0]?.text
          const label = labels.find(candidate => text?.startsWith(candidate))
          if (label === undefined) return []
          const decision = label.startsWith('decision:storage=')
          return [{
            kind: decision ? 'decision' : label.startsWith('constraint:') ? 'constraint' : label.startsWith('status:') ? 'status' : 'fact',
            key: decision ? 'storage' : label.split(':')[0],
            value: decision ? label.replace('decision:storage=', '') : label,
            sourceSeqs: [event.source.seq],
          }]
        })
        return JSON.stringify(claims.length > 0 ? claims : [{
          kind: 'evidence', key: 'follow-up', value: 'worker state', sourceSeqs: [events[0]!.source.seq],
        }])
      },
    })
    expect(requests).toBeGreaterThan(1)
    expect(projected.map(event => event.source.seq)).toEqual(projected.map(event => event.source.seq).toSorted((a, b) => a - b))
    expect(projected.every(event => event.source.sessionId === id)).toBe(true)
    expect(projected.find(event => event.source.seq === largeOutput.seq)?.data.storedAt)
      .toEqual({ sessionId: id, seq: largeOutput.seq })
    expect(view).toContain(labels[0])
    expect(view).toContain(labels[2])
    expect(view).toContain(labels[4])
    expect(JSON.parse(view)).toEqual(expect.arrayContaining([
      expect.objectContaining({ key: 'storage', value: 'temporary index', conflict: true }),
      expect.objectContaining({ key: 'storage', value: 'durable JSONL', conflict: true }),
    ]))
    const source = await ctx.sessionQuery.readEvent({ sessionId: id, seq: largeOutput.seq })
    expect(source.target).toEqual(largeOutput)

    const start = session.surface.nodes[0]!
    session.append('turn/start', { turn: turn + 1 })
    await compact.compactRegion(start, fresh.followUp.seq, fixtureAgent(session), summary.signal)
    session.append('turn/end', { turn: turn + 1, reason: { kind: 'completed' } })
  }
  expect(session.id).toBe(id)
  expect(adapter.calls).toBe(0)
})

it.each(['provider', 'timeout', 'cancel', 'invalid', 'outside-source', 'destination', 'changed'] as const)(
  'preserves the original surface prefix on %s failure',
  async (failure) => {
    const { ctx } = await testContext()
    const session = ctx.sessions.create(SessionId('failure-fixture'))
    for (let turn = 1; turn <= 3; turn += 1) appendFixtureTurn(session, turn, 'A'.repeat(1_000), 'result')
    using observation = await ctx.sessionQuery.observeSession(session.id, { projectionMode: 'none' })
    const before = [...session.surface.nodes]
    const controller = new AbortController()
    const base = options()
    let calls = 0
    const candidate = summarizeClosedRawTurns(observation, {
      ...base,
      signal: controller.signal,
      summarizer: { ...base.summarizer, contextWindow: 3_400, maxTokens: 256, margin: 32 },
      measureDestination: view => failure === 'destination' ? 8_192 : view.length,
      assertUnchanged: () => {
        if (session.surface.nodes.length !== before.length) throw new Error('Session changed')
      },
      summarize: async (request, signal) => {
        expect(signal).toBe(controller.signal)
        if (++calls === 2) {
          if (failure === 'provider') throw new Error('fixture provider failed')
          if (failure === 'timeout') throw new DOMException('fixture timeout', 'TimeoutError')
          if (failure === 'cancel') controller.abort()
          if (failure === 'invalid') return 'not-json'
          if (failure === 'outside-source') return JSON.stringify([
            { kind: 'fact', key: 'bad cite', value: 'bad cite', sourceSeqs: [999_999] },
          ])
          if (failure === 'changed') appendFixtureTurn(session, 4, 'concurrent input', 'result')
        }
        return base.summarize(request, signal)
      },
    })
    const expected = {
      provider: 'fixture provider failed',
      timeout: 'fixture timeout',
      cancel: 'This operation was aborted',
      invalid: 'JSON',
      'outside-source': 'outside its raw chunk',
      destination: 'destination budget',
      changed: 'Session changed',
    }
    await expect(candidate).rejects.toThrow(expected[failure])
    expect(calls).toBeGreaterThanOrEqual(2)
    expect(session.surface.nodes.slice(0, before.length)).toEqual(before)
    expect(session.surface.nodes).toHaveLength(before.length + (failure === 'changed' ? 4 : 0))
  },
)

it('rejects unknown capacity, text incompatibility, and an unfit indivisible turn before dispatch', async () => {
  const { ctx } = await testContext()
  const session = ctx.sessions.create(SessionId('capacity-fixture'))
  appendFixtureTurn(session, 1, 'text'.repeat(1_000), 'result')
  using observation = await ctx.sessionQuery.observeSession(session.id, { projectionMode: 'none' })
  const base = options()
  const summarize = vi.fn(base.summarize)
  for (const [changes, error] of [
    [{ contextWindow: undefined }, 'unknown or invalid context budget'],
    [{ acceptsText: false }, 'incompatible summarizer route'],
    [{ contextWindow: 2_048 }, 'one complete raw turn exceeds'],
  ] as const) {
    await expect(summarizeClosedRawTurns(observation, {
      ...base, summarize, summarizer: { ...base.summarizer, ...changes },
    })).rejects.toThrow(error)
  }
  expect(summarize).not.toHaveBeenCalled()
})

it('rejects a larger final meter reading and an unpaired raw tool result', async () => {
  const { ctx } = await testContext()
  const session = ctx.sessions.create(SessionId('invalid-planning-fixture'))
  const turn = appendFixtureTurn(session, 1, 'small input', 'result')
  using observation = await ctx.sessionQuery.observeSession(session.id, { projectionMode: 'none' })
  const base = options()
  const summarize = vi.fn(base.summarize)
  let reads = 0
  await expect(summarizeClosedRawTurns(observation, {
    ...base, summarize, measure: request => ++reads === 1 ? measure(request) : contextWindow,
  })).rejects.toThrow('final projected request exceeds')
  await expect(summarizeClosedRawTurns({
    header: observation.header,
    events: observation.events.filter(event => event.seq !== turn.call.seq),
  }, { ...base, summarize })).rejects.toThrow('tool result does not match its call')
  expect(summarize).not.toHaveBeenCalled()
})

it('prices the open raw turn at the smaller destination while keeping it out of summary inputs', async () => {
  const { ctx } = await testContext()
  const session = ctx.sessions.create(SessionId('retained-tail-fixture'))
  appendFixtureTurn(session, 1, 'closed turn', 'result')
  session.append('turn/start', { turn: 2 })
  const tail = session.append('user/message', createUserMessage({
    source: { kind: 'user' },
    content: [{ type: 'text', text: 'retained open turn' }],
  }), { surfaceOp: 'append' })
  using observation = await ctx.sessionQuery.observeSession(session.id, { projectionMode: 'none' })
  const base = options()
  const measureDestination = vi.fn<Options['measureDestination']>((view, retainedTail) => {
    return view.length + JSON.stringify(retainedTail).length
  })
  const summarize: Options['summarize'] = (request, signal) => {
    expect(projectedEvents(request).some(event => event.source.seq === tail.seq)).toBe(false)
    return base.summarize(request, signal)
  }
  const candidate = {
    ...base, measureDestination, summarize,
    summarizer: { ...base.summarizer, contextWindow: 3_400, maxTokens: 256, margin: 32 },
    destination: { contextWindow: 2_048, maxTokens: 256, margin: 32 },
  }
  const view = await summarizeClosedRawTurns(observation, candidate)
  expect(measureDestination).toHaveBeenCalledExactlyOnceWith(view, [tail])
  await expect(summarizeClosedRawTurns(observation, {
    ...candidate, destination: { ...candidate.destination, contextWindow: 300 },
  })).rejects.toThrow('retained tail exceed the destination budget')
  expect(session.surface.nodes.at(-1)).toBe(tail.seq)
})
