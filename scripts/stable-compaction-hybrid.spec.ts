import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { Context } from '@deepseek-ai/cordis'
import Include from '@deepseek-ai/cordis-plugin-include'
import Loader from '@deepseek-ai/cordis-plugin-loader'
import type { Agent } from '@deepseek-ai/dsh-agent'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import * as CommandCompact from '@deepseek-ai/dsh-command-compact'
import Commands from '@deepseek-ai/dsh-commands'
import { isCompactCheckpointSource } from '@deepseek-ai/dsh-compaction'
import HybridCompactionEngine from '@deepseek-ai/dsh-compaction-basic/hybrid'
import type { BasicCompactionConfig } from '@deepseek-ai/dsh-compaction-basic'
import { CONTEXT_WINDOW_EXCEEDED_CODE, createUserMessage, type GenerateOptions } from '@deepseek-ai/dsh-llm'
import { deriveReplayScript } from '@deepseek-ai/dsh-llm-replay'
import { SessionId, type SessionEvent } from '@deepseek-ai/dsh-session'
import * as JsonlSessionPersistence from '@deepseek-ai/dsh-session-persistence-jsonl'
import SqliteSessionQueryEngine from '@deepseek-ai/dsh-session-query-sqlite'
import TokenMeter from '@deepseek-ai/dsh-token-meter'
import { defineContentToolFixture } from '@deepseek-ai/dsh-tool-testkit'
import { expect, it, vi } from 'vitest'
import { MockAdapter, textResponse, toolCallResponse } from '../packages/core/agent-loop/tests/mock-adapter.ts'

class WindowAdapter extends MockAdapter {
  readonly windows = new Map([
    ['wide', 2_000_000], ['summary', 262_144], ['small', 8_192], ['narrow', 4_096],
  ])

  override async resolveModel(provider: string, model: string) {
    return { ...await super.resolveModel(provider, model), context: { contextWindow: this.windows.get(model)! } }
  }
}

const signal = new AbortController().signal
const wide = { provider: 'synthetic-chat', model: 'wide', maxTokens: 64 }
const fault = [{
  type: 'finish', reason: { kind: 'error', failure: { code: 'AUTH', message: 'synthetic rejection' } },
}] as const

async function withContext(
  run: (ctx: Context, root: string) => Promise<void>,
  config: BasicCompactionConfig = {},
) {
  const root = await mkdtemp(join(tmpdir(), 'dsh-hybrid-compaction-'))
  const ctx = new Context()
  try {
    await mountAgentLoopTestDependencies(ctx)
    await ctx.plugin(JsonlSessionPersistence, { root: join(root, 'sessions'), compression: 'zstd' })
    await ctx.plugin(SqliteSessionQueryEngine, { path: ':memory:', openAt: 'never' })
    await ctx.plugin(TokenMeter)
    await ctx.plugin(AgentLoop, { agents: [] })
    await ctx.plugin(Commands)
    await ctx.plugin(Loader)
    ctx.loader.builtins.include = Include
    ctx.loader.builtins['hybrid-fixture'] = HybridCompactionEngine
    ctx.loader.builtins['compact-fixture'] = CommandCompact
    const composition = join(root, 'cordis.yml')
    await writeFile(composition, [
      "- name: 'cordis:hybrid-fixture'",
      `  config: ${JSON.stringify({
        auto: false, maxTokens: 1_024, retainTokens: 0, headroomTokens: 0,
        summarizationProvider: 'synthetic-summary', summarizationModel: 'summary',
        summarizationFallback: 'error', ...config,
      })}`,
      "- name: 'cordis:compact-fixture'",
      '',
    ].join('\n'))
    await ctx.loader.create({ name: 'cordis:include', config: { path: pathToFileURL(composition).href } })
    await ctx.loader.await()
    expect(ctx.compaction).toBeInstanceOf(HybridCompactionEngine)
    await run(ctx, root)
  } finally {
    await ctx.fiber.dispose()
    await rm(root, { recursive: true, force: true })
  }
}

async function send(agent: Agent, text: string) {
  const message = createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } })
  agent.followup(message)
  await agent.whenIdle()
  return message
}

function checkpoints(events: readonly SessionEvent[]) {
  return events.filter(event => event.type === 'user/message' && isCompactCheckpointSource(event.data.source))
}

function requestText(request: GenerateOptions): string {
  return request.messages.flatMap(message => message.content)
    .flatMap(block => block.type === 'text' ? [block.text] : []).join('\n')
}

it('loads Hybrid and runs /compact three times under the selected chunk budget without prior summaries', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 6 }, () => textResponse('Acknowledged.')))
    const summary = new WindowAdapter(Array.from({ length: 20 }, (_, i) => textResponse(`SYNTHETIC_CHECKPOINT_${i}`)))
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-three-rounds')
    const owned = await ctx.agents.create({ sessionId, agentOptions: wide, meta: { cwd: root } })
    let estimatedOriginalInput = 0
    for (const [index, tokens] of [175_000, 175_000, 175_000, 178_679].entries()) {
      const label = `ORIGINAL_${index} `
      estimatedOriginalInput += ctx.tokenMeter.estimateMessage(await send(
        owned.agent, label + 'x'.repeat((tokens - 8) * 4 - label.length),
      ))
    }
    expect(estimatedOriginalInput).toBe(703_679)
    const original = (await ctx.sessionQuery.readSession(sessionId)).events
    const firstResult = await ctx.commands.execute(owned.agent, '/compact', [], signal)
    expect(firstResult?.result).toMatchObject({ kind: 'success', text: expect.stringContaining('Compacted') })
    expect(summary.requests.length).toBeGreaterThan(1)
    for (const word of ['second', 'third']) {
      await send(owned.agent, `Fresh ${word} round context.`.repeat(40))
      expect(await ctx.compaction.compactNow(owned.agent, signal)).not.toBeNull()
    }
    const landed = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(checkpoints(landed)).toHaveLength(3)
    expect(landed.slice(0, original.length)).toEqual(original)
    for (const request of summary.requests) {
      expect(request.maxTokens).toBe(1_024)
      expect(request.messages.reduce((sum, msg) => sum + ctx.tokenMeter.estimateMessage(msg), 0) + 1_024)
        .toBeLessThanOrEqual(262_144)
      expect(requestText(request)).not.toContain('SYNTHETIC_CHECKPOINT_')
    }
    for (const label of ['ORIGINAL_0', 'ORIGINAL_2', 'ORIGINAL_3']) {
      expect(summary.requests.filter(request => requestText(request).includes(label))).toHaveLength(3)
    }
    await owned.dispose()
    expect(ctx.sessions.get(sessionId)).toBeUndefined()
    const reopened = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(reopened).toEqual(landed)
    const replayed = deriveReplayScript([...reopened]).flatMap(entry => entry.kind === 'chunks'
      ? entry.chunks.flatMap(chunk => chunk.type === 'block-end' && chunk.block.type === 'text'
        && chunk.block.text.startsWith('SYNTHETIC_CHECKPOINT_') ? [chunk.block.text] : [])
      : [])
    expect(replayed).toEqual(summary.requests.map((_request, i) => `SYNTHETIC_CHECKPOINT_${i}`))
    const receipts = reopened.flatMap(event => event.type === 'compaction/summary' ? event.data.chunkCalls ?? [] : [])
    expect(receipts.map(call => [call.contextWindow, call.estimatedInputTokens])).toEqual(
      summary.requests.map(request => [262_144, request.messages.reduce((sum, msg) => sum + ctx.tokenMeter.estimateMessage(msg), 0)]),
    )
    const firstOriginal = original.find(event => event.type === 'user/message')!
    expect((await ctx.sessionQuery.traceEvent({ sessionId, seq: firstOriginal.seq })).replacementChain).toHaveLength(3)
  })
})

it('references an oversized result while preserving its tool pair and exact original bytes', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([toolCallResponse('fixture-pair', 'fixture_read', {}), textResponse('Read.')])
    const summary = new WindowAdapter([textResponse('The exact result is available by source reference.')])
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const bytes = 'ORIGINAL_TOOL_OUTPUT\n'.repeat(3_000)
    ctx.tools.register(defineContentToolFixture({
      name: 'fixture_read', description: 'Read a synthetic result.', parameters: {},
      async execute() { return [{ type: 'text', text: bytes }] },
    }))
    const sessionId = SessionId('hybrid-tool-result')
    const owned = await ctx.agents.create({ sessionId, agentOptions: wide, meta: { cwd: root } })
    await send(owned.agent, 'Read the fixture.')
    const before = (await ctx.sessionQuery.readSession(sessionId)).events
    const result = before.find(event => event.type === 'tool/result')!
    expect(await ctx.compaction.compactNow(owned.agent, signal)).not.toBeNull()
    const input = requestText(summary.requests[0]!)
    expect(input).toContain('fixture-pair')
    expect(input).toContain(`Original result retained in Session ${sessionId}, event ${result.seq}.`)
    expect(input).not.toContain('ORIGINAL_TOOL_OUTPUT')
    await owned.dispose()
    expect((await ctx.sessionQuery.readEvent({ sessionId, seq: result.seq })).target).toEqual(result)
    expect(result.data.message.content).toEqual([{ type: 'text', text: bytes }])
  }, { summarizationModel: 'narrow', maxTokens: 64 })
})

it.each([
  ['missing context', {}, 'SUMMARY_TARGET_UNVERIFIED'],
  ['invalid context', { context: { contextWindow: -1 } }, 'SUMMARY_TARGET_UNVERIFIED'],
  ['image-only input', { context: { contextWindow: 4_096 }, inputModalities: ['image'] as const }, 'SUMMARY_TARGET_NOT_CHAT'],
  ['Decision protocol', { context: { contextWindow: 4_096 }, requestProtocol: 'kinf-decision-v1' }, 'SUMMARY_TARGET_NOT_CHAT'],
] as const)('blocks /compact and Model Switch on prepared %s without dispatch or replacement', async (_case, metadata, code) => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 3 }, () => textResponse('Short reply.')))
    const summary = new WindowAdapter([textResponse('Must not dispatch.')])
    ctx.llm.registerAdapter([wide.provider], chat)
    const summaryProvider = 'synthetic-summary'
    ctx.llm.registerAdapter([summaryProvider], summary)
    expect((await ctx.llm.resolveModelInfo(summaryProvider, 'narrow')).context?.contextWindow).toBe(4_096)
    vi.spyOn(summary, 'prepareCall').mockResolvedValue({
      model: { provider: summaryProvider, id: 'narrow', name: 'narrow', ...metadata },
      stream: options => summary.stream(options),
    })

    const sessionId = SessionId('hybrid-unverified-switch')
    const owned = await ctx.agents.create({ sessionId, agentOptions: wide, meta: { cwd: root } })
    const { agent } = owned
    for (const label of ['early', 'middle', 'late']) await send(agent, `${label} `.repeat(1_700))
    const before = (await ctx.sessionQuery.readSession(sessionId)).events
    const surface = [...agent.session.surface.nodes]
    const header = agent.session.requestHeader()
    await expect(ctx.compaction.compactNow(agent, signal)).rejects.toMatchObject({ code: 'summary', cause: { code } })

    agent.model.select({ provider: wide.provider, model: 'small' })
    await send(agent, 'Switch to the lower-capacity model.')
    const after = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(summary.requests).toHaveLength(0)
    expect(chat.requests).toHaveLength(3)
    expect(after.at(-1)).toMatchObject({
      type: 'turn/end', data: { reason: { kind: 'error', error: { code: 'MODEL_SWITCH_COMPACTION_FAILED' } } },
    })
    expect(checkpoints(after)).toHaveLength(0)
    expect(after.some(event => event.type === 'compaction/summary')).toBe(false)
    expect(after.slice(0, before.length)).toEqual(before)
    expect(agent.session.surface.nodes).toEqual(surface)
    expect(agent.session.requestHeader()).toEqual(header)
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.5, summarizationModel: 'narrow', maxTokens: 64 })
})

it.each(['provider rejection', 'changed modality'] as const)('preserves the switch origin after second-chunk %s', async (failure) => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 3 }, () => textResponse('Short reply.')))
    const summary = new WindowAdapter([textResponse('First chunk.'), [...fault]])
    if (failure === 'changed modality') {
      const prepare = summary.prepareCall.bind(summary)
      vi.spyOn(summary, 'prepareCall').mockImplementation(async (...args) => {
        const call = await prepare(...args)
        return summary.requests.length === 0 ? call : { ...call, model: { ...call.model, inputModalities: ['image'] } }
      })
    }
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-failed-switch')
    const owned = await ctx.agents.create({ sessionId, agentOptions: wide, meta: { cwd: root } })
    for (const label of ['early', 'middle', 'late']) await send(owned.agent, `${label} `.repeat(1_700))
    const before = (await ctx.sessionQuery.readSession(sessionId)).events
    const surface = [...owned.agent.session.surface.nodes]
    const header = owned.agent.session.requestHeader()
    owned.agent.model.select({ provider: wide.provider, model: 'small' })
    await send(owned.agent, 'Switch to the lower-capacity model.')
    const after = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(summary.requests).toHaveLength(failure === 'changed modality' ? 1 : 2)
    expect(chat.requests).toHaveLength(3)
    expect(after.at(-1)).toMatchObject({
      type: 'turn/end', data: { reason: { kind: 'error', error: { code: 'MODEL_SWITCH_COMPACTION_FAILED' } } },
    })
    expect(checkpoints(after)).toHaveLength(0)
    expect(after.filter(event => event.type === 'compaction/summary')).toHaveLength(0)
    expect(after.slice(0, before.length)).toEqual(before)
    expect(owned.agent.session.surface.nodes).toEqual(surface)
    expect(owned.agent.session.requestHeader()).toEqual(header)
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.5, summarizationModel: 'narrow', maxTokens: 64 })
})

it('lands a bounded checkpoint and dispatches on the proposed destination after a switch', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 3 }, () => textResponse('Short reply.')))
    const summary = new WindowAdapter([textResponse('Early context.'), textResponse('Recent context.')])
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-successful-switch')
    const owned = await ctx.agents.create({ sessionId, agentOptions: wide, meta: { cwd: root } })
    await send(owned.agent, 'early '.repeat(1_600))
    await send(owned.agent, 'recent '.repeat(1_600))
    owned.agent.model.select({ provider: wide.provider, model: 'small' })
    await send(owned.agent, 'Continue on the smaller route.')
    expect(summary.requests).toHaveLength(2)
    const destination = chat.requests.at(-1)
    expect(destination?.model).toBe('small')
    if (destination === undefined) throw new Error('The destination did not receive a request')
    const requested = destination.messages.reduce((sum, message) => sum + ctx.tokenMeter.estimateMessage(message), 0)
      + ctx.tokenMeter.estimateTools(destination.tools) + (destination.maxTokens ?? 0)
    expect(requested).toBeLessThanOrEqual(8_192)
    expect(requestText(destination)).toContain('Early context.')
    expect(requestText(destination)).toContain('Recent context.')
    const after = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(checkpoints(after)).toHaveLength(1)
    expect(after.at(-1)).toMatchObject({ type: 'turn/end', data: { reason: { kind: 'completed' } } })
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.5, summarizationModel: 'narrow', maxTokens: 64 })
})

it('refuses a switch when an accepted pending message would exceed the destination budget', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([textResponse('Short reply.')])
    const summary = new WindowAdapter([textResponse('Compact historical context.')])
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    ctx.tools.register(defineContentToolFixture({
      name: 'fixture_original', description: 'A declaration on the committed request.', parameters: {},
      async execute() { return [] },
    }))
    const sessionId = SessionId('hybrid-refused-pending-input')
    const owned = await ctx.agents.create({ sessionId, agentOptions: wide, meta: { cwd: root } })
    await send(owned.agent, 'OLD_HISTORY '.repeat(400))
    const { session } = owned.agent
    const header = session.requestHeader()
    const toolHistory = session.toolHistory()
    const surface = [...session.surface.nodes]
    expect(ctx.tokenMeter.measure(session).totalTokens).toBeLessThan(4_096)
    expect(header?.tools?.map(tool => tool.name)).toContain('fixture_original')

    ctx.tools.register(defineContentToolFixture({
      name: 'fixture_pending', description: 'A new declaration for the pending attempt.', parameters: {},
      async execute() { return [] },
    }))
    owned.agent.model.select({ provider: wide.provider, model: 'small' })
    await send(owned.agent, 'PENDING_INPUT '.repeat(3_000))

    expect(chat.requests).toHaveLength(1)
    expect(summary.requests).toHaveLength(1)
    expect(requestText(summary.requests[0]!)).toContain('OLD_HISTORY')
    expect(requestText(summary.requests[0]!)).not.toContain('PENDING_INPUT')
    const { events } = await ctx.sessionQuery.readSession(sessionId)
    expect(events.at(-1)).toMatchObject({
      type: 'turn/end', data: { reason: { kind: 'error', error: { code: 'MODEL_SWITCH_COMPACTION_FAILED' } } },
    })
    expect(events.some(event => event.type === 'compaction/summary')).toBe(false)
    expect(session.surface.nodes).toEqual(surface)
    expect(session.requestHeader()).toEqual(header)
    expect(session.toolHistory()).toEqual(toolHistory)
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.5, summarizationModel: 'narrow', maxTokens: 64 })
})

it('runs the real overflow recovery hook through Hybrid and retries the request', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([
      textResponse('Reply before pressure.'),
      [{ type: 'finish', reason: {
        kind: 'error', failure: { code: CONTEXT_WINDOW_EXCEEDED_CODE, message: 'synthetic context overflow' },
      } }],
      textResponse('Reply after overflow.'),
    ])
    const summary = new WindowAdapter([textResponse('Recovered prior context.')])
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-overflow')
    const owned = await ctx.agents.create({ sessionId, agentOptions: wide, meta: { cwd: root } })
    await send(owned.agent, 'Original context. '.repeat(1_000))
    await send(owned.agent, 'Trigger a scripted provider overflow.')
    const after = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(summary.requests).toHaveLength(1)
    expect(chat.requests).toHaveLength(3)
    expect(checkpoints(after)).toHaveLength(1)
    expect(after.at(-1)).toMatchObject({ type: 'turn/end', data: { reason: { kind: 'completed' } } })
    await owned.dispose()
  }, { auto: true })
})

it('compacts before a step when the current route is under pressure', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([textResponse('First reply.'), textResponse('Second reply.')])
    const summary = new WindowAdapter([textResponse('Retained the original requirement.')])
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-pressure')
    const owned = await ctx.agents.create({ sessionId, agentOptions: { ...wide, model: 'small' }, meta: { cwd: root } })
    await send(owned.agent, 'Original pressure input. '.repeat(850))
    await send(owned.agent, 'Continue after measuring pressure.')
    const after = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(summary.requests).toHaveLength(1)
    expect(chat.requests).toHaveLength(2)
    expect(checkpoints(after)).toHaveLength(1)
    expect(requestText(chat.requests[1]!)).toContain('Retained the original requirement.')
    expect(after.at(-1)).toMatchObject({ type: 'turn/end', data: { reason: { kind: 'completed' } } })
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.5 })
})

it.each(['pressure', 'overflow'] as const)('refuses further dispatch after a %s compaction flush fails', async (trigger) => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([
      textResponse('Initial reply.'),
      [{ type: 'finish', reason: {
        kind: 'error', failure: { code: CONTEXT_WINDOW_EXCEEDED_CODE, message: 'synthetic context overflow' },
      } }],
      textResponse('Must not retry the unconfirmed checkpoint.'),
    ])
    const summary = new WindowAdapter([textResponse('A checkpoint with unconfirmed persistence.')])
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-failed-flush')
    const agentOptions = trigger === 'pressure' ? { ...wide, model: 'small' } : wide
    const owned = await ctx.agents.create({ sessionId, agentOptions, meta: { cwd: root } })
    const { agent } = owned
    await send(agent, 'Original context before the unconfirmed write. '.repeat(500))
    const before = (await ctx.sessionQuery.readSession(sessionId)).events
    const flush = vi.spyOn(ctx.sessions, 'flush').mockRejectedValueOnce(new Error('synthetic journal flush failure'))
    try {
      await send(agent, 'Continue from the original context.')
      expect(flush).toHaveBeenCalledWith(agent.session)
    } finally {
      flush.mockRestore()
    }
    const after = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(summary.requests).toHaveLength(1)
    expect(chat.requests).toHaveLength(trigger === 'pressure' ? 1 : 2)
    expect(after.slice(0, before.length)).toEqual(before)
    expect(checkpoints(after)).toHaveLength(1)
    expect(after.filter(event => event.type === 'compaction/end')).toHaveLength(1)
    expect(after.at(-1)).toMatchObject({
      type: 'turn/end',
      data: { reason: {
        kind: 'error',
        error: trigger === 'pressure'
          ? { message: expect.stringContaining('synthetic journal flush failure') }
          : { code: CONTEXT_WINDOW_EXCEEDED_CODE, message: 'synthetic context overflow' },
      } },
    })
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.5 })
})

it('uses the selected explicit region and preserves an agent-injected original', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([textResponse('First reply.'), textResponse('Retained reply.')])
    const summary = new WindowAdapter([textResponse('Selected region summary.')])
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-explicit-region')
    const owned = await ctx.agents.create({ sessionId, agentOptions: wide, meta: { cwd: root } })
    const { agent } = owned
    await send(agent, 'Selected raw history. '.repeat(400))
    const injected = agent.session.append('user/message', createUserMessage({
      source: { kind: 'runtime-context' },
      content: [{ type: 'text', text: 'INJECTED_ORIGINAL_MARKER' }],
    }), { surfaceOp: 'append' })
    await send(agent, 'RETAINED_UNSELECTED_TAIL')
    const before = (await ctx.sessionQuery.readSession(sessionId)).events
    const start = before.find(event => event.type === 'user/message')!
    agent.session.append('turn/start', { turn: 3 })
    const result = await ctx.compaction.compactRegion(start.seq, injected.seq, agent, signal)
    agent.session.append('turn/end', { turn: 3, reason: { kind: 'completed' } })
    expect(result.shadowedSeqs).toContain(injected.seq)
    expect(requestText(summary.requests[0]!)).toContain('INJECTED_ORIGINAL_MARKER')
    expect(requestText(summary.requests[0]!)).not.toContain('RETAINED_UNSELECTED_TAIL')
    expect(agent.session.deriveMessages().some(message => message.content.some(block =>
      block.type === 'text' && block.text === 'RETAINED_UNSELECTED_TAIL',
    ))).toBe(true)
    await owned.dispose()
  })
})

it.each(['cancellation', 'mutation'] as const)('refuses a checkpoint after %s during summarization', async (condition) => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([textResponse('A retained reply.')])
    const sessionId = SessionId('hybrid-inflight-refusal')
    const abort = new AbortController()
    ctx.llm.registerAdapter([wide.provider], chat)
    const owned = await ctx.agents.create({ sessionId, agentOptions: wide, meta: { cwd: root } })
    const { agent } = owned
    let mutation: SessionEvent | undefined
    const summary = new WindowAdapter([() => {
      if (condition === 'cancellation') abort.abort(new Error('Synthetic cancellation'))
      else {
        mutation = agent.session.append('user/message', createUserMessage({
          source: { kind: 'user' }, content: [{ type: 'text', text: 'Concurrent original.' }],
        }), { surfaceOp: 'append' })
      }
      return textResponse('This result cannot be committed.')
    }])
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    await send(agent, 'Original context before the attempt. '.repeat(200))
    const before = (await ctx.sessionQuery.readSession(sessionId)).events
    await expect(ctx.compaction.compactNow(agent, abort.signal)).rejects.toThrow()
    const after = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(summary.requests).toHaveLength(1)
    expect(after.slice(0, before.length)).toEqual(before)
    expect(checkpoints(after)).toHaveLength(0)
    expect(after.filter(event => event.type === 'compaction/summary')).toHaveLength(0)
    expect(after.at(-1)).toMatchObject({ type: 'compaction/end', data: { error: expect.any(String) } })
    if (condition === 'mutation') expect(after).toContainEqual(mutation)
    await owned.dispose()
  })
})
