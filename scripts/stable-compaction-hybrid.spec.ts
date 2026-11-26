import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { Context } from '@deepseek-ai/cordis'
import { AttachmentId } from '@deepseek-ai/dsh-attachment'
import Group from '@deepseek-ai/cordis-plugin-group'
import Include from '@deepseek-ai/cordis-plugin-include'
import Loader from '@deepseek-ai/cordis-plugin-loader'
import type { Agent } from '@deepseek-ai/dsh-agent'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import AgentPreset from '@deepseek-ai/dsh-agent-preset'
import AgentPresetRegistry from '@deepseek-ai/dsh-agent-preset-registry'
import * as CommandCompact from '@deepseek-ai/dsh-command-compact'
import Commands from '@deepseek-ai/dsh-commands'
import { CompactionId, compactCheckpointSource, isCompactCheckpointSource } from '@deepseek-ai/dsh-compaction'
import BasicCompactionEngine from '@deepseek-ai/dsh-compaction-basic'
import HybridCompactionEngine from '@deepseek-ai/dsh-compaction-basic/hybrid'
import type { BasicCompactionConfig } from '@deepseek-ai/dsh-compaction-basic'
import {
  CONTEXT_WINDOW_EXCEEDED_CODE, createUserMessage, textOnlyImageText,
  type GenerateOptions, type LlmImageRequestPricing, type ModelModality, type StreamChunk,
} from '@deepseek-ai/dsh-llm'
import * as LlmPiAi from '@deepseek-ai/dsh-llm-pi-ai'
import { deriveReplayScript } from '@deepseek-ai/dsh-llm-replay'
import { SessionId, type SessionEvent } from '@deepseek-ai/dsh-session'
import * as JsonlSessionPersistence from '@deepseek-ai/dsh-session-persistence-jsonl'
import SqliteSessionQueryEngine from '@deepseek-ai/dsh-session-query-sqlite'
import ToolResultPruner from '@deepseek-ai/dsh-compaction-tool-result-pruner'
import TokenMeter from '@deepseek-ai/dsh-token-meter'
import { defineContentToolFixture } from '@deepseek-ai/dsh-tool-testkit'
import { expect, it, onTestFinished, vi } from 'vitest'
import { advanceCoverage, deriveCommittedCoverage, digestCanonical } from '../packages/compaction/compaction-basic/src/incremental.ts'
import { validateLedger } from '../packages/compaction/compaction-basic/src/fact-ledger.ts'
import { createKinfResidentObservationSource } from '../packages/compaction/compaction-basic/src/kinf-resident-observation.ts'
import type { ResidentObservationRequest, ResidentObservationSource } from '../packages/compaction/compaction-basic/src/resident-observation.ts'
import {
  RESIDENT_ONLY_V1, type Candidate, type ResidentPrecondition,
} from '../packages/compaction/compaction-basic/src/resident-selector.ts'
import { MockAdapter, textResponse, toolCallResponse } from '../packages/core/agent-loop/tests/mock-adapter.ts'
import { closeMockServers, mockServer } from '../packages/llm/llm-pi-ai/tests/mock-server.ts'

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

type CompactionOverrides = { [Key in keyof BasicCompactionConfig]?: BasicCompactionConfig[Key] | undefined }

async function mountCompactionHarness(
  ctx: Context, root: string, extraBuiltins: Record<string, unknown> = {},
) {
  await mountAgentLoopTestDependencies(ctx)
  const persistence = await ctx.plugin(JsonlSessionPersistence, { root: join(root, 'sessions'), compression: 'zstd' })
  await ctx.plugin(SqliteSessionQueryEngine, { path: ':memory:', openAt: 'never' })
  await ctx.plugin(TokenMeter)
  await ctx.plugin(AgentLoop, { agents: [] })
  await ctx.plugin(Commands)
  await ctx.plugin(Loader)
  ctx.loader.builtins.include = Include
  ctx.loader.builtins['hybrid-fixture'] = HybridCompactionEngine
  Object.assign(ctx.loader.builtins, extraBuiltins)
  ctx.loader.builtins['compact-fixture'] = CommandCompact
  return persistence.dispose
}

async function loadCompactionComposition(
  ctx: Context, root: string, file: string, engine: 'hybrid' | 'basic', config: CompactionOverrides,
) {
  const composition = join(root, file)
  await writeFile(composition, [
    `- name: 'cordis:${engine}-fixture'`,
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
}

async function mountStandardPreset(ctx: Context, config: CompactionOverrides = {}) {
  ctx.loader.builtins.group = Group
  await ctx.plugin(AgentPresetRegistry, { default: 'standard' })
  await ctx.plugin(AgentPreset, {
    id: 'standard',
    plugins: [{
      id: 'compaction', name: 'cordis:group', group: true,
      isolate: { compaction: true },
      config: [{
        id: 'compaction-basic', name: 'cordis:hybrid-fixture',
        config: {
          auto: true, maxTokens: 1_024, retainTokens: 0, headroomTokens: 0,
          summarizationProvider: 'synthetic-summary', summarizationModel: 'summary',
          summarizationFallback: 'error', ...config,
        },
      }],
    }],
  })
}

async function withContext(
  run: (ctx: Context, root: string, disposePersistence: () => Promise<void>) => Promise<void>,
  config: CompactionOverrides = {},
) {
  const root = await mkdtemp(join(tmpdir(), 'dsh-hybrid-compaction-'))
  const ctx = new Context()
  try {
    const disposePersistence = await mountCompactionHarness(ctx, root)
    await loadCompactionComposition(ctx, root, 'cordis.yml', 'hybrid', config)
    expect(ctx.compaction).toBeInstanceOf(HybridCompactionEngine)
    await run(ctx, root, disposePersistence)
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

function citedSeqs(text: string): number[] {
  const cited = /source events: ([\d,\s]+)\./.exec(text)?.[1]?.trim() ?? ''
  return cited.split(',').map(value => Number(value.trim())).filter(value => Number.isSafeInteger(value))
}

function factLedgerAdapter(key: string, claim: string) {
  const respond = (options: GenerateOptions) => textResponse(JSON.stringify([{
    kind: 'observation', key, claim, sourceSeqs: [citedSeqs(requestText(options))[0] ?? 1], basis: 'asserted',
  }]))
  return new WindowAdapter(Array.from({ length: 8 }, () => respond))
}

function expectCompletedTurn(events: readonly SessionEvent[]) {
  expect(events.findLast(event => event.type === 'turn/end')?.data.reason.kind).toBe('completed')
}

function expectAdmissionRefusal(events: readonly SessionEvent[]) {
  expect(events.findLast(event => event.type === 'turn/end')?.data.reason).toMatchObject({
    kind: 'error',
    error: { code: 'CONTEXT_ADMISSION_REFUSED' },
  })
}

function watchRequestErrors(agent: Agent) {
  const errors = vi.fn()
  const dispose = agent.ctx.on('agent/request-error', (_event, next) => {
    errors()
    return next()
  })
  return { errors, dispose }
}

function expectPendingInput(agent: Agent, marker: string) {
  expect(agent.inbox.nextStep.some(message => message.content.some(block => block.type === 'text'
    && block.text.includes(marker)))).toBe(true)
}

async function expectCompactedTurn(ctx: Context, sessionId: SessionId, chat: MockAdapter, providerRequests: number) {
  const events = (await ctx.sessionQuery.readSession(sessionId)).events
  expectCompletedTurn(events)
  expect(checkpoints(events)).toHaveLength(1)
  expect(chat.requests).toHaveLength(providerRequests)
  return events
}

it.each([
  ['implicit', undefined],
  ['explicit', 960],
] as const)('uses the %s 960-token same-route summary cap under pressure', async (mode, maxTokens) => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([
      textResponse('Original reply.'),
      request => textResponse(request.purpose === 'compaction' ? 'Smaller summary.' : 'Ordinary continuation.'),
      textResponse('Ordinary continuation.'),
    ])
    ctx.llm.registerAdapter([wide.provider], chat)
    const sessionId = SessionId(`hybrid-pressure-cap-${mode}`)
    const agentOptions = { ...wide, model: 'small', maxTokens: 512 }
    const owned = await ctx.agents.create({ sessionId, agentOptions, meta: { cwd: root } })
    await send(owned.agent, 'SAME_ROUTE_ORIGINAL '.repeat(480))
    await send(owned.agent, 'Continue on the same 8,192-token route.')

    const summaryCalls = chat.requests.filter(request => request.purpose === 'compaction')
    expect(summaryCalls.map(request => request.maxTokens)).toEqual([960])
    const { events } = await ctx.sessionQuery.readSession(sessionId)
    expect(checkpoints(events)).toHaveLength(1)
  }, {
    auto: true, thresholdRatio: 0.1, headroomTokens: undefined, maxTokens,
    summarizationProvider: '', summarizationModel: '', summarizationFallback: 'latest-route',
  })
})

it('refuses an adaptive summary when the source completion reserve is unknown', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([textResponse('Original reply.')])
    ctx.llm.registerAdapter([wide.provider], chat)
    const sessionId = SessionId('hybrid-unknown-source-reserve')
    const agentOptions = { provider: wide.provider, model: wide.model }
    const owned = await ctx.agents.create({ sessionId, agentOptions, meta: { cwd: root } })
    await send(owned.agent, 'History with no known completion reserve. '.repeat(200))

    await expect(ctx.compaction.compactNow(owned.agent, signal)).rejects.toThrow()
    const { events } = await ctx.sessionQuery.readSession(sessionId)
    expect(events.findLast(event => event.type === 'compaction/end')?.data.error)
      .toContain('source completion reserve is unavailable')
    expect(chat.requests.filter(request => request.purpose === 'compaction')).toEqual([])
    expect(checkpoints(events)).toHaveLength(0)
  }, {
    headroomTokens: undefined, maxTokens: undefined,
    summarizationProvider: '', summarizationModel: '', summarizationFallback: 'latest-route',
  })
})

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
    expect(firstResult?.result.kind).toBe('success')
    expect(firstResult?.result.text).toContain('Compacted')
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

async function expectFittingSwitch(
  ctx: Context,
  sessionId: SessionId,
  adapters: { chat: WindowAdapter; summary: WindowAdapter },
  destination: string,
  retainedText: string,
) {
  const request = adapters.chat.requests.at(-1)
  expect(request?.model).toBe(destination)
  if (request === undefined) throw new Error('The destination did not receive a request')
  expect(requestText(request)).toContain(retainedText)
  expect(adapters.summary.requests).toHaveLength(0)
  const { events } = await ctx.sessionQuery.readSession(sessionId)
  expect(events.some(event => event.type === 'compaction/start')).toBe(false)
  expectCompletedTurn(events)
}

it.each([
  ['destination pressure', 'wide', 'small'],
  ['source pressure', 'small', 'wide'],
  ['destination transport fallback', 'wide', 'small'],
] as const)('switches without compacting when the request fits despite %s', async (condition, source, destination) => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([textResponse('Source reply.'), textResponse('Destination reply.')])
    const summary = new WindowAdapter([textResponse('Unnecessary summary.')])
    const fallback = condition === 'destination transport fallback'
    if (fallback) {
      const resolve = chat.resolveModel.bind(chat)
      vi.spyOn(chat, 'resolveModel').mockImplementation(async (provider, model) => {
        const info = await resolve(provider, model)
        return model === destination
          ? { ...info, transportMaxTokens: info.context.contextWindow }
          : { ...info, defaultMaxTokens: 64 }
      })
    }
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-fitting-switch')
    const owned = await ctx.agents.create({
      sessionId,
      agentOptions: { provider: wide.provider, model: source, ...fallback ? {} : { maxTokens: 64 } },
      meta: { cwd: root },
    })
    await send(owned.agent, 'ORIGINAL_HISTORY '.repeat(1_100))
    expect(ctx.tokenMeter.measure(owned.agent.session).totalTokens).toBeGreaterThan(4_096)

    owned.agent.model.select({ provider: wide.provider, model: destination })
    await send(owned.agent, 'Continue on the destination route.')

    await expectFittingSwitch(ctx, sessionId, { chat, summary }, destination, 'ORIGINAL_HISTORY')
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.5 })
})

it('switches without compacting when the text-only destination projects the image to text', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([textResponse('Source reply.'), textResponse('Destination reply.')])
    const summary = new WindowAdapter([textResponse('Unnecessary summary.')])
    const resolve = chat.resolveModel.bind(chat)
    vi.spyOn(chat, 'resolveModel').mockImplementation(async (provider, model) => {
      const info = await resolve(provider, model)
      return model === 'narrow' ? { ...info, inputModalities: ['text'] } : info
    })
    const visualTokens = 5_000
    vi.spyOn(chat, 'imageRequestPricing').mockImplementation((_provider, model) => model === 'narrow' ? {
      priceImages: images => images.map(() => ({ visualTokens, text: 'Detailed image pricing.' })),
    } : undefined)
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-fitting-text-only-switch')
    const owned = await ctx.agents.create({ sessionId, agentOptions: wide, meta: { cwd: root } })
    owned.agent.followup(attachmentAdmissionInput())
    await owned.agent.whenIdle()
    expect(chat.requests[0]?.messages.some(message => message.content.some(block => block.type === 'image'))).toBe(true)

    owned.agent.model.select({ provider: wide.provider, model: 'narrow' })
    await send(owned.agent, 'Continue on the text-only route.')

    await expectFittingSwitch(ctx, sessionId, { chat, summary }, 'narrow', textOnlyImageText(admissionImage))
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.5 })
})

it.each([
  ['missing context', {}, 'SUMMARY_TARGET_UNVERIFIED'],
  ['invalid context', { context: { contextWindow: -1 } }, 'SUMMARY_TARGET_UNVERIFIED'],
  ['image-only input', { context: { contextWindow: 4_096 }, inputModalities: ['image'] as const }, 'SUMMARY_TARGET_NOT_CHAT'],
  ['Decision protocol', { context: { contextWindow: 4_096 }, requestProtocol: 'kinf-decision-v1' }, 'SUMMARY_TARGET_NOT_CHAT'],
] as const)('blocks /compact and Model Switch on prepared %s without dispatch or replacement', async (_case, metadata, code) => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 3 }, () => textResponse('Short reply.')))
    chat.windows.set('small', 7_168)
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

it.each(['provider rejection', 'changed modality', 'changed capacity'] as const)('preserves the switch origin after second-chunk %s', async (failure) => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 3 }, () => textResponse('Short reply.')))
    chat.windows.set('small', 7_168)
    const summary = new WindowAdapter([textResponse('First chunk.'), [...fault]])
    if (failure !== 'provider rejection') {
      const prepare = summary.prepareCall.bind(summary)
      const changedMetadata = failure === 'changed modality'
        ? { inputModalities: ['image'] as const }
        : { context: { contextWindow: 8_192 } }
      vi.spyOn(summary, 'prepareCall').mockImplementation(async (...args) => {
        const call = await prepare(...args)
        return summary.requests.length === 0 ? call : { ...call, model: { ...call.model, ...changedMetadata } }
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
    expect(summary.requests).toHaveLength(failure === 'provider rejection' ? 2 : 1)
    expect(chat.requests).toHaveLength(3)
    expect(after.at(-1)).toMatchObject({
      type: 'turn/end', data: { reason: { kind: 'error', error: { code: 'MODEL_SWITCH_COMPACTION_FAILED' } } },
    })
    if (failure === 'changed capacity') {
      expect(after.findLast(event => event.type === 'compaction/end')?.data.error)
        .toBe('compaction: summarizer capacity changed')
    }
    expect(checkpoints(after)).toHaveLength(0)
    expect(after.filter(event => event.type === 'compaction/summary')).toHaveLength(0)
    expect(after.slice(0, before.length)).toEqual(before)
    expect(owned.agent.session.surface.nodes).toEqual(surface)
    expect(owned.agent.session.requestHeader()).toEqual(header)
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.5, summarizationModel: 'narrow', maxTokens: 64 })
})

it('lands a bounded checkpoint on switch when the completion reserve exceeds the remaining window', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 3 }, () => textResponse('Short reply.')))
    const destinationWindow = 5_300
    chat.windows.set('small', destinationWindow)
    const summary = new WindowAdapter([textResponse('Early context.'), textResponse('Recent context.')])
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-successful-switch')
    const owned = await ctx.agents.create({ sessionId, agentOptions: wide, meta: { cwd: root } })
    await send(owned.agent, 'early '.repeat(1_600))
    await send(owned.agent, 'recent '.repeat(1_600))
    const pending = 'Continue on the smaller route.'
    const pendingMessage = createUserMessage({ content: [{ type: 'text', text: pending }], source: { kind: 'user' } })
    const input = ctx.tokenMeter.measure(owned.agent.session).totalTokens + ctx.tokenMeter.estimateMessage(pendingMessage)
    expect(input).toBeLessThanOrEqual(destinationWindow)
    expect(input + wide.maxTokens).toBeGreaterThan(destinationWindow)
    owned.agent.model.select({ provider: wide.provider, model: 'small' })
    await send(owned.agent, pending)
    expect(summary.requests).toHaveLength(2)
    const destination = chat.requests.at(-1)
    expect(destination?.model).toBe('small')
    if (destination === undefined) throw new Error('The destination did not receive a request')
    const requested = destination.messages.reduce((sum, message) => sum + ctx.tokenMeter.estimateMessage(message), 0)
      + ctx.tokenMeter.estimateTools(destination.tools) + (destination.maxTokens ?? 0)
    expect(requested).toBeLessThanOrEqual(destinationWindow)
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
    const persist = ctx.sessions.flush.bind(ctx.sessions)
    let checkpointFlushRefused = false
    const flush = vi.spyOn(ctx.sessions, 'flush').mockImplementation(async (session) => {
      if (!checkpointFlushRefused && session.ownEvents().some(event => event.type === 'compaction/summary')) {
        checkpointFlushRefused = true
        throw new Error('synthetic journal flush failure')
      }
      return persist(session)
    })
    try {
      await send(agent, 'Continue from the original context.')
      expect(flush).toHaveBeenCalledWith(agent.session)
      expect(checkpointFlushRefused).toBe(true)
    } finally {
      flush.mockRestore()
    }
    const after = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(summary.requests).toHaveLength(1)
    expect(chat.requests).toHaveLength(trigger === 'pressure' ? 1 : 2)
    expect(after.slice(0, before.length)).toEqual(before)
    expect(checkpoints(after)).toHaveLength(1)
    expect(after.filter(event => event.type === 'compaction/end')).toHaveLength(1)
    const end = after.at(-1)
    const reason = end?.type === 'turn/end' ? end.data.reason : undefined
    const failure = reason?.kind === 'error' ? reason.error : undefined
    if (trigger === 'pressure') {
      expect(failure?.message).toContain('synthetic journal flush failure')
    } else {
      expect(failure).toMatchObject({ code: CONTEXT_WINDOW_EXCEEDED_CODE, message: 'synthetic context overflow' })
    }
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
    const end = after.at(-1)
    expect(typeof (end?.type === 'compaction/end' ? end.data.error : undefined)).toBe('string')
    if (condition === 'mutation') expect(after).toContainEqual(mutation)
    await owned.dispose()
  })
})

it('records disjoint incremental coverage and carries the checkpoint forward', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 12 }, () => textResponse('Short reply.')))
    const summary = new WindowAdapter(Array.from({ length: 8 }, (_, index) => textResponse(`CHECKPOINT_${index}`)))
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('incremental-three-pass')
    const owned = await ctx.agents.create({
      sessionId,
      agentOptions: { ...wide, maxTokens: 1_024 },
      meta: { cwd: root },
    })
    const compact = async () => { await ctx.compaction.compactNow(owned.agent, signal) }

    await send(owned.agent, 'EARLY_ALPHA '.repeat(40))
    await compact()
    await send(owned.agent, 'MIDDLE_BETA '.repeat(40))
    await send(owned.agent, 'MIDDLE_GAMMA '.repeat(40))
    await compact()
    await send(owned.agent, 'LATE_DELTA '.repeat(40))
    await compact()

    const events = (await ctx.sessionQuery.readSession(sessionId)).events
    const records = events.filter((event): event is SessionEvent<'compaction/summary'> =>
      event.type === 'compaction/summary' && event.data.incremental !== undefined)
    expect(records).toHaveLength(3)
    expect(records.map(record => record.data.incremental!.revision)).toEqual([1, 2, 3])

    const passes = records.map(record => (record.data.chunkCalls ?? []).flatMap(call =>
      call.groups.flatMap(group => group.eventSeqs.map(Number))))
    for (const [index, pass] of passes.entries()) {
      if (index === 0) continue
      expect(Math.min(...pass)).toBeGreaterThan(records[index - 1]!.data.incremental!.highWatermark)
    }
    const all = passes.flat()
    expect(new Set(all).size).toBe(all.length)

    for (const request of summary.requests.slice(1, 3)) {
      expect(requestText(request)).not.toContain('<compacted-summary>')
      expect(requestText(request)).not.toContain('CHECKPOINT_0')
    }
    const checkpointText = checkpoints(events).flatMap(event => event.type === 'user/message'
      ? event.data.content.flatMap(block => block.type === 'text' ? [block.text] : [])
      : []).join('\n')
    expect(checkpointText).toContain('CHECKPOINT_0')
    expect(checkpointText).toContain('CHECKPOINT_1')
    expect(checkpointText).toContain('CHECKPOINT_2')

    const live = (await ctx.sessionQuery.readSession(sessionId)).events
    await owned.dispose()
    const reopened = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(reopened).toEqual(live)
    expect(deriveCommittedCoverage(reopened)?.revision).toBe(3)
  }, {
    incremental: true, auto: true, thresholdRatio: 0.95, summarizationProvider: 'synthetic-summary',
    summarizationModel: 'summary', maxTokens: 1_024,
  })
})

it('dispatches no summarizer call when committed coverage reaches the last RAW event', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([textResponse('Covered reply.')])
    const summary = new WindowAdapter([textResponse('UNUSED')])
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('incremental-precovered')
    const owned = await ctx.agents.create({
      sessionId, agentOptions: { ...wide, maxTokens: 1_024 }, meta: { cwd: root },
    })
    await send(owned.agent, 'Covered history.')
    const events = owned.agent.session.snapshotEvents()
    const nodes = [...owned.agent.session.surface.nodes]
    const head = events.find(event => event.seq === nodes[0])
    const firstIdx = head?.type === 'system/message' ? 1 : 0
    const start = nodes[firstIdx]!
    const end = nodes.at(-1)!
    const last = events.at(-1)!.seq
    const coverage = advanceCoverage(undefined, events, start, last, { policyDigest: 'precommitted' })
    const id = CompactionId('precommitted')
    const startEvent = owned.agent.session.append('compaction/start', { compactionId: id, turn: null })
    const summaryEvent = owned.agent.session.append('compaction/summary', {
      compactionId: id,
      summary: [{ type: 'text', text: 'Precommitted checkpoint.' }],
      shadowedRange: { start, end },
      shadowedSeqs: nodes.slice(firstIdx),
      shadowedTokenCount: 1,
      provider: 'test-provider',
      model: 'test-model',
      chunkCalls: [],
      incremental: coverage,
    })
    owned.agent.session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'Precommitted checkpoint.' }],
      source: compactCheckpointSource(id),
    }), {
      surfaceOp: { op: 'replace', startSeq: start, endSeq: end },
      sourceEventSeqs: [startEvent.seq, summaryEvent.seq, ...nodes.slice(firstIdx)],
    })
    owned.agent.session.append('compaction/end', { compactionId: id, turn: null })

    const callsBefore = summary.requests.length
    expect(await ctx.compaction.compactNow(owned.agent, signal)).toBeNull()
    expect(summary.requests).toHaveLength(callsBefore)
    await owned.dispose()
  }, {
    incremental: true, auto: true, thresholdRatio: 0.95, summarizationProvider: 'synthetic-summary',
    summarizationModel: 'summary', maxTokens: 1_024,
  })
})

it('compacts through the fact ledger and keeps sourced facts across passes', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 8 }, () => textResponse('Reply.')))
    const summary = new WindowAdapter(Array.from({ length: 8 }, (_, index) => (options: GenerateOptions) => {
      const text = requestText(options)
      const seqs = citedSeqs(text)
      const key = text.includes('SECOND_SEGMENT') ? 'second' : 'first'
      return textResponse(JSON.stringify([{
        kind: 'decision', key, claim: `Claim ${index} keeps DSH-TSK-0012.`, sourceSeqs: [seqs[0] ?? 1],
        basis: 'asserted',
      }]))
    }))
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-fact-ledger')
    const owned = await ctx.agents.create({
      sessionId, agentOptions: { ...wide, maxTokens: 1_024 }, meta: { cwd: root },
    })

    await send(owned.agent, 'FIRST_SEGMENT '.repeat(150))
    await ctx.compaction.compactNow(owned.agent, signal)
    const firstPassRequests = summary.requests.length
    expect(firstPassRequests).toBeGreaterThan(0)
    expect(requestText(summary.requests[0]!)).toContain('Extract durable facts')

    await send(owned.agent, 'SECOND_SEGMENT raw history.')
    await ctx.compaction.compactNow(owned.agent, signal)
    for (const request of summary.requests.slice(firstPassRequests)) {
      expect(requestText(request)).not.toContain('Claim 0 keeps')
      expect(requestText(request)).not.toContain('Sourced facts from compacted history')
    }

    const events = (await ctx.sessionQuery.readSession(sessionId)).events
    const records = events.filter((event): event is SessionEvent<'compaction/summary'> =>
      event.type === 'compaction/summary' && event.data.ledger !== undefined)
    expect(records).toHaveLength(2)
    expect(records[0]!.data.ledger).toHaveLength(1)
    expect(records[1]!.data.ledger).toHaveLength(2)
    expect(records[1]!.data.ledger!.every(fact => fact.status === 'active')).toBe(true)
    expect(records[1]!.data.ledger![0]!.sourceEventIds.length).toBeGreaterThan(0)
    expect(records[1]!.data.ledger![0]!.evidenceDigest).toMatch(/^[0-9a-f]{64}$/)
    expect(records[1]!.data.incremental!.revision).toBe(2)

    const checkpointText = checkpoints(events).flatMap(event => event.type === 'user/message'
      ? event.data.content.flatMap(block => block.type === 'text' ? [block.text] : [])
      : []).join('\n')
    expect(checkpointText).toContain('Sourced facts from compacted history')
    expect(checkpointText).toContain('Claim 0 keeps DSH-TSK-0012')
    expect(checkpointText).toContain('Claim 1 keeps DSH-TSK-0012')
    expect(checkpointText).toContain('source events')
    await owned.dispose()
  }, {
    incremental: true, factLedger: true, auto: true, thresholdRatio: 0.95,
    summarizationProvider: 'synthetic-summary', summarizationModel: 'summary', maxTokens: 1_024,
  })
})

it('keeps seeded facts, updates and conflicts across a >1M fact-ledger compaction', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 70 }, () => textResponse('Reply.')))
    const legacyId = digestCanonical(['decision', 'auth-mode', 'legacy'])
    const claimsFor = (options: GenerateOptions): string => {
      const text = requestText(options)
      const seqs = citedSeqs(text)
      const first = seqs[0] ?? 1
      const claims: object[] = [{ kind: 'observation', key: `chunk-${String(first)}`, claim: `Chunk at ${String(first)} reviewed.`, sourceSeqs: [first] }]
      if (text.includes('EARLY_MARKER')) {
        claims.push({ kind: 'constraint', key: 'early', claim: 'Early fact keeps DSH-TSK-0012 and 1066429.', sourceSeqs: [first], basis: 'asserted' })
        claims.push({ kind: 'decision', key: 'auth-mode', claim: 'legacy', sourceSeqs: [first] })
      }
      if (text.includes('MIDDLE_MARKER')) {
        claims.push({ kind: 'identifier', key: 'middle', claim: 'Middle fact keeps DSH-TSK-0012.', sourceSeqs: [first] })
      }
      if (text.includes('CONFLICT_A')) {
        claims.push({ kind: 'decision', key: 'deploy-target', claim: 'staging', sourceSeqs: [first] })
      }
      if (text.includes('CONFLICT_B')) {
        claims.push({ kind: 'decision', key: 'deploy-target', claim: 'prod', sourceSeqs: [first] })
      }
      if (text.includes('LATE_MARKER')) {
        claims.push({ kind: 'decision', key: 'auth-mode', claim: 'resident', sourceSeqs: [first], supersedes: [legacyId] })
        claims.push({ kind: 'tool-evidence', key: 'late', claim: 'Late fact keeps DSH-TSK-0012.', sourceSeqs: [first] })
      }
      return JSON.stringify(claims)
    }
    const summary = new WindowAdapter(Array.from({ length: 10 }, () => (options: GenerateOptions) => textResponse(claimsFor(options))))
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-fact-ledger-million')
    const owned = await ctx.agents.create({
      sessionId, agentOptions: { ...wide, maxTokens: 1_024 }, meta: { cwd: root },
    })
    const filler = 'x'.repeat(72_000)
    for (let turn = 0; turn < 60; turn += 1) {
      const marker = turn === 0 ? 'EARLY_MARKER '
        : turn === 20 ? 'CONFLICT_A '
          : turn === 40 ? 'CONFLICT_B '
            : turn === 50 ? 'MIDDLE_MARKER '
              : turn === 59 ? 'LATE_MARKER ' : ''
      await send(owned.agent, `${marker}${filler}${turn}`)
    }
    expect(ctx.tokenMeter.measure(owned.agent.session).totalTokens).toBeGreaterThan(1_000_000)
    await ctx.compaction.compactNow(owned.agent, signal)
    const firstPassRequests = summary.requests.length
    expect(firstPassRequests).toBeGreaterThan(1)

    const before = (await ctx.sessionQuery.readSession(sessionId)).events
    const firstRecord = before.find(event => event.type === 'compaction/summary' && event.data.ledger !== undefined)
    expect(firstRecord?.type).toBe('compaction/summary')
    if (firstRecord?.type !== 'compaction/summary' || firstRecord.data.ledger === undefined) {
      throw new Error('The first compaction committed no ledger')
    }
    const ledger = firstRecord.data.ledger
    const fact = (key: string, claim: string) => ledger.find(entry => entry.key === key && entry.claim === claim)
    expect(fact('auth-mode', 'legacy')?.status).toBe('superseded')
    expect(fact('auth-mode', 'resident')?.status).toBe('active')
    expect(fact('deploy-target', 'staging')?.status).toBe('disputed')
    expect(fact('deploy-target', 'prod')?.status).toBe('disputed')
    expect(fact('early', 'Early fact keeps DSH-TSK-0012 and 1066429.')?.status).toBe('active')
    expect(fact('middle', 'Middle fact keeps DSH-TSK-0012.')?.status).toBe('active')
    expect(fact('late', 'Late fact keeps DSH-TSK-0012.')?.status).toBe('active')
    validateLedger(ledger, before)

    await send(owned.agent, `DELTA_MARKER ${filler}`)
    await ctx.compaction.compactNow(owned.agent, signal)
    for (const request of summary.requests.slice(firstPassRequests)) {
      expect(requestText(request)).not.toContain('Early fact keeps DSH-TSK-0012')
      expect(requestText(request)).not.toContain('Sourced facts from compacted history')
    }
    const after = (await ctx.sessionQuery.readSession(sessionId)).events
    const records = after.filter(event => event.type === 'compaction/summary' && event.data.ledger !== undefined)
    expect(records.length).toBe(2)
    const second = records[1]!
    if (second.type !== 'compaction/summary' || second.data.ledger === undefined) throw new Error('missing second ledger')
    expect(second.data.ledger.find(entry => entry.key === 'auth-mode' && entry.claim === 'resident')?.status).toBe('active')
    expect(second.data.ledger.find(entry => entry.key === 'early')?.status).toBe('active')
    validateLedger(second.data.ledger, after)
    await owned.dispose()
  }, {
    incremental: true, factLedger: true, auto: true, thresholdRatio: 0.95,
    summarizationProvider: 'synthetic-summary', summarizationModel: 'summary', maxTokens: 1_024,
  })
})

it('omits facts explicitly when the destination cannot hold the whole view', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 70 }, () => textResponse('Reply.')))
    const longText = (text: string): StreamChunk[] => [
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text },
      { type: 'block-end', index: 0, block: { type: 'text', text } },
      { type: 'usage', usage: { inputTokens: 10, outputTokens: 1_020 } },
      { type: 'finish', reason: { kind: 'stop' } },
    ]
    const claimsFor = (options: GenerateOptions): string => {
      const text = requestText(options)
      const seqs = citedSeqs(text)
      const first = seqs[0] ?? 1
      return JSON.stringify(Array.from({ length: 12 }, (_, index) => ({
        kind: 'observation', key: `fact-${String(first)}-${String(index)}`,
        claim: `Fact ${String(first)}-${String(index)} ${'y'.repeat(300)}`, sourceSeqs: [first],
      })))
    }
    const summary = new WindowAdapter(
      Array.from({ length: 8 }, () => (options: GenerateOptions) => longText(claimsFor(options))),
    )
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-fact-ledger-view-budget')
    const owned = await ctx.agents.create({
      sessionId, agentOptions: { ...wide, maxTokens: 1_024 }, meta: { cwd: root },
    })
    const filler = 'x'.repeat(72_000)
    for (let turn = 0; turn < 60; turn += 1) await send(owned.agent, `${filler}${turn}`)
    owned.agent.model.select({ provider: wide.provider, model: 'narrow' })
    await send(owned.agent, 'Switch to the narrow destination.')

    const events = (await ctx.sessionQuery.readSession(sessionId)).events
    const record = events.find(event => event.type === 'compaction/summary' && event.data.ledger !== undefined)
    expect(record?.type).toBe('compaction/summary')
    if (record?.type !== 'compaction/summary' || record.data.ledger === undefined) {
      throw new Error('The switch committed no ledger')
    }
    expect(record.data.ledger).toHaveLength(60)
    validateLedger(record.data.ledger, events)
    const checkpointText = checkpoints(events).flatMap(event => event.type === 'user/message'
      ? event.data.content.flatMap(block => block.type === 'text' ? [block.text] : [])
      : []).join('\n')
    expect(checkpointText).toContain('Omitted ')
    expect(checkpointText).toContain('exact originals remain retrievable from session events')
    const rendered = checkpointText.split('\n').filter(line => line.startsWith('- (observation fact-')).length
    expect(rendered).toBeGreaterThan(0)
    expect(rendered).toBeLessThan(60)
    expect(chat.requests.at(-1)?.model).toBe('narrow')
    await owned.dispose()
  }, {
    incremental: true, factLedger: true, auto: true, thresholdRatio: 0.95,
    summarizationProvider: 'synthetic-summary', summarizationModel: 'summary', maxTokens: 1_024,
  })
})

it('chunks a >1M-token switch through a 262,144-token summarizer inside the destination budget', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 64 }, () => () => textResponse('Continuing.')))
    chat.windows.set('source', 4_000_000)
    chat.windows.set('dest', 262_144)
    const summary = new WindowAdapter(
      Array.from({ length: 16 }, (_, index) => textResponse(`Chunk summary ${index}.`)),
    )
    ctx.llm.registerAdapter([wide.provider], chat)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    await mountStandardPreset(ctx, { thresholdRatio: 0.95 })
    const sessionId = SessionId('hybrid-million-token-switch')
    const owned = await ctx.agents.create({
      sessionId,
      agentOptions: { provider: wide.provider, model: 'source', maxTokens: 1_024 },
      meta: { cwd: root, agentPreset: 'standard' },
      setup: async (agentCtx) => { await ctx.agentPresets.mount(agentCtx, 'standard') },
    })
    expect(ctx.agentPresets.serviceFor(owned.agent, 'compaction')).toBeInstanceOf(HybridCompactionEngine)
    const cjk = '压缩上下文必须保留原始事件、精确标识符与工具事务。'.repeat(120)
    const json = JSON.stringify({
      ticket: 'DSH-TSK-0014', related: ['DSH-TSK-0012', 'DSH-TSK-0009'],
      policy: { provider: 'kinf-native-mservice', model: 'chat-qwen-27b-autoround-int4-vllm-dflash2-tp2@local' },
    })
    const filler = `${'x'.repeat(68_000)}${cjk}${json}`
    for (let turn = 0; turn < 60; turn += 1) await send(owned.agent, `${filler}${turn}`)
    const before = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(ctx.tokenMeter.measure(owned.agent.session).totalTokens).toBeGreaterThan(1_000_000)

    owned.agent.model.select({ provider: wide.provider, model: 'dest' })
    await send(owned.agent, 'Continue on the 262K destination.')

    expect(summary.requests.length).toBeGreaterThanOrEqual(4)
    for (const request of summary.requests) {
      const input = request.messages.reduce((sum, message) => sum + ctx.tokenMeter.estimateMessage(message), 0)
      expect(input + (request.maxTokens ?? 0)).toBeLessThanOrEqual(262_144)
      expect(requestText(request)).not.toContain('<compacted-summary>')
    }
    const after = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(checkpoints(after)).toHaveLength(1)
    expect(after.slice(0, before.length)).toEqual(before)
    expect(summary.requests.some(request => requestText(request).includes('压缩上下文'))).toBe(true)
    expect(summary.requests.some(request => requestText(request).includes('DSH-TSK-0014'))).toBe(true)
    const receipt = after.find(event => event.type === 'compaction/summary')
    expect(receipt?.data.chunkCalls).toHaveLength(summary.requests.length)
    const destination = chat.requests.at(-1)
    expect(destination?.model).toBe('dest')
    if (destination === undefined) throw new Error('The destination did not receive a request')
    const requested = destination.messages.reduce(
      (sum, message) => sum + ctx.tokenMeter.estimateMessage(message), 0,
    ) + ctx.tokenMeter.estimateTools(destination.tools) + (destination.maxTokens ?? 0)
    expect(requested).toBeLessThanOrEqual(262_144)
    await owned.dispose()
    const reopened = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(reopened).toEqual(after)
  }, { auto: false })
})

it('binds Hybrid through the product preset registry composition', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 4 }, () => textResponse('Preset reply.')))
    ctx.llm.registerAdapter([wide.provider], chat)
    await mountStandardPreset(ctx)
    const sessionId = SessionId('hybrid-preset-binding')
    const owned = await ctx.agents.create({
      sessionId,
      agentOptions: wide,
      meta: { cwd: root, agentPreset: 'standard' },
      setup: async (agentCtx) => { await ctx.agentPresets.mount(agentCtx, 'standard') },
    })
    expect(ctx.agentPresets.presetForSession(owned.agent.session)).toBe('standard')
    expect(ctx.agentPresets.composedPreset(owned.agent.ctx)).toBe('standard')
    expect(ctx.agentPresets.serviceFor(owned.agent, 'compaction')).toBeInstanceOf(HybridCompactionEngine)
    await send(owned.agent, 'Short history stays below the pressure threshold.')
    expect(ctx.agentPresets.inspectCompositions(owned.agent.ctx)[0]?.modules.map(row => row.moduleName))
      .toContain('cordis:hybrid-fixture')
    await owned.dispose()
  })
})

function residentRow(logicalModel: string): Candidate {
  const identity: ResidentPrecondition = {
    schema: RESIDENT_ONLY_V1,
    instance: 'fixture-instance',
    host_generation: 'host-1',
    serving_variant_id: 'variant-1',
    serving_variant_revision: 'variant-rev-1',
    model_artifact: 'artifact-1',
    artifact_revision: 'artifact-rev-1',
    target_config_revision: 'config-1',
    resident_generation: 'resident-1',
  }
  return {
    node: {
      id: 'fixture-local', name: 'fixture-alpha', bootGeneration: 'boot-1',
      registered: true, authorized: true, freshness: 'fresh', observedAtMs: Date.now() - 10,
    },
    host: { identity, mode: 'mservice', state: 'ready', enabled: true, usable: true, admissionOpen: true },
    model: {
      logicalModel, servingVariantId: 'variant-1', servingVariantRevision: 'variant-rev-1',
      artifactId: 'artifact-1', targetConfigRevision: 'config-1', loaded: true, ready: true,
      configurationValidated: true, operations: ['chat'], capabilitySource: 'runtime', capabilities: ['text'],
      budget: { contextTokens: 262_144, maxOutputTokens: 1_024, maxInputTokensAtMaxOutput: 261_120 },
    },
    route: {
      id: 'route-1', gatewayId: 'fixture-gateway', nodeId: 'fixture-local', bootGeneration: 'boot-1',
      instance: 'fixture-instance', logicalModel, servingVariantId: 'variant-1',
      alias: `${logicalModel}@fixture-alpha`, healthy: true, contract: RESIDENT_ONLY_V1,
      noRetry: true, noFallback: true, verifiedAtMs: Date.now() - 20,
    },
    queue: { pending: 0, active: 0, maxPending: 8 },
    measurement: {
      projectionId: '', hostGeneration: 'host-1', residentGeneration: 'resident-1',
      targetConfigRevision: 'config-1', promptTokens: 100,
    },
  }
}

it('resolves one resident target for every chunk behind fixed mode', async () => {
  await withContext(async (ctx, root) => {
    const alias = 'resident-model@fixture-alpha'
    const row = residentRow('resident-model')
    const source: ResidentObservationSource = {
      observe: (request: ResidentObservationRequest) => {
        const fresh = { ...row, node: { ...row.node, observedAtMs: Date.now() }, route: { ...row.route, verifiedAtMs: Date.now() } }
        return Promise.resolve([{
          ...fresh,
          measurement: { ...fresh.measurement!, projectionId: request.projection?.id ?? '' },
        }])
      },
    }
    ctx.provide('residentSummarySource', source)
    const chat = new WindowAdapter(Array.from({ length: 4 }, () => textResponse('Reply.')))
    ctx.llm.registerAdapter([wide.provider], chat)
    const summary = new WindowAdapter(Array.from({ length: 8 }, () => textResponse('Resident summary.')))
    summary.windows.set('fixed-model', 262_144)
    summary.windows.set(alias, 262_144)
    ctx.llm.registerAdapter(['fixture-provider'], summary)
    const sessionId = SessionId('hybrid-resident-target')
    const owned = await ctx.agents.create({
      sessionId, agentOptions: { ...wide, maxTokens: 1_024 }, meta: { cwd: root },
    })
    await send(owned.agent, 'RESIDENT_MODE_HISTORY '.repeat(200))
    await ctx.compaction.compactNow(owned.agent, signal)

    expect(summary.requests.length).toBeGreaterThan(0)
    expect(summary.requests.every(request => request.model === alias)).toBe(true)
    // Every chunk is dispatched under the same immutable resident identity.
    const bindings = summary.requests.map(request => request.residentBinding)
    for (const binding of bindings) {
      expect(binding).toMatchObject({
        schema: RESIDENT_ONLY_V1, instance: 'fixture-instance', hostGeneration: 'host-1',
        servingVariantId: 'variant-1', servingVariantRevision: 'variant-rev-1',
        modelArtifact: 'artifact-1', artifactRevision: 'artifact-rev-1',
        targetConfigRevision: 'config-1', residentGeneration: 'resident-1',
      })
    }
    expect(new Set(bindings.map(binding => JSON.stringify(binding))).size).toBe(1)
    const events = (await ctx.sessionQuery.readSession(sessionId)).events
    const summaryEvent = events.findLast(event => event.type === 'compaction/summary')
    expect(summaryEvent).toMatchObject({
      type: 'compaction/summary',
      data: { provider: 'fixture-provider', model: alias },
    })
    expect(checkpoints(events)).toHaveLength(1)
    await owned.dispose()
  }, {
    auto: true, thresholdRatio: 0.95, summarizationProvider: 'fixture-provider',
    summarizationModel: 'fixed-model', maxTokens: 1_024,
    summarizationMode: 'resident-preferred', residentGatewayId: 'fixture-gateway',
    residentPinnedNodeId: 'fixture-local', residentProvider: 'fixture-provider', residentMaxAgeMs: 500,
  })
})

it('measures the fact-extraction envelope when the resident target and fact ledger combine', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 4 }, () => textResponse('Reply.')))
    chat.windows.set('wide', 4_000_000)
    ctx.llm.registerAdapter([wide.provider], chat)
    const summary = factLedgerAdapter('resident-fact', 'Resident target and fact ledger combine.')
    summary.windows.set('fixed-model', 262_144)
    summary.windows.set('resident-fact@fixture-alpha', 262_144)
    ctx.llm.registerAdapter(['fixture-provider'], summary)

    const row = residentRow('resident-fact')
    const observations: ResidentObservationRequest[] = []
    ctx.provide('residentSummarySource', {
      observe: (request) => {
        observations.push(request)
        const fresh = {
          ...row,
          node: { ...row.node, observedAtMs: Date.now() },
          route: { ...row.route, verifiedAtMs: Date.now() },
        }
        return Promise.resolve([{
          ...fresh,
          measurement: { ...fresh.measurement!, projectionId: request.projection?.id ?? '' },
        }])
      },
    })
    const sessionId = SessionId('hybrid-resident-fact-ledger')
    const owned = await ctx.agents.create({
      sessionId, agentOptions: { ...wide, maxTokens: 1_024 }, meta: { cwd: root },
    })
    await send(owned.agent, 'RESIDENT_FACT_HISTORY '.repeat(200))
    expect(await ctx.compaction.compactNow(owned.agent, signal)).not.toBeNull()

    // The resident source is asked to measure the envelope that will actually run.
    expect(observations).toHaveLength(1)
    const projectionText = (observations[0]?.projection?.messages ?? [])
      .flatMap(message => message.content)
      .flatMap(block => block.type === 'text' ? [block.text] : []).join('\n')
    expect(projectionText).toContain('Extract durable facts')
    expect(projectionText).not.toContain('Summarize the quoted history')
    // And every dispatched chunk carries that same instruction shape.
    expect(summary.requests.length).toBeGreaterThan(0)
    for (const request of summary.requests) expect(requestText(request)).toContain('Extract durable facts')
    await owned.dispose()
  }, {
    incremental: true, factLedger: true, auto: true, thresholdRatio: 0.95,
    summarizationProvider: 'fixture-provider', summarizationModel: 'fixed-model', maxTokens: 1_024,
    summarizationMode: 'resident-preferred', residentGatewayId: 'fixture-gateway',
    residentPinnedNodeId: 'fixture-local', residentProvider: 'fixture-provider', residentMaxAgeMs: 500,
  })
}, 60_000)

const observationPath = '/v1/observations'
const residentPath = '/kinf/resident/v1/chat/completions'
const residentFact = 'The deployment approval code is cobalt-73.'
const residentSummary = 'The original deployment approval code was cobalt-73.'
const residentHistory = `${residentFact}\n${'Prior context. '.repeat(150)}`
const residentProvider = 'fixture-kinf'

const local27b = {
  id: '27b-node',
  name: 'qwen',
  logicalModel: 'chat-qwen-27b-uncensored-exl3-6bpw',
  variant: 'qwen27-uncensored-native-mtp',
}
const localFlash = {
  id: 'flash-node',
  name: 'flash',
  logicalModel: 'chat-qwen-flash-next',
  variant: 'flash-native-mtp',
}

type NativeResidentReply = 'observe' | 'unloaded' | 'summary' | 'stale' | 'ambiguous'

function residentIdentity(node: typeof local27b, generation: number): ResidentPrecondition {
  return {
    schema: RESIDENT_ONLY_V1,
    instance: 'default',
    host_generation: `host-${node.name}`,
    resident_generation: `resident-${node.name}-${generation}`,
    serving_variant_id: node.variant,
    serving_variant_revision: 'chat-v1',
    model_artifact: `artifact-${node.name}`,
    artifact_revision: 'artifact-v1',
    target_config_revision: `config-${node.name}`,
  }
}

function observationResponse(node: typeof local27b, identity: ResidentPrecondition, loaded: boolean) {
  return {
    schema: 'kinf.host-observation.v1',
    observed_at_unix_ms: Date.now(),
    complete: true,
    host: {
      ...identity,
      node_id: node.id,
      boot_generation: `boot-${node.name}`,
      route: null,
      mode: 'mservice',
      logical_model: node.logicalModel,
      state: loaded ? 'ready' : 'unloaded',
      enabled: true,
      admission_open: loaded,
      engine_usable: loaded,
      engine_generation: identity.resident_generation,
      resident_only_contract: RESIDENT_ONLY_V1,
      resident_only_admission_open: loaded,
      active_requests: 0,
      pending_requests: 0,
    },
    models: [{
      ...identity,
      logical_model: node.logicalModel,
      loaded,
      ready: loaded,
      configuration_validated: true,
      capability_source: 'runtime',
      capabilities: ['text'],
      operations: ['chat'],
      resident_only_budget: { context_tokens: 16_380, max_output_tokens: 2_048, max_input_tokens_at_max_output: 14_332 },
    }],
    queue: { active_requests: 0, pending_requests: 0, configured: { max_pending_requests: 4 } },
  }
}

async function mockResidentNode(node: typeof local27b, replies: readonly NativeResidentReply[]) {
  let generation = 0
  const server = await mockServer(replies.map((reply) => {
    if (reply === 'observe' || reply === 'unloaded') {
      generation++
      return { body: JSON.stringify(observationResponse(node, residentIdentity(node, generation), reply === 'observe')) }
    }
    if (reply === 'stale' || reply === 'ambiguous') {
      return {
        status: 409,
        body: JSON.stringify({
          request_contract: RESIDENT_ONLY_V1, accepted: reply === 'ambiguous', started: false,
          error: { code: 'resident_identity_mismatch' },
        }),
      }
    }
    const served = residentIdentity(node, generation)
    const replyIdentity = {
      request_contract: RESIDENT_ONLY_V1,
      accepted: true,
      started: true,
      served_identity: {
        instance: served.instance,
        host_generation: served.host_generation,
        resident_generation: served.resident_generation,
        serving_variant_id: served.serving_variant_id,
        serving_variant_revision: served.serving_variant_revision,
        model_artifact: served.model_artifact,
        artifact_revision: served.artifact_revision,
        target_config_revision: served.target_config_revision,
      },
    }
    return { events: [
      JSON.stringify({ ...replyIdentity, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] }),
      JSON.stringify({ choices: [{ index: 0, delta: { content: residentSummary }, finish_reason: null }] }),
      JSON.stringify({ ...replyIdentity, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }),
      '[DONE]',
    ] }
  }))
  const baseUrl = `${server.url}/v1`
  const alias = `${node.logicalModel}@${node.name}`
  return {
    server,
    alias,
    model: {
      id: alias, requestModel: node.logicalModel, baseURL: baseUrl, residentTransport: true,
      contextWindow: 16_380, maxTokens: 2_048, input: ['text'],
      headers: { 'X-KINF-Allow-Cold-Load': 'true', 'X-KINF-Allow-Eviction': 'true' },
    } satisfies LlmPiAi.PiAiModelProfile,
    route: {
      id: `${node.name}-summary`, nodeId: node.id, nodeName: node.name, instance: 'default',
      logicalModel: node.logicalModel, baseUrl,
    },
  }
}

async function withResidentComposition(
  replies27b: readonly NativeResidentReply[],
  repliesFlash: readonly NativeResidentReply[],
  run: (
    ctx: Context, root: string, qwen: Awaited<ReturnType<typeof mockResidentNode>>,
    flash: Awaited<ReturnType<typeof mockResidentNode>>,
  ) => Promise<void>,
  config: CompactionOverrides = {},
) {
  onTestFinished(closeMockServers)
  const flash = await mockResidentNode(localFlash, repliesFlash)
  const qwen = await mockResidentNode(local27b, replies27b)
  await withContext(async (ctx, root) => {
    await ctx.plugin(LlmPiAi, {
      providers: {
        [residentProvider]: { api: 'openai-completions', baseURL: qwen.route.baseUrl, models: [flash.model, qwen.model] },
      },
    })
    await run(ctx, root, qwen, flash)
  }, {
    auto: true,
    thresholdRatio: 0.5,
    maxTokens: 1_024,
    summarizationProvider: undefined,
    summarizationModel: undefined,
    summarizationMode: 'resident-preferred',
    residentGatewayId: 'fixture-local-kinf',
    residentProvider,
    residentMaxAgeMs: 60_000,
    residentAllowedNodeIds: [flash.route.nodeId, qwen.route.nodeId],
    residentPreferredVariantIds: [local27b.variant, localFlash.variant],
    residentObservationRoutes: [flash.route, qwen.route],
    ...config,
  })
}

function expectResidentRequests(
  host: Awaited<ReturnType<typeof mockResidentNode>>,
  node: typeof local27b,
  generations: readonly number[],
) {
  const indices = host.server.paths.flatMap((path, index) => path === residentPath ? [index] : [])
  expect(indices).toHaveLength(generations.length)
  for (const [i, index] of indices.entries()) {
    const body = host.server.requests[index]
    expect(body).toMatchObject({
      model: node.logicalModel, stream: true, max_tokens: 1_024,
      kinf_resident_only: residentIdentity(node, generations[i]!),
    })
    expect(body).not.toHaveProperty('tools')
    expect(host.server.headers[index]).not.toHaveProperty('x-kinf-allow-cold-load')
    expect(host.server.headers[index]).not.toHaveProperty('x-kinf-allow-eviction')
  }
}

it.each(['automatic', 'manual', 'overflow', 'model switch'] as const)('dispatches %s compaction to the preferred eligible 27B', async (entry) => {
  const flashObservation = entry === 'manual' ? 'unloaded' : 'observe'
  await withResidentComposition(['observe', 'summary'], [flashObservation], async (ctx, root, qwen, flash) => {
    const chat = new WindowAdapter([
      textResponse('Seeded context.'),
      entry === 'overflow' ? [{ type: 'finish', reason: {
        kind: 'error', failure: { code: CONTEXT_WINDOW_EXCEEDED_CODE, message: 'synthetic context overflow' },
      } }] : textResponse('Continued.'),
      textResponse('Recovered.'),
    ])
    ctx.llm.registerAdapter([wide.provider], chat)
    const sessionId = SessionId('hybrid-kinf-entry')
    const owned = await ctx.agents.create({
      sessionId, agentOptions: { ...wide, model: entry === 'automatic' ? 'small' : 'wide' }, meta: { cwd: root },
    })
    const pressureRepeats = entry === 'model switch' ? 1_700 : 850
    const history = entry === 'manual' || entry === 'overflow' ? residentHistory
      : `${residentFact}\n${'Original pressure input. '.repeat(pressureRepeats)}`
    await send(owned.agent, history)
    const before = (await ctx.sessionQuery.readSession(sessionId)).events
    if (entry === 'manual') {
      const result = await ctx.commands.execute(owned.agent, '/compact', [], signal)
      expect(result?.result.kind).toBe('success')
    } else {
      if (entry === 'model switch') owned.agent.model.select({ provider: wide.provider, model: 'small' })
      await send(owned.agent, 'Continue after preserving the original approval code.')
      expect(requestText(chat.requests.at(-1)!)).toContain(residentSummary)
    }
    const { events } = await ctx.sessionQuery.readSession(sessionId)
    expect(events.slice(0, before.length)).toEqual(before)
    expect(checkpoints(events)).toHaveLength(1)
    expect(events.findLast(event => event.type === 'compaction/summary')).toMatchObject({
      data: {
        provider: residentProvider,
        model: qwen.alias,
        residentSelection: {
          alias: qwen.alias,
          gatewayId: 'fixture-local-kinf',
          routeId: qwen.route.id,
          nodeId: local27b.id,
          nodeName: local27b.name,
          instance: 'default',
          servingVariantId: local27b.variant,
          servingVariantRevision: 'chat-v1',
          modelArtifact: 'artifact-qwen',
          artifactRevision: 'artifact-v1',
          targetConfigRevision: 'config-qwen',
          hostGeneration: 'host-qwen',
          residentGeneration: 'resident-qwen-1',
          reason: entry === 'manual' ? 'only-eligible' : 'configured-preference',
          projectionIds: [expect.stringMatching(/^[a-f0-9]{64}$/u)],
        },
      },
    })
    expect(JSON.stringify(qwen.server.requests[1])).toContain(residentFact)
    expect(qwen.server.paths).toEqual([observationPath, residentPath])
    expect(flash.server.paths).toEqual([observationPath])
    expectResidentRequests(qwen, local27b, [1])
    expect(owned.agent.session.requestHeader()?.config).toMatchObject({
      provider: wide.provider, model: entry === 'automatic' || entry === 'model switch' ? 'small' : 'wide',
    })
    await owned.dispose()
    expect((await ctx.sessionQuery.readSession(sessionId)).events).toEqual(events)
  })
})

it('refuses an unmapped session node before looking up any resident', async () => {
  await withResidentComposition(['observe', 'summary'], ['observe'], async (ctx, root, qwen, flash) => {
    const model = 'wide@unknown-node'
    const chat = new WindowAdapter([textResponse('Seeded context.')])
    chat.windows.set(model, 2_000_000)
    ctx.llm.registerAdapter([wide.provider], chat)
    const sessionId = SessionId('hybrid-kinf-unmapped-session-node')
    const owned = await ctx.agents.create({ sessionId, agentOptions: { ...wide, model }, meta: { cwd: root } })
    await send(owned.agent, residentHistory)
    const before = (await ctx.sessionQuery.readSession(sessionId)).events
    await expect(ctx.compaction.compactNow(owned.agent, signal)).rejects.toMatchObject({
      code: 'summary', cause: { code: 'no-eligible' },
    })
    const { events } = await ctx.sessionQuery.readSession(sessionId)
    expect(events.slice(0, before.length)).toEqual(before)
    expect(checkpoints(events)).toHaveLength(0)
    expect(qwen.server.paths).toEqual([])
    expect(flash.server.paths).toEqual([])
    await owned.dispose()
  }, { residentAllowedNodeIds: undefined, residentPinnedNodeId: localFlash.id })
})

it('refreshes once after a definite stale refusal and selects another already-loaded resident', async () => {
  await withResidentComposition(['observe', 'stale', 'unloaded'], ['observe', 'observe', 'summary'], async (ctx, root, qwen, flash) => {
    const chat = new WindowAdapter([textResponse('Seeded context.')])
    ctx.llm.registerAdapter([wide.provider], chat)
    const sessionId = SessionId('hybrid-kinf-reselect')
    const owned = await ctx.agents.create({ sessionId, agentOptions: wide, meta: { cwd: root } })
    await send(owned.agent, residentHistory)
    expect(await ctx.compaction.compactNow(owned.agent, signal)).not.toBeNull()
    expect(qwen.server.paths).toEqual([observationPath, residentPath, observationPath])
    expect(flash.server.paths).toEqual([observationPath, observationPath, residentPath])
    expectResidentRequests(qwen, local27b, [1])
    expectResidentRequests(flash, localFlash, [2])
    expect(chat.requests).toHaveLength(1)
    const { events } = await ctx.sessionQuery.readSession(sessionId)
    expect(checkpoints(events)).toHaveLength(1)
    expect(events.findLast(event => event.type === 'compaction/summary')).toMatchObject({
      data: {
        provider: residentProvider, model: flash.alias,
        residentSelection: { alias: flash.alias, nodeId: localFlash.id, residentGeneration: 'resident-flash-2' },
      },
    })
    await owned.dispose()
  })
})

it.each([
  { name: 'no eligible resident', qwenReplies: ['unloaded'], flashReplies: ['unloaded'], generations: [], code: 'no-eligible' },
  { name: 'a second stale refusal', qwenReplies: ['observe', 'stale', 'observe', 'stale'], flashReplies: ['unloaded', 'unloaded'], generations: [1, 2], code: 'RESIDENT_PRECONDITION_STALE' },
  { name: 'ambiguous acceptance', qwenReplies: ['observe', 'ambiguous'], flashReplies: ['observe'], generations: [1], code: 'RESIDENT_ACCEPTANCE_AMBIGUOUS' },
] as const)('preserves history without another dispatch after $name', async ({ qwenReplies, flashReplies, generations, code }) => {
  await withResidentComposition(qwenReplies, flashReplies, async (ctx, root, qwen, flash) => {
    const chat = new WindowAdapter([textResponse('Seeded context.')])
    ctx.llm.registerAdapter([wide.provider], chat)
    const sessionId = SessionId('hybrid-kinf-refusal')
    const owned = await ctx.agents.create({ sessionId, agentOptions: wide, meta: { cwd: root } })
    await send(owned.agent, residentHistory)
    const before = (await ctx.sessionQuery.readSession(sessionId)).events
    const surface = [...owned.agent.session.surface.nodes]
    const header = owned.agent.session.requestHeader()
    await expect(ctx.compaction.compactNow(owned.agent, signal)).rejects.toMatchObject({ code: 'summary', cause: { code } })
    expect(qwen.server.paths).toHaveLength(qwenReplies.length)
    expect(flash.server.paths).toEqual(flashReplies.map(() => observationPath))
    expectResidentRequests(qwen, local27b, generations)
    expect(chat.requests).toHaveLength(1)
    const { events } = await ctx.sessionQuery.readSession(sessionId)
    expect(events.slice(0, before.length)).toEqual(before)
    expect(checkpoints(events)).toHaveLength(0)
    expect(owned.agent.session.surface.nodes).toEqual(surface)
    expect(owned.agent.session.requestHeader()).toEqual(header)
    await owned.dispose()
  })
})

const live27bTargets = [
  {
    optIn: 'already-loaded-27b', origin: 'http://127.0.0.1:49362', instance: 'default',
    logicalModel: local27b.logicalModel, variant: local27b.variant,
  },
  {
    optIn: 'already-loaded-isolated-27b', origin: 'http://127.0.0.1:49371', instance: 'kinf-0262-private',
    logicalModel: 'chat-qwen-27b-standard-exl3-6bpw', variant: 'qwen27-standard-native-mtp',
  },
] as const

const live27bTarget = live27bTargets.find(target => target.optIn === process.env['DSH_KINF_RESIDENT_LIVE'])

it.skipIf(live27bTarget === undefined)(
  `compacts synthetic history through the already-loaded local 27B (${live27bTarget?.instance ?? 'opt-in'})`,
  async () => {
    if (live27bTarget === undefined) throw new Error('Live compaction requires an explicit 27B target')
    const { origin, instance, logicalModel, variant } = live27bTarget
    const baseUrl = `${origin}/v1`
    const gatewayId = 'local-kinf'
    const route = {
      id: 'local-27b', nodeId: 'kano-local', nodeName: 'local', instance, logicalModel, baseUrl,
    }
    const alias = `${route.logicalModel}@${route.nodeName}`
    const liveSignal = AbortSignal.timeout(45_000)
    const liveFetch = globalThis.fetch
    const requests: string[] = []
    const fetchGuard = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = new URL(input instanceof Request ? input.url : input)
      const method = init?.method ?? (input instanceof Request ? input.method : 'GET')
      const request = `${method} ${url.pathname}`
      if (url.origin !== origin || url.search !== ''
        || ![`GET ${observationPath}`, `POST ${residentPath}`].includes(request)) {
        throw new Error('Live compaction attempted an undeclared HTTP route')
      }
      requests.push(request)
      return liveFetch(input, { ...init, redirect: 'error' })
    })
    onTestFinished(() => { fetchGuard.mockRestore() })

    await withContext(async (ctx, root) => {
      const observation = createKinfResidentObservationSource(gatewayId, [route], {
        measure: messages => messages.reduce((total, message) => total + ctx.tokenMeter.estimateMessage(message), 0),
      })
      const observe = () => observation.observe({
        gatewayId, nodeIds: [route.nodeId], signal: liveSignal,
        projection: {
          id: 'live-resident-preflight',
          messages: [{ role: 'user', content: [{ type: 'text', text: residentHistory }] }],
          outputTokens: 1_024,
        },
      })
      const [before] = await observe()
      const budget = before?.model.budget
      if (before?.host.identity.serving_variant_id !== variant || budget == null || budget.maxOutputTokens < 1_024) {
        expect(requests).toEqual([`GET ${observationPath}`])
        throw new Error(`Live compaction requires an eligible resident 27B on ${instance}`)
      }

      await ctx.plugin(LlmPiAi, {
        providers: {
          [residentProvider]: {
            api: 'openai-completions', baseURL: baseUrl,
            models: [{
              id: alias, requestModel: route.logicalModel, contextWindow: budget.contextTokens,
              maxTokens: budget.maxOutputTokens, input: ['text'], residentTransport: true,
            }],
          },
        },
      })
      const chat = new WindowAdapter([textResponse('Seeded synthetic context.')])
      ctx.llm.registerAdapter([wide.provider], chat)
      const sessionId = SessionId('hybrid-kinf-live')
      const owned = await ctx.agents.create({ sessionId, agentOptions: wide, meta: { cwd: root } })
      await send(owned.agent, residentHistory)
      expect(await ctx.compaction.compactNow(owned.agent, liveSignal)).not.toBeNull()
      expect(chat.requests).toHaveLength(1)
      await owned.dispose()

      const { events } = await ctx.sessionQuery.readSession(sessionId)
      const summary = events.findLast(event => event.type === 'compaction/summary')
      const identity = before.host.identity
      expect(summary?.data).toMatchObject({
        provider: residentProvider, model: alias,
        residentSelection: {
          alias, gatewayId, routeId: route.id, nodeId: route.nodeId, nodeName: route.nodeName,
          instance: identity.instance, servingVariantId: identity.serving_variant_id,
          servingVariantRevision: identity.serving_variant_revision, modelArtifact: identity.model_artifact,
          artifactRevision: identity.artifact_revision, targetConfigRevision: identity.target_config_revision,
          hostGeneration: identity.host_generation, residentGeneration: identity.resident_generation,
          reason: 'only-eligible', projectionIds: [expect.stringMatching(/^[a-f0-9]{64}$/u)],
        },
      })
      expect(JSON.stringify(summary?.data.summary)).toContain('cobalt-73')
      expect(checkpoints(events)).toHaveLength(1)
      const [after] = await observe()
      expect(after?.host.identity).toEqual(identity)
      expect(requests).toEqual([
        `GET ${observationPath}`, `GET ${observationPath}`, `POST ${residentPath}`, `GET ${observationPath}`,
      ])
    }, {
      summarizationProvider: undefined, summarizationModel: undefined, summarizationMode: 'resident-preferred',
      residentGatewayId: gatewayId, residentProvider, residentPinnedNodeId: route.nodeId,
      residentPreferredVariantIds: [variant], residentObservationRoutes: [route],
    })
  },
  60_000,
)

it('refuses a first request whose envelope exceeds the window before any provider call', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 4 }, () => textResponse('Reply.')))
    chat.windows.set('tight', 4_096)
    ctx.llm.registerAdapter(['synthetic-chat'], chat)
    const sessionId = SessionId('hybrid-envelope-first-refusal')
    const owned = await ctx.agents.create({
      sessionId,
      agentOptions: { provider: 'synthetic-chat', model: 'tight', maxTokens: 3_900 },
      meta: { cwd: root },
    })
    await send(owned.agent, 'FIRST_REQUEST_REFUSAL '.repeat(40))
    const events = (await ctx.sessionQuery.readSession(sessionId)).events
    expectAdmissionRefusal(events)
    expect(chat.requests).toHaveLength(0)
    // The refused message stays pending instead of being consumed by the refusal.
    expectPendingInput(owned.agent, 'FIRST_REQUEST_REFUSAL')
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.95, maxTokens: 1_024 })
}, 60_000)

it('refuses a same-route next turn whose envelope exceeds the window', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter(Array.from({ length: 4 }, () => textResponse('Reply.')))
    chat.windows.set('small', 8_192)
    ctx.llm.registerAdapter(['synthetic-chat'], chat)
    const sessionId = SessionId('hybrid-envelope-next-turn-refusal')
    const owned = await ctx.agents.create({
      sessionId,
      agentOptions: { provider: 'synthetic-chat', model: 'small', maxTokens: 1_024 },
      meta: { cwd: root },
    })
    await send(owned.agent, 'accepted first turn')
    expect(chat.requests).toHaveLength(1)
    await send(owned.agent, 'NEXT_TURN_OVERSIZE '.repeat(3_000))
    const events = (await ctx.sessionQuery.readSession(sessionId)).events
    expectAdmissionRefusal(events)
    expect(chat.requests).toHaveLength(1)
    // The refused next-turn message stays pending.
    expectPendingInput(owned.agent, 'NEXT_TURN_OVERSIZE')
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.95, maxTokens: 1_024 })
}, 60_000)

it('restores a fact-ledger checkpoint in a fresh Basic context (rollback compatibility)', async () => {
  const root = await mkdtemp(join(tmpdir(), 'dsh-hybrid-cold-'))
  const sessionId = SessionId('hybrid-cold-ledger')
  const mount = async (engine: 'hybrid' | 'basic', config: CompactionOverrides) => {
    const ctx = new Context()
    await mountCompactionHarness(ctx, root, engine === 'basic' ? { 'basic-fixture': BasicCompactionEngine } : {})
    await loadCompactionComposition(ctx, root, `cordis-${engine}.yml`, engine, config)
    return ctx
  }
  try {
    let factCount = 0
    let coverageRevision = 0
    {
      const ctx = await mount('hybrid', { incremental: true, factLedger: true })
      try {
        const chat = new WindowAdapter(Array.from({ length: 4 }, () => textResponse('Reply.')))
        chat.windows.set('wide', 4_000_000)
        ctx.llm.registerAdapter([wide.provider], chat)
        const summary = factLedgerAdapter('cold-ledger', 'Cold resume keeps the ledger.')
        summary.windows.set('summary', 262_144)
        ctx.llm.registerAdapter(['synthetic-summary'], summary)
        const owned = await ctx.agents.create({
          sessionId, agentOptions: { ...wide, maxTokens: 1_024 }, meta: { cwd: root },
        })
        await send(owned.agent, 'COLD_LEDGER_HISTORY '.repeat(400))
        expect(await ctx.compaction.compactNow(owned.agent, signal)).not.toBeNull()
        const events = (await ctx.sessionQuery.readSession(sessionId)).events
        factCount = events.flatMap(event => event.type === 'compaction/summary' ? event.data.ledger ?? [] : []).length
        coverageRevision = deriveCommittedCoverage(events)?.revision ?? 0
        expect(factCount).toBeGreaterThan(0)
        expect(coverageRevision).toBeGreaterThan(0)
        await owned.dispose()
      } finally {
        await ctx.fiber.dispose()
      }
    }

    // Roll back to Basic: a fresh graph loads the same persisted session.
    const ctx = await mount('basic', {})
    try {
      const chat = new WindowAdapter(Array.from({ length: 4 }, () => textResponse('Reply.')))
      chat.windows.set('wide', 4_000_000)
      ctx.llm.registerAdapter([wide.provider], chat)
      const events = (await ctx.sessionQuery.readSession(sessionId)).events
      const ledger = events.flatMap(event => event.type === 'compaction/summary' ? event.data.ledger ?? [] : [])
      expect(ledger).toHaveLength(factCount)
      validateLedger(ledger, events)
      expect(deriveCommittedCoverage(events)?.revision).toBe(coverageRevision)
      expect(checkpoints(events)).toHaveLength(1)

      const handle = await ctx.agents.resume({
        resumeSessionId: sessionId, agentOptions: { ...wide, maxTokens: 1_024 },
      })
      await send(handle.agent, 'after rollback')
      const restored = (await ctx.sessionQuery.readSession(sessionId)).events
      expect(restored.findLast(event => event.type === 'turn/end')?.data.reason.kind).toBe('completed')
      // The restored surface still carries the Hybrid checkpoint text.
      expect(restored.filter(event => event.type === 'user/message' && isCompactCheckpointSource(event.data.source))).toHaveLength(1)
      await handle.dispose()
    } finally {
      await ctx.fiber.dispose()
    }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}, 120_000)

it('compacts for an over-budget envelope that stays under the pressure threshold', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([textResponse('First reply.'), textResponse('Second reply.')])
    chat.windows.set('roomy', 16_384)
    ctx.llm.registerAdapter(['synthetic-chat'], chat)
    const summary = new WindowAdapter(Array.from({ length: 4 }, () => textResponse('Compacted history.')))
    summary.windows.set('summary', 16_384)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-envelope-pressure-fallback')
    const owned = await ctx.agents.create({
      sessionId,
      agentOptions: { provider: 'synthetic-chat', model: 'roomy', maxTokens: 1_024 },
      meta: { cwd: root },
    })
    await send(owned.agent, 'HISTORY '.repeat(3_000))
    expect(chat.requests).toHaveLength(1)
    await send(owned.agent, 'PENDING '.repeat(5_000))

    await expectCompactedTurn(ctx, sessionId, chat, 2)
    // The fallback compacted the durable history, and the pending input survived.
    const second = requestText(chat.requests[1]!)
    expect(second).not.toContain('HISTORY HISTORY')
    expect(second).toContain('PENDING PENDING')
    await owned.dispose()
  }, {
    auto: true, thresholdRatio: 0.95, maxTokens: 1_024,
    summarizationProvider: 'synthetic-summary', summarizationModel: 'summary',
  })
}, 60_000)

it('prices a tool continuation whose result alone would not fit the window', async () => {
  await withContext(async (ctx, root) => {
    ctx.tools.register(defineContentToolFixture({
      name: 'fixture_huge', description: 'Return a large synthetic body.', parameters: {},
      async execute() { return [{ type: 'text', text: 'TOOL_BODY '.repeat(4_000) }] },
    }))
    const chat = new WindowAdapter([toolCallResponse('call-huge', 'fixture_huge', {}), textResponse('done')])
    chat.windows.set('tool-small', 8_192)
    ctx.llm.registerAdapter(['synthetic-chat'], chat)
    const summary = new WindowAdapter(Array.from({ length: 4 }, () => textResponse('Result retained by source reference.')))
    summary.windows.set('summary', 8_192)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-envelope-tool-continuation')
    const owned = await ctx.agents.create({
      sessionId,
      agentOptions: { provider: 'synthetic-chat', model: 'tool-small', maxTokens: 1_024 },
      meta: { cwd: root },
    })
    await send(owned.agent, 'Read the fixture.')
    const events = (await ctx.sessionQuery.readSession(sessionId)).events
    // Either the continuation is refused before dispatch, or compaction removed
    // the unwieldy result and the turn completed; both must be explicit.
    // The unwieldy result cannot be summarized below the window, so the
    // continuation is refused before a second provider request.
    expectAdmissionRefusal(events)
    expect(chat.requests).toHaveLength(1)
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.95, maxTokens: 1_024,
    summarizationProvider: 'synthetic-summary', summarizationModel: 'summary' })
}, 60_000)

it('prices the tool declarations the request will send, not the previous header', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([textResponse('First reply.'), textResponse('Second reply.')])
    chat.windows.set('roomy', 16_384)
    ctx.llm.registerAdapter(['synthetic-chat'], chat)
    const summary = new WindowAdapter(Array.from({ length: 4 }, () => textResponse('Compacted.')))
    summary.windows.set('summary', 16_384)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-envelope-tool-declarations')
    const owned = await ctx.agents.create({
      sessionId,
      agentOptions: { provider: 'synthetic-chat', model: 'roomy', maxTokens: 1_024 },
      meta: { cwd: root },
    })
    await send(owned.agent, 'HISTORY '.repeat(2_000))
    expect(chat.requests).toHaveLength(1)
    // The declaration arrives after the committed header, so a preview that read
    // the previous header would price zero tools.
    ctx.tools.register(defineContentToolFixture({
      name: 'fixture_wide', description: 'Wide declaration. ' + 'x'.repeat(8_000), parameters: {},
      async execute() { return [{ type: 'text', text: 'ok' }] },
    }))
    await send(owned.agent, 'PENDING '.repeat(5_000))

    // The preview saw the tool declarations, compacted the history and let the
    // request through; before the fix it priced zero tools and the verify phase
    // would have refused without attempting compaction.
    await expectCompactedTurn(ctx, sessionId, chat, 2)
    expect(requestText(chat.requests[1]!)).toContain('PENDING PENDING')
    await owned.dispose()
  }, {
    auto: true, thresholdRatio: 0.95, maxTokens: 1_024,
    summarizationProvider: 'synthetic-summary', summarizationModel: 'summary',
  })
}, 60_000)

it('trims oversized tool output at admission before shadowing history', async () => {
  await withContext(async (ctx, root) => {
    await ctx.plugin(ToolResultPruner)
    ctx.tools.register(defineContentToolFixture({
      name: 'fixture_prune', description: 'Return a large body.', parameters: {},
      async execute() { return [{ type: 'text', text: 'TOOL_BODY '.repeat(2_950) }] },
    }))
    const chat = new WindowAdapter([toolCallResponse('call-prune', 'fixture_prune', {}), textResponse('done')])
    chat.windows.set('small', 8_192)
    ctx.llm.registerAdapter(['synthetic-chat'], chat)
    const summary = new WindowAdapter([textResponse('Compacted.')])
    summary.windows.set('summary', 8_192)
    ctx.llm.registerAdapter(['synthetic-summary'], summary)
    const sessionId = SessionId('hybrid-envelope-prune')
    const owned = await ctx.agents.create({
      sessionId,
      agentOptions: { provider: 'synthetic-chat', model: 'small', maxTokens: 1_024 },
      meta: { cwd: root },
    })
    await send(owned.agent, 'Read it.')
    const events = (await ctx.sessionQuery.readSession(sessionId)).events
    expectCompletedTurn(events)
    // A trim, not a checkpoint: the pruner keeps the recent result readable.
    expect(checkpoints(events)).toHaveLength(0)
    expect(summary.requests).toHaveLength(0)
    expect(chat.requests).toHaveLength(2)
    expect(requestText(chat.requests[1]!)).toContain('tool result middle pruned')
    await owned.dispose()
  }, {
    auto: true, thresholdRatio: 0.99, maxTokens: 1_024,
    summarizationProvider: 'synthetic-summary', summarizationModel: 'summary',
  })
}, 60_000)

it('refuses at the frozen request when the file projection grows past the preview budget', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([textResponse('First reply.'), textResponse('Second reply.')])
    chat.windows.set('small', 4_096)
    ctx.llm.registerAdapter(['synthetic-chat'], chat)
    let path = '/sandbox/notes.txt'
    ctx.provide('attachments', { fileHostPath: () => '/host/notes.txt' } as never)
    ctx.provide('fs', { processPathFromHostPath: () => path } as never)
    const sessionId = SessionId('hybrid-envelope-projection-growth')
    const owned = await ctx.agents.create({
      sessionId,
      agentOptions: { provider: 'synthetic-chat', model: 'small', maxTokens: 512 },
      meta: { cwd: root },
    })
    await send(owned.agent, 'warm the header')
    expect(chat.requests).toHaveLength(1)

    const remove = ctx.on('llm/stream', (_request, next) => {
      path = `/sandbox/${'x'.repeat(16_384)}/notes.txt`
      return next()
    })
    owned.agent.followup(attachmentAdmissionInput())
    await owned.agent.whenIdle()
    remove()

    const events = (await ctx.sessionQuery.readSession(sessionId)).events
    expectAdmissionRefusal(events)
    // No follow-up provider dispatch, and the durable input is not re-queued.
    expect(chat.requests).toHaveLength(1)
    expect(events.filter(event => event.type === 'user/message' && event.data.source.kind === 'user')).toHaveLength(2)
    expect(events.filter(event => event.type === 'request/admission')).toHaveLength(1)
    expect(checkpoints(events)).toHaveLength(0)
    expect(owned.agent.inbox.nextStep).toHaveLength(0)
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.95, maxTokens: 512,
    summarizationProvider: 'synthetic-summary', summarizationModel: 'summary' })
}, 60_000)

it('bounds preview rebuilds when a listener keeps replacing the Surface', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([textResponse('First reply.'), textResponse('Second reply.')])
    chat.windows.set('roomy', 4_000_000)
    ctx.llm.registerAdapter(['synthetic-chat'], chat)
    const sessionId = SessionId('hybrid-envelope-rebuild-limit')
    const owned = await ctx.agents.create({
      sessionId, agentOptions: { ...wide, maxTokens: 1_024 }, meta: { cwd: root },
    })
    // A legal Surface replacement on every preview keeps changing the Surface, so
    // the loop must stop after its bounded previews instead of admitting forever.
    let replacements = 0
    await send(owned.agent, 'warm the surface')
    const remove = owned.agent.ctx.on('agent/request-envelope', ({ agent, phase }) => {
      if (phase !== 'preview') return
      const nodes = [...agent.session.surface.nodes].slice(1)
      const start = nodes.at(0)
      const end = nodes.at(-1)
      if (start === undefined || end === undefined) return
      const id = CompactionId(`rebuild-${replacements}`)
      const started = agent.session.append('compaction/start', { compactionId: id, turn: null })
      const summary = agent.session.append('compaction/summary', {
        compactionId: id, summary: [{ type: 'text', text: 'Rebuild checkpoint.' }],
        shadowedRange: { start, end }, shadowedSeqs: nodes, shadowedTokenCount: 0,
        provider: 'fixture', model: 'fixture',
      })
      agent.session.append('user/message', createUserMessage({
        content: [{ type: 'text', text: 'Rebuild checkpoint.' }], source: compactCheckpointSource(id),
      }), {
        surfaceOp: { op: 'replace', startSeq: start, endSeq: end },
        sourceEventSeqs: [started.seq, summary.seq, ...nodes],
      })
      replacements += 1
    })
    await send(owned.agent, 'REBUILD_LIMIT_INPUT')
    remove()
    const events = (await ctx.sessionQuery.readSession(sessionId)).events
    expectAdmissionRefusal(events)
    // Bounded previews: the initial admission plus MAX_ENVELOPE_REBUILDS.
    expect(replacements).toBe(3)
    // Only the warm-up turn reached the provider.
    expect(chat.requests).toHaveLength(1)
    expectPendingInput(owned.agent, 'REBUILD_LIMIT_INPUT')
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.95, maxTokens: 1_024 })
}, 60_000)

class FallbackAdapter extends WindowAdapter {
  override async resolveModel(provider: string, model: string) {
    const info = await super.resolveModel(provider, model)
    return { ...info, transportMaxTokens: info.context.contextWindow }
  }
}

/** A window-sized transport fallback cap must not refuse an ordinary request. */
it('admits a request whose transport fallback equals the whole window', async () => {
  await withContext(async (ctx, root) => {
    const chat = new FallbackAdapter([textResponse('Fallback reply.')])
    chat.windows.set('roomy', 4_096)
    ctx.llm.registerAdapter(['synthetic-chat'], chat)
    const sessionId = SessionId('hybrid-envelope-window-fallback')
    const owned = await ctx.agents.create({
      sessionId,
      agentOptions: { provider: 'synthetic-chat', model: 'roomy' },
      meta: { cwd: root },
    })
    await send(owned.agent, 'FALLBACK_INPUT')
    const events = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(events.findLast(event => event.type === 'turn/end')?.data.reason.kind).toBe('completed')
    expect(chat.requests).toHaveLength(1)
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.95 })
}, 60_000)

it.each([
  { name: 'full-window input with a positive transport fallback', window: 4_096, input: 4_096, output: undefined },
  { name: 'the 5,136-token excess from the explicit completion reserve', window: 1_048_576, input: 797_712, output: 256_000 },
])('refuses $name before dispatch', async ({ window, input, output }) => {
  await withContext(async (ctx, root) => {
    const Adapter = output === undefined ? FallbackAdapter : WindowAdapter
    const chat = new Adapter([textResponse('Must not dispatch.')])
    chat.windows.set('roomy', window)
    ctx.llm.registerAdapter(['synthetic-chat'], chat)
    const sessionId = SessionId('hybrid-envelope-no-fallback-room')
    const owned = await ctx.agents.create({
      sessionId,
      agentOptions: { provider: 'synthetic-chat', model: 'roomy', ...output === undefined ? {} : { maxTokens: output } },
      meta: { cwd: root },
    })
    const estimate = vi.spyOn(ctx.tokenMeter, 'estimateRequest').mockReturnValue(input)
    try {
      await send(owned.agent, 'FULL_WINDOW_INPUT')
      const { events } = await ctx.sessionQuery.readSession(sessionId)
      expectAdmissionRefusal(events)
      expect(chat.requests).toHaveLength(0)
      expect(events.find(event => event.type === 'request/admission')).toBeUndefined()
    } finally {
      estimate.mockRestore()
      await owned.dispose()
    }
  }, { auto: true, thresholdRatio: 0.95 })
}, 60_000)

/** One attempt prices through one materialized preview, however many listeners read it. */
it('shares one materialized preview across reads within an attempt', async () => {
  await withContext(async (ctx, root) => {
    const chat = new WindowAdapter([textResponse('Preview identity reply.')])
    chat.windows.set('roomy', 4_000_000)
    ctx.llm.registerAdapter(['synthetic-chat'], chat)
    const sessionId = SessionId('hybrid-envelope-preview-identity')
    const owned = await ctx.agents.create({
      sessionId, agentOptions: { ...wide, maxTokens: 1_024 }, meta: { cwd: root },
    })
    let previewsRead = 0
    let sharedAcrossReads = true
    const remove = owned.agent.ctx.on('agent/request-envelope', (envelope) => {
      if (envelope.phase !== 'preview') return
      const first = envelope.view()
      sharedAcrossReads &&= envelope.view() === first
      previewsRead += 1
    })
    await send(owned.agent, 'PREVIEW_IDENTITY_INPUT')
    remove()
    const events = (await ctx.sessionQuery.readSession(sessionId)).events
    expect(events.findLast(event => event.type === 'turn/end')?.data.reason.kind).toBe('completed')
    expect(previewsRead).toBe(1)
    expect(sharedAcrossReads).toBe(true)
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.95, maxTokens: 1_024 })
}, 60_000)

const ADMISSION_IMAGE_HANDLE = 'Synthetic image handle'
const ADMISSION_VISUAL_TOKENS = 1_200
const admissionFile = {
  attachmentId: AttachmentId(`sha256:${'ab'.repeat(32)}`),
  name: 'conference-notes-and-attachments.txt',
  bytes: 1_750_000,
}
const admissionImage = {
  attachmentId: AttachmentId(`sha256:${'cd'.repeat(32)}`),
  name: 'conference-diagram.png',
  mediaType: 'image/png',
  bytes: 8_000,
  width: 800,
  height: 600,
} as const

class AdmissionAdapter extends WindowAdapter {
  constructor(
    private readonly modality: 'vision' | 'text',
    private readonly visualTokens = ADMISSION_VISUAL_TOKENS,
  ) {
    super([textResponse('Admission reply.')])
  }

  override async resolveModel(provider: string, model: string) {
    const info = await super.resolveModel(provider, model)
    const inputModalities: readonly ModelModality[] = this.modality === 'vision' ? ['text', 'image'] : ['text']
    return { ...info, inputModalities }
  }

  override imageRequestPricing(): LlmImageRequestPricing {
    return {
      priceImages: images => images.map(() => ({
        visualTokens: this.visualTokens, text: ADMISSION_IMAGE_HANDLE,
      })),
    }
  }
}

function attachmentAdmissionInput() {
  return createUserMessage({
    content: [
      { type: 'text', text: 'Review the attached notes and diagram.' },
      { type: 'file', attachment: admissionFile },
      { type: 'image', attachment: admissionImage },
    ],
    source: { kind: 'user' },
  })
}

it.each(['vision', 'text'] as const)(
  'prices the exact projected %s adapter request and flushes its receipt before dispatch',
  async (modality) => {
    await withContext(async (ctx, root) => {
      const sessionId = SessionId(`hybrid-envelope-projected-${modality}`)
      const reader = new Context()
      try {
        await reader.plugin(JsonlSessionPersistence, { root: join(root, 'sessions'), compression: 'zstd' })
        let durableAtDispatch: readonly SessionEvent[] = []
        const chat = new class extends AdmissionAdapter {
          override async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
            const handle = await reader.sessionPersistence.open(sessionId, 'read')
            try {
              durableAtDispatch = (await handle.read()).events
            } finally {
              await handle.close()
            }
            yield* super.stream(options)
          }
        }(modality)
        ctx.llm.registerAdapter([wide.provider], chat)
        const owned = await ctx.agents.create({ sessionId, agentOptions: { ...wide, model: 'narrow' }, meta: { cwd: root } })
        const checked: GenerateOptions[] = []
        const remove = owned.agent.ctx.on('agent/request-envelope', (envelope) => {
          if (envelope.phase === 'verify') checked.push(envelope.request)
        })

        owned.agent.followup(attachmentAdmissionInput())
        await owned.agent.whenIdle()
        remove()
        const { events } = await ctx.sessionQuery.readSession(sessionId)
        expectCompletedTurn(events)
        expect(checked).toHaveLength(1)
        expect(chat.requests).toHaveLength(1)
        const [dispatched] = chat.requests
        if (dispatched === undefined) throw new Error('expected one adapter request')
        expect(checked[0]).toBe(dispatched)
        expect(Object.isFrozen(dispatched)).toBe(true)

        const imageProjection = modality === 'vision'
          ? { type: 'image', attachment: admissionImage }
          : { type: 'text', text: textOnlyImageText(admissionImage) }
        expect(dispatched.messages.findLast(message => message.role === 'user')?.content).toEqual([
          { type: 'text', text: 'Review the attached notes and diagram.' },
          { type: 'text', text: ctx.llm.fileRequestText(admissionFile) },
          imageProjection,
        ])

        const inputTokens = dispatched.messages.reduce((sum, message) => {
          if (modality === 'text') return sum + ctx.tokenMeter.estimateMessage(message)
          const content = message.content.map(block => block.type === 'image'
            ? { type: 'text' as const, text: ADMISSION_IMAGE_HANDLE }
            : block)
          const visual = message.content.filter(block => block.type === 'image').length * ADMISSION_VISUAL_TOKENS
          return sum + ctx.tokenMeter.estimateMessage({ ...message, content }) + visual
        }, ctx.tokenMeter.estimateTools(dispatched.tools))

        const receipt = durableAtDispatch.findLast(event => event.type === 'request/admission')
        expect(receipt).toMatchObject({
          type: 'request/admission',
          ignorable: true,
          data: {
            turn: 1, step: 1, provider: wide.provider, model: 'narrow', contextWindow: 4_096,
            inputTokens, completionMaxTokens: 64, completionSource: 'explicit',
          },
        })
        expect(events.findLast(event => event.type === 'request/admission')).toEqual(receipt)
        await owned.dispose()
      } finally {
        await reader.fiber.dispose()
      }
    }, { auto: true, thresholdRatio: 0.95 })
  }, 60_000,
)

it('refuses a vision input whose projected image alone exceeds the destination window', async () => {
  await withContext(async (ctx, root) => {
    const chat = new AdmissionAdapter('vision', 4_096)
    ctx.llm.registerAdapter([wide.provider], chat)
    const sessionId = SessionId('hybrid-envelope-visual-overflow')
    const owned = await ctx.agents.create({ sessionId, agentOptions: { ...wide, model: 'narrow' }, meta: { cwd: root } })
    owned.agent.followup(attachmentAdmissionInput())
    await owned.agent.whenIdle()

    const { events } = await ctx.sessionQuery.readSession(sessionId)
    expectAdmissionRefusal(events)
    expect(chat.requests).toHaveLength(0)
    expect(events.find(event => event.type === 'request/admission')).toBeUndefined()
    await owned.dispose()
  }, { auto: true, thresholdRatio: 0.95 })
}, 60_000)

it('withholds the adapter request when the accepted admission receipt cannot flush', async () => {
  await withContext(async (ctx, root) => {
    const chat = new AdmissionAdapter('text')
    ctx.llm.registerAdapter([wide.provider], chat)
    const sessionId = SessionId('hybrid-envelope-receipt-flush-failure')
    const owned = await ctx.agents.create({ sessionId, agentOptions: { ...wide, model: 'narrow' }, meta: { cwd: root } })
    const flush = ctx.sessions.flush.bind(ctx.sessions)
    let receiptFlushRefused = false
    const requestErrors = watchRequestErrors(owned.agent)
    const spy = vi.spyOn(ctx.sessions, 'flush').mockImplementation(async (session) => {
      if (!receiptFlushRefused && session.ownEvents().some(event => event.type === 'request/admission')) {
        receiptFlushRefused = true
        throw new Error('synthetic admission receipt flush failure')
      }
      return flush(session)
    })
    try {
      await send(owned.agent, 'Continue with the admitted request.')
      const { events } = await ctx.sessionQuery.readSession(sessionId)
      expect(receiptFlushRefused).toBe(true)
      const reason = events.findLast(event => event.type === 'turn/end')?.data.reason
      expect(reason?.kind).toBe('error')
      expect(reason?.kind === 'error' ? reason.error.message : undefined)
        .toContain('synthetic admission receipt flush failure')
      expect(chat.requests).toHaveLength(0)
      expect(requestErrors.errors).not.toHaveBeenCalled()
    } finally {
      requestErrors.dispose()
      spy.mockRestore()
      await owned.dispose()
    }
  }, { auto: true, thresholdRatio: 0.95 })
}, 60_000)

it.each(['removed', 'replaced'] as const)(
  'refuses an attached Session after its persisted writer is %s',
  async (mode) => {
    await withContext(async (ctx, root, disposePersistence) => {
      const chat = new WindowAdapter([textResponse('Stored warm-up.'), textResponse('Unexpected dispatch.')])
      ctx.llm.registerAdapter([wide.provider], chat)
      const sessionId = SessionId(`hybrid-envelope-writer-${mode}`)
      const owned = await ctx.agents.create({ sessionId, agentOptions: wide, meta: { cwd: root } })
      const { agent } = owned
      const original = ctx.sessionPersistence
      let observedFlushes = 0
      const removeObserver = ctx.on('session/flush', (session) => {
        if (session === agent.session) observedFlushes += 1
      })
      try {
        await send(agent, 'Write a durable warm-up turn.')
        expectCompletedTurn(agent.session.ownEvents())
        expect(chat.requests).toHaveLength(1)
        expect(observedFlushes).toBeGreaterThan(0)
        const reader = await original.open(sessionId, 'read')
        try {
          expect((await reader.read()).events).toContainEqual(expect.objectContaining({ type: 'request/admission' }))
        } finally {
          await reader.close()
        }

        await disposePersistence()
        expect(ctx.get('sessionPersistence')).toBeUndefined()
        expect(ctx.agents.get(sessionId)).toBe(agent)
        expect(ctx.sessions.get(sessionId)).toBe(agent.session)

        if (mode === 'replaced') {
          await ctx.plugin(JsonlSessionPersistence, { root: join(root, 'sessions'), compression: 'zstd' })
          expect(ctx.sessionPersistence).not.toBe(original)
        }
        const replacementWriter = mode === 'replaced' ? await ctx.sessionPersistence.open(sessionId, 'write') : undefined
        try {
          await send(agent, 'Only dispatch if the original writer acknowledges admission.')
          expectAdmissionRefusal(agent.session.ownEvents())
          expect(chat.requests).toHaveLength(1)
        } finally {
          await replacementWriter?.close()
        }
      } finally {
        removeObserver()
        await owned.dispose()
      }
    }, { auto: true, thresholdRatio: 0.95 })
  }, 60_000,
)

it('refuses a checked request after its Session surface changes during the receipt flush', async () => {
  await withContext(async (ctx, root) => {
    const chat = new AdmissionAdapter('text')
    ctx.llm.registerAdapter([wide.provider], chat)
    const sessionId = SessionId('hybrid-envelope-receipt-stale')
    const owned = await ctx.agents.create({ sessionId, agentOptions: { ...wide, model: 'narrow' }, meta: { cwd: root } })
    const pending = createUserMessage({
      content: [{ type: 'text', text: 'Continue once.' }], source: { kind: 'user' },
    })
    const concurrent = createUserMessage({
      content: [{ type: 'text', text: 'Concurrent surface change.' }], source: { kind: 'user' },
    })
    const flush = ctx.sessions.flush.bind(ctx.sessions)
    let changeInjected = false
    const requestErrors = watchRequestErrors(owned.agent)
    const spy = vi.spyOn(ctx.sessions, 'flush').mockImplementation(async (session) => {
      if (!changeInjected && session.ownEvents().some(event => event.type === 'request/admission')) {
        changeInjected = true
        await Promise.resolve()
        session.append('user/message', concurrent, { surfaceOp: 'append' })
      }
      return flush(session)
    })
    try {
      owned.agent.followup(pending)
      await owned.agent.whenIdle()
      const { events } = await ctx.sessionQuery.readSession(sessionId)
      expect(changeInjected).toBe(true)
      expectAdmissionRefusal(events)
      expect(chat.requests).toHaveLength(0)
      expect(requestErrors.errors).not.toHaveBeenCalled()
      const inputIds = events.flatMap(event => event.type === 'user/message' ? [event.data.id] : [])
      expect(inputIds.filter(id => id === pending.id)).toHaveLength(1)
      expect(inputIds).toContain(concurrent.id)
      expect(owned.agent.inbox.nextStep).toHaveLength(0)
    } finally {
      requestErrors.dispose()
      spy.mockRestore()
      await owned.dispose()
    }
  }, { auto: true, thresholdRatio: 0.95 })
}, 60_000)
