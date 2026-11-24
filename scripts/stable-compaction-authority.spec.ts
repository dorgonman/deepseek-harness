import { chmod, mkdtemp, readFile, readdir, rm, truncate, unlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import AgentRegistry from '@deepseek-ai/dsh-agent'
import type { Agent } from '@deepseek-ai/dsh-agent'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import TypertGatewayService from '@deepseek-ai/dsh-api-gateway'
import LocalAttachmentStore from '@deepseek-ai/dsh-attachment-local'
import { isCompactCheckpointSource } from '@deepseek-ai/dsh-compaction'
import BasicCompactionEngine from '@deepseek-ai/dsh-compaction-basic'
import LlmRuntime, { createUserMessage, ToolCallId } from '@deepseek-ai/dsh-llm'
import SessionStore, { SessionId, type Session, type SessionEvent } from '@deepseek-ai/dsh-session'
import * as JsonlSessionPersistence from '@deepseek-ai/dsh-session-persistence-jsonl'
import SessionProjectionRegistry from '@deepseek-ai/dsh-session-projection'
import SqliteSessionQueryEngine from '@deepseek-ai/dsh-session-query-sqlite'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import TokenMeter from '@deepseek-ai/dsh-token-meter'
import * as NativeToolExecutor from '@deepseek-ai/dsh-tool-executor-native'
import * as ToolSessionQuery from '@deepseek-ai/dsh-tool-session-query'
import { defineContentToolFixture, executeToolFixture } from '@deepseek-ai/dsh-tool-testkit'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import type { PeerId } from '@deepseek-ai/dsh-typert-protocol'
import TypertRegistry from '@deepseek-ai/dsh-typert-registry'
import { expect, it } from 'vitest'
import { createSessionTestController } from '../packages/api/session-controller/tests/test-remote.ts'
import { MockAdapter, textResponse, toolCallResponse } from '../packages/core/agent-loop/tests/mock-adapter.ts'
import { eventLine, eventLines } from '../packages/session/session-persistence-jsonl/src/format.ts'
import { compressZstdFrame, scanZstdFrames } from '../packages/session/session-persistence-jsonl/src/zstd.ts'
import { appendFixtureTurn, fixtureAgent } from './stable-compaction-fixture.ts'

class FixtureCompaction extends BasicCompactionEngine {
  protected override async summarize() {
    return { summary: [{ type: 'text' as const, text: 'Synthetic checkpoint.' }], provider: 'fixture', model: 'fixture' }
  }
}

class ModelSwitchAdapter extends MockAdapter {
  override async resolveModel(provider: string, model: string) {
    return {
      ...await super.resolveModel(provider, model),
      context: { contextWindow: model === 'small' ? 4_096 : 128_000 },
    }
  }
}

async function mountBasicRecovery(
  ctx: Context,
  persistence: { root: string; compression: 'none' | 'zstd' },
  summarizer: ModelSwitchAdapter,
) {
  await mountAgentLoopTestDependencies(ctx)
  await ctx.plugin(JsonlSessionPersistence, persistence)
  await ctx.plugin(SqliteSessionQueryEngine, { path: ':memory:', openAt: 'never' })
  await ctx.plugin(TokenMeter)
  await ctx.plugin(BasicCompactionEngine, {
    auto: true,
    thresholdRatio: 0.5,
    retainTokens: 0,
    headroomTokens: 0,
    maxTokens: 64,
    summarizationProvider: 'fixture-summary',
    summarizationModel: 'wide',
    summarizationFallback: 'error',
  })
  await ctx.plugin(AgentLoop, { agents: [] })
  ctx.llm.registerAdapter(['fixture-summary'], summarizer)
}

async function sendRecoveryPrompt(agent: Agent, text: string) {
  agent.followup(createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } }))
  await agent.whenIdle()
}

declare module '@deepseek-ai/dsh-typert-protocol' {
  interface PrincipalMap {
    'test-cold-history': { readonly kind: 'test-cold-history'; readonly caller: SessionId }
  }
}

it('reopens a compacted synthetic Session with intact originals, links, lineage, and a separate search index', async () => {
  const root = await mkdtemp(join(tmpdir(), 'dsh-compaction-authority-'))
  const sessionsRoot = join(root, 'sessions')
  const indexPath = join(root, 'search.db')
  const author = new Context()
  const reader = new Context()
  const failingSearch = new Context()
  try {
    await author.plugin(SessionStore)
    await author.plugin(SessionProjectionRegistry)
    await author.plugin(LlmRuntime)
    await author.plugin(TokenMeter)
    await author.plugin(JsonlSessionPersistence, { root: sessionsRoot, compression: 'zstd' })
    await author.plugin(SqliteSessionQueryEngine, { path: indexPath, openAt: 'never' })
    const compact = new FixtureCompaction(author, { auto: false })
    const parent = author.sessions.create(SessionId('fixture-parent'), { meta: { cwd: root } })
    const session = author.sessions.create(SessionId('fixture-child'), { meta: { cwd: root, parentSession: parent.id } })
    await using parentWriter = await author.sessionPersistence.create(parent.header)
    await using writer = await author.sessionPersistence.create(session.header)
    const original = appendFixtureTurn(session, 1, 'original needles', 'original artifact '.repeat(16_000))
    session.append('turn/start', { turn: 2 })
    expect(await author.sessions.flush(parent)).toBe(true)
    expect(await author.sessions.flush(session)).toBe(true)
    const checkpoint = await compact.compactRegion(
      original.input.seq, original.followUp.seq, fixtureAgent(session), new AbortController().signal,
    )
    expect(session.surface.nodes).toHaveLength(1)
    const [checkpointSeq] = session.surface.nodes
    if (checkpointSeq === undefined) throw new Error('missing checkpoint')
    expect(session.deriveMessages()).toHaveLength(1)
    const originals = Object.values(original)
    for (const source of originals) {
      expect((await author.sessionQuery.readEvent({ sessionId: session.id, seq: source.seq })).target).toEqual(source)
    }
    session.append('turn/end', { turn: 2, reason: { kind: 'completed' } })
    expect(await author.sessions.flush(session)).toBe(true)
    await writer.close()
    await parentWriter.close()

    await reader.plugin(SessionStore)
    await reader.plugin(SessionProjectionRegistry)
    await reader.plugin(JsonlSessionPersistence, { root: sessionsRoot, compression: 'zstd' })
    await reader.plugin(SqliteSessionQueryEngine, { path: indexPath, openAt: 'first-search' })
    for (const source of originals) {
      expect((await reader.sessionQuery.readEvent({ sessionId: session.id, seq: source.seq })).target).toEqual(source)
    }
    const trace = await reader.sessionQuery.traceEvent({ sessionId: session.id, seq: checkpointSeq })
    expect(trace.sourceEventSeqs).toEqual(expect.arrayContaining([
      original.input.seq, original.assistant.seq, original.output.seq, original.followUp.seq,
    ]))
    expect(trace.session).toMatchObject({ id: session.id, parentSession: parent.id })
    const event = (await reader.sessionQuery.readEvent({ sessionId: session.id, seq: checkpointSeq })).target
    if (event.type !== 'user/message' || !isCompactCheckpointSource(event.data.source)) {
      throw new Error('missing typed checkpoint source')
    }
    expect(event.data.source.compactionId).toBe(checkpoint.compactionId)
    expect(await reader.sessionQuery.traceSession(session.id)).toMatchObject({
      complete: true, root: { header: { id: parent.id } },
    })
    expect(await readdir(root)).not.toContain('search.db')
    await reader.sessionQuery.searchEvents({ sessionId: session.id, query: 'needles' })
    expect(await readdir(root)).toContain('search.db')

    const dataFiles = (await readdir(sessionsRoot, { recursive: true })).filter(path => path.endsWith('.jsonl.zstd'))
    expect(dataFiles.length).toBeGreaterThan(0)
    const bytes = await Promise.all(dataFiles.map(path => readFile(join(sessionsRoot, path))))
    await failingSearch.plugin(SessionStore)
    await failingSearch.plugin(SessionProjectionRegistry)
    await failingSearch.plugin(JsonlSessionPersistence, { root: sessionsRoot, compression: 'zstd' })
    await failingSearch.plugin(SqliteSessionQueryEngine, { path: root, openAt: 'first-search' })
    await expect(failingSearch.sessionQuery.searchEvents({ sessionId: session.id, query: 'needles' }))
      .rejects.toMatchObject({ code: 'SESSION_QUERY_INDEX_FAILED' })
    expect((await failingSearch.sessionQuery.readEvent({ sessionId: session.id, seq: original.output.seq })).target)
      .toEqual(original.output)
    expect(await Promise.all(dataFiles.map(path => readFile(join(sessionsRoot, path))))).toEqual(bytes)
  } finally {
    await failingSearch.fiber.dispose()
    await reader.fiber.dispose()
    await author.fiber.dispose()
    await rm(root, { recursive: true, force: true })
  }
})

it('authorizes cold compacted event reads and recovers the retained image through the Session Gateway', async () => {
  const root = await mkdtemp(join(tmpdir(), 'dsh-compaction-authorized-fetch-'))
  const persistence = { root: join(root, 'sessions'), compression: 'zstd' } as const
  const dshHome = join(root, 'home')
  const workspace = join(root, 'workspace')
  const author = new Context()
  const reader = new Context()
  const signal = new AbortController().signal
  try {
    await author.plugin(SessionStore)
    await author.plugin(SessionProjectionRegistry)
    await author.plugin(LlmRuntime)
    await author.plugin(TokenMeter)
    await author.plugin(LocalAttachmentStore, { dshHome })
    await author.plugin(JsonlSessionPersistence, persistence)
    const bytes = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAACXBIWXMAAAPoAAAD6AG1e1JrAAAADElEQVQImWNgZGIGAAAOAAeCcsnOAAAAAElFTkSuQmCC',
      'base64',
    )
    const image = await author.attachments.saveImage({ data: bytes, mediaType: 'image/png' })
    const sessionId = SessionId('cold-source')
    const source = author.sessions.create(sessionId, { meta: { cwd: workspace } })
    const originalText = 'Historical image evidence.'
    const writer = await author.sessionPersistence.create(source.header)
    const original = appendFixtureTurn(source, 1, [
      { type: 'text', text: originalText.repeat(200) },
      { type: 'image', attachment: image },
    ], 'Synthetic result.')
    source.append('turn/start', { turn: 2 })
    expect(await author.sessions.flush(source)).toBe(true)
    const compact = new FixtureCompaction(author, { auto: false })
    const result = await compact.compactRegion(original.input.seq, original.followUp.seq, fixtureAgent(source), signal)
    expect(result.shadowedSeqs).toContain(original.input.seq)
    expect(source.surface.nodes).toHaveLength(1)
    expect(source.surface.nodes).not.toContain(original.input.seq)
    source.append('turn/end', { turn: 2, reason: { kind: 'completed' } })
    expect(await author.sessions.flush(source)).toBe(true)
    await writer.close()
    await author.fiber.dispose()
    const logFiles = (await readdir(persistence.root, { recursive: true })).filter(path => path.endsWith('.jsonl.zstd'))
    expect(logFiles).toHaveLength(1)
    const logPath = join(persistence.root, logFiles[0]!)
    const logBytes = await readFile(logPath)

    await reader.plugin(SessionStore)
    await reader.plugin(SessionProjectionRegistry)
    await reader.plugin(AgentRegistry)
    await reader.plugin(SystemPrompt)
    await reader.plugin(ToolRuntime)
    await reader.plugin(NativeToolExecutor)
    await reader.plugin(JsonlSessionPersistence, persistence)
    await reader.plugin(SqliteSessionQueryEngine, { path: ':memory:', openAt: 'never' })
    await reader.plugin(ToolSessionQuery)
    await reader.plugin(LocalAttachmentStore, { dshHome })
    await reader.plugin(TypertRegistry)
    await createSessionTestController(reader, {
      defaultModelSelection: () => ({ provider: 'fixture', model: 'fixture' }),
      cwd: workspace,
      nativeOpen: false,
    })
    await reader.plugin(TypertGatewayService)

    const sameWorkspace = reader.sessions.create(SessionId('same-workspace'), { meta: { cwd: workspace } })
    const wrongWorkspace = reader.sessions.create(SessionId('wrong-workspace'), { meta: { cwd: join(root, 'outside') } })
    const noWorkspace = reader.sessions.create(SessionId('no-workspace'))
    const fetchEvent = (caller: Session) => executeToolFixture(reader, {
      name: 'session_event_read',
      arguments: { session_id: sessionId, seq: original.input.seq },
      callId: ToolCallId(`fetch-${caller.id}`),
      signal,
      agent: fixtureAgent(caller),
    })
    expect(reader.sessions.get(sessionId)).toBeUndefined()
    for (const caller of [wrongWorkspace, noWorkspace]) {
      const denied = await fetchEvent(caller)
      expect(denied.isError && denied.error.info?.code).toBe('SESSION_QUERY_TOOL_UNAUTHORIZED')
      expect(JSON.stringify(denied.content)).not.toContain(image.attachmentId)
      expect(JSON.stringify(denied.content)).not.toContain(originalText)
    }

    const found = await fetchEvent(sameWorkspace)
    expect(found.isError).toBe(false)
    const [block] = found.content
    const json = block?.type === 'text' ? /\x60{3}json\n([\s\S]+?)\n\x60{3}/u.exec(block.text)?.[1] : undefined
    if (json === undefined) throw new Error('Expected a JSON event in the history tool result')
    const recovered: unknown = JSON.parse(json)
    expect(recovered).toEqual(original.input)
    const event = recovered as typeof original.input
    const recoveredImage = event.data.content.find(part => part.type === 'image')
    if (recoveredImage === undefined) throw new Error('Expected an image reference in the recovered event')

    reader.remoteAccess.registerPolicy('test-cold-history', {
      bindings: principal => principal.caller === sameWorkspace.id ? [{
        role: { name: 'test:cold-image-reader', resource: 'session', actions: ['session/attachment'] },
        resource: { kind: 'session', id: sessionId },
      }] : [],
    })
    const fetchAttachment = (caller: Session) => reader.typertGateway.invoke({
      peer: {
        id: `peer-${caller.id}` as PeerId,
        ctx: reader,
        principal: { kind: 'test-cold-history', caller: caller.id },
        interactions: 'answer',
        signal,
        dispose: async () => {},
      },
      namespace: 'session',
      method: 'attachment',
      args: { request: { sessionId, attachment: recoveredImage.attachment } },
    })
    await expect(fetchAttachment(wrongWorkspace)).rejects.toMatchObject({ code: 'remote-access/denied' })
    await expect(fetchAttachment(sameWorkspace)).resolves.toEqual({
      attachment: recoveredImage.attachment,
      data: bytes.toString('base64'),
    })

    const path = reader.attachments.imageHostPath(recoveredImage.attachment)
    if (path === undefined) throw new Error('The local attachment has no persisted object path')
    await chmod(path, 0o600)
    await writeFile(path, Buffer.alloc(bytes.length))
    await expect(fetchAttachment(sameWorkspace)).rejects.toMatchObject({
      code: 'session/attachment-invalid', details: { reason: 'ATTACHMENT_CORRUPT' },
    })
    await unlink(path)
    await expect(fetchAttachment(sameWorkspace)).rejects.toMatchObject({
      code: 'session/attachment-invalid', details: { reason: 'ATTACHMENT_NOT_FOUND' },
    })
    await expect(fetchEvent(sameWorkspace)).resolves.toEqual(found)
    expect(await readFile(logPath)).toEqual(logBytes)
    expect(reader.sessions.get(sessionId)).toBeUndefined()
  } finally {
    await reader.fiber.dispose()
    await author.fiber.dispose()
    await rm(root, { recursive: true, force: true })
  }
})

it('retains original tool history through repeated Basic compaction, a refused switch, and a cold retry', async () => {
  const root = await mkdtemp(join(tmpdir(), 'dsh-compaction-switch-recovery-'))
  const persistence = { root: join(root, 'sessions'), compression: 'zstd' } as const
  const sourceRoute = { provider: 'fixture-source', model: 'wide', maxTokens: 64 }
  const destinationRoute = { provider: 'fixture-destination', model: 'small' }
  const sessionId = SessionId('repeated-compaction')
  const resultText = 'Retained synthetic tool bytes — exact.\n'.repeat(1_000)
  const callIds = [1, 2, 3, 4].map(n => `recovery-call-${n}`)
  const sourceAdapter = new ModelSwitchAdapter(callIds.flatMap(id => [
    toolCallResponse(id, 'fixture_read', {}),
    textResponse('Tool read complete.'),
  ]))
  const failedSummarizer = new ModelSwitchAdapter([
    textResponse('First checkpoint.'),
    textResponse('Second checkpoint.'),
    textResponse('Third checkpoint.'),
    [{ type: 'finish', reason: { kind: 'error', failure: { code: 'AUTH', message: 'Synthetic summary refusal' } } }],
  ])
  const retrySummarizer = new ModelSwitchAdapter([textResponse('Recovered checkpoint.')])
  const destinationAdapter = new ModelSwitchAdapter([textResponse('Destination answered.')])
  const author = new Context()
  const reader = new Context()
  let toolExecutions = 0
  async function mount(ctx: Context, summarizer: ModelSwitchAdapter) {
    await mountBasicRecovery(ctx, persistence, summarizer)
    ctx.llm.registerAdapter([sourceRoute.provider], sourceAdapter)
    ctx.llm.registerAdapter([destinationRoute.provider], destinationAdapter)
    ctx.tools.register(defineContentToolFixture({
      name: 'fixture_read',
      description: 'Read the synthetic tool fixture.',
      parameters: {},
      async execute() {
        toolExecutions += 1
        return [{ type: 'text', text: resultText }]
      },
    }))
  }
  try {
    await mount(author, failedSummarizer)
    const owned = await author.agents.create({ sessionId, agentOptions: sourceRoute, meta: { cwd: root } })
    const { agent } = owned
    for (const id of callIds.slice(0, 3)) {
      await sendRecoveryPrompt(agent, `Read ${id}.`)
      expect(await author.compaction.compactNow(agent, new AbortController().signal)).not.toBeNull()
    }
    await sendRecoveryPrompt(agent, 'Read one more original before switching.')
    const before = (await author.sessionQuery.readSession(sessionId)).events
    const calls = before.filter(event => event.type === 'tool/call')
    const results = before.filter(event => event.type === 'tool/result')
    const checkpoints = before.filter(event => event.type === 'user/message' && isCompactCheckpointSource(event.data.source))
    expect(checkpoints).toHaveLength(3)
    expect(calls.map(event => event.data.callId)).toEqual(callIds)
    expect(results.map(event => event.data.message.toolCallId)).toEqual(callIds)
    for (const result of results) {
      expect(result.data.message).toMatchObject({ content: [{ type: 'text', text: resultText }], isError: false })
    }
    const committedHeader = agent.session.requestHeader()
    expect(committedHeader?.config).toMatchObject(sourceRoute)
    const priorSurface = [...agent.session.surface.nodes]

    agent.model.select(destinationRoute)
    await sendRecoveryPrompt(agent, 'Try the smaller route.')
    const failed = (await author.sessionQuery.readSession(sessionId)).events
    expect(failed.slice(0, before.length)).toEqual(before)
    const attempt = failed.slice(before.length)
    expect(attempt.filter(event => event.type.startsWith('compaction/')).map(event => event.type))
      .toEqual(['compaction/start', 'compaction/end'])
    expect(attempt.at(-1)).toMatchObject({
      type: 'turn/end', data: { reason: { kind: 'error', error: { code: 'MODEL_SWITCH_COMPACTION_FAILED' } } },
    })
    expect(agent.session.surface.nodes).toEqual(priorSurface)
    expect(agent.session.requestHeader()).toEqual(committedHeader)
    expect(failedSummarizer.requests).toHaveLength(4)
    expect(destinationAdapter.requests).toHaveLength(0)
    expect(toolExecutions).toBe(4)
    await author.sessions.flush(agent.session)
    await owned.dispose()
    expect(author.sessions.get(sessionId)).toBeUndefined()

    await mount(reader, retrySummarizer)
    expect(reader.sessions.get(sessionId)).toBeUndefined()
    expect((await reader.sessionQuery.readSession(sessionId)).events).toEqual(failed)
    const oldestResult = results[0]!
    const trace = await reader.sessionQuery.traceEvent({ sessionId, seq: oldestResult.seq })
    expect(trace.replacementChain).toEqual(checkpoints.map(event => event.seq))
    expect(reader.sessions.get(sessionId)).toBeUndefined()
    const resumed = await reader.agents.resume({ resumeSessionId: sessionId, agentOptions: sourceRoute })
    expect(resumed.agent.session.requestHeader()).toEqual(committedHeader)
    expect(resumed.agent.model.current()).toMatchObject(destinationRoute)
    await sendRecoveryPrompt(resumed.agent, 'Retry the smaller route from the stored history.')
    expect(retrySummarizer.requests).toHaveLength(1)
    expect(destinationAdapter.requests).toHaveLength(1)
    expect(toolExecutions).toBe(4)
    expect(resumed.agent.session.requestHeader()?.config).toMatchObject(destinationRoute)
    await reader.sessions.flush(resumed.agent.session)
    await resumed.dispose()
    expect(reader.sessions.get(sessionId)).toBeUndefined()
    const recovered = (await reader.sessionQuery.readSession(sessionId)).events
    expect(recovered.slice(0, failed.length)).toEqual(failed)
    expect(recovered.filter(event => event.type === 'compaction/summary')).toHaveLength(4)
    expect(recovered.at(-1)).toMatchObject({ type: 'turn/end', data: { reason: { kind: 'completed' } } })
  } finally {
    await reader.fiber.dispose()
    await author.fiber.dispose()
    await rm(root, { recursive: true, force: true })
  }
})

async function tearPlainRecord(logPath: string, cut: SessionEvent) {
  const rows = (await readFile(logPath, 'utf8')).split('\n')
  const cutIndex = rows.findIndex(row => row.includes(`"seq":${cut.seq},`) && row.includes(`"type":"${cut.type}"`))
  if (cutIndex < 0) throw new Error('The omitted record was not written into the fixture log')
  const committed = Buffer.from(rows.slice(0, cutIndex).join('\n') + '\n')
  const fragment = Buffer.from(rows[cutIndex]!).subarray(0, 16)
  await truncate(logPath, committed.length + fragment.length)
  return { committed, torn: Buffer.concat([committed, fragment]) }
}

async function tearCompressedFrame(
  logPath: string, committed: Buffer, completeRecords: readonly SessionEvent[], cut: SessionEvent,
) {
  // Complete transaction records share a final frame with the incomplete next row.
  const frame = await compressZstdFrame(eventLines(completeRecords) + '\n' + eventLine(cut).slice(0, 16))
  const torn = Buffer.concat([committed, frame.subarray(0, -1)])
  expect(scanZstdFrames(torn).tornStart).toBe(committed.length)
  await writeFile(logPath, torn)
  return { committed, torn }
}

it.each([
  { compression: 'none', boundary: 'replacement', replacementStored: false },
  { compression: 'none', boundary: 'close marker', replacementStored: true },
  { compression: 'zstd', boundary: 'replacement', replacementStored: false },
  { compression: 'zstd', boundary: 'close marker', replacementStored: true },
] as const)('retains raw history and retries after a torn Basic $boundary ($compression)', async ({
  compression, replacementStored,
}) => {
  const root = await mkdtemp(join(tmpdir(), 'dsh-compaction-torn-tail-'))
  const persistence = { root: join(root, 'sessions'), compression }
  const route = { provider: 'fixture-chat', model: 'wide', maxTokens: 64 }
  const sessionId = SessionId('torn-compaction')
  const signal = new AbortController().signal
  const author = new Context()
  const reader = new Context()
  try {
    const firstSummary = new ModelSwitchAdapter([textResponse('Detailed checkpoint of the first tool result and its source.')])
    await mountBasicRecovery(author, persistence, firstSummary)
    author.llm.registerAdapter([route.provider], new ModelSwitchAdapter([
      toolCallResponse('torn-tool-call', 'fixture_read', {}),
      textResponse('Tool read complete.'),
    ]))
    const resultText = 'Exact synthetic bytes from the original tool result.\n'.repeat(600)
    author.tools.register(defineContentToolFixture({
      name: 'fixture_read',
      description: 'Read the synthetic tool fixture.',
      parameters: {},
      async execute() { return [{ type: 'text', text: resultText }] },
    }))
    const owned = await author.agents.create({ sessionId, agentOptions: route, meta: { cwd: root } })
    const { agent } = owned
    await sendRecoveryPrompt(agent, 'Read the original tool result before compaction.')
    const rawSurface = [...agent.session.surface.nodes]
    const original = await author.sessionQuery.readSession(sessionId)
    const originalResult = original.events.find(event => event.type === 'tool/result')
    expect(originalResult?.data.message).toMatchObject({
      toolCallId: 'torn-tool-call', isError: false, content: [{ type: 'text', text: resultText }],
    })
    expect(await author.sessions.flush(agent.session)).toBe(true)
    const extension = compression === 'zstd' ? '.jsonl.zstd' : '.jsonl'
    const logs = (await readdir(persistence.root, { recursive: true })).filter(path => path.endsWith(extension))
    expect(logs).toHaveLength(1)
    const logPath = join(persistence.root, logs[0]!)
    const preCompactionFrames = compression === 'zstd' ? await readFile(logPath) : undefined
    const first = await author.compaction.compactNow(agent, signal)
    const compacted = await author.sessionQuery.readSession(sessionId)
    const replacement = compacted.events.find(event => event.type === 'user/message' && isCompactCheckpointSource(event.data.source))
    const end = compacted.events.find(event => event.type === 'compaction/end')
    if (first === null || originalResult === undefined || replacement === undefined || end === undefined) {
      throw new Error('The synthetic writer did not produce a complete compaction')
    }
    expect(end.data.compactionId).toBe(first.compactionId)
    const replayedSurface = replacementStored ? [...agent.session.surface.nodes] : rawSurface
    await owned.dispose()

    const cut = replacementStored ? end : replacement
    const prefix = compacted.events.filter(event => event.seq < cut.seq)
    const { committed, torn } = preCompactionFrames === undefined
      ? await tearPlainRecord(logPath, cut)
      : await tearCompressedFrame(logPath, preCompactionFrames, prefix.slice(original.events.length), cut)

    const retrySummary = new ModelSwitchAdapter([textResponse('Recovered.')])
    await mountBasicRecovery(reader, persistence, retrySummary)
    expect((await reader.sessionQuery.readSession(sessionId)).events).toEqual(prefix)
    expect(await readFile(logPath)).toEqual(torn)
    const trace = await reader.sessionQuery.traceEvent({ sessionId, seq: originalResult.seq })
    expect(trace.replacementChain).toEqual(replacementStored ? [replacement.seq] : [])
    expect(reader.sessions.get(sessionId)).toBeUndefined()

    const resumed = await reader.agents.resume({ resumeSessionId: sessionId, agentOptions: route })
    expect(resumed.agent.session.surface.nodes).toEqual(replayedSurface)
    expect(resumed.agent.session.requestHeader()?.config).toMatchObject(route)
    expect(retrySummary.requests).toHaveLength(0)
    const retry = await reader.compaction.compactNow(resumed.agent, signal)
    expect(retry).not.toBeNull()
    expect(retry?.compactionId).not.toBe(first.compactionId)
    expect(retrySummary.requests).toHaveLength(1)
    await resumed.dispose()
    const recovered = (await reader.sessionQuery.readSession(sessionId)).events
    expect(recovered.slice(0, prefix.length)).toEqual(prefix)
    expect(recovered.filter(event => event.type === 'session/end-seed' && event.seq >= cut.seq)).toHaveLength(1)
    expect(recovered.filter(event => event.type === 'compaction/end').map(event => event.data.compactionId))
      .toEqual([retry?.compactionId])
    expect((await reader.sessionQuery.readEvent({ sessionId, seq: originalResult.seq })).target).toEqual(originalResult)
    const repaired = await readFile(logPath)
    expect(repaired.subarray(0, committed.length)).toEqual(committed)
    if (compression === 'zstd') expect(scanZstdFrames(repaired).tornStart).toBeUndefined()
  } finally {
    await reader.fiber.dispose()
    await author.fiber.dispose()
    await rm(root, { recursive: true, force: true })
  }
})
