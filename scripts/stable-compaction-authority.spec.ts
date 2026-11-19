import { chmod, mkdtemp, readFile, readdir, rm, unlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import AgentRegistry from '@deepseek-ai/dsh-agent'
import TypertGatewayService from '@deepseek-ai/dsh-api-gateway'
import LocalAttachmentStore from '@deepseek-ai/dsh-attachment-local'
import { isCompactCheckpointSource } from '@deepseek-ai/dsh-compaction'
import BasicCompactionEngine from '@deepseek-ai/dsh-compaction-basic'
import LlmRuntime, { ToolCallId } from '@deepseek-ai/dsh-llm'
import SessionStore, { SessionId, type Session } from '@deepseek-ai/dsh-session'
import * as JsonlSessionPersistence from '@deepseek-ai/dsh-session-persistence-jsonl'
import SessionProjectionRegistry from '@deepseek-ai/dsh-session-projection'
import SqliteSessionQueryEngine from '@deepseek-ai/dsh-session-query-sqlite'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import TokenMeter from '@deepseek-ai/dsh-token-meter'
import * as NativeToolExecutor from '@deepseek-ai/dsh-tool-executor-native'
import * as ToolSessionQuery from '@deepseek-ai/dsh-tool-session-query'
import { executeToolFixture } from '@deepseek-ai/dsh-tool-testkit'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import type { PeerId } from '@deepseek-ai/dsh-typert-protocol'
import TypertRegistry from '@deepseek-ai/dsh-typert-registry'
import { expect, it } from 'vitest'
import { createSessionTestController } from '../packages/api/session-controller/tests/test-remote.ts'
import { appendFixtureTurn, fixtureAgent } from './stable-compaction-fixture.ts'

class FixtureCompaction extends BasicCompactionEngine {
  protected override async summarize() {
    return { summary: [{ type: 'text' as const, text: 'Synthetic checkpoint.' }], provider: 'fixture', model: 'fixture' }
  }
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
