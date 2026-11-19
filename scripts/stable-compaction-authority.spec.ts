import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import { isCompactCheckpointSource } from '@deepseek-ai/dsh-compaction'
import BasicCompactionEngine from '@deepseek-ai/dsh-compaction-basic'
import LlmRuntime from '@deepseek-ai/dsh-llm'
import SessionStore, { SessionId } from '@deepseek-ai/dsh-session'
import * as JsonlSessionPersistence from '@deepseek-ai/dsh-session-persistence-jsonl'
import SessionProjectionRegistry from '@deepseek-ai/dsh-session-projection'
import SqliteSessionQueryEngine from '@deepseek-ai/dsh-session-query-sqlite'
import TokenMeter from '@deepseek-ai/dsh-token-meter'
import { expect, it } from 'vitest'
import { appendFixtureTurn, fixtureAgent } from './stable-compaction-fixture.ts'

class FixtureCompaction extends BasicCompactionEngine {
  protected override async summarize() {
    return { summary: [{ type: 'text' as const, text: 'Synthetic checkpoint.' }], provider: 'fixture', model: 'fixture' }
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
