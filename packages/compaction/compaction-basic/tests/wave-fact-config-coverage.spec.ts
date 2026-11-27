import { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { CompactionIncrementalCoverage, CompactionResidentSelection } from '@deepseek-ai/dsh-compaction'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { Session, SessionId, SessionSeq } from '@deepseek-ai/dsh-session'
import SessionProjectionRegistry from '@deepseek-ai/dsh-session-projection'
import TokenMeter from '@deepseek-ai/dsh-token-meter'
import { describe, expect, it, vi } from 'vitest'
import { resolveConfig, resolveSummaryMaxTokens, resolveTargetPolicy } from '../src/config.ts'
import { parseFactExtraction } from '../src/fact-extraction.ts'
import { composeCompactView, createFacts, mergeFacts, validateLedger } from '../src/fact-ledger.ts'
import type { FactInput, FactRecord } from '../src/fact-ledger.ts'
import { digestCanonical, digestRange } from '../src/incremental.ts'
import { compactSurfaceRegion, type CompactionChunkCall } from '../src/region.ts'
import type { BasicCompactionConfig } from '../src/types.ts'

const resident = {
  summarizationMode: 'resident-preferred',
  residentGatewayId: 'fixture-gateway',
  residentProvider: 'fixture-provider',
  residentPinnedNodeId: 'node-a',
} satisfies BasicCompactionConfig

describe('configuration accepted from the Loader', () => {
  it.each([
    ['a string in incremental', { incremental: 'false' }, /incremental must be a boolean/],
    ['a string in factLedger', { factLedger: 'false' }, /factLedger must be a boolean/],
    ['an unknown mode', { summarizationMode: 'unrecognized' }, /summarizationMode must be fixed or resident-preferred/],
    ['a blank provider', { residentProvider: ' ' }, /residentProvider is required/],
    ['a blank pinned node', { residentPinnedNodeId: ' ' }, /residentPinnedNodeId must be a non-empty string/],
    ['a blank allowed node', { residentAllowedNodeIds: ['node-a', ' '] }, /residentAllowedNodeIds entries must be non-empty/],
    ['duplicate variants', { residentPreferredVariantIds: ['variant-a', 'variant-a'] }, /residentPreferredVariantIds repeats variant-a/],
  ])('rejects %s', (_label, input, message) => {
    expect(() => resolveConfig({ ...resident, ...input } as BasicCompactionConfig)).toThrow(message)
  })

  it('detaches the resident observation route records from Loader input', () => {
    const route = {
      id: 'route-a',
      nodeId: 'node-a',
      nodeName: 'node',
      instance: 'mservice/default',
      logicalModel: 'fixture-model',
      baseUrl: 'http://127.0.0.1:10001',
    }
    const input = {
      ...resident,
      residentAllowedNodeIds: ['node-a', 'node-b'],
      residentPreferredVariantIds: ['fixture-variant'],
      residentObservationRoutes: [route],
    }
    const resolved = resolveConfig(input)

    route.baseUrl = 'http://127.0.0.1:10002'
    expect(resolved.residentAllowedNodeIds).toEqual(['node-a', 'node-b'])
    expect(resolved.residentPreferredVariantIds).toEqual(['fixture-variant'])
    expect(resolved.residentObservationRoutes).toEqual([{ ...route, baseUrl: 'http://127.0.0.1:10001' }])
    expect(Object.isFrozen(resolved.residentObservationRoutes?.[0])).toBe(true)
  })

  it('uses the remaining conversation budget when the caller omitted the summary cap', () => {
    const target = { provider: 'fixture-provider', model: 'fixture-model' }
    const defaultPolicy = resolveTargetPolicy(resolveConfig(), target)
    const explicitPolicy = resolveTargetPolicy(resolveConfig({ maxTokens: 512 }), target)

    expect(resolveSummaryMaxTokens(defaultPolicy, 1024)).toBe(128)
    expect(resolveSummaryMaxTokens(explicitPolicy, 1024)).toBe(512)
    expect(resolveSummaryMaxTokens(defaultPolicy, 0)).toBe(defaultPolicy.maxTokens)
  })
})

describe('structured fact extraction', () => {
  it('rejects a disabled extraction budget before parsing the response', () => {
    expect(() => parseFactExtraction('invalid JSON', [], { maxFacts: 0, maxClaimChars: 20 }))
      .toThrow(expect.objectContaining({ code: 'bounds', message: 'fact extraction: bounds must be positive integers' }))
  })

  it('rejects a null item in a valid JSON array', () => {
    expect(() => parseFactExtraction('[null]', [SessionSeq(0)], { maxFacts: 1, maxClaimChars: 20 }))
      .toThrow(expect.objectContaining({ code: 'malformed', message: 'fact extraction: claim 0 is not an object' }))
  })
})

function appendMessage(session: Session, text: string): SessionSeq {
  return session.append('user/message', createUserMessage({
    content: [{ type: 'text', text }],
    source: { kind: 'user' },
  }), { surfaceOp: 'append' }).seq
}

function factFixture(sessionId: string) {
  const session = Session.create(SessionId(sessionId))
  const source = appendMessage(session, 'Use the fixture model.')
  const events = session.snapshotEvents()
  const input: FactInput = {
    kind: 'decision',
    key: 'summary-model',
    claim: 'Use the fixture model.',
    sourceEventIds: [source],
    basis: 'asserted',
  }
  return { events, input }
}

describe('Fact Ledger validation', () => {
  it.each([
    ['an unknown kind', { kind: 'opinion' }, /unknown fact kind opinion/],
    ['an unknown basis', { basis: 'guess' }, /basis must be asserted or inferred/],
    ['omitted citations', { sourceEventIds: [] }, /at least one source event/],
  ])('rejects %s in facts crossing the extractor boundary', (_label, override, message) => {
    const { events, input } = factFixture('extractor-input')
    expect(() => createFacts([{ ...input, ...override } as FactInput], events)).toThrow(message)
  })

  it.each([
    ['empty fact id', { factId: '' }, /fact id is missing/],
    ['unknown kind', { kind: 'opinion' }, /unknown fact kind opinion/],
    ['empty claim', { claim: ' ' }, /fact key or claim is empty/],
    ['no sources', { sourceEventIds: [] }, /fact has no source event/],
    ['duplicate sources', { sourceEventIds: [SessionSeq(0), SessionSeq(0)] }, /fact source ids are not unique/],
    ['unknown source', { sourceEventIds: [SessionSeq(999)] }, /source event 999 is outside the log/],
    ['missing digest', { evidenceDigest: '' }, /fact evidence digest is missing/],
    ['incorrect seen bounds', { lastSeenSeq: SessionSeq(999) }, /fact seen bounds do not match its sources/],
    ['unknown status', { status: 'pending' }, /fact status is invalid/],
    ['unknown basis', { basis: 'guess' }, /fact basis is invalid/],
    ['blank branch', { branch: ' ' }, /fact branch is empty/],
    ['invalid lineage list', { supersedes: [''] }, /fact supersedes list is invalid/],
    ['dangling supersede', { supersedes: ['missing-fact'] }, /supersedes unknown missing-fact/],
  ])('refuses a durable snapshot with %s', (_label, override, message) => {
    const { events, input } = factFixture('snapshot-record')
    const [fact] = createFacts([input], events)
    expect(() => { validateLedger([{ ...fact!, ...override } as FactRecord], events) }).toThrow(message)
  })

  it('refuses duplicate records in a durable snapshot', () => {
    const { events, input } = factFixture('snapshot-duplicate')
    const [fact] = createFacts([input], events)
    expect(() => { validateLedger([fact!, fact!], events) }).toThrow(/duplicate fact id/)
  })

  it('requires a named replacer to cite the replaced fact', () => {
    const { events, input } = factFixture('snapshot-lineage')
    const [older, newer] = createFacts([input, { ...input, claim: 'Use the alternate model.' }], events)
    const conflicting = { ...older!, status: 'superseded', supersededBy: newer!.factId } satisfies FactRecord

    expect(() => { validateLedger([conflicting, newer!], events) }).toThrow(/supersededBy lineage is inconsistent/)
    const merged = mergeFacts([older!], [{ ...newer!, supersedes: [older!.factId] }], events)
    expect(() => { validateLedger(merged, events) }).not.toThrow()
    expect(merged).toContainEqual(conflicting)
  })

  it('combines explicit lineage when the same fact is extracted again', () => {
    const { events, input } = factFixture('merge-lineage')
    const [older, alternate, newest] = createFacts([
      input,
      { ...input, claim: 'Use the alternate model.' },
      { ...input, claim: 'Use the replacement model.' },
    ], events)
    const merged = mergeFacts(
      mergeFacts([older!, alternate!], [{ ...newest!, supersedes: [older!.factId] }], events),
      [{ ...newest!, supersedes: [alternate!.factId] }],
      events,
    )

    expect(merged.find(fact => fact.factId === newest!.factId)?.supersedes).toEqual([
      older!.factId, alternate!.factId,
    ])
    expect(merged.filter(fact => fact.status === 'superseded').map(fact => fact.supersededBy))
      .toEqual([newest!.factId, newest!.factId])
    expect(() => { validateLedger(merged, events) }).not.toThrow()
  })

  it('orders separate subjects independently of their input order', () => {
    const { events, input } = factFixture('merge-order')
    const facts = createFacts([
      { ...input, key: 'enrollment' },
      { ...input, key: 'summary-model' },
    ], events)

    for (const records of [facts, [...facts].reverse()]) {
      expect(mergeFacts([], records, events).map(fact => fact.key)).toEqual(['enrollment', 'summary-model'])
    }
  })

  it('prices disputed claims in stable identity order regardless of input order', () => {
    const { events, input } = factFixture('disputed-order')
    const disputed = mergeFacts([], createFacts([input, { ...input, claim: 'Use the alternate model.' }], events), events)
    const ids = disputed.map(fact => fact.factId).sort()
    const options = { maxTokens: 1, tokensOf: () => 1 }

    for (const claims of [disputed, [...disputed].reverse()]) {
      const view = composeCompactView(claims, options)
      expect(view.facts.map(fact => fact.factId)).toEqual(ids.slice(0, 1))
      expect(view.facts.map(fact => fact.status)).toEqual(['disputed'])
      expect(view.omittedFactIds).toEqual(ids.slice(1))
    }
  })
})

function openRegion(sessionId: string) {
  const ctx = new Context()
  new SessionProjectionRegistry(ctx)
  const meter = new TokenMeter(ctx)
  const session = Session.create(SessionId(sessionId))
  session.append('turn/start', { turn: 1 })
  const seq = appendMessage(session, 'Detailed request content. '.repeat(100))
  const agent = { session } as Agent
  const dependencies = {
    meter,
    summarize: async () => ({
      summary: [{ type: 'text' as const, text: 'Request summarized.' }],
      provider: 'fixture-provider',
      model: 'fixture-model',
    }),
    recover: () => false,
  }
  return { session, seq, agent, dependencies }
}

describe('automatic compaction durability', () => {
  it('rebuilds the input and retries when the recovery hook accepts a summary failure', async () => {
    const { session, seq, agent, dependencies } = openRegion('automatic-recovery')
    const failure = new Error('retryable fixture failure')
    const summarize = vi.fn(dependencies.summarize).mockRejectedValueOnce(failure)
    const recover = vi.fn(() => true)

    const result = await compactSurfaceRegion({ ...dependencies, summarize, recover }, session, seq, seq, agent, {
      owner: 'current-turn',
      stability: 'whole-surface',
    })

    expect(recover).toHaveBeenCalledWith(failure, agent, [seq], undefined)
    expect(summarize).toHaveBeenCalledTimes(2)
    expect(result.shadowedSeqs).toEqual([seq])
    expect(session.surface.nodes).not.toContain(seq)
    expect(session.snapshotEvents().at(-1)).toMatchObject({
      seq: result.endSeq, type: 'compaction/end', data: { compactionId: result.compactionId },
    })
  })

  it.each([false, true])('commits ordered call records and sourced coverage with resident dispatch %s', async (residentDispatch) => {
    const { session, seq, agent, dependencies } = openRegion('automatic-coverage')
    const events = session.snapshotEvents()
    const incremental: CompactionIncrementalCoverage = {
      revision: 1,
      highWatermark: seq,
      covered: [{ start: seq, end: seq, digest: digestRange(events, seq, seq) }],
      policyDigest: digestCanonical({ maxTokens: 32 }),
    }
    const ledger = createFacts([{
      kind: 'decision', key: 'summary-model', claim: 'Use the fixture model.',
      sourceEventIds: [seq], basis: 'asserted',
    }], events)
    const chunkCalls: CompactionChunkCall[] = [{
      groups: [{ eventSeqs: [seq], toolBodiesOmitted: false }],
      contextWindow: 4096,
      estimatedInputTokens: 512,
      rawOutput: [{ type: 'text', text: 'Request summarized.' }],
    }]
    const residentSelection: CompactionResidentSelection = {
      alias: 'fixture-model-node',
      gatewayId: 'fixture-gateway',
      routeId: 'route-a',
      nodeId: 'node-a',
      nodeName: 'node',
      instance: 'mservice/default',
      servingVariantId: 'fixture-variant',
      servingVariantRevision: 'variant-revision',
      modelArtifact: 'fixture-artifact',
      artifactRevision: 'artifact-revision',
      targetConfigRevision: 'target-revision',
      hostGeneration: 'host-generation',
      residentGeneration: 'resident-generation',
      reason: 'only-eligible',
      projectionIds: [digestCanonical(['fixture-model', seq])],
    }
    const selection = residentDispatch ? { residentSelection } : {}

    const result = await compactSurfaceRegion({
      ...dependencies,
      summarize: async () => ({ ...await dependencies.summarize(), incremental, ledger, chunkCalls, ...selection }),
    }, session, seq, seq, agent, { owner: 'current-turn', stability: 'whole-surface' })

    const record = session.snapshotEvents().find(event => event.seq === result.summarySeq)
    expect(record).toMatchObject({ type: 'compaction/summary', data: { incremental, ledger, chunkCalls, ...selection } })
    if (!residentDispatch) expect(record?.data).not.toHaveProperty('residentSelection')
  })

  it('reports a failed durability flush after a successful close', async () => {
    const { session, seq, agent, dependencies } = openRegion('automatic-durability')
    const diskError = new Error('disk full')

    await expect(compactSurfaceRegion(dependencies, session, seq, seq, agent, {
      owner: 'current-turn',
      stability: 'whole-surface',
      flush: async () => { throw diskError },
    })).rejects.toMatchObject({
      code: 'persistence', message: 'compaction durability checkpoint failed', cause: diskError,
    })

    expect(session.surface.nodes).not.toContain(seq)
    expect(session.snapshotEvents().at(-1)).toMatchObject({ type: 'compaction/end', data: { turn: 1 } })
  })

  it('reports the failed flush when the failed summary was recorded', async () => {
    const { session, seq, agent, dependencies } = openRegion('automatic-summary-flush')
    const diskError = new Error('disk full')

    await expect(compactSurfaceRegion({
      ...dependencies,
      summarize: async () => { throw new Error('summary unavailable') },
    }, session, seq, seq, agent, {
      owner: 'current-turn',
      stability: 'whole-surface',
      flush: async () => { throw diskError },
    })).rejects.toMatchObject({ code: 'persistence', cause: diskError })

    expect(session.surface.nodes).toEqual([seq])
    const end = session.snapshotEvents().at(-1)
    expect(end).toMatchObject({ type: 'compaction/end', data: { turn: 1 } })
    expect(end?.type === 'compaction/end' && end.data.error).toContain('summary unavailable')
  })

  it('reports a commit failure even after the error close flushes successfully', async () => {
    const { session, seq, agent, dependencies } = openRegion('automatic-commit-flush')
    const commitError = new Error('summary record rejected')
    const call: CompactionChunkCall = {
      groups: [{ eventSeqs: [seq], toolBodiesOmitted: false }],
      get contextWindow(): number { throw commitError },
      estimatedInputTokens: 512,
      rawOutput: [{ type: 'text', text: 'Request summarized.' }],
    }
    const summarize = async () => ({ ...await dependencies.summarize(), chunkCalls: [call] })
    const flush = vi.fn(async () => {
      const end = session.snapshotEvents().at(-1)
      expect(end).toMatchObject({ type: 'compaction/end', data: { turn: 1 } })
      expect(end?.type === 'compaction/end' && end.data.error).toContain('summary record rejected')
    })

    await expect(compactSurfaceRegion({ ...dependencies, summarize }, session, seq, seq, agent, {
      owner: 'current-turn', stability: 'whole-surface', flush,
    })).rejects.toMatchObject({ code: 'commit', message: 'compaction did not commit cleanly', cause: commitError })

    expect(session.surface.nodes).toEqual([seq])
    expect(flush).toHaveBeenCalledOnce()
  })
})
