import { CompactionId, compactCheckpointSource, type CompactionIncrementalCoverage } from '@deepseek-ai/dsh-compaction'
import { createMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { Session, SessionId, SessionSeq } from '@deepseek-ai/dsh-session'
import { describe, expect, it } from 'vitest'
import {
  advanceCoverage,
  assertCoverageRevision,
  deriveCommittedCoverage,
  digestCanonical,
  digestEventIds,
  digestPolicy,
  digestRange,
  hasUnprocessedRawEvents,
  IncrementalCoverageError,
  selectUnprocessedRawEvents,
  summaryTransactionCommitted,
  validateCoverage,
} from '../src/incremental.ts'

let nextSession = 0
function conversation(turns: number): Session {
  const session = Session.create(SessionId(`incremental-${nextSession++}`))
  for (let turn = 1; turn <= turns; turn += 1) {
    session.append('turn/start', { turn })
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: `turn ${turn}` }], source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    session.append('assistant/message', {
      stream: [],
      turn,
      step: 1,
      message: createMessage({
        role: 'assistant',
        content: [{ type: 'text', text: 'reply' }],
        source: { kind: 'model', provider: 'test-provider', model: 'test-model' },
      }),
    }, { surfaceOp: 'append' })
    session.append('turn/end', { turn, reason: { kind: 'completed' } })
  }
  return session
}

function eventsOf(session: Session) {
  return session.snapshotEvents()
}

function commitCoverage(session: Session, coverage: CompactionIncrementalCoverage) {
  const id = CompactionId(`incremental-${String(coverage.revision)}`)
  const nodes = [...session.surface.nodes]
  const start = nodes[0]!
  const end = nodes.at(-1)!
  session.append('compaction/start', { compactionId: id, turn: null })
  session.append('compaction/summary', {
    compactionId: id,
    summary: [{ type: 'text', text: 'checkpoint' }],
    shadowedRange: { start, end },
    shadowedSeqs: nodes,
    shadowedTokenCount: 1,
    provider: 'test-provider',
    model: 'test-model',
    chunkCalls: [],
    incremental: coverage,
  })
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'checkpoint' }],
    source: compactCheckpointSource(id),
  }), { surfaceOp: { op: 'replace', startSeq: start, endSeq: end }, sourceEventSeqs: nodes })
  session.append('compaction/end', { compactionId: id, turn: null })
  return id
}

describe('incremental coverage', () => {
  it('advances, commits and derives a complete record', () => {
    const session = conversation(2)
    const events = eventsOf(session)
    const end = SessionSeq(events.at(-1)!.seq)
    const coverage = advanceCoverage(undefined, events, SessionSeq(0), end, { policyDigest: 'policy-a' })
    expect(coverage).toMatchObject({ revision: 1, highWatermark: end, policyDigest: 'policy-a' })
    expect(coverage.covered).toEqual([{ start: 0, end, digest: digestRange(events, SessionSeq(0), end) }])

    commitCoverage(session, coverage)
    const derived = deriveCommittedCoverage(eventsOf(session))
    expect(derived).toEqual(coverage)
    validateCoverage(derived!, eventsOf(session))
  })

  it('coalesces adjacent ranges and recomputes the merged digest', () => {
    const session = conversation(2)
    const events = eventsOf(session)
    const first = advanceCoverage(undefined, events, SessionSeq(0), SessionSeq(3), { policyDigest: 'policy-a' })
    const second = advanceCoverage(first, events, SessionSeq(4), SessionSeq(7), { policyDigest: 'policy-a' })
    expect(second.revision).toBe(2)
    expect(second.covered).toEqual([{ start: 0, end: 7, digest: digestRange(events, SessionSeq(0), SessionSeq(7)) }])
    validateCoverage(second, events)
  })

  it('binds exact event bytes independently of object-key and selection order', () => {
    const events = eventsOf(conversation(2))
    const ids = [SessionSeq(1), SessionSeq(2)]
    const digest = digestEventIds(events, ids)

    expect(digest).toBe(digestRange(events, SessionSeq(1), SessionSeq(2)))
    expect(digestEventIds(events, [...ids].reverse())).toBe(digest)
    expect(digestEventIds(events, [SessionSeq(1), SessionSeq(3)])).not.toBe(digest)
    expect(digestPolicy({ z: 1, a: [null, { y: true, x: 'text' }] }))
      .toBe(digestCanonical({ a: [null, { x: 'text', y: true }], z: 1 }))
    expect(() => digestCanonical({ revision: 1n })).toThrow(/non-JSON value/)
    expect(() => digestEventIds(events, [SessionSeq(1), SessionSeq(1)])).toThrow(/ids are not unique/)
    expect(() => digestEventIds(events, [SessionSeq(8)])).toThrow(/events outside the log/)
  })

  it('refuses inverted, gapped, and incomplete source ranges', () => {
    const events = eventsOf(conversation(2))
    const reversed = [SessionSeq(2), SessionSeq(1)] as const
    expect(() => digestRange(events, ...reversed)).toThrow(/start after end/)
    expect(() => advanceCoverage(undefined, events, ...reversed, { policyDigest: 'policy-a' }))
      .toThrow(/start after end/)
    expect(() => digestRange(events.filter(event => event.seq !== 2), SessionSeq(1), SessionSeq(3)))
      .toThrow(/not contiguous at 3/)
    expect(() => digestRange(events, SessionSeq(6), SessionSeq(8))).toThrow(/not fully present/)
  })

  it('keeps disjoint coverage and validates summarizer and carried identities', () => {
    const events = eventsOf(conversation(2))
    const first = advanceCoverage(undefined, events, SessionSeq(0), SessionSeq(2), { policyDigest: 'policy-a' })
    const identity = {
      summarizer: { provider: 'summary-provider', model: 'summary-model', maxTokens: 64 },
      carried: { summarySeq: SessionSeq(3), digest: 'prior-checkpoint' },
    }
    const second = advanceCoverage(first, events, SessionSeq(4), SessionSeq(7), { ...identity, policyDigest: 'policy-b' })

    expect(second).toEqual({
      ...identity, policyDigest: 'policy-b', revision: 2, highWatermark: 7,
      covered: [...first.covered, { start: 4, end: 7, digest: digestRange(events, SessionSeq(4), SessionSeq(7)) }],
    })
    expect(() => { validateCoverage(second, events) }).not.toThrow()
    expect(first.covered).toHaveLength(1)
  })

  it.each([
    ['a fractional revision', { revision: 1.5 }, 'revision is not a non-negative integer'],
    ['a missing policy digest', { policyDigest: '' }, 'policy digest is missing'],
    ['a fractional watermark', { highWatermark: 2.5 }, 'high watermark is not a non-negative integer'],
    ['a watermark above the last covered end', { highWatermark: 4 }, 'high watermark 4 does not equal the last covered end 3'],
    ['a non-array range list', { covered: {} }, 'covered ranges are not an array'],
    ['a null range', { covered: [null] }, 'contains a non-object range'],
    ['a reversed range', { covered: [{ start: 2, end: 1, digest: 'digest' }] }, 'contains an invalid range boundary'],
    ['a range without a digest', { covered: [{ start: 1, end: 2 }] }, 'contains a range without a digest'],
    ['a range outside the log', { highWatermark: 4, covered: [{ start: 1, end: 4, digest: 'digest' }] }, 'range 1-4 is beyond the log'],
    ['a watermark without ranges', { covered: [] }, 'has no ranges but a non-zero watermark or revision'],
    ['a null summarizer', { summarizer: null }, 'summarizer is not an object'],
    ['an incomplete summarizer', { summarizer: { provider: 'summary-provider', maxTokens: 16 } }, 'summarizer identity is incomplete'],
    ['a null carried reference', { carried: null }, 'carried checkpoint reference is not an object'],
    ['an incomplete carried reference', { carried: { summarySeq: 1 } }, 'carried checkpoint reference is incomplete'],
  ] as const)('refuses a stored record with %s', (_name, fields, message) => {
    const events = eventsOf(conversation(1))
    const baseline = advanceCoverage(undefined, events, SessionSeq(0), SessionSeq(3), { policyDigest: 'policy-a' })
    const stored = JSON.parse(JSON.stringify({ ...baseline, ...fields })) as CompactionIncrementalCoverage

    expect(() => { validateCoverage(stored, events) }).toThrow(expect.objectContaining({
      code: 'malformed', message: `compaction: incremental coverage ${message}`,
    }))
  })

  it('keeps empty initial coverage valid', () => {
    const initial = { revision: 0, highWatermark: SessionSeq(0), covered: [], policyDigest: 'policy-a' }
    expect(() => { validateCoverage(initial, []) }).not.toThrow()
    expect(() => { validateCoverage({ ...initial, highWatermark: SessionSeq(3) }, []) }).not.toThrow()
  })

  it('ignores a dangling summary and keeps the earlier committed record', () => {
    const session = conversation(2)
    const events = eventsOf(session)
    const end = SessionSeq(events.at(-1)!.seq)
    const coverage = advanceCoverage(undefined, events, SessionSeq(0), end, { policyDigest: 'policy-a' })
    commitCoverage(session, coverage)

    const danglingId = CompactionId('dangling')
    session.append('compaction/start', { compactionId: danglingId, turn: null })
    session.append('compaction/summary', {
      compactionId: danglingId,
      summary: [{ type: 'text', text: 'dangling checkpoint' }],
      shadowedRange: { start: SessionSeq(0), end: SessionSeq(0) },
      shadowedSeqs: [SessionSeq(0)],
      shadowedTokenCount: 1,
      provider: 'test-provider',
      model: 'test-model',
      chunkCalls: [],
      incremental: {
        revision: 2, highWatermark: end, covered: coverage.covered, policyDigest: 'policy-a',
      },
    })
    expect(deriveCommittedCoverage(eventsOf(session))).toEqual(coverage)
  })

  it('requires the checkpoint and matching successful end before a summary commits', () => {
    const session = conversation(1)
    const coverage = advanceCoverage(undefined, eventsOf(session), SessionSeq(0), SessionSeq(3), { policyDigest: 'policy-a' })
    commitCoverage(session, coverage)
    const events = eventsOf(session)
    const summaryIndex = events.findIndex(event => event.type === 'compaction/summary')
    const restart = events.find(event => event.type === 'compaction/start')!
    const end = events.find(event => event.type === 'compaction/end')!
    const beforeEnd = events.slice(0, -1)
    const failed = { ...end, data: { ...end.data, error: 'cancelled' } }
    const unrelatedEnd = { ...end, data: { ...end.data, compactionId: CompactionId('unrelated') } }
    const interrupted = [...beforeEnd]
    interrupted.length += 1
    interrupted.push(end)

    expect(summaryTransactionCommitted(events, summaryIndex)).toBe(true)
    expect(summaryTransactionCommitted([...beforeEnd, unrelatedEnd, end], summaryIndex)).toBe(true)
    expect(summaryTransactionCommitted(events, 0)).toBe(false)
    for (const unfinished of [beforeEnd, [...beforeEnd, failed], [...beforeEnd, restart, end], interrupted]) {
      expect(summaryTransactionCommitted(unfinished, summaryIndex)).toBe(false)
      expect(deriveCommittedCoverage(unfinished)).toBeUndefined()
    }
    const unrelated = eventsOf(conversation(1)).find(event => event.type === 'user/message')!
    expect(summaryTransactionCommitted([...events.slice(0, summaryIndex + 1), unrelated, end], summaryIndex)).toBe(false)
  })

  it('refuses an overlapping range, a wrong digest and a watermark mismatch', () => {
    const session = conversation(2)
    const events = eventsOf(session)
    const coverage = advanceCoverage(undefined, events, SessionSeq(0), SessionSeq(7), { policyDigest: 'policy-a' })

    expect(() => {
      validateCoverage({
        ...coverage,
        covered: [
          { start: SessionSeq(0), end: SessionSeq(3), digest: digestRange(events, SessionSeq(0), SessionSeq(3)) },
          { start: SessionSeq(3), end: SessionSeq(7), digest: digestRange(events, SessionSeq(3), SessionSeq(7)) },
        ],
      }, events)
    }).toThrow(IncrementalCoverageError)
    expect(() => {
      validateCoverage({
        ...coverage, covered: [{ start: SessionSeq(0), end: SessionSeq(7), digest: '0'.repeat(64) }],
      }, events)
    }).toThrow(/does not match its digest/)
    expect(() => {
      validateCoverage({
        ...coverage, highWatermark: SessionSeq(6),
      }, events)
    }).toThrow(/high watermark/)
  })

  it('selects only unprocessed append-origin events and never a checkpoint', () => {
    const session = conversation(2)
    const events = eventsOf(session)
    const coverage = advanceCoverage(undefined, events, SessionSeq(0), SessionSeq(7), { policyDigest: 'policy-a' })
    commitCoverage(session, coverage)

    expect(selectUnprocessedRawEvents(eventsOf(session), deriveCommittedCoverage(eventsOf(session)), { maxEvents: 10 }))
      .toEqual([])
    expect(hasUnprocessedRawEvents(eventsOf(session), coverage)).toBe(false)

    session.append('turn/start', { turn: 3 })
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'new turn' }], source: { kind: 'user' },
    }), { surfaceOp: 'append' })

    const all = eventsOf(session)
    const selected = selectUnprocessedRawEvents(all, coverage, { maxEvents: 10 })
    expect(selected).toHaveLength(1)
    expect(selected[0]).toMatchObject({ type: 'user/message' })

    const checkpoint = all.find(event => event.type === 'user/message'
      && event.data.source.kind === 'compact-checkpoint')
    expect(checkpoint).toBeDefined()
    expect(selectUnprocessedRawEvents(all, undefined, { maxEvents: 10 })).not.toContainEqual(checkpoint)
    expect(hasUnprocessedRawEvents(all, undefined)).toBe(true)
    expect(hasUnprocessedRawEvents(all, coverage)).toBe(true)
  })

  it('refuses a work cap instead of returning a partial range', () => {
    const session = conversation(2)
    const events = eventsOf(session)
    expect(() => selectUnprocessedRawEvents(events, undefined, { maxEvents: 1 }))
      .toThrow(IncrementalCoverageError)
    expect(() => selectUnprocessedRawEvents(events, undefined, { maxEvents: 1 }))
      .toThrow(/more than 1 unprocessed RAW events/)
    expect(() => selectUnprocessedRawEvents(events, undefined, { maxEvents: -1 }))
      .toThrow(/work cap is not a non-negative integer/)
  })

  it('refuses stale revisions and non-advancing ranges', () => {
    const session = conversation(2)
    const events = eventsOf(session)
    const coverage = advanceCoverage(undefined, events, SessionSeq(0), SessionSeq(7), { policyDigest: 'policy-a' })
    expect(() => { assertCoverageRevision(undefined, 0) }).not.toThrow()
    expect(() => { assertCoverageRevision(coverage, 0) }).toThrow(/revision changed/)
    expect(() => advanceCoverage(coverage, events, SessionSeq(0), SessionSeq(7), { policyDigest: 'policy-a' }))
      .toThrow(/does not advance/)
  })

  it('treats a legacy summary without incremental coverage as no coverage', () => {
    const session = conversation(2)
    const nodes = [...session.surface.nodes]
    const start = nodes[0]!
    const end = nodes.at(-1)!
    const id = CompactionId('legacy-basic')
    session.append('compaction/start', { compactionId: id, turn: null })
    session.append('compaction/summary', {
      compactionId: id,
      summary: [{ type: 'text', text: 'legacy checkpoint' }],
      shadowedRange: { start, end },
      shadowedSeqs: nodes,
      shadowedTokenCount: 1,
      provider: 'test-provider',
      model: 'test-model',
      chunkCalls: [],
    })
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'legacy checkpoint' }],
      source: compactCheckpointSource(id),
    }), { surfaceOp: { op: 'replace', startSeq: start, endSeq: end }, sourceEventSeqs: nodes })
    session.append('compaction/end', { compactionId: id, turn: null })

    const committed = eventsOf(session)
    expect(deriveCommittedCoverage(committed)).toBeUndefined()
    const selected = selectUnprocessedRawEvents(committed, undefined, { maxEvents: 20 })
    expect(selected.length).toBeGreaterThan(0)
    expect(selected.every(event => event.type === 'user/message' || event.type === 'assistant/message')).toBe(true)
    const coverage = advanceCoverage(undefined, committed, SessionSeq(0), SessionSeq(committed.at(-1)!.seq), {
      policyDigest: 'policy-a',
    })
    expect(coverage.revision).toBe(1)
  })
})
