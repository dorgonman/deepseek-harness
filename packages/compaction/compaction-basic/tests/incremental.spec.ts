import { CompactionId, compactCheckpointSource, type CompactionIncrementalCoverage } from '@deepseek-ai/dsh-compaction'
import { createMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { Session, SessionId, SessionSeq } from '@deepseek-ai/dsh-session'
import { describe, expect, it } from 'vitest'
import {
  advanceCoverage,
  assertCoverageRevision,
  deriveCommittedCoverage,
  digestRange,
  IncrementalCoverageError,
  selectUnprocessedRawEvents,
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
  })

  it('refuses a work cap instead of returning a partial range', () => {
    const session = conversation(2)
    const events = eventsOf(session)
    expect(() => selectUnprocessedRawEvents(events, undefined, { maxEvents: 1 }))
      .toThrow(IncrementalCoverageError)
    expect(() => selectUnprocessedRawEvents(events, undefined, { maxEvents: 1 }))
      .toThrow(/more than 1 unprocessed RAW events/)
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
