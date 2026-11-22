import { CompactionId, compactCheckpointSource } from '@deepseek-ai/dsh-compaction'
import { createMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { Session, SessionId, SessionSeq } from '@deepseek-ai/dsh-session'
import { expect, it } from 'vitest'
import {
  composeCompactView,
  createFacts,
  deriveCommittedLedger,
  FactLedgerError,
  mergeFacts,
  validateLedger,
  type FactInput,
} from '../src/fact-ledger.ts'

let nextSession = 0

function conversation(turns = 2): Session {
  const session = Session.create(SessionId(`fact-ledger-${nextSession++}`))
  for (let turn = 1; turn <= turns; turn += 1) {
    session.append('turn/start', { turn })
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: `request ${turn}` }], source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    session.append('assistant/message', {
      stream: [],
      turn,
      step: 1,
      message: createMessage({
        role: 'assistant',
        content: [{ type: 'text', text: `answer ${turn}` }],
        source: { kind: 'model', provider: 'test-provider', model: 'test-model' },
      }),
    }, { surfaceOp: 'append' })
    session.append('turn/end', { turn, reason: { kind: 'completed' } })
  }
  return session
}

function inputs(session: Session): FactInput[] {
  const events = session.snapshotEvents()
  const user = events.find(event => event.type === 'user/message')!.seq
  const assistant = events.find(event => event.type === 'assistant/message')!.seq
  return [
    { kind: 'decision', key: 'auth-mode', claim: 'Use local resident summarizers.', sourceEventIds: [user, assistant], basis: 'asserted' },
    { kind: 'constraint', key: 'no-cloud', claim: 'Never fall back to a cloud summarizer.', sourceEventIds: [user], basis: 'asserted', branch: 'main' },
  ]
}

it('creates immutable sourced facts with a deterministic identity and digest', () => {
  const session = conversation()
  const events = session.snapshotEvents()
  const [first] = createFacts(inputs(session), events)
  const [reordered] = createFacts([{
    ...inputs(session)[0]!,
    sourceEventIds: [...inputs(session)[0]!.sourceEventIds].reverse(),
  }], events)
  expect(first).toMatchObject({
    kind: 'decision', key: 'auth-mode', status: 'active', basis: 'asserted',
  })
  expect(first!.factId).toBe(reordered!.factId)
  expect(first!.evidenceDigest).toBe(reordered!.evidenceDigest)
  expect(first!.firstSeenSeq).toBeLessThan(first!.lastSeenSeq)
  expect(Object.isFrozen(first)).toBe(true)
  expect(Object.isFrozen(first!.sourceEventIds)).toBe(true)
})

it('rejects empty claims, duplicate sources, unknown sources and dangling supersedes', () => {
  const session = conversation()
  const events = session.snapshotEvents()
  const base = inputs(session)[0]!
  expect(() => createFacts([{ ...base, claim: '   ' }], events)).toThrow(FactLedgerError)
  expect(() => createFacts([{ ...base, sourceEventIds: [base.sourceEventIds[0]!, base.sourceEventIds[0]!] }], events))
    .toThrow(/unique/)
  expect(() => createFacts([{ ...base, sourceEventIds: [SessionSeq(9999)] }], events))
    .toThrow(/outside the log/)
  const [fact] = createFacts([base], events)
  expect(() => mergeFacts([fact!], [{ ...fact!, supersedes: ['unknown-fact'] }]))
    .toThrow(/supersedes unknown/)
})

it('deduplicates identical facts and unions their sources and seen bounds', () => {
  const session = conversation(3)
  const events = session.snapshotEvents()
  const base = inputs(session)[0]!
  const later = events.filter(event => event.type === 'user/message')[2]!.seq
  const [first] = createFacts([base], events)
  const [sameFactLater] = createFacts([{ ...base, sourceEventIds: [later] }], events)
  const merged = mergeFacts([first!], [sameFactLater!])
  expect(merged).toHaveLength(1)
  expect(merged[0]!.sourceEventIds).toEqual([...first!.sourceEventIds, later].sort((a, b) => a - b))
  expect(merged[0]!.firstSeenSeq).toBe(first!.firstSeenSeq)
  expect(merged[0]!.lastSeenSeq).toBe(later)
  expect(merged[0]!.status).toBe('active')
})

it('applies an explicit update and excludes the superseded fact from the view', () => {
  const session = conversation()
  const events = session.snapshotEvents()
  const base = inputs(session)[0]!
  const [oldFact] = createFacts([base], events)
  const [newFact] = createFacts([{
    ...base, claim: 'Use only the 27B resident summarizer.', sourceEventIds: [base.sourceEventIds[1]!],
    supersedes: [oldFact!.factId],
  }], events)
  const merged = mergeFacts([oldFact!], [newFact!])
  expect(merged.find(fact => fact.factId === oldFact!.factId)).toMatchObject({
    status: 'superseded', supersededBy: newFact!.factId,
  })
  expect(merged.find(fact => fact.factId === newFact!.factId)?.status).toBe('active')
  const view = composeCompactView(merged, { maxTokens: 1000, tokensOf: () => 1 })
  expect(view.facts.map(fact => fact.factId)).toEqual([newFact!.factId])
})

it('keeps conflicting claims as disputed rather than choosing a winner', () => {
  const session = conversation()
  const events = session.snapshotEvents()
  const base = inputs(session)[0]!
  const [first] = createFacts([base], events)
  const [conflicting] = createFacts([{ ...base, claim: 'Use a cloud summarizer.' }], events)
  const merged = mergeFacts([first!], [conflicting!])
  expect(merged.map(fact => fact.status)).toEqual(['disputed', 'disputed'])
  const view = composeCompactView(merged, { maxTokens: 1000, tokensOf: () => 1 })
  expect(view.facts).toHaveLength(2)
})

it('bounds the view and reports omitted facts and source ids explicitly', () => {
  const session = conversation(3)
  const events = session.snapshotEvents()
  const base = inputs(session)[0]!
  const facts = createFacts([
    { ...base, key: 'a', claim: 'first' },
    { ...base, key: 'b', claim: 'second' },
    { ...base, key: 'c', claim: 'third' },
  ], events)
  const view = composeCompactView(facts, { maxTokens: 25, tokensOf: () => 10 })
  expect(view.facts.map(fact => fact.key)).toEqual(['a', 'b'])
  expect(view.omittedFactIds).toHaveLength(1)
  const omitted = facts.find(fact => fact.factId === view.omittedFactIds[0])!
  expect(view.omittedSourceEventIds).toEqual([...omitted.sourceEventIds])
  expect(() => composeCompactView(facts, { maxTokens: -1, tokensOf: () => 1 })).toThrow(/maxTokens/)
  expect(() => composeCompactView(facts, { maxTokens: 10, tokensOf: () => Number.NaN })).toThrow(/tokensOf/)
})

it('preserves inferred basis and branch scope without promoting either', () => {
  const session = conversation()
  const events = session.snapshotEvents()
  const [fact] = createFacts([{
    ...inputs(session)[0]!, basis: 'inferred', branch: 'feature-x',
  }], events)
  expect(fact!.basis).toBe('inferred')
  expect(fact!.branch).toBe('feature-x')
  expect(fact!.status).toBe('active')
})

function commitLedger(session: Session, facts: ReturnType<typeof createFacts>, dangling = false): void {
  const id = CompactionId(`ledger-${facts[0]?.factId.slice(0, 8) ?? 'empty'}-${nextSession}`)
  const nodes = [...session.surface.nodes]
  const start = nodes[0]!
  const end = nodes.at(-1)!
  session.append('compaction/start', { compactionId: id, turn: null })
  session.append('compaction/summary', {
    compactionId: id,
    summary: [{ type: 'text', text: 'ledger checkpoint' }],
    shadowedRange: { start, end },
    shadowedSeqs: nodes,
    shadowedTokenCount: 1,
    provider: 'test-provider',
    model: 'test-model',
    chunkCalls: [],
    ledger: facts,
  })
  if (dangling) return
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'ledger checkpoint' }],
    source: compactCheckpointSource(id),
  }), { surfaceOp: { op: 'replace', startSeq: start, endSeq: end }, sourceEventSeqs: nodes })
  session.append('compaction/end', { compactionId: id, turn: null })
}

it('derives a committed ledger only from a complete transaction', () => {
  const session = conversation()
  const facts = createFacts(inputs(session), session.snapshotEvents())
  commitLedger(session, facts)
  const committed = deriveCommittedLedger(session.snapshotEvents())
  expect(committed).toHaveLength(facts.length)
  expect(committed?.[0]).toMatchObject({ factId: facts[0]!.factId, status: 'active' })
  validateLedger(committed!, session.snapshotEvents())

  const danglingSession = conversation()
  const danglingFacts = createFacts(inputs(danglingSession), danglingSession.snapshotEvents())
  commitLedger(danglingSession, danglingFacts, true)
  expect(deriveCommittedLedger(danglingSession.snapshotEvents())).toBeUndefined()
})

it('refuses a tampered ledger snapshot and inconsistent lineage', () => {
  const session = conversation()
  const facts = createFacts(inputs(session), session.snapshotEvents())
  const tampered = [{ ...facts[0]!, evidenceDigest: '0'.repeat(64) }]
  expect(() => validateLedger(tampered, session.snapshotEvents()))
    .toThrow(/does not match its source digest/)
  expect(() => validateLedger([{ ...facts[0]!, supersededBy: 'missing' }], session.snapshotEvents()))
    .toThrow(FactLedgerError)
})
