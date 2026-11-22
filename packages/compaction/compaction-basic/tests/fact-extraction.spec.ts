import { createMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { Session, SessionId } from '@deepseek-ai/dsh-session'
import { expect, it } from 'vitest'
import {
  FactExtractionError,
  mergeExtractedFacts,
  parseFactExtraction,
} from '../src/fact-extraction.ts'
import { createFacts, FactLedgerError } from '../src/fact-ledger.ts'

let nextSession = 0

function conversation(turns = 2): Session {
  const session = Session.create(SessionId(`fact-extraction-${nextSession++}`))
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

function sourcesOf(session: Session) {
  const events = session.snapshotEvents()
  return {
    events,
    user: events.find(event => event.type === 'user/message')!.seq,
    assistant: events.find(event => event.type === 'assistant/message')!.seq,
  }
}

const OPTIONS = { maxFacts: 8, maxClaimChars: 500 }

it('parses validated claims with inferred default and explicit assertions', () => {
  const session = conversation()
  const { user, assistant } = sourcesOf(session)
  const text = JSON.stringify([
    { kind: 'decision', key: 'auth-mode', claim: 'Use DSH-TSK-0014 IDs exactly.', sourceSeqs: [user, assistant] },
    { kind: 'constraint', key: 'no-cloud', claim: 'Never cloud fallback.', sourceSeqs: [user], provenance: 'asserted' },
  ])
  const parsed = parseFactExtraction(text, [user, assistant], OPTIONS)
  expect(parsed).toHaveLength(2)
  expect(parsed[0]).toMatchObject({ kind: 'decision', key: 'auth-mode', provenance: 'inferred' })
  expect(parsed[1]).toMatchObject({ provenance: 'asserted' })
  expect(parsed[0]!.claim).toBe('Use DSH-TSK-0014 IDs exactly.')
})

it('refuses invalid JSON, non-arrays and out-of-bounds counts', () => {
  const session = conversation()
  const { user } = sourcesOf(session)
  expect(() => parseFactExtraction('not json', [user], OPTIONS)).toThrow(FactExtractionError)
  expect(() => parseFactExtraction('not json', [user], OPTIONS)).toThrow(/not valid JSON/)
  expect(() => parseFactExtraction(JSON.stringify({ kind: 'decision' }), [user], OPTIONS)).toThrow(/1-8 claims/)
  expect(() => parseFactExtraction('[]', [user], OPTIONS)).toThrow(/1-8 claims/)
  const many = Array.from({ length: 9 }, () => ({ kind: 'decision', key: 'k', claim: 'c', sourceSeqs: [user] }))
  expect(() => parseFactExtraction(JSON.stringify(many), [user], OPTIONS)).toThrow(/1-8 claims/)
})

it('refuses unknown kinds, empty fields, oversized claims and bad provenance', () => {
  const session = conversation()
  const { user } = sourcesOf(session)
  const parse = (claim: object) => parseFactExtraction(JSON.stringify([claim]), [user], OPTIONS)
  expect(() => parse({ kind: 'rumor', key: 'k', claim: 'c', sourceSeqs: [user] })).toThrow(/unknown kind/)
  expect(() => parse({ kind: 'decision', key: ' ', claim: 'c', sourceSeqs: [user] })).toThrow(/empty key or claim/)
  expect(() => parse({ kind: 'decision', key: 'k', claim: 'x'.repeat(501), sourceSeqs: [user] })).toThrow(/exceeds/)
  expect(() => parse({ kind: 'decision', key: 'k', claim: 'c', sourceSeqs: [user], provenance: 'trusted' }))
    .toThrow(/invalid provenance/)
  expect(() => parse({ kind: 'decision', key: 'k', claim: 'c', sourceSeqs: [user], supersedes: [''] }))
    .toThrow(/supersedes/)
})

it('refuses sources outside the summarized chunk and duplicate sources', () => {
  const session = conversation()
  const { user, assistant } = sourcesOf(session)
  const parse = (seqs: unknown[]) => parseFactExtraction(
    JSON.stringify([{ kind: 'decision', key: 'k', claim: 'c', sourceSeqs: seqs }]), [user], OPTIONS,
  )
  expect(() => parse([assistant])).toThrow(/outside its chunk/)
  expect(() => parse([user, user])).toThrow(/repeats a source/)
  expect(() => parse(['event-1'])).toThrow(FactExtractionError)
})

it('merges an explicit update into the ledger and rejects a dangling supersede', () => {
  const session = conversation()
  const { events, user, assistant } = sourcesOf(session)
  const [prior] = createFacts([{
    kind: 'decision', key: 'auth-mode', claim: 'Use the fixed summarizer.', sourceEventIds: [user],
    provenance: 'asserted',
  }], events)
  const updated = JSON.stringify([{
    kind: 'decision', key: 'auth-mode', claim: 'Use the resident selector.', sourceSeqs: [assistant],
    provenance: 'asserted', supersedes: [prior!.factId],
  }])
  const merged = mergeExtractedFacts([prior!], updated, [user, assistant], events, OPTIONS)
  expect(merged.find(fact => fact.factId === prior!.factId)).toMatchObject({
    status: 'superseded', supersededBy: expect.any(String),
  })
  expect(merged.filter(fact => fact.status === 'active')).toHaveLength(1)

  const dangling = JSON.stringify([{
    kind: 'decision', key: 'auth-mode', claim: 'Use the resident selector.', sourceSeqs: [assistant],
    supersedes: ['missing-fact'],
  }])
  expect(() => mergeExtractedFacts([prior!], dangling, [user, assistant], events, OPTIONS))
    .toThrow(FactLedgerError)
})

it('derives the same fact identity regardless of cited source order', () => {
  const session = conversation()
  const { events, user, assistant } = sourcesOf(session)
  const [first] = parseFactExtraction(JSON.stringify([{
    kind: 'identifier', key: 'ticket', claim: 'DSH-TSK-0012', sourceSeqs: [user, assistant],
  }]), [user, assistant], OPTIONS)
  const [second] = parseFactExtraction(JSON.stringify([{
    kind: 'identifier', key: 'ticket', claim: 'DSH-TSK-0012', sourceSeqs: [assistant, user],
  }]), [user, assistant], OPTIONS)
  expect(createFacts([first!], events)[0]!.factId).toBe(createFacts([second!], events)[0]!.factId)
})
