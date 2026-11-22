/**
 * Typed Fact Ledger: sourced facts derived from RAW events, with validation,
 * deterministic dedup/conflict handling and a bounded view. The ledger is a
 * derived index; it never replaces the append-only Session history.
 *
 * @module @deepseek-ai/dsh-compaction-basic/fact-ledger
 */

import { SessionSeq, type SessionEvent } from '@deepseek-ai/dsh-session'
import { digestCanonical, digestEventIds } from './incremental.ts'

/** Closed set of fact classes the ledger accepts. */
export const FACT_KINDS = [
  'decision', 'constraint', 'identifier', 'open-work', 'observation', 'tool-evidence',
] as const

/** One accepted fact class. */
export type FactKind = typeof FACT_KINDS[number]

/** Lifecycle of a sourced fact. */
export type FactStatus = 'active' | 'superseded' | 'disputed'

/** How the claim reached the ledger. An inference is never promoted to RAW truth. */
export type FactProvenance = 'asserted' | 'inferred'

/** Input supplied by an extractor before validation. */
export interface FactInput {
  /** Fact class. */
  readonly kind: FactKind
  /** Stable subject key within the class, for example `auth-mode`. */
  readonly key: string
  /** Canonical claim or exact value, as extracted. */
  readonly claim: string
  /** Exact source event ids the claim cites. */
  readonly sourceEventIds: readonly SessionSeq[]
  /** Extraction provenance. */
  readonly provenance: FactProvenance
  /** Session branch scope, when the fact is branch-limited. */
  readonly branch?: string
  /** Earlier fact ids this fact explicitly replaces. */
  readonly supersedes?: readonly string[]
}

/** One validated, sourced fact. */
export interface FactRecord {
  /** Deterministic identity derived from kind, key and claim. */
  readonly factId: string
  readonly kind: FactKind
  readonly key: string
  readonly claim: string
  readonly sourceEventIds: readonly SessionSeq[]
  /** Digest binding the exact source event content. */
  readonly evidenceDigest: string
  readonly firstSeenSeq: SessionSeq
  readonly lastSeenSeq: SessionSeq
  readonly status: FactStatus
  readonly provenance: FactProvenance
  readonly branch?: string
  readonly supersedes?: readonly string[]
  readonly supersededBy?: string
}

/** Failure classes for ledger validation and composition. */
export class FactLedgerError extends Error {
  constructor(
    /** Machine-readable cause. */
    readonly code: 'malformed' | 'unknown-source' | 'dangling-supersede' | 'budget',
    message: string,
  ) {
    super(message)
    this.name = 'FactLedgerError'
  }
}

function malformed(message: string): never {
  throw new FactLedgerError('malformed', `fact ledger: ${message}`)
}

function nonEmpty(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

function factsOf(input: FactInput, events: readonly SessionEvent[]): FactRecord {
  if (!FACT_KINDS.includes(input.kind)) malformed(`unknown fact kind ${String(input.kind)}`)
  if (!nonEmpty(input.key) || !nonEmpty(input.claim)) malformed('key and claim must be non-empty')
  if (input.provenance !== 'asserted' && input.provenance !== 'inferred') {
    malformed('provenance must be asserted or inferred')
  }
  if (!Array.isArray(input.sourceEventIds) || input.sourceEventIds.length === 0) {
    malformed('at least one source event is required')
  }
  const ids = [...new Set(input.sourceEventIds)]
  if (ids.length !== input.sourceEventIds.length) malformed('source event ids must be unique')
  const first = SessionSeq(Math.min(...ids))
  const last = SessionSeq(Math.max(...ids))
  const known = new Set(events.map(event => event.seq))
  for (const id of ids) {
    if (!known.has(id)) {
      throw new FactLedgerError('unknown-source', `fact ledger: source event ${id} is outside the log`)
    }
  }
  return Object.freeze<FactRecord>({
    factId: digestCanonical([input.kind, input.key, input.claim]),
    kind: input.kind,
    key: input.key,
    claim: input.claim,
    sourceEventIds: Object.freeze([...ids].sort((left, right) => left - right)),
    evidenceDigest: digestEventIds(events, ids),
    firstSeenSeq: first,
    lastSeenSeq: last,
    status: 'active' as const,
    provenance: input.provenance,
    ...input.branch === undefined ? {} : { branch: input.branch },
    ...input.supersedes === undefined ? {} : { supersedes: Object.freeze([...input.supersedes]) },
  })
}

/**
 * Validate inputs and create sourced fact records.
 * @param inputs - extractor claims with exact source ids.
 * @param events - the session log those ids address.
 * @returns validated, immutable facts in input order.
 */
export function createFacts(
  inputs: readonly FactInput[],
  events: readonly SessionEvent[],
): FactRecord[] {
  return inputs.map(input => factsOf(input, events))
}

/** Merge sources and seen bounds for two records of the same fact. */
function mergeSame(left: FactRecord, right: FactRecord): FactRecord {
  const ids = [...new Set([...left.sourceEventIds, ...right.sourceEventIds])].sort((a, b) => a - b)
  const supersedes = [...new Set([...(left.supersedes ?? []), ...(right.supersedes ?? [])])]
  return Object.freeze<FactRecord>({
    ...left,
    sourceEventIds: Object.freeze(ids),
    firstSeenSeq: SessionSeq(Math.min(left.firstSeenSeq, right.firstSeenSeq)),
    lastSeenSeq: SessionSeq(Math.max(left.lastSeenSeq, right.lastSeenSeq)),
    ...supersedes.length === 0 ? {} : { supersedes: Object.freeze(supersedes) },
  })
}

/**
 * Merge new facts into an existing ledger deterministically.
 *
 * Identical facts merge their sources. An explicit `supersedes` marks the
 * earlier fact superseded. Two active facts with the same kind and key but
 * different claims stay in the ledger as `disputed`; the ledger never picks a
 * winner without an explicit update.
 * @param existing - committed ledger records.
 * @param incoming - newly validated records (source order).
 * @returns the merged ledger in stable order.
 */
export function mergeFacts(existing: readonly FactRecord[], incoming: readonly FactRecord[]): FactRecord[] {
  const byId = new Map(existing.map(fact => [fact.factId, fact]))
  for (const fact of incoming) {
    const prior = byId.get(fact.factId)
    byId.set(fact.factId, prior === undefined ? fact : mergeSame(prior, fact))
  }
  const supersededBy = new Map<string, string>()
  for (const fact of byId.values()) {
    for (const target of fact.supersedes ?? []) {
      if (!byId.has(target)) {
        throw new FactLedgerError('dangling-supersede', `fact ledger: ${fact.factId} supersedes unknown ${target}`)
      }
      supersededBy.set(target, fact.factId)
    }
  }
  const groups = new Map<string, string[]>()
  for (const fact of byId.values()) {
    const key = JSON.stringify([fact.kind, fact.key])
    const ids = groups.get(key) ?? []
    ids.push(fact.factId)
    groups.set(key, ids)
  }
  const disputed = new Set<string>()
  for (const ids of groups.values()) {
    const live = ids.filter(id => !supersededBy.has(id))
    if (live.length > 1) for (const id of live) disputed.add(id)
  }
  const merged = [...byId.values()].map((fact): FactRecord => {
    const replacedBy = supersededBy.get(fact.factId)
    const status: FactStatus = replacedBy === undefined
      ? disputed.has(fact.factId) ? 'disputed' : 'active'
      : 'superseded'
    return Object.freeze<FactRecord>({
      ...fact,
      status,
      ...replacedBy === undefined ? {} : { supersededBy: replacedBy },
    })
  })
  return merged.sort((left, right) => {
    const key = JSON.stringify([left.kind, left.key])
    const other = JSON.stringify([right.kind, right.key])
    return key < other ? -1 : key > other ? 1 : left.factId < right.factId ? -1 : left.factId > right.factId ? 1 : 0
  })
}

/** Bounded Compact View over active sourced facts. */
export interface CompactView {
  /** Included facts, active before disputed, in stable order. */
  readonly facts: readonly FactRecord[]
  /** Facts that did not fit the budget. */
  readonly omittedFactIds: readonly string[]
  /** Source events reachable only through omitted facts. */
  readonly omittedSourceEventIds: readonly SessionSeq[]
}

/** Budget inputs for {@link composeCompactView}. */
export interface CompactViewOptions {
  /** Maximum tokens the facts may consume. */
  readonly maxTokens: number
  /** Token price of one rendered fact. */
  readonly tokensOf: (fact: FactRecord) => number
}

/**
 * Compose a destination-budgeted view from ledger facts.
 *
 * Superseded facts are excluded. Disputed facts are included and remain marked
 * so the model sees the conflict instead of a silently chosen winner. Omitted
 * facts and their source ids stay explicit so exact originals remain
 * retrievable through Session Query.
 * @param facts - merged ledger records.
 * @param options - token budget and per-fact pricing.
 * @returns included facts plus explicit omissions.
 */
export function composeCompactView(facts: readonly FactRecord[], options: CompactViewOptions): CompactView {
  if (!Number.isSafeInteger(options.maxTokens) || options.maxTokens < 0) {
    throw new FactLedgerError('budget', 'fact ledger: maxTokens must be a non-negative integer')
  }
  const rank = (fact: FactRecord): number => fact.status === 'active' ? 0 : 1
  const candidates = facts
    .filter(fact => fact.status !== 'superseded')
    .sort((left, right) => rank(left) - rank(right)
      || (left.kind < right.kind ? -1 : left.kind > right.kind ? 1 : 0)
      || (left.key < right.key ? -1 : left.key > right.key ? 1 : 0)
      || (left.factId < right.factId ? -1 : left.factId > right.factId ? 1 : 0))
  const included: FactRecord[] = []
  const omitted: FactRecord[] = []
  let used = 0
  for (const fact of candidates) {
    const cost = options.tokensOf(fact)
    if (!Number.isSafeInteger(cost) || cost < 0) {
      throw new FactLedgerError('budget', 'fact ledger: tokensOf returned an invalid price')
    }
    if (used + cost > options.maxTokens) {
      omitted.push(fact)
      continue
    }
    used += cost
    included.push(fact)
  }
  const omittedSources = [...new Set(omitted.flatMap(fact => fact.sourceEventIds))].sort((a, b) => a - b)
  return Object.freeze<CompactView>({
    facts: Object.freeze([...included]),
    omittedFactIds: Object.freeze(omitted.map(fact => fact.factId)),
    omittedSourceEventIds: Object.freeze(omittedSources),
  })
}
