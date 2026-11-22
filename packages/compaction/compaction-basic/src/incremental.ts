/**
 * Incremental RAW coverage for repeated compaction: derive committed coverage
 * from the log, validate it, and select only append-origin events a later
 * summarization has not processed.
 *
 * @module @deepseek-ai/dsh-compaction-basic/incremental
 */

import { createHash } from 'node:crypto'
import {
  isCompactCheckpointSource,
  type CompactionCoveredRange,
  type CompactionIncrementalCoverage,
} from '@deepseek-ai/dsh-compaction'
import { isAppendSurfaceEvent, type SessionEvent, type SessionSeq } from '@deepseek-ai/dsh-session'

/** Failure class for incremental coverage derivation and selection. */
export class IncrementalCoverageError extends Error {
  constructor(
    /** Machine-readable cause. */
    readonly code: 'malformed' | 'overlap' | 'digest' | 'stale-revision' | 'work-limit',
    message: string,
  ) {
    super(message)
    this.name = 'IncrementalCoverageError'
  }
}

function malformed(message: string): never {
  throw new IncrementalCoverageError('malformed', `compaction: incremental coverage ${message}`)
}

/**
 * Stable digest of one resolved policy object.
 * @param value - JSON-serializable policy identity.
 * @returns lowercase hex SHA-256 of its canonical JSON.
 */
export function digestPolicy(value: unknown): string {
  return digestCanonical(value)
}

/**
 * Stable digest of one JSON-serializable value under canonical key order.
 * @param value - JSON-serializable content.
 * @returns lowercase hex SHA-256.
 */
export function digestCanonical(value: unknown): string {
  return createHash('sha256').update(canonicalJson(value)).digest('hex')
}

/** JSON with recursively sorted object keys, so equal content hashes equally. */
function canonicalJson(value: unknown): string {
  if (value === null) return 'null'
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return JSON.stringify(value)
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>
    const entries = Object.keys(record)
      .filter(key => record[key] !== undefined)
      .sort()
      .map(key => `${JSON.stringify(key)}:${canonicalJson(record[key])}`)
    return `{${entries.join(',')}}`
  }
  return malformed('contains a non-JSON value')
}

/**
 * SHA-256 over the canonical content of one contiguous event range.
 * @param events - the session log, in sequence order.
 * @param start - first seq, inclusive.
 * @param end - last seq, inclusive.
 * @returns lowercase hex digest binding the exact range content.
 */
export function digestRange(
  events: readonly SessionEvent[],
  start: SessionSeq,
  end: SessionSeq,
): string {
  if (start > end) malformed(`range ${start}-${end} has start after end`)
  const hash = createHash('sha256')
  let expected: number = start
  for (const event of events) {
    if (event.seq < start) continue
    if (event.seq > end) break
    if (event.seq !== expected) malformed(`range ${start}-${end} is not contiguous at ${event.seq}`)
    hash.update(canonicalJson({ type: event.type, seq: event.seq, time: event.time, data: event.data }))
    hash.update('\n')
    expected += 1
  }
  if (expected !== end + 1) malformed(`range ${start}-${end} is not fully present`)
  return hash.digest('hex')
}

/**
 * SHA-256 over the canonical content of an explicit set of event ids.
 * @param events - the session log, in sequence order.
 * @param ids - the exact source event ids to bind.
 * @returns lowercase hex digest over those events in log order.
 */
export function digestEventIds(events: readonly SessionEvent[], ids: readonly SessionSeq[]): string {
  const wanted = new Set<number>(ids)
  if (wanted.size !== ids.length) malformed('event ids are not unique')
  const hash = createHash('sha256')
  let found = 0
  for (const event of events) {
    if (!wanted.has(event.seq)) continue
    hash.update(canonicalJson({ type: event.type, seq: event.seq, time: event.time, data: event.data }))
    hash.update('\n')
    found += 1
  }
  if (found !== wanted.size) malformed('event ids reference events outside the log')
  return hash.digest('hex')
}
function assertRevision(revision: unknown): asserts revision is number {
  if (typeof revision !== 'number' || !Number.isInteger(revision) || revision < 0) {
    malformed('revision is not a non-negative integer')
  }
}

function assertRangeShape(range: unknown): asserts range is CompactionCoveredRange {
  if (typeof range !== 'object' || range === null) malformed('contains a non-object range')
  const candidate = range as Partial<CompactionCoveredRange>
  if (typeof candidate.start !== 'number' || typeof candidate.end !== 'number'
    || !Number.isInteger(candidate.start) || !Number.isInteger(candidate.end)
    || candidate.start < 0 || candidate.start > candidate.end) {
    malformed('contains an invalid range boundary')
  }
  if (typeof candidate.digest !== 'string' || candidate.digest.length === 0) {
    malformed('contains a range without a digest')
  }
}

function assertSummarizer(summarizer: unknown): void {
  if (summarizer === undefined) return
  if (typeof summarizer !== 'object' || summarizer === null) malformed('summarizer is not an object')
  const candidate = summarizer as Partial<NonNullable<CompactionIncrementalCoverage['summarizer']>>
  if (typeof candidate.provider !== 'string' || typeof candidate.model !== 'string'
    || typeof candidate.maxTokens !== 'number' || !Number.isInteger(candidate.maxTokens)
    || candidate.maxTokens < 0) {
    malformed('summarizer identity is incomplete')
  }
}

function assertCarried(carried: unknown): void {
  if (carried === undefined) return
  if (typeof carried !== 'object' || carried === null) malformed('carried provenance is not an object')
  const candidate = carried as Partial<NonNullable<CompactionIncrementalCoverage['carried']>>
  if (typeof candidate.summarySeq !== 'number' || !Number.isInteger(candidate.summarySeq)
    || typeof candidate.digest !== 'string' || candidate.digest.length === 0) {
    malformed('carried provenance is incomplete')
  }
}

/**
 * Validate one durable coverage record against the log it claims.
 *
 * Checks range shape and order, disjointness, the high-watermark relation and
 * every range digest. A record that fails any check is refused rather than
 * repaired, because a wrong coverage silently drops or duplicates RAW history.
 * @param coverage - the durable record read from a summary event.
 * @param events - the session log the record claims to describe.
 */
export function validateCoverage(
  coverage: CompactionIncrementalCoverage,
  events: readonly SessionEvent[],
): void {
  assertRevision(coverage.revision)
  if (typeof coverage.policyDigest !== 'string' || coverage.policyDigest.length === 0) {
    malformed('policy digest is missing')
  }
  assertSummarizer(coverage.summarizer)
  assertCarried(coverage.carried)
  if (typeof coverage.highWatermark !== 'number' || !Number.isInteger(coverage.highWatermark)
    || coverage.highWatermark < 0) {
    malformed('high watermark is not a non-negative integer')
  }
  if (!Array.isArray(coverage.covered)) malformed('covered ranges are not an array')
  let previousEnd = -1
  for (const range of coverage.covered) {
    assertRangeShape(range)
    if (range.start <= previousEnd) {
      throw new IncrementalCoverageError('overlap', `compaction: covered range ${range.start}-${range.end} overlaps the previous range`)
    }
    if (range.end > coverage.highWatermark) {
      malformed(`range ${range.start}-${range.end} ends beyond the high watermark ${coverage.highWatermark}`)
    }
    if (range.end >= events.length) malformed(`range ${range.start}-${range.end} is beyond the log`)
    if (digestRange(events, range.start, range.end) !== range.digest) {
      throw new IncrementalCoverageError('digest', `compaction: covered range ${range.start}-${range.end} does not match its digest`)
    }
    previousEnd = range.end
  }
  const last = coverage.covered.at(-1)
  if (last !== undefined && last.end !== coverage.highWatermark) {
    malformed(`high watermark ${coverage.highWatermark} does not equal the last covered end ${last.end}`)
  }
  if (coverage.covered.length === 0 && coverage.highWatermark !== 0 && coverage.revision !== 0) {
    malformed('has no ranges but a non-zero watermark or revision')
  }
}

/**
 * Whether a summary event's log transaction closed successfully with the
 * replacement message that carries the checkpoint. A dangling summary never
 * proves that its checkpoint became the surface.
 */
function transactionCommitted(
  events: readonly SessionEvent[],
  index: number,
  compactionId: string,
): boolean {
  const replacement = events[index + 1]
  if (replacement?.type !== 'user/message' || !isCompactCheckpointSource(replacement.data.source)) {
    return false
  }
  for (let cursor = index + 2; cursor < events.length; cursor += 1) {
    const event = events[cursor]
    if (event === undefined) break
    if (event.type === 'compaction/start' && event.data.compactionId === compactionId) return false
    if (event.type === 'compaction/end' && event.data.compactionId === compactionId) {
      return event.data.error === undefined
    }
  }
  return false
}

/**
 * Whether the summary event at `index` closed with its checkpoint replacement
 * and a clean `compaction/end`. Dangling or failed summaries never commit
 * derived state.
 * @param events - the session log, in sequence order.
 * @param index - candidate summary event index.
 * @returns true only for a complete successful transaction.
 */
export function summaryTransactionCommitted(events: readonly SessionEvent[], index: number): boolean {
  const event = events[index]
  if (event?.type !== 'compaction/summary') return false
  return transactionCommitted(events, index, event.data.compactionId)
}
/**
 * Read the latest committed incremental coverage from the session log.
 * @param events - the session log, in sequence order.
 * @returns the newest complete record, or undefined when none is committed.
 */
export function deriveCommittedCoverage(
  events: readonly SessionEvent[],
): CompactionIncrementalCoverage | undefined {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]
    if (event?.type !== 'compaction/summary' || event.data.incremental === undefined) continue
    if (!transactionCommitted(events, index, event.data.compactionId)) continue
    validateCoverage(event.data.incremental, events)
    return event.data.incremental
  }
  return undefined
}

/**
 * Refuse a stale compare-and-set generation.
 * @param current - committed coverage, or undefined before the first commit.
 * @param expected - the generation captured when the attempt started.
 */
export function assertCoverageRevision(
  current: CompactionIncrementalCoverage | undefined,
  expected: number,
): void {
  const actual = current?.revision ?? 0
  if (actual !== expected) {
    throw new IncrementalCoverageError(
      'stale-revision',
      `compaction: coverage revision changed from ${expected} to ${actual}`,
    )
  }
}

/**
 * Select append-origin surface events above the committed high watermark.
 *
 * Checkpoint replacements are excluded because they are derived views, not new
 * RAW evidence. A work cap refuses the attempt instead of returning a partial
 * range, so total extraction work is bounded by the caller's policy.
 * @param events - the session log, in sequence order.
 * @param coverage - committed coverage, or undefined for a full RAW selection.
 * @param options - inclusive maximum number of selected events.
 * @returns the unprocessed append-origin events, in log order.
 */
export function selectUnprocessedRawEvents(
  events: readonly SessionEvent[],
  coverage: CompactionIncrementalCoverage | undefined,
  options: { readonly maxEvents: number },
): SessionEvent[] {
  if (!Number.isInteger(options.maxEvents) || options.maxEvents < 0) malformed('work cap is not a non-negative integer')
  const from = coverage === undefined ? 0 : coverage.highWatermark + 1
  const selected: SessionEvent[] = []
  for (const event of events) {
    if (event.seq < from || !isAppendSurfaceEvent(event)) continue
    selected.push(event)
    if (selected.length > options.maxEvents) {
      throw new IncrementalCoverageError(
        'work-limit',
        `compaction: more than ${options.maxEvents} unprocessed RAW events`,
      )
    }
  }
  return selected
}

/**
 * Whether any append-origin event above the committed watermark exists.
 * @param events - the session log, in sequence order.
 * @param coverage - committed coverage, or undefined for a full RAW selection.
 * @returns true when a summarizer call could process new RAW evidence.
 */
export function hasUnprocessedRawEvents(
  events: readonly SessionEvent[],
  coverage: CompactionIncrementalCoverage | undefined,
): boolean {
  const from = coverage === undefined ? 0 : coverage.highWatermark + 1
  return events.some(event => event.seq >= from && isAppendSurfaceEvent(event))
}

/** Identity and provenance supplied by the backend when it commits coverage. */
export interface IncrementalAdvanceMeta {
  /** Digest of the resolved compaction policy. */
  readonly policyDigest: string
  /** Summarizer identity, when a model call produced new text. */
  readonly summarizer?: CompactionIncrementalCoverage['summarizer']
  /** Provenance of checkpoint text carried forward from an earlier summary. */
  readonly carried?: CompactionIncrementalCoverage['carried']
}

/**
 * Advance committed coverage over one newly processed contiguous range.
 *
 * Adjacent ranges coalesce so the payload stays bounded across many small
 * compactions. The digest of a coalesced range is recomputed over the whole
 * span, so every recorded digest still verifies against the log.
 * @param previous - committed coverage, or undefined for the first commit.
 * @param events - the session log, in sequence order.
 * @param start - first newly covered seq, inclusive.
 * @param end - last newly covered seq, inclusive.
 * @param meta - policy and provenance identity for the new record.
 * @returns the next coverage record with an incremented revision.
 */
export function advanceCoverage(
  previous: CompactionIncrementalCoverage | undefined,
  events: readonly SessionEvent[],
  start: SessionSeq,
  end: SessionSeq,
  meta: IncrementalAdvanceMeta,
): CompactionIncrementalCoverage {
  if (start > end) malformed(`range ${start}-${end} has start after end`)
  const floor = previous?.highWatermark ?? -1
  if (start <= floor) {
    throw new IncrementalCoverageError(
      'stale-revision',
      `compaction: range ${start}-${end} does not advance past watermark ${floor}`,
    )
  }
  const covered = [...(previous?.covered ?? [])]
  const last = covered.at(-1)
  if (last !== undefined && last.end + 1 === start) {
    covered[covered.length - 1] = { start: last.start, end, digest: digestRange(events, last.start, end) }
  } else {
    covered.push({ start, end, digest: digestRange(events, start, end) })
  }
  return {
    revision: (previous?.revision ?? 0) + 1,
    highWatermark: end,
    covered,
    policyDigest: meta.policyDigest,
    ...meta.summarizer === undefined ? {} : { summarizer: meta.summarizer },
    ...meta.carried === undefined ? {} : { carried: meta.carried },
  }
}
