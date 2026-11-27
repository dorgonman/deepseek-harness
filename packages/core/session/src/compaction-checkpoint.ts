/** Publication state of one incremental compaction on a Session. */
import { brandString, type Branded } from '@deepseek-ai/dsh-brand'
import { SessionSeq, type SessionEvent } from './types.ts'

type CheckpointId = Branded<'CompactionId'>

/** Summary or replacement waiting for a matching successful close. */
export type PendingCompactionCheckpoint =
  | {
    readonly phase: 'summary'
    readonly id: CheckpointId
    readonly seq: SessionSeq
    readonly start: SessionSeq
    readonly end: SessionSeq
  }
  | { readonly phase: 'replacement'; readonly id: CheckpointId; readonly seq: SessionSeq }

/** Change to checkpoint publication state. An absent transition preserves the state. */
export type CompactionCheckpointTransition =
  | { readonly kind: 'arm'; readonly checkpoint: Extract<PendingCompactionCheckpoint, { phase: 'summary' }> }
  | { readonly kind: 'defer'; readonly checkpoint: Extract<PendingCompactionCheckpoint, { phase: 'replacement' }> }
  | { readonly kind: 'suppress' | 'clear' }
  | { readonly kind: 'publish'; readonly seq: SessionSeq }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

/**
 * Identify compaction summaries that commit derived state with their replacement.
 * @param event - a Session record, including records declared by packages.
 * @returns whether its replacement requires a successful close.
 */
export function isGatedCompactionSummary(
  event: { type: string; data: unknown },
): event is { type: 'compaction/summary'; data: Record<string, unknown> } {
  return event.type === 'compaction/summary' && isRecord(event.data)
    && (event.data['incremental'] !== undefined || event.data['ledger'] !== undefined)
}

/**
 * Interpret a checkpoint event in log order. A Session has one compaction
 * owner. Returning from a seed or starting another compaction abandons its
 * incomplete transaction.
 * @param pending - checkpoint state before the event.
 * @param event - next contiguous Session event.
 * @returns a state change or publication, if any.
 */
export function advanceCompactionCheckpoint(
  pending: PendingCompactionCheckpoint | undefined,
  event: SessionEvent,
): CompactionCheckpointTransition | undefined {
  // Compaction declares these durable records above Session in the package graph.
  const record: { type: string; data: unknown } = event
  if (isGatedCompactionSummary(record)) {
    const { compactionId, shadowedRange } = record.data
    if (typeof compactionId !== 'string' || !isRecord(shadowedRange)
      || typeof shadowedRange['start'] !== 'number' || typeof shadowedRange['end'] !== 'number') {
      throw new Error('Incremental compaction summary must identify its checkpoint and replacement range')
    }
    return { kind: 'arm', checkpoint: {
      phase: 'summary', id: brandString<CheckpointId>(compactionId), seq: event.seq,
      start: SessionSeq(shadowedRange['start']), end: SessionSeq(shadowedRange['end']),
    } }
  }
  if (pending === undefined) return undefined
  if (record.type === 'compaction/start' || record.type === 'session/end-seed') return { kind: 'clear' }
  if (pending.phase === 'summary') {
    if (event.type !== 'user/message') return { kind: 'clear' }
    const source: { kind: string } = event.data.source
    if (source.kind !== 'compact-checkpoint') return { kind: 'clear' }
    const op = event.surfaceOp
    if (event.seq !== pending.seq + 1 || !('compactionId' in source) || source.compactionId !== pending.id
      || op === 'append' || op.startSeq !== pending.start
      || op.endSeq !== pending.end) return { kind: 'suppress' }
    return { kind: 'defer', checkpoint: { phase: 'replacement', id: pending.id, seq: event.seq } }
  }
  if (record.type !== 'compaction/end' || !isRecord(record.data)
    || record.data['compactionId'] !== pending.id) return undefined
  return record.data['error'] === undefined ? { kind: 'publish', seq: pending.seq } : { kind: 'clear' }
}
