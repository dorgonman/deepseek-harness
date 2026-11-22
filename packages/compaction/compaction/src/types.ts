/**
 * Compaction vocabulary: the result type and the `compaction/*` session events.
 * Those declaration-merged events record the lock and summary inputs without entering the surface, so they are not
 * surface events; a separate replacement `user/message` carries the summary.
 * Backend packages own configuration and retention policy; see the
 * [compaction reference](../README.md#understand-the-implementation).
 * @module @deepseek-ai/dsh-compaction/types
 */

import type { ContentBlock, ReasoningEffortId, TokenUsage } from '@deepseek-ai/dsh-llm'
import type { CommandId } from '@deepseek-ai/dsh-commands/brand'
import type { SessionSeq } from '@deepseek-ai/dsh-session/types'
import type { CompactionId } from './brand.ts'

export type { CompactionId }

/** The selected input representation of one auxiliary summarization call. */
export type SummaryRequestProjection = 'conversation-prefix-v1' | 'text-transcript-v1'

/** One contiguous append-origin event range represented by a checkpoint. */
export interface CompactionCoveredRange {
  /** First append-origin seq, inclusive. */
  start: SessionSeq
  /** Last append-origin seq, inclusive. */
  end: SessionSeq
  /** SHA-256 over the canonical content of every event in the range. */
  digest: string
}

/**
 * Durable incremental RAW coverage carried by a committed summary. Its
 * presence marks a summary produced by incremental extraction; absence keeps
 * the full-surface behavior of earlier backends and never implies coverage.
 */
export interface CompactionIncrementalCoverage {
  /** Compare-and-set generation; incremented once per committed incremental summary. */
  revision: number
  /** Highest append-origin seq whose content this checkpoint accounts for. */
  highWatermark: SessionSeq
  /** Disjoint, ordered raw ranges represented by the checkpoint. */
  covered: CompactionCoveredRange[]
  /** Digest of the resolved compaction policy that produced the coverage. */
  policyDigest: string
  /** Summarizer identity that produced the new text, when one was called. */
  summarizer?: { provider: string; model: string; maxTokens: number }
  /** Provenance of checkpoint text carried forward from an earlier summary. */
  carried?: { summarySeq: SessionSeq; digest: string }
}

/** Replay facts for one successful Hybrid summarizer call. */
interface CompactionChunkCall {
  /** Original event groups in request order, with any tool-body substitution. */
  groups: { eventSeqs: SessionSeq[]; toolBodiesOmitted: boolean }[]
  /** Selected summarizer window and measured projected request size. */
  contextWindow: number
  estimatedInputTokens: number
  /** Adapter-resolved reasoning control, when applied. */
  reasoningEffort?: ReasoningEffortId
  /** Complete output before checkpoint projection and ordered merge. */
  rawOutput: Extract<ContentBlock, { type: 'text' | 'reasoning' }>[]
  usage?: TokenUsage
}

declare module '@deepseek-ai/dsh-session/types' {
  interface SessionEventMap {
    /**
     * Marks the start of a compaction — log-only, holds the lock until
     * `compaction/end`. A numbered owner is strictly enclosed by that open turn;
     * `null` identifies a standalone manual transaction between turns.
     */
    'compaction/start': { compactionId: CompactionId; sourceCommandId?: CommandId; turn: number | null }
    /**
     * Completed summary, its inputs, and its model call facts — log-only, no surfaceOp.
     * The summary content is in `data.summary`; the actual surface replacement
     * is performed by the immediately following `user/message` event that
     * shadows the compacted range. That adjacency is contractual — the
     * shadowed pricing fields are the replacement's shadow price, so a
     * consumer may pair a replacement with the metering event directly
     * before it (`compaction/prune` documents the shared protocol).
     */
    'compaction/summary': {
      compactionId: CompactionId
      sourceCommandId?: CommandId
      summary: ContentBlock[]
      shadowedRange: { start: SessionSeq; end: SessionSeq }
      shadowedSeqs: SessionSeq[]
      shadowedTokenCount: number
      /** The provider route that wrote the summary. */
      provider: string
      /**
       * The model that wrote the summary — the summarize call's envelope,
       * reported by the backend that made the call, logged so the one-shot
       * request is reconstructable from log + code and "which model wrote
       * this summary" has a durable answer (the reconstructability Agent Note).
       */
      model: string
      /** The generation cap the summarize call sent, when one applied. */
      maxTokens?: number
      /**
       * Selected auxiliary input representation. Absence retains recorded historical
       * or unspecified custom-summarizer behavior; never infer it from current capabilities.
       */
      requestProjection?: SummaryRequestProjection
      /** Provider-reported token usage for the summarization request, when emitted. */
      usage?: TokenUsage
      /**
       * Incremental RAW coverage committed with this checkpoint. Absent on a
       * full-surface summary; readers must not infer coverage for that case.
       */
      incremental?: CompactionIncrementalCoverage
    } & (
      | {
        /** Complete provider output before the backend's safe summary projection. */
        rawOutput: ContentBlock[]
        /** Identifies exactly one call through this context's `ctx.llm.stream()`. */
        llmStreamCall: true
        chunkCalls?: never
      }
      | {
        /** Optional complete output from an unmarked template, remote, or other summarizer. */
        rawOutput?: ContentBlock[]
        /** An unmarked summary does not identify a call through this context's LLM seam. */
        llmStreamCall?: never
        chunkCalls?: never
      }
      | {
        /** Successful Hybrid requests in dispatch order. */
        chunkCalls: CompactionChunkCall[]
        rawOutput?: never
        llmStreamCall?: never
      }
    )
    /**
     * Marks the end of a compaction — log-only, releases the lock. Its owner
     * matches `compaction/start`; `error` records an unsuccessful attempt.
     */
    'compaction/end': { compactionId: CompactionId; sourceCommandId?: CommandId; turn: number | null; error?: string }
    /**
     * Shadow price of one model-free prune replacement — log-only, no
     * surfaceOp. The shared shadow-price protocol: a surface `replace` event
     * is priced by the metering event immediately before it (`compaction/summary`
     * for a summarizing compaction, this event for a prune), which states the
     * heuristic token price of the exact replaced range so a pure consumer
     * can subtract it without retaining per-node prices. The replacement MUST
     * be appended synchronously right after this event.
     */
    'compaction/prune': {
      /** The replaced range's first and last surface-node seqs (a surface-position span, like {@link CompactionResult.shadowedRange}). */
      shadowedRange: { start: SessionSeq; end: SessionSeq }
      /** The seqs of all shadowed surface nodes, in surface order. */
      shadowedSeqs: SessionSeq[]
      /** Heuristic price of the shadowed content under the token-meter's fixed estimator. */
      shadowedTokenCount: number
    }
  }
}

/** Result of a successful compaction operation. */
export interface CompactionResult {
  /** Stable identity shared by this compaction's complete durable lifecycle. */
  compactionId: CompactionId
  /** Human command that initiated this compaction, when it was manual. */
  sourceCommandId?: CommandId
  /** The seq of the appended `compaction/start` event. */
  startSeq: SessionSeq
  /** The seq of the appended `compaction/summary` event. */
  summarySeq: SessionSeq
  /** The seq of the appended `compaction/end` event. */
  endSeq: SessionSeq
  /** The summary content blocks produced by the backend. */
  summary: ContentBlock[]
  /**
   * The surface-boundary pair that was shadowed: the seqs of the first
   * (`start`) and last (`end`) surface nodes of the replaced range. A
   * surface-POSITION span, not a numeric seq interval — after a prior replace
   * lands a fresh high-seq summary node at an older range's position, `start`
   * can be GREATER than `end`. {@link CompactionResult.shadowedSeqs} is the
   * authoritative set of shadowed nodes, in surface order.
   */
  shadowedRange: { start: SessionSeq; end: SessionSeq }
  /** The seqs of all shadowed surface nodes, in surface order. */
  shadowedSeqs: SessionSeq[]
  /** Estimated token count of the shadowed content. */
  shadowedTokenCount: number
}
