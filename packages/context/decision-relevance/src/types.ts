import type { MessageId, ToolCallId } from '@deepseek-ai/dsh-llm'

/** How one eligible tool result was dispositioned for the outbound request. */
export type RelevanceDisposition =
  /** The Decision model judged the content relevant; it was retained. */
  | 'retained-relevant'
  /** The Decision model judged the content irrelevant; its body was elided for this request only. */
  | 'elided-irrelevant'
  /** Evaluation was skipped by a bounded limit; the content was retained. */
  | 'retained-limit'
  /** Evaluation was uncertain (score inside the uncertainty band); the content was retained. */
  | 'retained-uncertain'
  /** Evaluation failed or was cancelled; the content was retained. */
  | 'retained-error'

/** Validated, detached, deeply immutable relevance-filter configuration. */
export interface ResolvedConfig {
  /** `off` disables the feature; `on` applies validated selections to the outbound projection. */
  readonly mode: 'off' | 'on'
  /** Explicit Decision route; absent disables the feature regardless of `mode`. */
  readonly provider?: string
  /** Explicit Decision model; absent disables the feature regardless of `mode`. */
  readonly model?: string
  /** Minimum relevance score required to retain a tool result. */
  readonly retainThreshold: number
  /** Half-width of the symmetric uncertainty band around `retainThreshold`. */
  readonly uncertaintyMargin: number
  /** Maximum eligible tool results evaluated per request. */
  readonly maxCandidatesPerRequest: number
  /** Minimum text size in Unicode code points before a tool result is eligible. */
  readonly minCandidateChars: number
  /** Maximum text size in Unicode code points sent to the Decision model. */
  readonly maxInputChars: number
  /** End-to-end deadline for one Decision evaluation in milliseconds. */
  readonly timeoutMs: number
  /** Replacement text for an elided tool-result body. */
  readonly elisionText: string
}

/**
 * Sparse relevance-filter configuration as a composition declares it.
 *
 * Every field is optional and falls back to {@link DEFAULTS}; `provider` and
 * `model` must be configured together, and the feature stays inactive unless
 * `mode` is `on` **and** both route fields are present.
 */
export interface DecisionRelevanceConfig {
  /** `off` disables the feature; `on` applies validated selections to the outbound projection. */
  readonly mode?: 'off' | 'on'
  /** Canonical provider of the Decision route. */
  readonly provider?: string
  /** Canonical model of the Decision route. */
  readonly model?: string
  /** Minimum relevance score required to retain a tool result. */
  readonly retainThreshold?: number
  /** Half-width of the symmetric uncertainty band around `retainThreshold`. */
  readonly uncertaintyMargin?: number
  /** Maximum eligible tool results evaluated per request. */
  readonly maxCandidatesPerRequest?: number
  /** Minimum text size in Unicode code points before a tool result is eligible. */
  readonly minCandidateChars?: number
  /** Maximum text size in Unicode code points sent to the Decision model. */
  readonly maxInputChars?: number
  /** End-to-end deadline for one Decision evaluation in milliseconds. */
  readonly timeoutMs?: number
  /** Replacement text for an elided tool-result body. */
  readonly elisionText?: string
}

/** One bounded durable judgment record; carries no credentials or history dump. */
export interface RelevanceJudgment {
  /** Stable identity of the tool-result message this judgment concerns. */
  readonly messageId: MessageId
  /** Tool call the candidate answers; preserves call/result correlation identity. */
  readonly callId: ToolCallId
  /** Candidate text size in Unicode code points before any input bound. */
  readonly candidateChars: number
  /** Candidate text size actually sent after the input bound. */
  readonly inputChars: number
  /** Disposition applied to the outbound projection. */
  readonly disposition: RelevanceDisposition
  /** Short bounded reason for the disposition. */
  readonly reason: string
  /** Requested canonical Decision route. */
  readonly requested: { readonly provider: string; readonly model: string }
  /** Served canonical Decision identity fields, bounded and credential-free. */
  readonly served?: { readonly model: string; readonly mode: string; readonly instance: string }
  /** Validated relevance score in `[0, 1]`. */
  readonly score?: number
  /** Validated served label. */
  readonly label?: 'yes' | 'no'
}

/** Aggregate outcome of one outbound relevance-projection pass. */
export interface ProjectionResult {
  /** Whether any content was actually elided for this request. */
  readonly applied: boolean
  /** Judgments recorded for this pass, in candidate order. */
  readonly judgments: readonly RelevanceJudgment[]
}
