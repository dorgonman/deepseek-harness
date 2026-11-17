/** Wire values for profile-owned preset summary policy. */

/** An exact conversation route whose existing override the caller owns. */
export interface CompactionPolicyRoute {
  readonly provider: string
  readonly model: string
}

/** Explicit summary fields; absence preserves the preset's configured inheritance. */
export interface CompactionSummaryPolicy {
  readonly summarizationProvider?: string
  readonly summarizationModel?: string
  readonly summarizationFallback?: 'latest-route' | 'error'
  readonly maxTokens?: number
}

/** One preset's current summary fields, separate from its conversation default. */
export interface PresetCompactionPolicy {
  readonly entryId: string
  readonly presetId: string
  /** SHA256 of the declaration identity and default/exact-route summary fields. */
  readonly revision: string
  readonly policy: CompactionSummaryPolicy
  readonly modelPolicies: readonly (CompactionPolicyRoute & { readonly policy: CompactionSummaryPolicy })[]
}

/** A revision-checked change to one preset and explicitly selected existing overrides. */
export interface PresetCompactionPolicyUpdate {
  readonly entryId: string
  readonly expectedRevision: string
  readonly target: CompactionPolicyRoute & { readonly maxTokens: number }
  /** Existing override keys to update; every omitted override remains unchanged. */
  readonly modelPolicyRoutes: readonly CompactionPolicyRoute[]
}

/** Persisted policy after the replacement preset passed activation verification. */
export interface PresetCompactionPolicyWrite extends PresetCompactionPolicy {
  readonly persisted: true
  readonly activated: true
}

declare module '@deepseek-ai/dsh-typert-protocol' {
  interface RemoteErrorDetailsMap {
    /** The requested declaration is absent or not uniquely profile-owned. */
    'compaction-policy/not-found': { readonly entryId: string }
    /** The declaration or requested summary policy is invalid. */
    'compaction-policy/invalid': { readonly entryId: string; readonly reason: string }
    /** Declaration identity or summary fields changed after the caller read them. */
    'compaction-policy/conflict': { readonly entryId: string; readonly expectedRevision: string; readonly actualRevision: string }
    /** The requested ordinary Chat target cannot be verified. */
    'compaction-policy/unverified': { readonly entryId: string; readonly provider: string; readonly model: string }
    /** Persistence or replacement-preset activation failed. */
    'compaction-policy/rejected': { readonly entryId: string }
  }
}
