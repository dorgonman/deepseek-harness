/**
 * Structured fact extraction: parse a summarizer's JSON claim list into
 * validated `FactInput` rows, then merge them into the Fact Ledger. Malformed
 * output and out-of-chunk citations are refused, never repaired.
 *
 * @module @deepseek-ai/dsh-compaction-basic/fact-extraction
 */

import type { SessionEvent, SessionSeq } from '@deepseek-ai/dsh-session'
import {
  createFacts,
  FACT_KINDS,
  mergeFacts,
  type FactInput,
  type FactBasis,
  type FactRecord,
} from './fact-ledger.ts'

/** Failure classes for structured fact extraction. */
export class FactExtractionError extends Error {
  constructor(
    /** Machine-readable cause. */
    readonly code: 'invalid-json' | 'malformed' | 'unknown-kind' | 'unknown-source' | 'bounds',
    message: string,
  ) {
    super(message)
    this.name = 'FactExtractionError'
  }
}

/** Extraction bounds supplied by configuration. */
export interface FactExtractionOptions {
  /** Maximum claims accepted from one summarizer response. */
  readonly maxFacts: number
  /** Maximum claim text length in characters. */
  readonly maxClaimChars: number
}

function malformed(message: string): never {
  throw new FactExtractionError('malformed', `fact extraction: ${message}`)
}

function nonEmpty(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

function isSeqSet(value: unknown): value is number[] {
  return Array.isArray(value) && value.every(item => typeof item === 'number' && Number.isSafeInteger(item) && item >= 0)
}

/**
 * Parse one structured claim response.
 *
 * Accepted shape: a JSON array of
 * `{ kind, key, claim, sourceSeqs, basis?, supersedes? }`. Basis
 * defaults to `inferred` so extracted prose is never silently promoted to a
 * RAW assertion. Every cited source must belong to the summarized chunk.
 * @param text - raw summarizer output.
 * @param allowedSources - event ids present in the summarized chunk.
 * @param options - count and length bounds.
 * @returns validated inputs in response order.
 */
export function parseFactExtraction(
  text: string,
  allowedSources: readonly SessionSeq[],
  options: FactExtractionOptions,
): FactInput[] {
  if (!Number.isSafeInteger(options.maxFacts) || options.maxFacts < 1
    || !Number.isSafeInteger(options.maxClaimChars) || options.maxClaimChars < 1) {
    throw new FactExtractionError('bounds', 'fact extraction: bounds must be positive integers')
  }
  let output: unknown
  try {
    output = JSON.parse(text)
  } catch {
    throw new FactExtractionError('invalid-json', 'fact extraction: output is not valid JSON')
  }
  if (!Array.isArray(output) || output.length === 0 || output.length > options.maxFacts) {
    throw new FactExtractionError('bounds', `fact extraction: expected 1-${options.maxFacts} claims`)
  }
  const allowed = new Set<number>(allowedSources)
  const items: readonly unknown[] = output
  return items.map((item, index) => {
    if (typeof item !== 'object' || item === null) malformed(`claim ${index} is not an object`)
    const record = item as Record<string, unknown>
    const kind = record.kind
    if (typeof kind !== 'string' || !FACT_KINDS.some(candidate => candidate === kind)) {
      throw new FactExtractionError('unknown-kind', `fact extraction: claim ${index} has an unknown kind`)
    }
    const key = record.key
    const claim = record.claim
    if (!nonEmpty(key) || !nonEmpty(claim)) malformed(`claim ${index} has an empty key or claim`)
    if (claim.length > options.maxClaimChars) {
      throw new FactExtractionError('bounds', `fact extraction: claim ${index} exceeds ${options.maxClaimChars} characters`)
    }
    const sourceSeqs = record.sourceSeqs
    if (!isSeqSet(sourceSeqs) || sourceSeqs.length === 0) malformed(`claim ${index} has no source events`)
    const sources = [...new Set(sourceSeqs)]
    if (sources.length !== sourceSeqs.length) malformed(`claim ${index} repeats a source event`)
    for (const source of sources) {
      if (!allowed.has(source)) {
        throw new FactExtractionError('unknown-source', `fact extraction: claim ${index} cites event ${source} outside its chunk`)
      }
    }
    const basis = record.basis ?? 'inferred'
    if (basis !== 'asserted' && basis !== 'inferred') {
      malformed(`claim ${index} has an invalid basis`)
    }
    const supersedes = record.supersedes
    if (supersedes !== undefined
      && (!Array.isArray(supersedes) || supersedes.some(target => !nonEmpty(target)))) {
      malformed(`claim ${index} has an invalid supersedes list`)
    }
    return {
      kind: kind as FactInput['kind'],
      key: key.trim(),
      claim: claim.trim(),
      sourceEventIds: sources.map(source => source as SessionSeq),
      basis: basis as FactBasis,
      ...supersedes === undefined ? {} : { supersedes: (supersedes as string[]).map(target => target.trim()) },
    }
  })
}

/**
 * Parse one response and merge its facts into an existing ledger.
 * @param prior - committed ledger records.
 * @param text - raw summarizer output.
 * @param allowedSources - event ids present in the summarized chunk.
 * @param events - the session log those ids address.
 * @param options - count and length bounds.
 * @returns the merged ledger, with dedup, explicit updates and conflicts applied.
 */
export function mergeExtractedFacts(
  prior: readonly FactRecord[],
  text: string,
  allowedSources: readonly SessionSeq[],
  events: readonly SessionEvent[],
  options: FactExtractionOptions,
): FactRecord[] {
  const parsed = parseFactExtraction(text, allowedSources, options)
  return mergeFacts(prior, createFacts(parsed, events))
}
