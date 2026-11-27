/**
 * KINF resident-only connected transport.
 *
 * A resident-bound call must be served by the exact loaded deployment the
 * caller measured, over the dedicated `/kinf/resident/v1/chat/completions`
 * path rather than the ordinary pi-ai dispatch a merely resolved model alias
 * would take. This module owns that request end to end — the
 * `kinf_resident_only` precondition, the first-frame and finish-frame served
 * identity checks, and the SSE-to-`StreamChunk` projection — with no Cordis
 * context, no DSH configuration layer, and no pi-ai SDK, so the same function
 * can also be driven directly by a controlled acceptance run.
 *
 * Nothing from an unverified stream is published: text is buffered until the
 * finish frame has reasserted the binding, and any contract violation fails
 * the call without emitting chunks.
 *
 * @module dsh-llm-pi-ai/resident
 */

import { randomUUID } from 'node:crypto'
import { LlmError } from '@deepseek-ai/dsh-llm'
import type {
  BuiltinLlmFailureCode,
  FinishReason,
  ResidentCallBinding,
  StreamChunk,
  TokenUsage,
} from '@deepseek-ai/dsh-llm'

/** Dedicated resident-only connected endpoint, appended to the model row's origin. */
const RESIDENT_PATH = '/kinf/resident/v1/chat/completions'

/** Resident-only contract schema the binding and every accepted reply frame declare. */
const RESIDENT_SCHEMA = 'kinf.resident-only.v1'

/** Request-lifetime, cold-load, and eviction headers the host rejects on this path. */
const FORBIDDEN_HEADERS = [
  'x-kinf-request-lifetime', 'x-kinf-allow-cold-load', 'x-kinf-allow-eviction', 'allow-cold-load', 'allow-eviction',
] as const

/** Token grammar every `kinf_resident_only` value must match. */
const TOKEN = /^[A-Za-z0-9_.:+/-]{1,128}$/

/** Media type an accepted resident response must carry. */
const EVENT_STREAM = /^text\/event-stream(?:;|$)/i

/** One KINF resident-only connected request. */
export interface KinfResidentRequest {
  /** The selected model row's configured endpoint; only its origin is used. */
  baseURL: string
  /** Static request headers resolved for the row, including any credential. */
  headers: Record<string, string>
  /** Wire model id sent as `model`. */
  requestModel: string
  /** Plain-text conversation history in OpenAI chat-message shape. */
  messages: readonly { role: 'system' | 'user' | 'assistant'; content: string }[]
  /** Completion cap sent as `max_tokens`; omission leaves the host default. */
  maxTokens?: number
  /** Sampling temperature; the contract only accepts zero, sent when set. */
  temperature?: 0
  /** Immutable resident dispatch identity this request must be served under. */
  binding: ResidentCallBinding
  /** Caller cancellation forwarded to the request. */
  signal?: AbortSignal
  /** Reports received nonempty text while the reply awaits terminal validation. */
  onTextProgress?: () => void
}

/** A contract violation in a resident reply. */
function malformed(): LlmError {
  return new LlmError('KINF resident reply violated the resident-only connected contract.', 'MALFORMED_RESPONSE')
}

/**
 * Narrow one JSON value to a plain object without throwing.
 * @param value - any decoded JSON value.
 * @returns the value as a record, or undefined for arrays, null, and scalars.
 */
function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined
}

/**
 * Require one JSON value to be a plain object.
 * @param value - any decoded JSON value.
 * @returns the value as a record.
 * @throws LlmError when the value is not a plain object.
 */
function object(value: unknown): Record<string, unknown> {
  const result = record(value)
  if (result === undefined) throw malformed()
  return result
}

/**
 * Validate one resident precondition value against the contract's token grammar.
 * @param value - the binding value to send.
 * @param field - binding field name used in the diagnostic.
 * @returns the validated value.
 * @throws LlmError when the value is not a non-empty contract token.
 */
function token(value: string, field: string): string {
  if (typeof value !== 'string' || !TOKEN.test(value)) {
    throw new LlmError(`KINF resident binding ${field} must match ${TOKEN.source}`, 'INVALID_REQUEST')
  }
  return value
}

/**
 * Build the exactly nine wire members of `kinf_resident_only` from one binding.
 * `projectionId` is DSH-internal pricing identity and never leaves the process.
 * @param binding - immutable resident dispatch identity.
 * @returns the precondition object sent with the request.
 * @throws LlmError when the binding does not declare the resident contract or a value is not a token.
 */
function precondition(binding: ResidentCallBinding): Record<string, string> {
  if (binding.schema !== RESIDENT_SCHEMA) {
    throw new LlmError(`KINF resident binding schema must be "${RESIDENT_SCHEMA}"`, 'INVALID_REQUEST')
  }
  return {
    schema: RESIDENT_SCHEMA,
    instance: token(binding.instance, 'instance'),
    host_generation: token(binding.hostGeneration, 'hostGeneration'),
    serving_variant_id: token(binding.servingVariantId, 'servingVariantId'),
    serving_variant_revision: token(binding.servingVariantRevision, 'servingVariantRevision'),
    model_artifact: token(binding.modelArtifact, 'modelArtifact'),
    artifact_revision: token(binding.artifactRevision, 'artifactRevision'),
    target_config_revision: token(binding.targetConfigRevision, 'targetConfigRevision'),
    resident_generation: token(binding.residentGeneration, 'residentGeneration'),
  }
}

/**
 * Derive the resident-only endpoint from the selected model row's endpoint.
 * @param baseURL - the row's configured endpoint; only its origin is kept.
 * @returns the dedicated resident-only URL.
 * @throws LlmError when the endpoint is not an absolute HTTP(S) URL.
 */
function endpoint(baseURL: string): string {
  let url: URL
  try {
    url = new URL(baseURL)
  } catch (error) {
    throw new LlmError('KINF resident transport requires an absolute HTTP(S) model endpoint', 'INVALID_CONFIG', { cause: error })
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new LlmError('KINF resident transport requires an absolute HTTP(S) model endpoint', 'INVALID_CONFIG')
  }
  return `${url.origin}${RESIDENT_PATH}`
}

/**
 * Compose the resident request headers: the row's static headers, a JSON
 * content type, and a fresh request id, never the cold-load headers.
 * @param staticHeaders - provider and model headers resolved for this call.
 * @returns the headers to send.
 */
function wireHeaders(staticHeaders: Readonly<Record<string, string>>): Headers {
  const headers = new Headers()
  for (const [name, value] of Object.entries(staticHeaders)) headers.set(name, value)
  for (const name of FORBIDDEN_HEADERS) headers.delete(name)
  headers.set('content-type', 'application/json')
  headers.set('x-request-id', randomUUID())
  return headers
}

/**
 * Dispatch one resident request, mapping abort and transport failures.
 * @param request - the resident request being dispatched.
 * @param url - the dedicated resident endpoint.
 * @param headers - composed request headers.
 * @param body - serialized request body.
 * @returns the HTTP response.
 * @throws LlmError `ABORTED` when the caller aborted, otherwise `TRANSPORT`.
 */
async function send(request: KinfResidentRequest, url: string, headers: Headers, body: string): Promise<Response> {
  try {
    return await fetch(url, {
      method: 'POST',
      headers,
      body,
      redirect: 'error',
      ...request.signal === undefined ? {} : { signal: request.signal },
    })
  } catch (error) {
    if (request.signal?.aborted === true) {
      throw new LlmError('KINF resident request aborted by caller', 'ABORTED', { cause: error })
    }
    throw new LlmError('KINF resident transport failed', 'TRANSPORT', { cause: error })
  }
}

/**
 * Map one non-2xx status without a resident verdict onto the shared taxonomy.
 * @param status - the HTTP status.
 * @returns the provider-neutral failure code.
 */
function httpCode(status: number): BuiltinLlmFailureCode {
  if (status === 401 || status === 403) return 'AUTH'
  if (status === 429) return 'RATE_LIMIT'
  if (status === 408) return 'TIMEOUT'
  return status >= 500 ? 'SERVER' : 'INVALID_REQUEST'
}

/**
 * Read one non-2xx body and classify its pre-admission or post-admission verdict.
 * @param response - the failed HTTP response.
 * @returns the typed failure to throw.
 */
async function failure(response: Response): Promise<LlmError> {
  const body = record(await response.json().catch(() => undefined))
  const accepted = body?.['accepted']
  const hostCode = record(body?.['error'])?.['code']
  const detail = typeof hostCode === 'string' && TOKEN.test(hostCode) ? `; host error "${hostCode}"` : ''
  if (accepted === false && body?.['started'] === false && body['request_contract'] === RESIDENT_SCHEMA
    && response.status === 409
    && (hostCode === 'resident_identity_mismatch' || hostCode === 'resident_target_changed' || hostCode === 'resident_not_loaded')) {
    return new LlmError(
      `KINF resident request was refused before admission (HTTP ${response.status}${detail})`,
      'RESIDENT_PRECONDITION_STALE',
      { status: response.status },
    )
  }
  if (accepted === true) {
    return new LlmError(
      `KINF resident request failed after admission (HTTP ${response.status}${detail})`,
      'RESIDENT_ACCEPTANCE_AMBIGUOUS',
      { status: response.status },
    )
  }
  return new LlmError(
    `KINF resident request failed (HTTP ${response.status}${detail})`,
    httpCode(response.status),
    { status: response.status },
  )
}

/**
 * Require an accepted resident response to be an SSE event stream.
 * @param response - the successful HTTP response.
 * @returns the response body stream.
 * @throws LlmError when the media type is not SSE or the body is absent.
 */
async function eventStream(response: Response): Promise<ReadableStream<Uint8Array>> {
  const body = response.body
  if (!EVENT_STREAM.test(response.headers.get('content-type') ?? '') || body === null) {
    if (body !== null) await body.cancel()
    throw malformed()
  }
  return body
}

/**
 * The first streamed choice of one reply frame, when the frame carries any.
 * @param frame - one parsed SSE frame.
 * @returns the first choice record, or undefined for a frame without choices.
 */
function firstChoice(frame: Record<string, unknown>): Record<string, unknown> | undefined {
  const choices = frame['choices']
  if (!Array.isArray(choices) || choices.length === 0) return undefined
  return object(choices[0])
}

/**
 * Require one acceptance or finish frame to reassert the full served identity.
 * @param frame - the frame to validate.
 * @param binding - the identity this call was bound to.
 * @throws LlmError when the frame violates the resident-only contract.
 */
function validateFrame(frame: Record<string, unknown>, binding: ResidentCallBinding): void {
  if (frame['request_contract'] !== RESIDENT_SCHEMA || frame['accepted'] !== true || frame['started'] !== true) {
    throw malformed()
  }
  const served = object(frame['served_identity'])
  if (served['instance'] !== binding.instance
    || served['host_generation'] !== binding.hostGeneration
    || served['resident_generation'] !== binding.residentGeneration
    || served['target_config_revision'] !== binding.targetConfigRevision
    || served['model_artifact'] !== binding.modelArtifact
    || served['artifact_revision'] !== binding.artifactRevision
    || served['serving_variant_id'] !== binding.servingVariantId
    || served['serving_variant_revision'] !== binding.servingVariantRevision) {
    throw malformed()
  }
}

/**
 * Map one terminal resident choice onto the harness finish reason.
 * @param value - the frame's finish reason.
 * @returns the harness reason.
 * @throws LlmError for any reason the contract does not define.
 */
function finishReason(value: unknown): FinishReason {
  if (value === 'stop') return { kind: 'stop' }
  if (value === 'length') return { kind: 'max-tokens' }
  throw malformed()
}

/**
 * Require a non-negative safe integer token count.
 * @param value - any decoded JSON value.
 * @returns the validated count.
 * @throws LlmError when the value is not a token count.
 */
function count(value: unknown): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) throw malformed()
  return value
}

/**
 * Read one frame's usage, when it reports any.
 * @param frame - one parsed SSE frame.
 * @returns harness usage counts, or undefined when the frame carries none.
 * @throws LlmError when a reported usage block is malformed.
 */
function usageOf(frame: Record<string, unknown>): TokenUsage | undefined {
  const usage = frame['usage']
  if (usage === undefined) return undefined
  const counts = object(usage)
  const inputTokens = count(counts['prompt_tokens'])
  const outputTokens = count(counts['completion_tokens'])
  const totalTokens = counts['total_tokens']
  return {
    inputTokens,
    outputTokens,
    totalTokens: totalTokens === undefined ? inputTokens + outputTokens : count(totalTokens),
  }
}

/**
 * Read one response body as SSE data frames, stopping at the `[DONE]` terminal.
 * @param body - the response body stream.
 * @param state - records that the `[DONE]` terminal was observed.
 * @returns parsed JSON frames in arrival order.
 * @throws LlmError when a data frame is not a JSON object.
 */
async function* framesOf(body: ReadableStream<Uint8Array>, state: { done: boolean }): AsyncGenerator<Record<string, unknown>> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let pending = ''
  try {
    for (;;) {
      const chunk = await reader.read()
      if (chunk.done) return
      pending += decoder.decode(chunk.value, { stream: true })
      for (;;) {
        const newline = pending.indexOf('\n')
        if (newline < 0) break
        const line = pending.slice(0, newline).trim()
        pending = pending.slice(newline + 1)
        if (!line.startsWith('data:')) continue
        const payload = line.slice('data:'.length).trim()
        if (payload === '[DONE]') {
          state.done = true
          return
        }
        let parsed: unknown
        try {
          parsed = JSON.parse(payload)
        } catch (error) {
          throw new LlmError('KINF resident stream carried a malformed frame', 'MALFORMED_RESPONSE', { cause: error })
        }
        yield object(parsed)
      }
    }
  } finally {
    await reader.cancel().catch((_closed: unknown) => {
      // The read outcome is settled; cancellation only releases the response body.
    })
  }
}

/**
 * Stream one KINF resident-only connected call as harness chunks.
 *
 * The first frame and the finish frame must both declare the resident-only
 * contract, an acceptance, and the served identity this call was bound to.
 * Text is buffered until the finish frame has been validated, so a reply that
 * violates the contract publishes nothing. The yielded sequence is the same
 * text sequence ordinary pi-ai text dispatch produces.
 * @param request - binding, endpoint, headers, plain-text history, and controls.
 * @returns text block chunks, usage when reported, and a successful finish.
 * @throws LlmError when the request is not contract-valid, the transport fails,
 *   the host refuses the resident precondition, or a reply frame violates the
 *   served identity. A refused pre-admission failure is `RESIDENT_PRECONDITION_STALE`;
 *   a failure after admission is `RESIDENT_ACCEPTANCE_AMBIGUOUS`.
 */
export async function* streamKinfResident(request: KinfResidentRequest): AsyncGenerator<StreamChunk> {
  const residentOnly = precondition(request.binding)
  if (request.messages.length === 0) {
    throw new LlmError('KINF resident requests require at least one message', 'INVALID_REQUEST')
  }
  const response = await send(request, endpoint(request.baseURL), wireHeaders(request.headers), JSON.stringify({
    model: request.requestModel,
    messages: request.messages,
    stream: true,
    ...request.maxTokens === undefined ? {} : { max_tokens: request.maxTokens },
    ...request.temperature === undefined ? {} : { temperature: request.temperature },
    kinf_resident_only: residentOnly,
  }))
  if (!response.ok) throw await failure(response)
  const body = await eventStream(response)
  const state = { done: false }
  const deltas: string[] = []
  let first = false
  let reason: FinishReason | undefined
  let usage: TokenUsage | undefined
  try {
    for await (const frame of framesOf(body, state)) {
      const choice = firstChoice(frame)
      const terminal = choice?.['finish_reason']
      const finishing = terminal !== null && terminal !== undefined
      if (!first || finishing) validateFrame(frame, request.binding)
      first = true
      if (choice !== undefined) {
        const content = record(choice['delta'])?.['content']
        if (typeof content === 'string' && content.length > 0) {
          deltas.push(content)
          request.onTextProgress?.()
        }
        if (finishing) reason = finishReason(terminal)
      }
      const frameUsage = usageOf(frame)
      if (frameUsage !== undefined) usage = frameUsage
    }
  } catch (error) {
    if (error instanceof LlmError) throw error
    if (request.signal?.aborted === true) {
      throw new LlmError('KINF resident request aborted by caller', 'ABORTED', { cause: error })
    }
    throw new LlmError('KINF resident stream failed', 'TRANSPORT', { cause: error })
  }
  if (!state.done || !first || reason === undefined) throw malformed()
  const text = deltas.join('')
  if (text.length === 0) {
    throw new LlmError(`KINF resident completion for model "${request.requestModel}" carried no text`, 'EMPTY_RESPONSE')
  }
  yield { type: 'block-start', index: 0, blockType: 'text' }
  for (const delta of deltas) yield { type: 'text-delta', index: 0, text: delta }
  yield { type: 'block-end', index: 0, block: { type: 'text', text } }
  if (usage !== undefined) yield { type: 'usage', usage }
  yield { type: 'finish', reason }
}
