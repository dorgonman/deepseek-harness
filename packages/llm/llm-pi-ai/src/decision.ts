/** Bounded KINF first-token Decision requests and complete, durable readouts. @module */
import { LlmError } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'

// A first-token readout and its fixed identity fields must fit this protocol bound.
const MAX_RESPONSE_BYTES = 64 * 1024
const IDENTITY_FIELDS = [
  'mode', 'instance', 'logical_model', 'requested_model', 'requested_serving_variant_id',
  'serving_variant_id', 'serving_variant_revision', 'model_artifact', 'artifact_revision',
  'backend', 'runtime_profile_id', 'runtime_generation', 'host_generation', 'engine_generation',
  'resident_generation', 'config_revision', 'target_config_revision',
] as const

interface DecisionRequest {
  options: GenerateOptions
  model: string
  baseURL: string
  headers: Readonly<Record<string, string>>
  apiKey?: string
  timeoutMs?: number
  signal: AbortSignal
}

function malformed(): LlmError {
  return new LlmError('KINF returned an invalid complete Decision readout or served identity.', 'MALFORMED_RESPONSE')
}

function object(value: unknown): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw malformed()
  return value as Record<string, unknown>
}

function integer(value: unknown): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1) throw malformed()
  return value
}

function finite(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw malformed()
  return value
}

function pair(value: unknown): { yes: number; no: number } {
  const record = object(value)
  if (Object.keys(record).length !== 2 || !Object.hasOwn(record, 'yes') || !Object.hasOwn(record, 'no')) throw malformed()
  return { yes: finite(record['yes']), no: finite(record['no']) }
}

function text(value: unknown, secret: string | undefined): string {
  if (typeof value !== 'string' || value.length === 0 || value.length > 2048
    || /[\u0000-\u001f\u007f]/u.test(value) || (secret !== undefined && value.includes(secret))) throw malformed()
  return value
}

function readout(raw: unknown, request: DecisionRequest, headers: Headers) {
  const reply = object(raw)
  const choices = reply['choices']
  if (reply['model'] !== request.model || !Array.isArray(choices) || choices.length !== 1) throw malformed()
  const choice = object(choices[0])
  const message = object(choice['message'])
  const decision = object(reply['decision'])
  const usage = object(reply['usage'])
  const inputTokens = integer(usage['prompt_tokens'])
  const outputTokens = integer(usage['completion_tokens'])
  const totalTokens = integer(usage['total_tokens'])
  const logits = pair(decision['logits'])
  const scores = pair(decision['label_scores'])
  const label = decision['label']
  const yesWeight = Math.exp(logits.yes - Math.max(logits.yes, logits.no))
  const noWeight = Math.exp(logits.no - Math.max(logits.yes, logits.no))
  if (message['role'] !== 'assistant' || choice['finish_reason'] !== 'length'
    || Object.hasOwn(message, 'tool_calls') || outputTokens !== 1 || totalTokens !== inputTokens + 1
    || decision['schema'] !== 'kinf.decision.v1' || decision['mode'] !== 'label_scores'
    || (label !== 'yes' && label !== 'no') || message['content'] !== label
    || decision['token_position'] !== inputTokens || decision['calibration'] !== 'uncalibrated'
    || (logits.yes > logits.no && label !== 'yes') || (logits.no > logits.yes && label !== 'no')
    || scores.yes < 0 || scores.yes > 1 || scores.no < 0 || scores.no > 1
    || Math.abs(scores.yes - yesWeight / (yesWeight + noWeight)) > 1e-12
    || Math.abs(scores.no - noWeight / (yesWeight + noWeight)) > 1e-12) throw malformed()

  const identity = object(reply['served_identity'])
  const servedIdentity: Record<string, string | number> = {}
  const credential = headers.get('authorization')?.replace(/^Bearer\s+/i, '')
  for (const field of IDENTITY_FIELDS) servedIdentity[field] = text(identity[field], credential)
  servedIdentity['worker_pid'] = integer(identity['worker_pid'])
  if (!['service', 'mservice'].includes(String(identity['mode']))
    || ['logical_model', 'requested_model', 'requested_serving_variant_id', 'serving_variant_id']
      .some(field => identity[field] !== request.model)
    || identity['host_generation'] !== reply['generation']
    || identity['engine_generation'] !== reply['engine_generation']
    || identity['resident_generation'] !== identity['engine_generation']
    || !/^[0-9a-f]{1,16}-[0-9a-f]{1,16}-[0-9a-f]{1,16}$/.test(String(identity['host_generation']))
    || !/^[0-9a-f]{1,16}-[0-9a-f]{1,16}-[0-9a-f]{1,16}$/.test(String(identity['engine_generation']))
    || !/^[0-9a-f]{64}$/.test(String(identity['config_revision']))
    || !/^[0-9a-f]{64}$/.test(String(identity['target_config_revision']))
    || (headers.has('x-kinf-mode') && identity['mode'] !== headers.get('x-kinf-mode'))
    || (headers.has('x-kinf-instance') && identity['instance'] !== headers.get('x-kinf-instance'))) throw malformed()
  return {
    model: request.model,
    decision: {
      schema: 'kinf.decision.v1', mode: 'label_scores', label, token_position: inputTokens,
      logits, label_scores: scores, calibration: 'uncalibrated',
    },
    served_identity: servedIdentity,
    usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: totalTokens },
  }
}

/**
 * Convert Decision history to plain text, validating each block as its text is read.
 * @param options - request options with system text and typed message history.
 * @returns the ordered plain-text messages sent to KINF.
 */
function messages(options: GenerateOptions) {
  if ((options.tools?.length ?? 0) > 0) {
    throw new LlmError('KINF Decision requests do not support native tools.', 'UNSUPPORTED_TOOLS')
  }
  if ((options.maxTokens !== undefined && options.maxTokens !== 1)
    || (options.temperature !== undefined && options.temperature !== 0)
    || options.reasoningEffort !== undefined || options.stop !== undefined) {
    throw new LlmError('KINF Decision requests require one output token, temperature 0, and no reasoning or stop controls.', 'UNSUPPORTED_OPTION')
  }
  const result: { role: 'system' | 'user' | 'assistant'; content: string }[] = []
  if (options.system !== undefined) result.push({ role: 'system', content: options.system })
  const unsupportedHistory = 'KINF Decision requests require plain text system, user, or assistant history.'
  for (const message of options.messages) {
    if (message.role !== 'system' && message.role !== 'user' && message.role !== 'assistant') {
      throw new LlmError(unsupportedHistory, 'UNSUPPORTED_CONTENT')
    }
    result.push({
      role: message.role,
      content: message.content.map((block) => {
        if (block.type !== 'text') throw new LlmError(unsupportedHistory, 'UNSUPPORTED_CONTENT')
        return block.text
      }).join(''),
    })
  }
  if (result.length === 0) throw new LlmError('KINF Decision requests require a message.', 'INVALID_REQUEST')
  return result
}

async function body(response: Response): Promise<string> {
  const oversized = () => new LlmError('KINF Decision response exceeds the protocol byte limit.', 'MALFORMED_RESPONSE')
  if (Number(response.headers.get('content-length')) > MAX_RESPONSE_BYTES) {
    await response.body?.cancel()
    throw oversized()
  }
  if (response.body === null) throw malformed()
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let length = 0
  try {
    for (;;) {
      const chunk = await reader.read()
      if (chunk.done) break
      length += chunk.value.byteLength
      if (length > MAX_RESPONSE_BYTES) throw oversized()
      chunks.push(chunk.value)
    }
  } finally {
    await reader.cancel().catch((_alreadyClosed: unknown) => {
      // The read outcome is settled; cancellation only releases the response body.
    })
  }
  const bytes = new Uint8Array(length)
  let offset = 0
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength }
  return new TextDecoder('utf-8', { fatal: true }).decode(bytes)
}

/**
 * Complete one strict native Decision request, then publish its full readout as DSH chunks.
 * @param request - exact selected route, resolved credentials, cancellation, and request controls.
 * @returns one text block, authoritative token usage, and a successful finish; invalid replies emit no content.
 */
export async function* decisionChunks(request: DecisionRequest): AsyncGenerator<StreamChunk> {
  const history = messages(request.options)
  const headers = new Headers(request.headers)
  if (request.apiKey !== undefined && !headers.has('authorization')) headers.set('authorization', `Bearer ${request.apiKey}`)
  if (!headers.has('authorization')) throw new LlmError('KINF Decision requires a configured credential or authorization header.', 'MISSING_CREDENTIAL')
  if (headers.has('x-kinf-service') && headers.get('x-kinf-service') !== request.model) {
    throw new LlmError('KINF Decision model and service selection must match.', 'INVALID_CONFIG')
  }
  headers.set('content-type', 'application/json')
  headers.set('accept', 'application/json')
  const timeout = request.timeoutMs !== undefined && request.timeoutMs > 0 ? AbortSignal.timeout(request.timeoutMs) : undefined
  const signal = timeout === undefined ? request.signal : AbortSignal.any([request.signal, timeout])
  try {
    const response = await fetch(`${request.baseURL.replace(/\/+$/, '')}/chat/completions`, {
      method: 'POST', headers, signal, redirect: 'error',
      body: JSON.stringify({ model: request.model, messages: history, stream: false, max_tokens: 1, kinf_decision: 'label_scores' }),
    })
    if (!response.ok) {
      await response.body?.cancel()
      const code = response.status === 401 || response.status === 403 ? 'AUTH'
        : response.status === 402 ? 'QUOTA'
          : response.status === 429 ? 'RATE_LIMIT'
            : response.status === 408 ? 'TIMEOUT'
              : response.status >= 500 ? 'SERVER' : 'INVALID_REQUEST'
      throw new LlmError(`KINF Decision request failed (HTTP ${response.status}).`, code, { status: response.status })
    }
    if (!/^application\/(?:[a-z0-9.+-]+\+)?json(?:;|$)/i.test(response.headers.get('content-type') ?? '')) {
      await response.body?.cancel()
      throw malformed()
    }
    let raw: unknown
    try { raw = JSON.parse(await body(response)) }
    catch (error: unknown) {
      if (error instanceof LlmError || signal.aborted) throw error
      throw malformed()
    }
    const result = readout(raw, request, headers)
    signal.throwIfAborted()
    const content = JSON.stringify(result, null, 2)
    yield { type: 'block-start', index: 0, blockType: 'text' }
    yield { type: 'text-delta', index: 0, text: content }
    yield { type: 'block-end', index: 0, block: { type: 'text', text: content } }
    yield { type: 'usage', usage: { inputTokens: result.usage.prompt_tokens, outputTokens: 1, totalTokens: result.usage.total_tokens } }
    // Native length is the successful first-token operation's required terminal state.
    yield { type: 'finish', reason: { kind: 'stop' } }
  } catch (error: unknown) {
    if (request.signal.aborted) throw error
    if (timeout?.aborted) throw new LlmError('KINF Decision request exceeded its configured timeout.', 'TIMEOUT', { cause: error })
    if (error instanceof LlmError) throw error
    throw new LlmError('KINF Decision transport failed.', 'TRANSPORT', { cause: error })
  }
}
