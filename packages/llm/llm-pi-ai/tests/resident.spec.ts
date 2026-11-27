/**
 * KINF resident-only connected transport: the dedicated wire path, the
 * `kinf_resident_only` precondition, served-identity enforcement, and the
 * pre-admission/post-admission refusal distinction.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createAssistantMessage, createDeveloperMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, ResidentCallBinding, StreamChunk } from '@deepseek-ai/dsh-llm'
import type { PiAiModelProfile } from '../src/config.ts'
import { PiAiAdapter } from '../src/adapter.ts'
import { Config, resolveProfiles } from '../src/config.ts'
import { streamKinfResident } from '../src/resident.ts'
import type { KinfResidentRequest } from '../src/resident.ts'
import { memoryAuth } from './auth-double.ts'
import { closeMockServers, controlledSseResponse, mockServer } from './mock-server.ts'

const MODEL = 'fixture-resident-chat'
const KEY = 'resident-fixture-key'
const PATH = '/kinf/resident/v1/chat/completions'

const BINDING: ResidentCallBinding = {
  schema: 'kinf.resident-only.v1',
  instance: 'fixture-instance',
  hostGeneration: '1ab-2bc-3cd',
  servingVariantId: 'fixture-variant',
  servingVariantRevision: 'variant-rev-1',
  modelArtifact: 'fixture-artifact',
  artifactRevision: 'artifact-rev-1',
  targetConfigRevision: 'a'.repeat(64),
  residentGeneration: '2ab-3bc-4cd',
  projectionId: 'projection-secret',
}

afterEach(async () => {
  await closeMockServers()
  vi.unstubAllGlobals()
})

/** One user turn in the shape the LLM seam receives. */
function userMessage(text: string) {
  return createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text }] })
}

/** The served identity a conforming reply frame reasserts. */
function identity(): Record<string, unknown> {
  return {
    instance: BINDING.instance,
    host_generation: BINDING.hostGeneration,
    resident_generation: BINDING.residentGeneration,
    target_config_revision: BINDING.targetConfigRevision,
    model_artifact: BINDING.modelArtifact,
    artifact_revision: BINDING.artifactRevision,
    serving_variant_id: BINDING.servingVariantId,
    serving_variant_revision: BINDING.servingVariantRevision,
  }
}

/** One accepted opening frame with an empty role delta. */
function firstFrame(): Record<string, unknown> {
  return {
    request_contract: 'kinf.resident-only.v1',
    accepted: true,
    started: true,
    served_identity: identity(),
    choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
  }
}

/** One streamed text delta frame. */
function textFrame(content: string): Record<string, unknown> {
  return { choices: [{ index: 0, delta: { content }, finish_reason: null }] }
}

/** The terminal frame; `overrides` replaces any contract or choice member. */
function finishFrame(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    request_contract: 'kinf.resident-only.v1',
    accepted: true,
    started: true,
    served_identity: identity(),
    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
    ...overrides,
  }
}

/** Encode frames and an optional terminal marker as the mock server's SSE events. */
function sse(frames: readonly Record<string, unknown>[], done = true): string[] {
  return [...frames.map(frame => JSON.stringify(frame)), ...done ? ['[DONE]'] : []]
}

/** A standalone request against one mock-server endpoint. */
function request(overrides: Partial<KinfResidentRequest> = {}): KinfResidentRequest {
  return {
    baseURL: 'https://fixture.invalid/v1',
    headers: { Authorization: `Bearer ${KEY}` },
    requestModel: MODEL,
    messages: [{ role: 'user', content: 'hello resident' }],
    maxTokens: 4_096,
    binding: BINDING,
    ...overrides,
  }
}

/** The adapter over one declared (or undeclared) model row. */
function adapter(baseURL: string, models: PiAiModelProfile[]): PiAiAdapter {
  return new PiAiAdapter({
    profiles: () => resolveProfiles({ fixture: { api: 'openai-completions', baseURL, models } }),
    resolveApiKey: () => Promise.resolve(KEY),
    auth: memoryAuth(),
  })
}

const RESIDENT_MODEL: PiAiModelProfile = {
  id: MODEL, requestModel: MODEL, residentTransport: true, contextWindow: 65_536, maxTokens: 4_096,
}

/** Collect a chunk stream, retaining whatever was emitted before an error. */
async function drain(
  stream: AsyncIterable<StreamChunk>,
  chunks: StreamChunk[] = [],
): Promise<{ chunks: StreamChunk[]; error?: unknown }> {
  try {
    for await (const chunk of stream) chunks.push(chunk)
  } catch (error) {
    return { chunks, error }
  }
  return { chunks }
}

describe('KINF resident-only transport', () => {
  it.each(['finishes', 'stalls'] as const)('rearms idle while resident text is buffered and the stream %s', async (outcome) => {
    const wire = controlledSseResponse()
    let aborted = false
    const fetchMock = vi.fn((_input: string | URL | Request, init?: RequestInit) => {
      const signal = init?.signal
      if (signal === undefined || signal === null) throw new Error('expected a resident cancellation signal')
      signal.addEventListener('abort', () => {
        aborted = true
        wire.abort(signal.reason)
      }, { once: true })
      return Promise.resolve(wire.response)
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
    const emitted: StreamChunk[] = []
    const pending = drain(adapter('https://fixture.invalid/v1', [{ ...RESIDENT_MODEL, streamIdleTimeoutMs: 20 }]).stream({
      provider: 'fixture', model: MODEL, messages: [userMessage('hi')], residentBinding: BINDING,
    }), emitted)
    try {
      await wire.readRequested(1)
      wire.write(`data: ${JSON.stringify(firstFrame())}\n\n`)
      await wire.readRequested(2)
      await vi.advanceTimersByTimeAsync(15)
      wire.write(`data: ${JSON.stringify(textFrame('hello'))}\n\n`)
      await wire.readRequested(3)
      await vi.advanceTimersByTimeAsync(15)
      expect(aborted).toBe(false)
      expect(emitted).toEqual([])
      if (outcome === 'finishes') {
        wire.write(`data: ${JSON.stringify(textFrame(' resident'))}\n\n`)
        await wire.readRequested(4)
        await vi.advanceTimersByTimeAsync(15)
        expect(emitted).toEqual([])
        wire.write(`data: ${JSON.stringify(finishFrame())}\n\ndata: [DONE]\n\n`)
        expect((await pending).error).toBeUndefined()
        expect(emitted).toContainEqual({ type: 'block-end', index: 0, block: { type: 'text', text: 'hello resident' } })
      } else {
        wire.write('data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":0}}\n\n')
        await wire.readRequested(4)
        await vi.advanceTimersByTimeAsync(4)
        wire.write(': heartbeat\n\n')
        await wire.readRequested(5)
        await vi.advanceTimersByTimeAsync(1)
        expect((await pending).error).toMatchObject({ code: 'TIMEOUT' })
        expect(emitted).toEqual([])
      }
      expect(fetchMock).toHaveBeenCalledOnce()
    } finally {
      wire.abort(new Error('resident fixture complete'))
      vi.useRealTimers()
      await pending
    }
  })

  it('refuses an undeclared row before any request', async () => {
    const server = await mockServer([{ events: sse([firstFrame(), textFrame('hi'), finishFrame()]) }])
    const result = await drain(adapter(server.url, [{ id: MODEL, contextWindow: 65_536 }]).stream({
      provider: 'fixture', model: MODEL, messages: [userMessage('hi')], residentBinding: BINDING,
    }))
    expect(result.error).toMatchObject({ code: 'UNSUPPORTED_RESIDENT_TRANSPORT' })
    expect(result.chunks).toEqual([])
    expect(server.requests).toHaveLength(0)
  })

  it('sends one resident request with the exact contract body and no cold-load headers', async () => {
    const terminal = finishFrame({ usage: { prompt_tokens: 7, completion_tokens: 2, total_tokens: 9 } })
    const server = await mockServer([{ events: sse([firstFrame(), textFrame('hel'), textFrame('lo'), terminal]) }])
    const result = await drain(adapter(server.url, [{
      ...RESIDENT_MODEL,
      headers: {
        Authorization: `Bearer ${KEY}`,
        'X-KINF-Request-Lifetime': '30s',
        'X-KINF-Allow-Cold-Load': '1',
        'X-KINF-Allow-Eviction': '1',
        'Allow-Cold-Load': '1',
        'Allow-Eviction': '1',
      },
    }]).stream({
      provider: 'fixture', model: MODEL, messages: [userMessage('hello resident')],
      maxTokens: 4_096, residentBinding: BINDING,
    }))
    expect(result.error).toBeUndefined()
    expect(result.chunks).toEqual([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'hel' },
      { type: 'text-delta', index: 0, text: 'lo' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'hello' } },
      { type: 'usage', usage: { inputTokens: 7, outputTokens: 2, totalTokens: 9 } },
      { type: 'finish', reason: { kind: 'stop' } },
    ])
    expect(server.paths).toEqual([PATH])
    expect(server.requests).toEqual([{
      model: MODEL,
      messages: [{ role: 'user', content: 'hello resident' }],
      stream: true,
      max_tokens: 4_096,
      kinf_resident_only: {
        schema: 'kinf.resident-only.v1',
        instance: BINDING.instance,
        host_generation: BINDING.hostGeneration,
        serving_variant_id: BINDING.servingVariantId,
        serving_variant_revision: BINDING.servingVariantRevision,
        model_artifact: BINDING.modelArtifact,
        artifact_revision: BINDING.artifactRevision,
        target_config_revision: BINDING.targetConfigRevision,
        resident_generation: BINDING.residentGeneration,
      },
    }])
    const headers = server.headers[0]!
    expect(headers['authorization']).toBe(`Bearer ${KEY}`)
    expect(headers['content-type']).toBe('application/json')
    expect(headers['x-request-id']).toMatch(/^[A-Za-z0-9_-]{1,100}$/)
    expect(headers['x-kinf-request-lifetime']).toBeUndefined()
    expect(headers['x-kinf-allow-cold-load']).toBeUndefined()
    expect(headers['x-kinf-allow-eviction']).toBeUndefined()
    expect(headers['allow-cold-load']).toBeUndefined()
    expect(headers['allow-eviction']).toBeUndefined()
    // projectionId is DSH-internal and never leaves the process.
    expect(JSON.stringify(server.requests[0])).not.toContain('projection-secret')
  })

  it('falls back to the selector id as the wire model and sends temperature zero', async () => {
    const server = await mockServer([{ events: sse([firstFrame(), textFrame('ok'), finishFrame()]) }])
    const result = await drain(adapter(server.url, [{
      id: MODEL, residentTransport: true, contextWindow: 65_536, maxTokens: 4_096,
    }]).stream({
      provider: 'fixture', model: MODEL, messages: [userMessage('hi')],
      maxTokens: 4_096, temperature: 0, residentBinding: BINDING,
    }))
    expect(result.error).toBeUndefined()
    expect(server.requests[0]).toMatchObject({ model: MODEL, temperature: 0, max_tokens: 4_096 })
  })

  it('mints a distinct request id for every resident call', async () => {
    const events = sse([firstFrame(), textFrame('hi'), finishFrame()])
    const server = await mockServer([{ events }, { events }])
    const pi = adapter(server.url, [RESIDENT_MODEL])
    const call = () => drain(pi.stream({
      provider: 'fixture', model: MODEL, messages: [userMessage('hi')], residentBinding: BINDING,
    }))
    await call()
    await call()
    const ids = server.headers.map(headers => headers['x-request-id'])
    expect(ids).toHaveLength(2)
    expect(ids[0]).toMatch(/^[A-Za-z0-9_-]{1,100}$/)
    expect(ids[1]).not.toBe(ids[0])
  })

  it('projects plain-text system and multi-block user history', async () => {
    const server = await mockServer([{ events: sse([firstFrame(), textFrame('ok'), finishFrame()]) }])
    const result = await drain(adapter(server.url, [RESIDENT_MODEL]).stream({
      provider: 'fixture', model: MODEL, system: 'be brief',
      messages: [createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'a' }, { type: 'text', text: 'b' }] })],
      residentBinding: BINDING,
    }))
    expect(result.error).toBeUndefined()
    expect(server.requests[0]).toMatchObject({
      messages: [{ role: 'system', content: 'be brief' }, { role: 'user', content: 'ab' }],
    })
    expect(server.requests[0]).not.toHaveProperty('max_tokens')
    expect(server.requests[0]).not.toHaveProperty('temperature')
  })

  it.each([
    { name: 'native tools', override: { tools: [{ name: 'echo', description: 'Echo.', parameters: {} }] }, code: 'UNSUPPORTED_TOOLS' },
    { name: 'a non-chat role', override: { messages: [createDeveloperMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'x' }] })] }, code: 'UNSUPPORTED_CONTENT' },
    {
      name: 'non-text content',
      override: { messages: [createAssistantMessage({
        source: { provider: 'fixture', model: MODEL }, content: [{ type: 'reasoning', text: 'hidden' }],
      })] },
      code: 'UNSUPPORTED_CONTENT',
    },
    { name: 'an empty history', override: { messages: [] }, code: 'INVALID_REQUEST' },
    { name: 'a nonzero temperature', override: { temperature: 0.5 }, code: 'UNSUPPORTED_OPTION' },
  ] satisfies { name: string; override: Partial<GenerateOptions>; code: string }[])('refuses $name before dispatch', async ({ override, code }) => {
    const server = await mockServer([{ events: sse([firstFrame(), textFrame('hi'), finishFrame()]) }])
    const result = await drain(adapter(server.url, [RESIDENT_MODEL]).stream({
      provider: 'fixture', model: MODEL, messages: [userMessage('hi')], residentBinding: BINDING,
      ...override,
    }))
    expect(result.error).toMatchObject({ code })
    expect(server.requests).toHaveLength(0)
  })

  it.each([
    ['schema', { ...BINDING, schema: 'kinf.decision.v1' }],
    ['empty value', { ...BINDING, instance: '' }],
    ['punctuation', { ...BINDING, hostGeneration: 'boot generation' }],
    ['non-string', { ...BINDING, residentGeneration: 7 as never }],
  ])('refuses an invalid binding (%s) before dispatch', async (_name, binding) => {
    const fetch = vi.fn()
    vi.stubGlobal('fetch', fetch)
    const result = await drain(streamKinfResident(request({ binding })))
    expect(result.error).toMatchObject({ code: 'INVALID_REQUEST' })
    expect(fetch).not.toHaveBeenCalled()
  })

  it.each(['fixture', 'ftp://fixture.invalid/v1'])('refuses the non-HTTP endpoint %s', async (baseURL) => {
    const fetch = vi.fn()
    vi.stubGlobal('fetch', fetch)
    const result = await drain(streamKinfResident(request({ baseURL })))
    expect(result.error).toMatchObject({ code: 'INVALID_CONFIG' })
    expect(fetch).not.toHaveBeenCalled()
  })

  it('refuses an empty standalone history before dispatch', async () => {
    const fetch = vi.fn()
    vi.stubGlobal('fetch', fetch)
    const result = await drain(streamKinfResident(request({ messages: [] })))
    expect(result.error).toMatchObject({ code: 'INVALID_REQUEST' })
    expect(fetch).not.toHaveBeenCalled()
  })

  it('maps a caller abort during the stream to ABORTED', async () => {
    const controller = new AbortController()
    const encoder = new TextEncoder()
    const waiting = Promise.withResolvers<undefined>()
    const body = new ReadableStream<Uint8Array>({
      start(stream) {
        stream.enqueue(encoder.encode(`data: ${JSON.stringify(firstFrame())}\n\n`))
      },
      pull() {
        waiting.resolve(undefined)
        return new Promise<void>((_resolve, reject) => {
          controller.signal.addEventListener('abort', () => { reject(new Error('caller aborted')) }, { once: true })
        })
      },
    })
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(body, {
      status: 200, headers: { 'content-type': 'text/event-stream' },
    }))))
    const pending = drain(streamKinfResident(request({ signal: controller.signal })))
    await waiting.promise
    controller.abort()
    const result = await pending
    expect(result.error).toMatchObject({ code: 'ABORTED' })
    expect(result.chunks).toEqual([])
  })

  it('forwards caller cancellation while the resident response headers are outstanding', async () => {
    const dispatched = Promise.withResolvers<'dispatched'>()
    const fetchMock = vi.fn((_input: string | URL | Request, init?: RequestInit) => new Promise<Response>((_resolve, reject) => {
      const signal = init?.signal
      if (signal === undefined || signal === null) throw new Error('expected a resident cancellation signal')
      signal.addEventListener('abort', () => { reject(signal.reason) }, { once: true })
      dispatched.resolve('dispatched')
    }))
    vi.stubGlobal('fetch', fetchMock)
    const controller = new AbortController()
    const pending = drain(adapter('https://fixture.invalid/v1', [RESIDENT_MODEL]).stream({
      provider: 'fixture', model: MODEL, messages: [userMessage('hi')], residentBinding: BINDING,
      signal: controller.signal,
    }))
    try {
      expect(await Promise.race([dispatched.promise, pending])).toBe('dispatched')
      controller.abort()
      expect(await pending).toMatchObject({ chunks: [], error: { code: 'ABORTED' } })
      expect(fetchMock).toHaveBeenCalledOnce()
    } finally {
      controller.abort()
      await pending
    }
  })

  it('maps an accepted:false refusal to RESIDENT_PRECONDITION_STALE without retrying', async () => {
    const server = await mockServer([{
      status: 409,
      body: JSON.stringify({
        error: { code: 'resident_identity_mismatch' },
        request_contract: 'kinf.resident-only.v1', accepted: false, started: false,
      }),
    }])
    const result = await drain(streamKinfResident(request({ baseURL: server.url })))
    expect(result.error).toMatchObject({ code: 'RESIDENT_PRECONDITION_STALE', failure: { status: 409 } })
    expect(server.requests).toHaveLength(1)
  })

  it('does not infer stale identity from an arbitrary pre-admission refusal', async () => {
    const server = await mockServer([{
      status: 409,
      body: JSON.stringify({
        error: { code: 'invalid_request' },
        request_contract: 'kinf.resident-only.v1', accepted: false, started: false,
      }),
    }])
    const result = await drain(streamKinfResident(request({ baseURL: server.url })))
    expect(result.error).toMatchObject({ code: 'INVALID_REQUEST', failure: { status: 409 } })
    expect(server.requests).toHaveLength(1)
  })

  it('refuses a model endpoint other than the observed origin before sending', async () => {
    const server = await mockServer([])
    const result = await drain(adapter(server.url, [RESIDENT_MODEL]).stream({
      provider: 'fixture', model: MODEL, messages: [userMessage('hi')],
      residentBinding: { ...BINDING, observedOrigin: 'http://127.0.0.1:1' },
    }))
    expect(result.error).toMatchObject({ code: 'INVALID_CONFIG' })
    expect(server.requests).toHaveLength(0)
  })

  it('maps an accepted:true failure to RESIDENT_ACCEPTANCE_AMBIGUOUS without retrying', async () => {
    const server = await mockServer([{
      status: 503,
      body: JSON.stringify({ error: { code: 'resident_generation_rotated' }, accepted: true, started: false }),
    }])
    const result = await drain(streamKinfResident(request({ baseURL: server.url })))
    expect(result.error).toMatchObject({ code: 'RESIDENT_ACCEPTANCE_AMBIGUOUS', failure: { status: 503 } })
    expect(server.requests).toHaveLength(1)
  })

  it.each([
    [401, 'AUTH'],
    [429, 'RATE_LIMIT'],
    [408, 'TIMEOUT'],
    [503, 'SERVER'],
    [400, 'INVALID_REQUEST'],
  ])('maps HTTP %i without a resident verdict to %s', async (status, code) => {
    const server = await mockServer([{ status, body: '{}' }])
    const result = await drain(streamKinfResident(request({ baseURL: server.url })))
    expect(result.error).toMatchObject({ code })
    expect(server.requests).toHaveLength(1)
  })

  it('maps an unparseable failure body by its status', async () => {
    const server = await mockServer([{ status: 502, body: 'not json' }])
    expect((await drain(streamKinfResident(request({ baseURL: server.url })))).error).toMatchObject({ code: 'SERVER' })
  })

  it.each([
    ['request contract', (frame: Record<string, unknown>) => { frame['request_contract'] = 'kinf.decision.v1' }],
    ['acceptance', (frame: Record<string, unknown>) => { frame['accepted'] = false }],
    ['started', (frame: Record<string, unknown>) => { frame['started'] = false }],
    ['missing identity', (frame: Record<string, unknown>) => { delete frame['served_identity'] }],
    ['instance', (frame: Record<string, unknown>) => { identityOf(frame)['instance'] = 'other-instance' }],
    ['host generation', (frame: Record<string, unknown>) => { identityOf(frame)['host_generation'] = '9ab-8bc-7cd' }],
    ['resident generation', (frame: Record<string, unknown>) => { identityOf(frame)['resident_generation'] = '9ab-8bc-7cd' }],
    ['target config revision', (frame: Record<string, unknown>) => { identityOf(frame)['target_config_revision'] = 'b'.repeat(64) }],
    ['model artifact', (frame: Record<string, unknown>) => { identityOf(frame)['model_artifact'] = 'wrong-artifact' }],
    ['serving variant', (frame: Record<string, unknown>) => { identityOf(frame)['serving_variant_id'] = 'wrong-variant' }],
    ['artifact revision', (frame: Record<string, unknown>) => { identityOf(frame)['artifact_revision'] = 'artifact-rev-2' }],
    ['serving variant revision', (frame: Record<string, unknown>) => { identityOf(frame)['serving_variant_revision'] = 'variant-rev-2' }],
  ])('rejects a first frame with a mismatched %s and emits no text', async (_name, mutate) => {
    const frame = firstFrame()
    mutate(frame)
    const server = await mockServer([{ events: sse([frame, textFrame('leak'), finishFrame()]) }])
    const result = await drain(streamKinfResident(request({ baseURL: server.url })))
    expect(result.error).toMatchObject({ code: 'MALFORMED_RESPONSE' })
    expect(result.chunks).toEqual([])
    expect(server.requests).toHaveLength(1)
  })

  it('rejects a finish frame that changes the served identity before emitting buffered text', async () => {
    const terminal = finishFrame()
    identityOf(terminal)['resident_generation'] = 'rotated-generation'
    const server = await mockServer([{ events: sse([firstFrame(), textFrame('leak'), terminal]) }])
    const result = await drain(streamKinfResident(request({ baseURL: server.url })))
    expect(result.error).toMatchObject({ code: 'MALFORMED_RESPONSE' })
    expect(result.chunks).toEqual([])
  })

  it('streams buffered deltas and maps a native length finish without usage', async () => {
    const terminal = finishFrame({ choices: [{ index: 0, delta: {}, finish_reason: 'length' }] })
    const server = await mockServer([{ events: sse([firstFrame(), textFrame('par'), textFrame('tial'), terminal]) }])
    const { maxTokens: _cap, ...withoutCap } = request({ baseURL: server.url })
    const result = await drain(streamKinfResident(withoutCap))
    expect(result.error).toBeUndefined()
    expect(result.chunks).toEqual([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'par' },
      { type: 'text-delta', index: 0, text: 'tial' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'partial' } },
      { type: 'finish', reason: { kind: 'max-tokens' } },
    ])
    expect(server.requests[0]).not.toHaveProperty('max_tokens')
  })

  it('accepts an acceptance-only first frame and usage-only terminal frame', async () => {
    const acceptance = firstFrame()
    delete acceptance['choices']
    const usage = { choices: [], usage: { prompt_tokens: 5, completion_tokens: 2 } }
    const server = await mockServer([{ events: sse([acceptance, textFrame('hi'), finishFrame(), usage]) }])
    const result = await drain(streamKinfResident(request({ baseURL: server.url, temperature: 0 })))
    expect(result.error).toBeUndefined()
    expect(result.chunks).toEqual([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'hi' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'hi' } },
      { type: 'usage', usage: { inputTokens: 5, outputTokens: 2, totalTokens: 7 } },
      { type: 'finish', reason: { kind: 'stop' } },
    ])
    expect(server.requests[0]).toMatchObject({ temperature: 0 })
  })

  it.each([
    { name: 'only the terminal marker', events: ['[DONE]'], code: 'MALFORMED_RESPONSE' },
    { name: 'no finish frame', events: sse([firstFrame(), textFrame('x')]), code: 'MALFORMED_RESPONSE' },
    { name: 'no terminal marker', events: sse([firstFrame(), textFrame('x'), finishFrame()], false), code: 'MALFORMED_RESPONSE' },
    { name: 'a malformed JSON frame', events: [JSON.stringify(firstFrame()), 'not json', '[DONE]'], code: 'MALFORMED_RESPONSE' },
    { name: 'a non-object frame', events: [JSON.stringify(firstFrame()), 'null', '[DONE]'], code: 'MALFORMED_RESPONSE' },
    { name: 'an empty completion', events: sse([firstFrame(), finishFrame()]), code: 'EMPTY_RESPONSE' },
    { name: 'an undefined finish reason', events: sse([firstFrame(), textFrame('x'), finishFrame({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] })]), code: 'MALFORMED_RESPONSE' },
    { name: 'a malformed usage block', events: sse([firstFrame(), textFrame('x'), finishFrame({ usage: { prompt_tokens: 'x', completion_tokens: 1 } })]), code: 'MALFORMED_RESPONSE' },
  ])('rejects a stream with $name', async ({ events, code }) => {
    const server = await mockServer([{ events }])
    const result = await drain(streamKinfResident(request({ baseURL: server.url })))
    expect(result.error).toMatchObject({ code })
    expect(result.chunks).toEqual([])
  })

  it('rejects a non-SSE response body', async () => {
    const server = await mockServer([{ events: sse([firstFrame(), finishFrame()]), headers: { 'content-type': 'application/json' } }])
    expect((await drain(streamKinfResident(request({ baseURL: server.url })))).error)
      .toMatchObject({ code: 'MALFORMED_RESPONSE' })
  })

  it('rejects an SSE response without a body', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(null, {
      status: 200, headers: { 'content-type': 'text/event-stream' },
    }))))
    expect((await drain(streamKinfResident(request()))).error).toMatchObject({ code: 'MALFORMED_RESPONSE' })
  })

  it('rejects a successful response without an SSE media type', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(new ReadableStream<Uint8Array>({
      start(controller) { controller.close() },
    }), { status: 200 }))))
    expect((await drain(streamKinfResident(request()))).error).toMatchObject({ code: 'MALFORMED_RESPONSE' })
  })

  it('maps an errored response stream to TRANSPORT', async () => {
    const encoder = new TextEncoder()
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(firstFrame())}\n\n`))
        controller.error(new TypeError('socket dropped'))
      },
    }), { status: 200, headers: { 'content-type': 'text/event-stream' } }))))
    const result = await drain(streamKinfResident(request()))
    expect(result.error).toMatchObject({ code: 'TRANSPORT' })
    expect(result.chunks).toEqual([])
  })

  it('maps a transport failure to TRANSPORT', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('network down'))))
    expect((await drain(streamKinfResident(request()))).error).toMatchObject({ code: 'TRANSPORT' })
  })

  it('carries the resident declaration through the configuration schema', () => {
    expect(JSON.stringify(Config.toJSON())).toContain('residentTransport')
    const configured = Config({ providers: { fixture: {
      api: 'openai-completions', baseURL: 'https://fixture.invalid/v1',
      models: [{ id: MODEL, residentTransport: true }],
    } } })
    expect(configured.providers.get()['fixture']?.models[0]?.residentTransport).toBe(true)
  })

  it('rejects a non-boolean residentTransport declaration', () => {
    const model = { id: MODEL, residentTransport: 'yes' as never } as PiAiModelProfile
    expect(() => resolveProfiles({
      fixture: { api: 'openai-completions', baseURL: 'https://fixture.invalid/v1', models: [model] },
    })).toThrow(/residentTransport must be a boolean/)
  })
})

/** The mutable served identity carried by one frame. */
function identityOf(frame: Record<string, unknown>): Record<string, unknown> {
  return frame['served_identity'] as Record<string, unknown>
}
