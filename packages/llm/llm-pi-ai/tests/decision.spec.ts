/** Strict native Decisions remain separate from ordinary pi-ai chat dispatch. */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { createToolResultMessage, createUserMessage, ToolCallId } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import * as LlmPiAi from '@deepseek-ai/dsh-llm-pi-ai'
import { PiAiAdapter } from '../src/adapter.ts'
import { Config, resolveProfiles } from '../src/config.ts'
import { decisionChunks } from '../src/decision.ts'
import { memoryAuth } from './auth-double.ts'
import { assemble } from './assemble.ts'
import { closeMockServers, mockServer, textEvents } from './mock-server.ts'

const MODEL = 'flash-native-mtp-decision'
const SELECTOR = 'chat-qwen-flash-next-exl3-decision@local'
const KEY = 'decision-fixture-key'
const contexts: Context[] = []
afterEach(async () => {
  for (const ctx of contexts.splice(0)) await ctx.fiber.dispose()
  await closeMockServers()
  vi.unstubAllGlobals()
})

/** Native local_chat_tests.cpp DecisionReply, with deterministic catalog identities. */
function reply() {
  return {
    model: MODEL, generation: '1ab-2bc-3cd', engine_generation: '2ab-3bc-4cd',
    choices: [{ message: { role: 'assistant', content: 'yes' }, finish_reason: 'length' }],
    usage: { prompt_tokens: 8, completion_tokens: 1, total_tokens: 9, private: KEY },
    decision: {
      schema: 'kinf.decision.v1', mode: 'label_scores', label: 'yes', token_position: 8,
      logits: { yes: 1.25, no: 1.25 }, label_scores: { yes: 0.5, no: 0.5 }, calibration: 'uncalibrated',
    },
    served_identity: {
      mode: 'mservice', instance: 'fixture-managed', logical_model: MODEL, requested_model: MODEL,
      requested_serving_variant_id: MODEL, serving_variant_id: MODEL, serving_variant_revision: 'decision-v1',
      model_artifact: 'fixture-artifact', artifact_revision: 'fixture-artifact-v1', backend: 'exllamav3',
      runtime_profile_id: 'fixture-profile', runtime_generation: '1.6.4+cu132.torch2.13.0',
      host_generation: '1ab-2bc-3cd', engine_generation: '2ab-3bc-4cd', resident_generation: '2ab-3bc-4cd',
      worker_pid: 42, config_revision: 'a'.repeat(64), target_config_revision: 'b'.repeat(64), private: KEY,
    },
  }
}

function model() {
  return { id: SELECTOR, requestModel: MODEL, requestProtocol: 'kinf-decision-v1' as const,
    contextWindow: 262144, maxTokens: 1, input: ['text' as const], reasoningEfforts: false as const }
}

function options(overrides: Partial<GenerateOptions> = {}): GenerateOptions {
  return { provider: 'fixture', model: SELECTOR, messages: [
    createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'Is the evidence relevant?' }] }),
  ], ...overrides }
}

const headers = { 'X-Kinf-Mode': 'mservice', 'X-Kinf-Instance': 'fixture-managed', 'X-Kinf-Service': MODEL }

async function collect(chunks: AsyncIterable<StreamChunk>) {
  const result: StreamChunk[] = []
  for await (const chunk of chunks) result.push(chunk)
  return result
}

function directDecision(overrides: Partial<Parameters<typeof decisionChunks>[0]> = {}) {
  return decisionChunks({
    options: options(), model: MODEL, baseURL: 'https://fixture.invalid/v1',
    headers: { ...headers, Authorization: `Bearer ${KEY}` },
    signal: new AbortController().signal, ...overrides,
  })
}

async function harness(baseURL: string) {
  const ctx = new Context()
  contexts.push(ctx)
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(LlmPiAi, { providers: { fixture: {
    api: 'openai-completions', baseURL, headers: { Authorization: `Bearer ${KEY}`, ...headers },
    models: [model(), { id: 'ordinary', contextWindow: 65536 },
      { id: 'no-tools', nativeTools: false }, { id: 'with-tools', nativeTools: true }],
  } } })
  return ctx
}

describe('KINF Decision adaptation', () => {
  it('publishes both scores and exact served identity without private metadata', async () => {
    const server = await mockServer([{ body: JSON.stringify(reply()) }])
    const ctx = await harness(server.url)
    expect(await ctx.llm.resolveModelInfo('fixture', SELECTOR)).toMatchObject({
      nativeTools: false, inputProjectionPolicy: 'reject', defaultMaxTokens: 1,
      requestProtocol: 'kinf-decision-v1',
    })
    expect(await ctx.llm.resolveModelInfo('fixture', 'ordinary')).not.toHaveProperty('requestProtocol')
    const result = await assemble(ctx, options())
    expect(result.finish).toEqual({ kind: 'stop' })
    expect(result.usage).toEqual({ inputTokens: 8, outputTokens: 1, totalTokens: 9 })
    const block = result.message.content[0]
    expect(block?.type).toBe('text')
    if (block?.type !== 'text') throw new Error('expected visible Decision text')
    const displayed = JSON.parse(block.text) as Record<string, unknown>
    expect(displayed['model']).toBe(MODEL)
    expect(displayed['decision']).toEqual(reply().decision)
    expect(displayed['served_identity']).toMatchObject({
      logical_model: MODEL, requested_model: MODEL, requested_serving_variant_id: MODEL,
      serving_variant_id: MODEL, worker_pid: 42,
    })
    expect(block.text).not.toContain(KEY)
    expect(block.text).not.toContain('private')
    expect(result.message.source).toMatchObject({ model: SELECTOR })
    expect(server.requests).toEqual([{
      model: MODEL, messages: [{ role: 'user', content: 'Is the evidence relevant?' }],
      stream: false, max_tokens: 1, kinf_decision: 'label_scores',
    }])
    expect(server.headers[0]?.['x-kinf-service']).toBe(MODEL)
  })

  it('leaves ordinary chat dispatch streaming with its tool declarations', async () => {
    const server = await mockServer([{ events: textEvents }])
    const ctx = await harness(server.url)
    expect(await ctx.llm.resolveModelInfo('fixture', 'ordinary')).not.toHaveProperty('nativeTools')
    const result = await assemble(ctx, options({ model: 'ordinary', tools: [{ name: 'echo', description: 'Echo.', parameters: { type: 'object' } }] }))
    expect(result.finish).toEqual({ kind: 'stop' })
    expect(server.requests[0]).toMatchObject({ model: 'ordinary', stream: true, tools: [{ type: 'function', function: { name: 'echo' } }] })
    expect(server.requests[0]).not.toHaveProperty('kinf_decision')
  })

  it('uses the selected model id without a wire override and forwards the configured timeout', async () => {
    const { requestModel: _requestModel, ...modelProfile } = model()
    const profiles = resolveProfiles({ fixture: {
      api: 'openai-completions', baseURL: 'https://fixture.invalid/v1', timeoutMs: 2500, headers,
      models: [{ ...modelProfile, id: MODEL }],
    } })
    const adapter = new PiAiAdapter({ profiles: () => profiles, resolveApiKey: () => Promise.resolve(KEY), auth: memoryAuth() })
    const fetch = vi.fn(() => Promise.resolve(Response.json(reply())))
    vi.stubGlobal('fetch', fetch)
    const timeout = vi.spyOn(AbortSignal, 'timeout')
    try {
      const result = await collect(adapter.stream(options({ model: MODEL, system: 'Classify the evidence.' })))
      expect(result.at(-1)).toEqual({ type: 'finish', reason: { kind: 'stop' } })
      expect(timeout).toHaveBeenCalledWith(2500)
      expect(fetch).toHaveBeenCalledWith('https://fixture.invalid/v1/chat/completions', expect.objectContaining({
        body: JSON.stringify({ model: MODEL, messages: [
          { role: 'system', content: 'Classify the evidence.' },
          { role: 'user', content: 'Is the evidence relevant?' },
        ], stream: false, max_tokens: 1, kinf_decision: 'label_scores' }),
      }))
    } finally {
      timeout.mockRestore()
    }
  })

  it('keeps no-tools Chat on ordinary streaming and refuses explicit tools before dispatch', async () => {
    const server = await mockServer([{ events: textEvents }])
    const ctx = await harness(server.url)
    const info = await ctx.llm.resolveModelInfo('fixture', 'no-tools')
    expect(info?.nativeTools).toBe(false)
    expect(info).not.toHaveProperty('requestProtocol')
    expect(info).not.toHaveProperty('inputProjectionPolicy')
    const text = await assemble(ctx, options({ model: 'no-tools' }))
    expect(text.finish).toEqual({ kind: 'stop' })
    expect(server.requests[0]).toMatchObject({ model: 'no-tools', stream: true })
    expect(server.requests[0]).not.toHaveProperty('kinf_decision')
    const refused = await assemble(ctx, options({ model: 'no-tools', tools: [{ name: 'echo', description: 'Echo.', parameters: {} }] }))
    expect(refused.finish).toMatchObject({ kind: 'error', failure: { code: 'UNSUPPORTED_TOOLS' } })
    expect(server.requests).toHaveLength(1)
  })

  it('keeps explicit tool support independent of no-tools and Decision siblings', async () => {
    const server = await mockServer([{ events: textEvents }])
    const ctx = await harness(server.url)
    expect(await ctx.llm.resolveModelInfo('fixture', 'with-tools')).toMatchObject({ nativeTools: true })
    const result = await assemble(ctx, options({ model: 'with-tools', tools: [{ name: 'echo', description: 'Echo.', parameters: {} }] }))
    expect(result.finish).toEqual({ kind: 'stop' })
    expect(server.requests[0]).toMatchObject({ model: 'with-tools', stream: true, tools: [{ type: 'function', function: { name: 'echo' } }] })
  })

  it.each([
    { override: { maxTokens: 2 }, code: 'UNSUPPORTED_OPTION' },
    { override: { temperature: 0.5 }, code: 'UNSUPPORTED_OPTION' },
    { override: { tools: [{ name: 'echo', description: 'Echo.', parameters: {} }] }, code: 'UNSUPPORTED_TOOLS' },
    { override: { messages: [{ role: 'user' as const, content: [{ type: 'reasoning' as const, text: 'hidden' }] }] }, code: 'UNSUPPORTED_CONTENT' },
    { override: { messages: [] }, code: 'INVALID_REQUEST' },
  ])('rejects unsupported direct-call controls before network I/O (%j)', async ({ override, code }) => {
    const fetch = vi.fn()
    vi.stubGlobal('fetch', fetch)
    const adapter = new PiAiAdapter({
      profiles: () => resolveProfiles({ fixture: { api: 'openai-completions', baseURL: 'https://fixture.invalid/v1', headers, models: [model()] } }),
      resolveApiKey: () => Promise.resolve(KEY), auth: memoryAuth(),
    })
    await expect(collect(adapter.stream(options(override)))).rejects.toMatchObject({ code })
    expect(fetch).not.toHaveBeenCalled()
  })

  it.each([
    (value: ReturnType<typeof reply>) => { value.model = 'ordinary' },
    (value: ReturnType<typeof reply>) => { value.choices[0]!.message.content = 'no' },
    (value: ReturnType<typeof reply>) => { value.decision.logits.no = 10 },
    (value: ReturnType<typeof reply>) => { value.decision.label_scores.no = 0.4 },
    (value: ReturnType<typeof reply>) => { value.decision.logits.yes = 10 },
    (value: ReturnType<typeof reply>) => { value.usage.completion_tokens = 2 },
    (value: ReturnType<typeof reply>) => { value.decision.token_position = 7 },
    (value: ReturnType<typeof reply>) => { value.served_identity.serving_variant_id = 'ordinary' },
    (value: ReturnType<typeof reply>) => { value.served_identity.instance = 'another-instance' },
    (value: ReturnType<typeof reply>) => { value.served_identity.host_generation = 'f-f-f' },
    (value: ReturnType<typeof reply>) => { value.served_identity.runtime_profile_id = KEY },
  ])('rejects inconsistent native readouts without publishing content', async (mutate) => {
    const response = reply()
    mutate(response)
    const server = await mockServer([{ body: JSON.stringify(response) }])
    const result = await assemble(await harness(server.url), options())
    expect(result.finish).toMatchObject({ kind: 'error', failure: { code: 'MALFORMED_RESPONSE' } })
    expect(result.message.content).toEqual([])
    expect(server.requests).toHaveLength(1)
  })

  it.each([
    { body: '{invalid' },
    { body: 'x'.repeat(65537) },
    { events: textEvents },
  ])('rejects malformed, oversized, or streamed responses', async (response) => {
    const server = await mockServer([response])
    const result = await assemble(await harness(server.url), options())
    expect(result.finish).toMatchObject({ kind: 'error', failure: { code: 'MALFORMED_RESPONSE' } })
    expect(result.message.content).toEqual([])
  })

  it.each([
    { name: 'array instead of an object', payload: [reply()] },
    { name: 'zero input tokens', payload: { ...reply(), usage: { ...reply().usage, prompt_tokens: 0 } } },
    { name: 'non-numeric logit', payload: { ...reply(), decision: { ...reply().decision, logits: { yes: '1.25', no: 1.25 } } } },
    { name: 'missing label', payload: { ...reply(), decision: { ...reply().decision, logits: { yes: 1.25 } } } },
    { name: 'unrecognized decision label', payload: { ...reply(), decision: { ...reply().decision, label: 'maybe' } } },
  ])('rejects a malformed Decision reply with $name', async ({ payload }) => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(Response.json(payload))))
    await expect(directDecision().next()).rejects.toMatchObject({ code: 'MALFORMED_RESPONSE' })
  })

  it.each([
    { name: 'missing content type', response: () => new Response(null) },
    { name: 'missing JSON body', response: () => new Response(null, { headers: { 'content-type': 'application/json' } }) },
  ])('rejects a successful HTTP reply with $name', async ({ response }) => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(response())))
    await expect(directDecision().next()).rejects.toMatchObject({ code: 'MALFORMED_RESPONSE' })
  })

  it('cancels a response with a declared byte count above the limit before reading it', async () => {
    const cancel = vi.fn()
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(new ReadableStream({ cancel }), {
      headers: { 'content-type': 'application/json', 'content-length': '65537' },
    }))))
    await expect(directDecision().next()).rejects.toMatchObject({ code: 'MALFORMED_RESPONSE' })
    expect(cancel).toHaveBeenCalledOnce()
  })

  it('rejects a response whose body fails during a read', async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) { controller.error(new TypeError('fixture response body failed')) },
    })
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(body, {
      headers: { 'content-type': 'application/json' },
    }))))
    await expect(directDecision().next()).rejects.toMatchObject({ code: 'MALFORMED_RESPONSE' })
  })

  it.each([
    { overrides: { headers }, code: 'MISSING_CREDENTIAL' },
    { overrides: { headers: { ...headers, 'X-Kinf-Service': 'ordinary' }, apiKey: KEY }, code: 'INVALID_CONFIG' },
    { overrides: { options: options({ messages: [
      createToolResultMessage({ callId: ToolCallId('fixture-call'), content: [{ type: 'text', text: 'evidence' }], isError: false }),
    ] }) }, code: 'UNSUPPORTED_CONTENT' },
  ])('rejects an invalid Decision request before dispatch ($code)', async ({ overrides, code }) => {
    const fetch = vi.fn()
    vi.stubGlobal('fetch', fetch)
    await expect(directDecision(overrides).next()).rejects.toMatchObject({ code })
    expect(fetch).not.toHaveBeenCalled()
  })

  it.each([
    [401, 'AUTH'], [403, 'AUTH'], [402, 'QUOTA'], [429, 'RATE_LIMIT'], [408, 'TIMEOUT'], [503, 'SERVER'],
  ])('maps a Decision HTTP %i response to %s', async (status, code) => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(null, { status }))))
    await expect(directDecision().next()).rejects.toMatchObject({ code, failure: { status } })
  })

  it('reports fetch failure as a transport error', async () => {
    const failure = new TypeError('fixture transport closed')
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(failure)))
    await expect(directDecision().next()).rejects.toMatchObject({ code: 'TRANSPORT', cause: failure })
  })

  it('reports HTTP rejection without exposing its body or choosing another model', async () => {
    const server = await mockServer([{ status: 400, body: JSON.stringify({ error: { message: KEY } }) }])
    const result = await assemble(await harness(server.url), options())
    expect(result.finish).toMatchObject({ kind: 'error', failure: { code: 'INVALID_REQUEST' } })
    expect(JSON.stringify(result)).not.toContain(KEY)
    expect(server.requests).toHaveLength(1)
  })

  it.each([65536, 65537])('bounds actual streamed response bytes at %i', async (bytes) => {
    const json = JSON.stringify(reply())
    const payload = new TextEncoder().encode(json + ' '.repeat(bytes - new TextEncoder().encode(json).byteLength))
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(payload); controller.close() },
    }), { headers: { 'content-type': 'application/json' } }))))
    const response = collect(decisionChunks({
      options: options(), model: MODEL, baseURL: 'https://fixture.invalid/v1', headers, apiKey: KEY,
      signal: new AbortController().signal,
    }))
    if (bytes === 65536) expect((await response).at(-1)).toEqual({ type: 'finish', reason: { kind: 'stop' } })
    else await expect(response).rejects.toMatchObject({ code: 'MALFORMED_RESPONSE' })
  })

  it('cancels an outstanding request and enforces the configured HTTP timeout', async () => {
    const signals: AbortSignal[] = []
    vi.stubGlobal('fetch', vi.fn((_url: unknown, init: RequestInit) => new Promise<Response>((_resolve, reject) => {
      const signal = init.signal
      if (signal === null || signal === undefined) throw new Error('missing cancellation')
      signals.push(signal)
      signal.addEventListener('abort', () => {
        reject(signal.reason instanceof Error ? signal.reason : new Error('cancelled'))
      }, { once: true })
    })))
    const controller = new AbortController()
    const pending = collect(decisionChunks({ options: options(), model: MODEL, baseURL: 'https://fixture.invalid/v1', headers, apiKey: KEY, signal: controller.signal }))
    controller.abort('cancelled')
    await expect(pending).rejects.toThrow('cancelled')
    await expect(collect(decisionChunks({ options: options(), model: MODEL, baseURL: 'https://fixture.invalid/v1', headers, apiKey: KEY, timeoutMs: 10, signal: new AbortController().signal })))
      .rejects.toMatchObject({ code: 'TIMEOUT' })
    expect(signals.every(signal => signal.aborted)).toBe(true)
  })

  it('advertises the exact per-model schema literal and refuses incompatible declarations', () => {
    const schema = JSON.stringify(Config.toJSON())
    expect(schema).toContain('requestProtocol')
    expect(schema).toContain('kinf-decision-v1')
    expect(schema).toContain('nativeTools')
    const withoutReasoning: LlmPiAi.PiAiModelProfile = model()
    delete withoutReasoning.reasoningEfforts
    for (const invalidModel of [
      { ...model(), maxTokens: 32768 }, { ...model(), input: ['text' as const, 'image' as const] }, withoutReasoning,
    ]) {
      expect(() => resolveProfiles({ fixture: { api: 'openai-completions', baseURL: 'https://fixture.invalid/v1', models: [invalidModel] } }))
        .toThrow(/kinf-decision-v1 requires/)
    }
  })

  it('keeps Decision tools disabled and rejects contradictory capability metadata', () => {
    const providers = (nativeTools: boolean) => ({ fixture: {
      api: 'openai-completions' as const, baseURL: 'https://fixture.invalid/v1', models: [{ ...model(), nativeTools }],
    } })
    expect(resolveProfiles(providers(false)).get('fixture')?.nativeTools.get(SELECTOR)).toBe(false)
    expect(() => resolveProfiles(providers(true))).toThrow(/kinf-decision-v1 cannot enable nativeTools/)
  })

  it('rejects an absolute per-model endpoint whose provider endpoint cannot be compared', () => {
    expect(() => resolveProfiles({ fixture: {
      api: 'openai-completions', baseURL: '/relative-provider',
      models: [{ ...model(), baseURL: 'https://fixture.invalid/v1' }],
    } })).toThrow(/needs an absolute provider baseURL/)
  })
})
