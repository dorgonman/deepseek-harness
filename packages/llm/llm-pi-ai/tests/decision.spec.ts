/** Strict native Decisions remain separate from ordinary pi-ai chat dispatch. */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import * as LlmPiAi from '@deepseek-ai/dsh-llm-pi-ai'
import { PiAiAdapter } from '../src/adapter.ts'
import { Config, resolveProfiles } from '../src/config.ts'
import { decisionChunks } from '../src/decision.ts'
import { memoryAuth } from './auth-double.ts'
import { assemble } from './assemble.ts'
import { closeMockServers, mockServer, textEvents } from './mock-server.ts'

const MODEL = 'flash-native-mtp-decision'
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
  return { id: `${MODEL}@fixture`, requestModel: MODEL, requestProtocol: 'kinf-decision-v1' as const,
    contextWindow: 262144, maxTokens: 1, input: ['text' as const], reasoningEfforts: false as const }
}

function options(overrides: Partial<GenerateOptions> = {}): GenerateOptions {
  return { provider: 'fixture', model: `${MODEL}@fixture`, messages: [
    createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'Is the evidence relevant?' }] }),
  ], ...overrides }
}

const headers = { 'X-Kinf-Mode': 'mservice', 'X-Kinf-Instance': 'fixture-managed', 'X-Kinf-Service': MODEL }

async function collect(chunks: AsyncIterable<StreamChunk>) {
  const result: StreamChunk[] = []
  for await (const chunk of chunks) result.push(chunk)
  return result
}

async function harness(baseURL: string) {
  const ctx = new Context()
  contexts.push(ctx)
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(LlmPiAi, { providers: { fixture: {
    api: 'openai-completions', baseURL, headers: { Authorization: `Bearer ${KEY}`, ...headers },
    models: [model(), { id: 'ordinary', contextWindow: 65536 }],
  } } })
  return ctx
}

describe('KINF Decision adaptation', () => {
  it('publishes both scores and exact served identity without private metadata', async () => {
    const server = await mockServer([{ body: JSON.stringify(reply()) }])
    const ctx = await harness(server.url)
    expect(await ctx.llm.resolveModelInfo('fixture', `${MODEL}@fixture`)).toMatchObject({
      nativeTools: false, inputProjectionPolicy: 'reject', defaultMaxTokens: 1,
    })
    const result = await assemble(ctx, options())
    expect(result.finish).toEqual({ kind: 'stop' })
    expect(result.usage).toEqual({ inputTokens: 8, outputTokens: 1, totalTokens: 9 })
    const block = result.message.content[0]
    expect(block?.type).toBe('text')
    if (block?.type !== 'text') throw new Error('expected visible Decision text')
    const displayed = JSON.parse(block.text) as Record<string, unknown>
    expect(displayed['decision']).toEqual(reply().decision)
    expect(displayed['served_identity']).toMatchObject({ serving_variant_id: MODEL, worker_pid: 42 })
    expect(block.text).not.toContain(KEY)
    expect(block.text).not.toContain('private')
    expect(result.message.source).toMatchObject({ model: `${MODEL}@fixture` })
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
    const withoutReasoning: LlmPiAi.PiAiModelProfile = model()
    delete withoutReasoning.reasoningEfforts
    for (const invalidModel of [
      { ...model(), maxTokens: 32768 }, { ...model(), input: ['text' as const, 'image' as const] }, withoutReasoning,
    ]) {
      expect(() => resolveProfiles({ fixture: { api: 'openai-completions', baseURL: 'https://fixture.invalid/v1', models: [invalidModel] } }))
        .toThrow(/kinf-decision-v1 requires/)
    }
  })
})
