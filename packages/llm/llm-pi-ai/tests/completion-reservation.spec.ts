/**
 * DSH-BUG-0011: the completion reservation admission prices must be the cap the
 * transport is actually asked for. pi-ai's payload builder falls back to the
 * catalog capability when a request omits a cap, so a route with no configured
 * request default must still reserve that transport fallback.
 */
import { afterEach, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime from '@deepseek-ai/dsh-llm'
import type { PreparedLlmCall, StreamChunk } from '@deepseek-ai/dsh-llm'
import * as PiAi from '../src/index.ts'
import { closeMockServers, mockServer, textEvents } from './mock-server.ts'

afterEach(async () => {
  await closeMockServers()
  vi.unstubAllEnvs()
})

interface ModelRow {
  id: string
  contextWindow?: number
  maxTokens?: number
}

async function harness(baseURL: string, models: ModelRow[], transportCapability?: number): Promise<Context> {
  vi.stubEnv('PI_RESERVATION_KEY', 'fixture-key')
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(PiAi, { providers: { gateway: {
    apiKeyEnv: 'PI_RESERVATION_KEY',
    api: 'openai-completions',
    baseURL,
    ...transportCapability === undefined ? {} : { defaultMaxTokens: transportCapability },
    models,
  } } })
  return ctx
}

/** Dispatch one prepared call and return the exact JSON body the provider received. */
async function dispatchBody(prepared: PreparedLlmCall, server: { requests: unknown[] }): Promise<Record<string, unknown>> {
  for await (const chunk of prepared.stream({ ...prepared.config, messages: [] })) void chunk
  expect(server.requests).toHaveLength(1)
  return server.requests[0] as Record<string, unknown>
}

/** The completion cap the wire request carries, under either compatibility spelling. */
function wireCompletionCap(body: Record<string, unknown>): number | undefined {
  return (body['max_completion_tokens'] ?? body['max_tokens']) as number | undefined
}

it('reserves the transport fallback before dispatch when no request default exists', async () => {
  const server = await mockServer([{ events: textEvents }])
  const ctx = await harness(server.url, [{ id: 'canonical', contextWindow: 65_536 }], 4_096)

  const info = await ctx.llm.resolveModelInfo('gateway', 'canonical')
  // A deployment-configured request default does not exist on this route…
  expect(info.defaultMaxTokens).toBeUndefined()
  // …but its transport still asks for the catalog capability when omitted.
  expect(info.transportMaxTokens).toBe(4_096)

  const prepared = await ctx.llm.prepareCall({ provider: 'gateway', model: 'canonical' })
  expect(prepared.completion).toEqual({ maxTokens: 4_096, source: 'transport-fallback' })
  expect(prepared.config.maxTokens).toBe(4_096)

  const body = await dispatchBody(prepared, server)
  expect(wireCompletionCap(body)).toBe(prepared.config.maxTokens)
  expect(wireCompletionCap(body)).toBeLessThanOrEqual(prepared.completion.maxTokens ?? 0)
}, 20_000)

it('keeps an explicit caller cap ahead of the transport fallback', async () => {
  const server = await mockServer([{ events: textEvents }])
  const ctx = await harness(server.url, [{ id: 'canonical', contextWindow: 65_536 }], 4_096)

  const prepared = await ctx.llm.prepareCall({ provider: 'gateway', model: 'canonical', maxTokens: 512 })
  expect(prepared.completion).toEqual({ maxTokens: 512, source: 'explicit' })
  expect(prepared.config.maxTokens).toBe(512)
  expect(prepared.adapterDefaults.maxTokens).toBeUndefined()

  const body = await dispatchBody(prepared, server)
  expect(wireCompletionCap(body)).toBe(512)
}, 20_000)

it('prefers a deployment request default over the transport fallback', async () => {
  const server = await mockServer([{ events: textEvents }])
  const ctx = await harness(server.url, [{ id: 'capped', contextWindow: 65_536, maxTokens: 2_048 }], 4_096)

  const info = await ctx.llm.resolveModelInfo('gateway', 'capped')
  expect(info.defaultMaxTokens).toBe(2_048)

  const prepared = await ctx.llm.prepareCall({ provider: 'gateway', model: 'capped' })
  expect(prepared.completion).toEqual({ maxTokens: 2_048, source: 'adapter-default' })
  expect(prepared.config.maxTokens).toBe(2_048)
  expect(prepared.adapterDefaults.maxTokens).toBe(true)

  const body = await dispatchBody(prepared, server)
  expect(wireCompletionCap(body)).toBe(2_048)
}, 20_000)

it('refuses a resident dispatch binding before any provider request', async () => {
  const server = await mockServer([{ events: textEvents }])
  const ctx = await harness(server.url, [{ id: 'canonical', contextWindow: 65_536 }], 4_096)
  const prepared = await ctx.llm.prepareCall({ provider: 'gateway', model: 'canonical' })
  const residentBinding = {
    schema: 'kinf.resident-only.v1', instance: 'fixture-instance', hostGeneration: 'boot-1',
    servingVariantId: 'variant-1', servingVariantRevision: 'variant-rev-1',
    modelArtifact: 'artifact-1', artifactRevision: 'artifact-rev-1',
    targetConfigRevision: 'config-1', residentGeneration: 'resident-1', projectionId: 'projection-1',
  }
  const chunks: StreamChunk[] = []
  for await (const chunk of prepared.stream(
    { ...prepared.config, messages: [], residentBinding },
  )) chunks.push(chunk)
  expect(chunks.at(-1)).toMatchObject({
    type: 'finish',
    reason: { kind: 'error', failure: { code: 'UNSUPPORTED_RESIDENT_TRANSPORT' } },
  })
  expect(server.requests).toHaveLength(0)
}, 20_000)
