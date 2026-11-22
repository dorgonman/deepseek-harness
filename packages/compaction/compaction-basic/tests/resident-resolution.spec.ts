import { createUserMessage, type RequestMessage } from '@deepseek-ai/dsh-llm'
import { expect, it } from 'vitest'
import {
  ResidentResolutionError,
  residentProjectionId,
  resolveResidentSummary,
  type ResidentProjection,
  type ResidentResolutionConfig,
} from '../src/resident-resolution.ts'
import type { ResidentObservationRequest, ResidentObservationSource } from '../src/resident-observation.ts'
import { RESIDENT_ONLY_V1, type Candidate, type ResidentPrecondition } from '../src/resident-selector.ts'

const NOW_MS = 10_000
const GATEWAY_ID = 'fixture-gateway'
const LOCAL = { id: 'fixture-local', name: 'fixture-alpha', bootGeneration: 'fixture-boot-alpha-3' }
const FLASH = { label: 'flash', variantId: 'fixture-flash-v1', parameters: 8_000_000_000 }

type Mutable<T> = { -readonly [Key in keyof T]: Mutable<T[Key]> }

function candidate(): Mutable<Candidate> {
  const identity: Mutable<ResidentPrecondition> = {
    schema: RESIDENT_ONLY_V1,
    instance: 'fixture-instance-flash',
    host_generation: 'host-fixture-local-flash-1',
    serving_variant_id: FLASH.variantId,
    serving_variant_revision: 'variant-flash-2',
    model_artifact: 'artifact-flash',
    artifact_revision: 'artifact-flash-3',
    target_config_revision: 'config-flash-4',
    resident_generation: 'resident-fixture-local-flash-5',
  }
  const logicalModel = `fixture-model-${FLASH.label}`
  return {
    node: { ...LOCAL, registered: true, authorized: true, freshness: 'fresh', observedAtMs: NOW_MS - 10 },
    host: { identity, mode: 'mservice', state: 'ready', enabled: true, usable: true, admissionOpen: true },
    model: {
      logicalModel,
      servingVariantId: FLASH.variantId,
      servingVariantRevision: identity.serving_variant_revision,
      artifactId: identity.model_artifact,
      targetConfigRevision: identity.target_config_revision,
      loaded: true,
      ready: true,
      configurationValidated: true,
      operations: ['chat'],
      capabilitySource: 'runtime',
      capabilities: ['text'],
      budget: { contextTokens: 8_192, maxOutputTokens: 1_024, maxInputTokensAtMaxOutput: 7_168 },
      size: { group: 'fixture-comparable-parameters', units: 'parameters', value: FLASH.parameters, source: 'authoritative' },
    },
    route: {
      id: 'route-fixture-local-flash',
      gatewayId: GATEWAY_ID,
      nodeId: LOCAL.id,
      bootGeneration: LOCAL.bootGeneration,
      instance: identity.instance,
      logicalModel,
      servingVariantId: FLASH.variantId,
      alias: `${logicalModel}@${LOCAL.name}`,
      healthy: true,
      contract: RESIDENT_ONLY_V1,
      noRetry: true,
      noFallback: true,
      verifiedAtMs: NOW_MS - 20,
    },
    queue: { pending: 0, active: 0, maxPending: 8 },
    measurement: {
      projectionId: '',
      hostGeneration: identity.host_generation,
      residentGeneration: identity.resident_generation,
      targetConfigRevision: identity.target_config_revision,
      renderedPromptTokens: 600,
    },
  }
}

class FakeSource implements ResidentObservationSource {
  readonly requests: ResidentObservationRequest[] = []

  constructor(private readonly rows: readonly Candidate[]) {}

  async observe(request: ResidentObservationRequest): Promise<readonly Candidate[]> {
    this.requests.push(request)
    return this.rows.map(row => row.measurement === undefined ? row : ({
      ...row,
      measurement: { ...row.measurement, projectionId: request.projection?.id ?? row.measurement.projectionId },
    }))
  }
}

function config(overrides: Partial<ResidentResolutionConfig> = {}): ResidentResolutionConfig {
  return {
    mode: 'resident-preferred',
    gatewayId: GATEWAY_ID,
    pinnedNodeId: LOCAL.id,
    provider: 'fixture-provider',
    maxAgeMs: 500,
    ...overrides,
  }
}

function projection(): ResidentProjection {
  return {
    messages: [createUserMessage({
      content: [{ type: 'text', text: 'Summarize this projection.' }], source: { kind: 'user' },
    })] as RequestMessage[],
    maxTokens: 512,
    reserveTokens: 16,
  }
}

const clock = () => NOW_MS

it('resolves one frozen provider/model target with a receipt and projection identity', async () => {
  const row = candidate()
  const source = new FakeSource([row])
  const input = projection()
  const resolved = await resolveResidentSummary(source, config(), input, { now: clock })
  expect(resolved.provider).toBe('fixture-provider')
  expect(resolved.model).toBe(row.route.alias)
  expect(resolved.receipt).toMatchObject({
    alias: row.route.alias, nodeId: LOCAL.id, instance: row.host.identity.instance,
    budgetSource: 'resident-only-observation', projectionId: residentProjectionId(input),
  })
  expect(source.requests).toHaveLength(1)
  expect(source.requests[0]?.projection).toMatchObject({
    id: residentProjectionId(input), outputTokens: 512,
  })
  expect(source.requests[0]?.projection?.messages).toEqual(input.messages)
})

it('refuses fixed mode, a missing source and incomplete configuration', async () => {
  await expect(resolveResidentSummary(new FakeSource([candidate()]), config({ mode: 'fixed' }), projection(), { now: clock }))
    .rejects.toMatchObject({ code: 'mode-fixed' })
  await expect(resolveResidentSummary(undefined, config(), projection(), { now: clock }))
    .rejects.toMatchObject({ code: 'source-unavailable' })
  const { gatewayId: _gateway, ...noGateway } = config()
  await expect(resolveResidentSummary(new FakeSource([candidate()]), noGateway, projection(), { now: clock }))
    .rejects.toMatchObject({ code: 'not-configured' })
  const { provider: _provider, ...noProvider } = config()
  await expect(resolveResidentSummary(new FakeSource([candidate()]), noProvider, projection(), { now: clock }))
    .rejects.toMatchObject({ code: 'not-configured' })
})

it('refuses a missing scope before observing and maps ineligibility to a typed error', async () => {
  const { pinnedNodeId: _pinned, ...unscoped } = config()
  const source = new FakeSource([candidate()])
  await expect(resolveResidentSummary(source, unscoped, projection(), { now: clock }))
    .rejects.toMatchObject({ code: 'scope-required' })
  expect(source.requests).toHaveLength(0)

  const unloaded = candidate()
  unloaded.host.state = 'unloaded'
  unloaded.model.loaded = false
  unloaded.model.ready = false
  await expect(resolveResidentSummary(new FakeSource([unloaded]), config(), projection(), { now: clock }))
    .rejects.toMatchObject({ code: 'no-eligible' })
})

it('forwards cancellation and rejects a projection without a stable identity', async () => {
  const controller = new AbortController()
  let seen: AbortSignal | undefined
  const source: ResidentObservationSource = {
    observe: (request) => {
      seen = request.signal
      const row = candidate()
      row.measurement!.projectionId = request.projection?.id ?? ''
      return Promise.resolve([row])
    },
  }
  await resolveResidentSummary(source, config(), projection(), { now: clock, signal: controller.signal })
  expect(seen).toBe(controller.signal)
  expect(() => residentProjectionId({ ...projection(), maxTokens: -1 })).not.toThrow()
  expect(residentProjectionId(projection())).toMatch(/^[0-9a-f]{64}$/)
})

it('keeps typed refusal codes distinct from generic errors', async () => {
  const error = await resolveResidentSummary(undefined, config(), projection(), { now: clock })
    .catch((reason: unknown) => reason)
  expect(error).toBeInstanceOf(ResidentResolutionError)
})
