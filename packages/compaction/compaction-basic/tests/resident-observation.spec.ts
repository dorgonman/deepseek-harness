import { expect, it } from 'vitest'
import {
  selectResidentFromSource,
  type ResidentObservationRequest,
  type ResidentObservationSource,
  type ResidentSelectionRequest,
} from '../src/resident-observation.ts'
import { RESIDENT_ONLY_V1, type Candidate, type ResidentPrecondition } from '../src/resident-selector.ts'

const NOW_MS = 10_000
const GATEWAY_ID = 'fixture-gateway'
const PROJECTION_ID = 'fixture-projection-v1'
const LOCAL = { id: 'fixture-local', name: 'fixture-alpha', bootGeneration: 'fixture-boot-alpha-3' }
const PEER = { id: 'fixture-peer', name: 'fixture-beta', bootGeneration: 'fixture-boot-beta-9' }
const FLASH = { label: 'flash', variantId: 'fixture-flash-v1', parameters: 8_000_000_000 }
const LARGE = { label: '27b', variantId: 'fixture-27b-v1', parameters: 27_000_000_000 }

type Mutable<T> = { -readonly [Key in keyof T]: Mutable<T[Key]> }

function candidate(
  variant: typeof FLASH | typeof LARGE = FLASH,
  placement: typeof LOCAL | typeof PEER = LOCAL,
): Mutable<Candidate> {
  const logicalModel = `fixture-model-${variant.label}`
  const identity: Mutable<ResidentPrecondition> = {
    schema: RESIDENT_ONLY_V1,
    instance: `fixture-instance-${variant.label}`,
    host_generation: `host-${placement.id}-${variant.label}-1`,
    serving_variant_id: variant.variantId,
    serving_variant_revision: `variant-${variant.label}-2`,
    model_artifact: `artifact-${variant.label}`,
    artifact_revision: `artifact-${variant.label}-3`,
    target_config_revision: `config-${variant.label}-4`,
    resident_generation: `resident-${placement.id}-${variant.label}-5`,
  }
  return {
    node: { ...placement, registered: true, authorized: true, freshness: 'fresh', observedAtMs: NOW_MS - 10 },
    host: { identity, mode: 'mservice', state: 'ready', enabled: true, usable: true, admissionOpen: true },
    model: {
      logicalModel,
      servingVariantId: variant.variantId,
      servingVariantRevision: identity.serving_variant_revision,
      artifactId: identity.model_artifact,
      targetConfigRevision: identity.target_config_revision,
      loaded: true,
      ready: true,
      configurationValidated: true,
      operations: ['chat'],
      capabilitySource: 'runtime',
      capabilities: ['text'],
      budget: { contextTokens: 2048, maxOutputTokens: 256, maxInputTokensAtMaxOutput: 1792 },
      size: {
        group: 'fixture-comparable-parameters',
        units: 'parameters',
        value: variant.parameters,
        source: 'authoritative',
      },
    },
    route: {
      id: `route-${placement.id}-${variant.label}`,
      gatewayId: GATEWAY_ID,
      nodeId: placement.id,
      bootGeneration: placement.bootGeneration,
      instance: identity.instance,
      logicalModel,
      servingVariantId: variant.variantId,
      alias: `${logicalModel}@${placement.name}`,
      healthy: true,
      contract: RESIDENT_ONLY_V1,
      noRetry: true,
      noFallback: true,
      verifiedAtMs: NOW_MS - 20,
    },
    queue: { pending: 0, active: 0, maxPending: 8 },
    measurement: {
      projectionId: PROJECTION_ID,
      hostGeneration: identity.host_generation,
      residentGeneration: identity.resident_generation,
      targetConfigRevision: identity.target_config_revision,
      promptTokens: 600,
    },
  }
}

class FakeSource implements ResidentObservationSource {
  readonly requests: ResidentObservationRequest[] = []

  constructor(private readonly rows: readonly Candidate[]) {}

  async observe(request: ResidentObservationRequest): Promise<readonly Candidate[]> {
    this.requests.push(request)
    return this.rows
  }
}

function request(overrides: Partial<ResidentSelectionRequest> = {}): ResidentSelectionRequest {
  return {
    gatewayId: GATEWAY_ID,
    pinnedNodeId: LOCAL.id,
    projectionId: PROJECTION_ID,
    outputTokens: 128,
    reserveTokens: 16,
    maxAgeMs: 500,
    ...overrides,
  }
}

const clock = () => NOW_MS

it('observes exactly the pinned node and returns a frozen receipt', async () => {
  const local = candidate()
  const source = new FakeSource([local])
  const selection = await selectResidentFromSource(request(), source, { now: clock })
  expect(source.requests).toEqual([{ gatewayId: GATEWAY_ID, nodeIds: [LOCAL.id] }])
  expect(selection.ok).toBe(true)
  if (!selection.ok) return
  expect(selection.receipt).toEqual({
    alias: local.route.alias,
    gatewayId: GATEWAY_ID,
    routeId: local.route.id,
    nodeId: LOCAL.id,
    nodeName: LOCAL.name,
    instance: local.host.identity.instance,
    servingVariantId: local.host.identity.serving_variant_id,
    servingVariantRevision: local.host.identity.serving_variant_revision,
    modelArtifact: local.host.identity.model_artifact,
    artifactRevision: local.host.identity.artifact_revision,
    targetConfigRevision: local.host.identity.target_config_revision,
    residentGeneration: local.host.identity.resident_generation,
    budgetSource: 'resident-only-observation',
    projectionId: PROJECTION_ID,
    reason: 'only-eligible',
    precondition: local.host.identity,
  })
  expect(Object.isFrozen(selection.receipt)).toBe(true)
})

it('observes only the explicit allowlist and deduplicates repeated node ids', async () => {
  const local = candidate(LARGE)
  const remote = candidate(FLASH, PEER)
  const source = new FakeSource([local, remote])
  const selection = await selectResidentFromSource(
    request({ pinnedNodeId: LOCAL.id, allowedNodeIds: [LOCAL.id, PEER.id, PEER.id], preferredVariantIds: [FLASH.variantId] }),
    source,
    { now: clock },
  )
  expect(source.requests).toHaveLength(1)
  expect(source.requests[0]?.nodeIds).toEqual([LOCAL.id, PEER.id])
  expect(selection.ok).toBe(true)
  if (!selection.ok) return
  expect(selection.target.nodeId).toBe(PEER.id)
  expect(selection.receipt.reason).toBe('configured-preference')
})

it('refuses an unscoped request before observing anything', async () => {
  const source = new FakeSource([candidate()])
  const { pinnedNodeId: _pinned, ...unscoped } = request()
  const selection = await selectResidentFromSource(unscoped, source, { now: clock })
  expect(selection).toEqual({ ok: false, code: 'resident_summary_scope_required' })
  expect(source.requests).toHaveLength(0)
})

it('propagates a typed refusal without a receipt', async () => {
  const unloaded = candidate()
  unloaded.host.state = 'unloaded'
  unloaded.model.loaded = false
  unloaded.model.ready = false
  const selection = await selectResidentFromSource(request(), new FakeSource([unloaded]), { now: clock })
  expect(selection).toEqual({ ok: false, code: 'resident_summary_no_eligible' })
})

it('applies the clock to observation freshness', async () => {
  const stale = candidate()
  stale.node.observedAtMs = NOW_MS - 501
  const selection = await selectResidentFromSource(request(), new FakeSource([stale]), { now: clock })
  expect(selection).toEqual({ ok: false, code: 'resident_summary_no_eligible' })
})

it('forwards cancellation and propagates a source failure', async () => {
  const controller = new AbortController()
  let seen: AbortSignal | undefined
  const failing: ResidentObservationSource = {
    observe: (observation) => {
      seen = observation.signal
      return Promise.reject(new Error('observation transport failed'))
    },
  }
  await expect(selectResidentFromSource(request(), failing, { now: clock, signal: controller.signal }))
    .rejects.toThrow('observation transport failed')
  expect(seen).toBe(controller.signal)
})
