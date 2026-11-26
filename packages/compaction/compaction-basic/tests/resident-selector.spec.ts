import { describe, expect, it } from 'vitest'
import {
  RESIDENT_ONLY_V1,
  selectResidentSummary,
  type Candidate,
  type ResidentPrecondition,
  type SelectionInput,
} from '../src/resident-selector.ts'

const NOW_MS = 10_000
const MAX_AGE_MS = 500
const GATEWAY_ID = 'fixture-gateway'
const PROJECTION_ID = 'fixture-projection-v1'
const LOCAL = { id: 'fixture-local', name: 'fixture-alpha', bootGeneration: 'fixture-boot-alpha-3' }
const PEER = { id: 'fixture-peer', name: 'fixture-beta', bootGeneration: 'fixture-boot-beta-9' }
const FLASH = { label: 'flash', variantId: 'fixture-flash-v1', parameters: 8_000_000_000 }
const LARGE = { label: '27b', variantId: 'fixture-27b-v1', parameters: 27_000_000_000 }

type Mutable<T> = { -readonly [Key in keyof T]: Mutable<T[Key]> }
type FixtureCandidate = Mutable<Candidate>

function candidate(
  variant: typeof FLASH | typeof LARGE = FLASH,
  placement: typeof LOCAL | typeof PEER = LOCAL,
): FixtureCandidate {
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

function request(
  candidates: readonly FixtureCandidate[],
  overrides: Partial<Omit<SelectionInput, 'candidates'>> = {},
): SelectionInput {
  return {
    nowMs: NOW_MS,
    maxAgeMs: MAX_AGE_MS,
    gatewayId: GATEWAY_ID,
    pinnedNodeId: LOCAL.id,
    projectionId: PROJECTION_ID,
    outputTokens: 128,
    reserveTokens: 16,
    ...overrides,
    candidates,
  }
}

function unscoped(candidates: readonly FixtureCandidate[]): SelectionInput {
  const { pinnedNodeId: _pinned, ...rest } = request(candidates)
  return rest
}

function edit(base: FixtureCandidate, change: (row: FixtureCandidate) => void): FixtureCandidate {
  const copy = structuredClone(base)
  change(copy)
  return copy
}

function unloaded(row: FixtureCandidate): void {
  row.host.state = 'unloaded'
  row.model.loaded = false
  row.model.ready = false
}

function expectTarget(
  input: SelectionInput,
  winner: FixtureCandidate,
  reason: 'only-eligible' | 'configured-preference' | 'comparable-size' | 'queue-pressure' | 'stable-identity',
): void {
  expect(selectResidentSummary(input)).toEqual({
    ok: true,
    target: {
      nodeId: winner.node.id,
      gatewayId: winner.route.gatewayId,
      routeId: winner.route.id,
      alias: `${winner.model.logicalModel}@${winner.node.name}`,
      precondition: { ...winner.host.identity },
    },
    reason,
  })
}

describe('resident summary selector', () => {
  it('uses the only resident deployment despite a preferred unloaded variant', () => {
    const flash = candidate()
    const large = edit(candidate(LARGE), unloaded)
    expectTarget(request([large, flash], { preferredVariantIds: [LARGE.variantId] }), flash, 'only-eligible')
    expectTarget(request([flash, large], { preferredVariantIds: [LARGE.variantId] }), flash, 'only-eligible')
  })

  it('ranks a larger comparable resident above configured preference only by size tier', () => {
    const flash = candidate()
    const large = candidate(LARGE)
    expectTarget(request([flash, large], { allowedNodeIds: [LOCAL.id] }), large, 'comparable-size')
    expectTarget(request([large, flash], { allowedNodeIds: [LOCAL.id] }), large, 'comparable-size')
  })

  it('fails closed when nothing is eligible', () => {
    expect(selectResidentSummary(request([edit(candidate(), unloaded)])))
      .toEqual({ ok: false, code: 'resident_summary_no_eligible' })
    expect(selectResidentSummary(request([])))
      .toEqual({ ok: false, code: 'resident_summary_no_eligible' })
  })

  it('keeps the pinned node by default and widens only through an explicit allowlist', () => {
    const local = candidate(LARGE)
    const remote = candidate(FLASH, PEER)
    expectTarget(request([remote, local], { preferredVariantIds: [FLASH.variantId] }), local, 'only-eligible')
    const widened = { allowedNodeIds: [LOCAL.id, PEER.id], preferredVariantIds: [FLASH.variantId] }
    expectTarget(request([local, remote], widened), remote, 'configured-preference')
    expect(selectResidentSummary(request([local, remote], { allowedNodeIds: [] })))
      .toEqual({ ok: false, code: 'resident_summary_no_eligible' })
  })

  it('excludes unauthorized and unloaded observations', () => {
    const local = candidate()
    const unauthorized = edit(candidate(FLASH, PEER), (row) => { row.node.authorized = false })
    expectTarget(request([unauthorized, local], { allowedNodeIds: [LOCAL.id, PEER.id] }), local, 'only-eligible')
    const unloadedRemote = edit(candidate(FLASH, PEER), unloaded)
    expectTarget(request([unloadedRemote, local], { allowedNodeIds: [LOCAL.id, PEER.id] }), local, 'only-eligible')
  })

  it('excludes non-Chat, non-text, unknown-capacity and stale observations', () => {
    const rejects = [
      (row: FixtureCandidate) => { row.model.operations = ['decision'] },
      (row: FixtureCandidate) => { row.model.capabilities = ['image'] },
      (row: FixtureCandidate) => { row.model.budget = null },
      (row: FixtureCandidate) => { row.node.observedAtMs = NOW_MS - MAX_AGE_MS - 1 },
      (row: FixtureCandidate) => { row.route.verifiedAtMs = NOW_MS - MAX_AGE_MS - 1 },
    ]
    for (const change of rejects) {
      expect(selectResidentSummary(request([edit(candidate(), change)])))
        .toEqual({ ok: false, code: 'resident_summary_no_eligible' })
    }
  })

  it('enforces the measured prompt plus reserve boundary exactly', () => {
    const fits = edit(candidate(), (row) => { row.measurement!.promptTokens = 1904 })
    expectTarget(request([fits]), fits, 'only-eligible')
    const over = edit(candidate(), (row) => { row.measurement!.promptTokens = 1905 })
    expect(selectResidentSummary(request([over])))
      .toEqual({ ok: false, code: 'resident_summary_no_eligible' })
  })

  it('fails closed on contradictory residency evidence instead of pooling rows', () => {
    const first = candidate()
    const contradictory = edit(candidate(), (row) => {
      row.model.logicalModel = 'conflicting-model'
      row.route.logicalModel = 'conflicting-model'
      row.route.alias = 'conflicting-model@fixture-alpha'
    })
    expect(selectResidentSummary(request([first, contradictory])))
      .toEqual({ ok: false, code: 'resident_summary_no_eligible' })
  })

  it('breaks a fully tied tier deterministically by stable identity', () => {
    const first = candidate(FLASH, LOCAL)
    const peer = candidate(FLASH, PEER)
    const allowedNodeIds = [LOCAL.id, PEER.id]
    expectTarget(request([peer, first], { allowedNodeIds }), first, 'stable-identity')
    expectTarget(request([first, peer], { allowedNodeIds }), first, 'stable-identity')
  })

  it('requires a scope and rejects invalid input', () => {
    expect(selectResidentSummary(unscoped([candidate()])))
      .toEqual({ ok: false, code: 'resident_summary_scope_required' })
    expect(selectResidentSummary(request([candidate()], { maxAgeMs: 60_001 })))
      .toEqual({ ok: false, code: 'resident_summary_invalid_input' })
    expect(selectResidentSummary(request([candidate()], { projectionId: '' })))
      .toEqual({ ok: false, code: 'resident_summary_invalid_input' })
  })
})
