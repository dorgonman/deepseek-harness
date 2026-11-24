import { expect, it } from 'vitest'
import {
  selectResidentSummary,
  type Candidate,
  type ResidentPrecondition,
  type SelectionInput,
  type SelectionResult,
} from '../src/resident-selector.ts'

const CONTRACT = 'kinf.resident-only.v1'
const NOW_MS = 10_000
const MAX_AGE_MS = 500
const GATEWAY_ID = 'fixture-gateway'
const PROJECTION_ID = 'fixture-summary-projection-v2'
const LOCAL = { id: 'fixture-local', name: 'fixture-alpha', bootGeneration: 'fixture-boot-alpha-3' }
const PEER = { id: 'fixture-peer', name: 'fixture-beta', bootGeneration: 'fixture-boot-beta-9' }
const FLASH = { label: 'flash', variantId: 'fixture-flash-v1', parameters: 8_000_000_000 }
const TWENTY_SEVEN = { label: '27b', variantId: 'fixture-27b-v1', parameters: 27_000_000_000 }
type Mutable<T> = { -readonly [K in keyof T]: Mutable<T[K]> }
type FixtureCandidate = Mutable<Candidate>
type FixtureInput = Mutable<SelectionInput>
type NodePlacement = Pick<Candidate['node'], 'id' | 'name' | 'bootGeneration'>
type Success = Extract<SelectionResult, { ok: true }>
type RefusalCode = Extract<SelectionResult, { ok: false }>['code']
type CandidateEdit = (candidate: FixtureCandidate) => void

function candidate(variant = FLASH, placement: NodePlacement = LOCAL): FixtureCandidate {
  const logicalModel = `fixture-model-${variant.label}`
  const identity: Mutable<ResidentPrecondition> = {
    schema: CONTRACT,
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
      logicalModel, servingVariantId: variant.variantId,
      servingVariantRevision: identity.serving_variant_revision, artifactId: identity.model_artifact,
      targetConfigRevision: identity.target_config_revision, loaded: true, ready: true, configurationValidated: true,
      operations: ['chat'], capabilitySource: 'runtime', capabilities: ['text'],
      budget: { contextTokens: 2048, maxOutputTokens: 256, maxInputTokensAtMaxOutput: 1792 },
      size: { group: 'fixture-comparable-parameters', units: 'parameters', value: variant.parameters, source: 'authoritative' },
    },
    route: {
      id: `route-${placement.id}-${variant.label}`, gatewayId: GATEWAY_ID, nodeId: placement.id,
      bootGeneration: placement.bootGeneration, instance: identity.instance, logicalModel,
      servingVariantId: variant.variantId, alias: `${logicalModel}@${placement.name}`, healthy: true,
      contract: CONTRACT, noRetry: true, noFallback: true, verifiedAtMs: NOW_MS - 20,
    },
    queue: { pending: 0, active: 0, maxPending: 8 },
    measurement: {
      projectionId: PROJECTION_ID, hostGeneration: identity.host_generation,
      residentGeneration: identity.resident_generation, targetConfigRevision: identity.target_config_revision,
      renderedPromptTokens: 600,
    },
  }
}

function request(candidates: FixtureCandidate[], overrides: Partial<Omit<FixtureInput, 'candidates'>> = {}): FixtureInput {
  return {
    nowMs: NOW_MS, maxAgeMs: MAX_AGE_MS, gatewayId: GATEWAY_ID, pinnedNodeId: LOCAL.id,
    projectionId: PROJECTION_ID, outputTokens: 128, reserveTokens: 16, ...overrides, candidates,
  }
}

function edit(base: FixtureCandidate, change: CandidateEdit): FixtureCandidate {
  const copy = structuredClone(base)
  change(copy)
  return copy
}

function strip<T extends object>(object: T, field: keyof T): void {
  // Intentionally violate the static shape to exercise untrusted producer metadata.
  expect(Reflect.deleteProperty(object, field)).toBe(true)
}

function expectTarget(input: SelectionInput, winner: Candidate, reason: Success['reason']): void {
  const expected: Success = {
    ok: true,
    target: {
      nodeId: winner.node.id, gatewayId: winner.route.gatewayId, routeId: winner.route.id,
      alias: `${winner.model.logicalModel}@${winner.node.name}`, precondition: { ...winner.host.identity },
    },
    reason,
  }
  expect(selectResidentSummary(input)).toEqual(expected)
}

function expectRefusal(input: SelectionInput, code: RefusalCode = 'resident_summary_no_eligible'): void {
  expect(selectResidentSummary(input)).toEqual({ ok: false, code })
}

function rejectRows(cases: readonly [description: string, change: CandidateEdit][]): void {
  for (const [description, change] of cases) {
    it(`no eligible resident: ${description}`, () => { expectRefusal(request([edit(candidate(), change)])) })
  }
}

function* permutations<T>(values: readonly T[]): Generator<T[]> {
  if (values.length === 0) {
    yield []
    return
  }
  for (const [index, value] of values.entries()) {
    for (const rest of permutations(values.filter((_, i) => i !== index))) yield [value, ...rest]
  }
}

const unloaded: CandidateEdit = (row) => {
  row.host.state = 'unloaded'
  row.model.loaded = row.model.ready = false
}

it('Flash only: a nonresident 27B cannot win despite explicit preference', () => {
  const flash = candidate(FLASH)
  const large = edit(candidate(TWENTY_SEVEN), unloaded)
  expectTarget(request([large, flash], { preferredVariantIds: [TWENTY_SEVEN.variantId] }), flash, 'only-eligible')
})

it('27B only: a nonresident Flash is not loaded to satisfy preference', () => {
  const flash = edit(candidate(FLASH), unloaded)
  const large = candidate(TWENTY_SEVEN)
  expectTarget(request([flash, large], { preferredVariantIds: [FLASH.variantId] }), large, 'only-eligible')
})

it('both resident in separate managed instances: larger comparable model wins before queue pressure', () => {
  const flash = candidate(FLASH)
  const large = candidate(TWENTY_SEVEN)
  large.queue = { pending: 6, active: 3, maxPending: 8 }
  expect(flash.host.identity.instance).not.toBe(large.host.identity.instance)
  for (const rows of permutations([flash, large])) expectTarget(request(rows), large, 'comparable-size')
})

it('neither resident and no inventory both fail closed', () => {
  expectRefusal(request([edit(candidate(FLASH), unloaded), edit(candidate(TWENTY_SEVEN), unloaded)]))
  expectRefusal(request([]))
})

it('pinned default is local; explicit two-node authorization can widen it', () => {
  const local = candidate(TWENTY_SEVEN)
  const remote = candidate(FLASH, PEER)
  const preferredVariantIds = [FLASH.variantId]
  expectTarget(request([remote, local], { preferredVariantIds }), local, 'only-eligible')
  const allowedNodeIds = [LOCAL.id, PEER.id]
  for (const rows of permutations([local, remote])) {
    expectTarget(request(rows, { allowedNodeIds, preferredVariantIds }), remote, 'configured-preference')
  }
  const unauthorized = edit(remote, (row) => { row.node.authorized = false })
  expectTarget(request([unauthorized, local], { allowedNodeIds, preferredVariantIds }), local, 'only-eligible')
})

it('explicit membership overrides rather than adds the pin; an empty list admits none', () => {
  const local = candidate(TWENTY_SEVEN)
  const remote = candidate(FLASH, PEER)
  const explicit = { allowedNodeIds: [PEER.id], preferredVariantIds: [TWENTY_SEVEN.variantId] }
  expectTarget(request([local, remote], explicit), remote, 'only-eligible')
  expectRefusal(request([local, remote], { allowedNodeIds: [] }))
  expectRefusal(request([local, remote], { allowedNodeIds: ['fixture-not-observed'] }))
  const unpinned = request([local, remote], { allowedNodeIds: [PEER.id] })
  delete unpinned.pinnedNodeId
  expectTarget(unpinned, remote, 'only-eligible')
  delete unpinned.allowedNodeIds
  expectRefusal(unpinned, 'resident_summary_scope_required')
})

it('both node observation and route verification use inclusive age and present-time bounds', () => {
  for (const at of [NOW_MS - MAX_AGE_MS, NOW_MS]) {
    const row = candidate()
    row.node.observedAtMs = row.route.verifiedAtMs = at
    expectTarget(request([row]), row, 'only-eligible')
  }
})

rejectRows([
  ['unregistered node', (row) => { row.node.registered = false }],
  ['unauthorized node', (row) => { row.node.authorized = false }],
  ['producer says stale despite a recent observation', (row) => { row.node.freshness = 'stale' }],
  ['producer says unavailable', (row) => { row.node.freshness = 'unavailable' }],
  ['node observation expired by one millisecond', (row) => { row.node.observedAtMs = NOW_MS - MAX_AGE_MS - 1 }],
  ['node observation is in the future', (row) => { row.node.observedAtMs = NOW_MS + 1 }],
  ['route verification expired by one millisecond', (row) => { row.route.verifiedAtMs = NOW_MS - MAX_AGE_MS - 1 }],
  ['route verification is in the future', (row) => { row.route.verifiedAtMs = NOW_MS + 1 }],
  ['not a managed host', (row) => { row.host.mode = 'service' }],
  ['host disabled', (row) => { row.host.enabled = false }],
  ['host unusable', (row) => { row.host.usable = false }],
  ['host admission closed', (row) => { row.host.admissionOpen = false }],
  ['host unloaded despite ready-looking model row', (row) => { row.host.state = 'unloaded' }],
  ['host loading', (row) => { row.host.state = 'loading' }],
  ['host draining', (row) => { row.host.state = 'draining' }],
  ['host stopping', (row) => { row.host.state = 'stopping' }],
  ['host requires recovery', (row) => { row.host.state = 'recovery_required' }],
  ['model not loaded', (row) => { row.model.loaded = false }],
  ['model not ready', (row) => { row.model.ready = false }],
  ['configuration not validated', (row) => { row.model.configurationValidated = false }],
  ['no advertised operation', (row) => { row.model.operations = [] }],
  ['Decision-only operation, even with runtime text', (row) => { row.model.operations = ['decision'] }],
  ['Decision operation is not ordinary Chat even when Chat is also advertised', (row) => { row.model.operations = ['chat', 'decision'] }],
  ['no text capability', (row) => { row.model.capabilities = [] }],
  ['vision without text', (row) => { row.model.capabilities = ['vision'] }],
  ['Decision capability instead of text', (row) => { row.model.capabilities = ['decision'] }],
  ['wrongly cased text capability', (row) => { row.model.capabilities = ['Text'] }],
  ['text declared only by configuration', (row) => { row.model.capabilitySource = 'configuration' }],
  ['text capability source unknown', (row) => { row.model.capabilitySource = 'unknown' }],
])

it('ordinary Chat with verified text may also provide vision', () => {
  const row = edit(candidate(), (value) => { value.model.capabilities = ['vision', 'text'] })
  expectTarget(request([row]), row, 'only-eligible')
})

// Stripped or inconsistent producer fields reject the row, not the well-formed request.
rejectRows([
  ['alias stripped', (row) => { strip(row.route, 'alias') }],
  ['generic alias lacks node suffix', (row) => { row.route.alias = row.model.logicalModel }],
  ['alias claims a different node name', (row) => { row.route.alias = `${row.model.logicalModel}@${PEER.name}` }],
  ['alias is not an exact ordinal match', (row) => { row.route.alias = row.route.alias.toUpperCase() }],
  ['route id stripped', (row) => { strip(row.route, 'id') }],
  ['gateway binding mismatch', (row) => { row.route.gatewayId = 'fixture-wrong-gateway' }],
  ['route node binding mismatch', (row) => { row.route.nodeId = PEER.id }],
  ['route boot generation mismatch', (row) => { row.route.bootGeneration = PEER.bootGeneration }],
  ['route instance mismatch', (row) => { row.route.instance = 'fixture-other-instance' }],
  ['route logical model mismatch', (row) => { row.route.logicalModel = 'fixture-other-model' }],
  ['route serving variant mismatch', (row) => { row.route.servingVariantId = TWENTY_SEVEN.variantId }],
  ['route not healthy', (row) => { row.route.healthy = false }],
  ['resident-only route contract stripped', (row) => { strip(row.route, 'contract') }],
  ['wrong resident-only route contract', (row) => { row.route.contract = 'kinf.resident-only.v2' }],
  ['route could retry', (row) => { row.route.noRetry = false }],
  ['route could fall back', (row) => { row.route.noFallback = false }],
  ['no-retry proof stripped', (row) => { strip(row.route, 'noRetry') }],
  ['no-fallback proof stripped', (row) => { strip(row.route, 'noFallback') }],
  ['entire resident precondition stripped', (row) => { strip(row.host, 'identity') }],
  ['camelCase artifact revision is not the snake_case wire field', (row) => {
    Reflect.set(row.host.identity, 'artifactRevision', row.host.identity.artifact_revision)
    strip(row.host.identity, 'artifact_revision')
  }],
  ['unsupported precondition schema', (row) => { Reflect.set(row.host.identity, 'schema', 'kinf.resident-only.v2') }],
  ['host and model variant ids disagree', (row) => { row.host.identity.serving_variant_id = TWENTY_SEVEN.variantId }],
  ['host and model variant revisions disagree', (row) => { row.model.servingVariantRevision = 'fixture-wrong-variant-revision' }],
  ['host and model artifact ids disagree', (row) => { row.model.artifactId = 'fixture-other-artifact' }],
  ['host and model config revisions disagree', (row) => { row.model.targetConfigRevision = 'fixture-wrong-config-revision' }],
])

rejectRows([
  ['measurement missing', (row) => { delete row.measurement }],
  ['measured a different projection', (row) => { row.measurement!.projectionId = 'fixture-other-projection' }],
  ['measured a different host generation', (row) => { row.measurement!.hostGeneration = 'fixture-old-host' }],
  ['measured a different resident generation', (row) => { row.measurement!.residentGeneration = 'fixture-old-resident' }],
  ['measured a different target configuration', (row) => { row.measurement!.targetConfigRevision = 'fixture-old-config' }],
  ['zero rendered prompt tokens', (row) => { row.measurement!.renderedPromptTokens = 0 }],
  ['negative rendered prompt tokens', (row) => { row.measurement!.renderedPromptTokens = -1 }],
  ['fractional rendered prompt tokens', (row) => { row.measurement!.renderedPromptTokens = 600.5 }],
  ['unsafe rendered prompt count', (row) => { row.measurement!.renderedPromptTokens = Number.MAX_SAFE_INTEGER + 1 }],
  ['no authoritative token budget', (row) => { row.model.budget = null }],
  ['non-finite context budget', (row) => { row.model.budget!.contextTokens = Infinity }],
  ['negative advertised input budget', (row) => { row.model.budget!.maxInputTokensAtMaxOutput = -1 }],
  ['max output exceeds context', (row) => { row.model.budget!.maxOutputTokens = 2049 }],
  ['advertised input plus max output exceeds context', (row) => { row.model.budget!.maxInputTokensAtMaxOutput = 1793 }],
  ['advertised input plus max output undershoots context', (row) => { row.model.budget!.maxInputTokensAtMaxOutput = 1791 }],
])

it('measured prompt + requested output + reserve can exactly fill context, not exceed it', () => {
  const row = edit(candidate(), (value) => { value.measurement!.renderedPromptTokens = 1760 })
  const atLimit = { outputTokens: 256, reserveTokens: 32 }
  expectTarget(request([row], atLimit), row, 'only-eligible')
  const oneOver = edit(row, (value) => { value.measurement!.renderedPromptTokens++ })
  expectRefusal(request([oneOver], atLimit))
  expectRefusal(request([row], { ...atLimit, reserveTokens: 33 }))
  expectTarget(request([oneOver], { ...atLimit, outputTokens: 255 }), oneOver, 'only-eligible')
})

it('output cap is independent of context fit; zero reserve is allowed', () => {
  const row = candidate()
  expectTarget(request([row], { outputTokens: 256, reserveTokens: 0 }), row, 'only-eligible')
  expectRefusal(request([row], { outputTokens: 257, reserveTokens: 0 }))
})

it('fit is per-target tokenization, not one rendered count reused for every resident', () => {
  const flash = edit(candidate(FLASH), (row) => { row.measurement!.renderedPromptTokens = 1761 })
  const large = edit(candidate(TWENTY_SEVEN), (row) => { row.measurement!.renderedPromptTokens = 1760 })
  const overrides = { outputTokens: 256, reserveTokens: 32, preferredVariantIds: [FLASH.variantId] }
  for (const rows of permutations([flash, large])) expectTarget(request(rows, overrides), large, 'only-eligible')
})

for (const [label, overrides] of [
  ['non-finite clock', { nowMs: NaN }],
  ['negative freshness interval', { maxAgeMs: -1 }],
  ['missing gateway identity', { gatewayId: '' }],
  ['missing projection identity', { projectionId: '' }],
  ['zero output', { outputTokens: 0 }],
  ['negative output', { outputTokens: -1 }],
  ['fractional output', { outputTokens: 1.5 }],
  ['non-finite output', { outputTokens: Infinity }],
  ['negative reserve', { reserveTokens: -1 }],
  ['fractional reserve', { reserveTokens: 1.5 }],
  ['unsafe reserve', { reserveTokens: Number.MAX_SAFE_INTEGER + 1 }],
] as const) {
  it(`invalid request: ${label}`, () => {
    expectRefusal(request([candidate()], overrides), 'resident_summary_invalid_input')
  })
}

it('busy accepts a free bounded pending slot without requiring active telemetry', () => {
  for (const queue of [{ pending: 2, maxPending: 3 }, { pending: 0, active: 0, maxPending: 1 }]) {
    const row = edit(candidate(), (value) => { value.host.state = 'busy'; value.queue = queue })
    expectTarget(request([row]), row, 'only-eligible')
  }
})

it('ready can admit without queue telemetry; known full refuses ready and busy', () => {
  const ready = edit(candidate(), (row) => { delete row.queue })
  expectTarget(request([ready]), ready, 'only-eligible')
  for (const state of ['ready', 'busy'] as const) {
    const full = edit(ready, (row) => { row.host.state = state; row.queue = { pending: 3, active: 1, maxPending: 3 } })
    expectRefusal(request([full]))
  }
})

for (const [label, queue] of [
  ['missing queue', undefined],
  ['missing pending', { maxPending: 3 }],
  ['missing bound', { pending: 0 }],
  ['fractional pending', { pending: 0.5, maxPending: 3 }],
  ['negative pending', { pending: -1, maxPending: 3 }],
  ['unbounded capacity', { pending: 0, maxPending: Infinity }],
  ['zero capacity', { pending: 0, maxPending: 0 }],
] as const) {
  it(`busy fails closed with ${label}`, () => {
    const row = edit(candidate(), (value) => {
      value.host.state = 'busy'
      if (queue === undefined) delete value.queue
      else value.queue = { ...queue }
    })
    expectRefusal(request([row]))
  })
}

it('the ordered configured preference wins over both size and queue advantages', () => {
  const flash = edit(candidate(FLASH), (row) => { row.queue = { pending: 4, active: 3, maxPending: 8 } })
  const large = candidate(TWENTY_SEVEN)
  for (const rows of permutations([flash, large])) {
    expectTarget(request(rows, { preferredVariantIds: ['fixture-unseen-variant', FLASH.variantId, TWENTY_SEVEN.variantId] }), flash, 'configured-preference')
    expectTarget(request(rows, { preferredVariantIds: ['fixture-unseen-variant'] }), large, 'comparable-size')
  }
})

for (const [label, makeIncomparable] of [
  ['different size groups', (row: FixtureCandidate) => { row.model.size!.group = 'fixture-other-parameter-group' }],
  ['unverified size', (row: FixtureCandidate) => { row.model.size!.source = 'unknown' }],
  ['missing size', (row: FixtureCandidate) => { delete row.model.size }],
] as const) {
  it(`${label}: no partial size ordering of comparable pairs among three tied candidates`, () => {
    const flash = candidate(FLASH)
    const large = edit(candidate(TWENTY_SEVEN), (row) => { row.queue!.pending = 4 })
    const other = edit(candidate(FLASH, PEER), (row) => { row.queue!.pending = 2; makeIncomparable(row) })
    for (const rows of permutations([flash, large, other])) {
      expectTarget(request(rows, { allowedNodeIds: [LOCAL.id, PEER.id] }), flash, 'queue-pressure')
    }
  })
}

for (const [label, localQueue, peerQueue] of [
  ['pending orders before active', { pending: 0, active: 9 }, { pending: 1, active: 0 }],
  ['active breaks equal pending', { pending: 1, active: 0 }, { pending: 1, active: 2 }],
  ['known pending outranks unknown pending', { pending: 1, active: 4 }, { active: 0 }],
  ['known active outranks unknown active at equal pending', { pending: 1, active: 0 }, { pending: 1 }],
] as const) {
  it(`queue pressure: ${label}`, () => {
    const local = edit(candidate(FLASH), (row) => { row.queue = { ...localQueue } })
    const remote = edit(candidate(FLASH, PEER), (row) => { row.queue = { ...peerQueue } })
    for (const rows of permutations([local, remote])) {
      expectTarget(request(rows, { allowedNodeIds: [PEER.id, LOCAL.id] }), local, 'queue-pressure')
    }
  })
}

it('complete ranking tie uses ordinal identity, is permutation-invariant, and leaves the snapshot intact', () => {
  const placements = [
    { id: 'Z-node', name: 'Z-host', bootGeneration: 'Z-boot' },
    { id: 'a-node', name: 'a-host', bootGeneration: 'a-boot' },
    { id: 'b-node', name: 'b-host', bootGeneration: 'b-boot' },
  ]
  const rows = placements.map(placement => candidate(FLASH, placement))
  for (const order of permutations(rows)) {
    const input = request(order, { allowedNodeIds: placements.map(({ id }) => id).reverse() })
    const before = structuredClone(input)
    expectTarget(input, rows[0]!, 'stable-identity')
    expect(input).toEqual(before)
  }
})

it('same exact alias and identity count once; minimum eligible route id is ordinal', () => {
  const base = candidate()
  const earlier = edit(base, (row) => { row.route.id = 'Z-verified-route' })
  const later = edit(base, (row) => { row.route.id = 'a-verified-route' })
  const unverified = edit(base, (row) => { row.route.id = '0-unverified-route'; row.route.healthy = false })
  for (const rows of permutations([base, earlier, later])) {
    expectTarget(request([unverified, ...rows, earlier]), earlier, 'only-eligible')
  }
})

for (const [label, conflict] of [
  ['artifact id', (row: FixtureCandidate) => { row.model.artifactId = row.host.identity.model_artifact = 'fixture-other-artifact' }],
  ['artifact revision', (row: FixtureCandidate) => { row.host.identity.artifact_revision = 'fixture-other-artifact-revision' }],
  ['serving variant revision', (row: FixtureCandidate) => {
    row.model.servingVariantRevision = row.host.identity.serving_variant_revision = 'fixture-other-variant-revision'
  }],
  ['target configuration revision', (row: FixtureCandidate) => {
    row.measurement!.targetConfigRevision = row.model.targetConfigRevision =
      row.host.identity.target_config_revision = 'fixture-other-config-revision'
  }],
  ['host generation', (row: FixtureCandidate) => {
    row.measurement!.hostGeneration = row.host.identity.host_generation = 'fixture-other-host-generation'
  }],
  ['resident generation', (row: FixtureCandidate) => {
    row.measurement!.residentGeneration = row.host.identity.resident_generation = 'fixture-other-resident-generation'
  }],
] as const) {
  it(`conflicting loaded claims on a node/instance fail closed: ${label}`, () => {
    const base = candidate()
    const contradiction = edit(base, (row) => { row.route.id += '-second-claim'; conflict(row) })
    // Both claims are individually complete. Route sorting must not arbitrarily bless one.
    expectTarget(request([base]), base, 'only-eligible')
    expectTarget(request([contradiction]), contradiction, 'only-eligible')
    for (const rows of permutations([base, contradiction])) expectRefusal(request(rows))
  })
}

it('a contradictory fresh loaded revision still poisons its instance when its queue is full', () => {
  const base = candidate()
  const full = edit(base, (row) => {
    row.route.id += '-full'
    row.host.identity.artifact_revision = 'fixture-contradictory-revision'
    row.queue = { pending: 2, maxPending: 2 }
  })
  expectTarget(request([base]), base, 'only-eligible')
  expectRefusal(request([full]))
  for (const rows of permutations([base, full])) expectRefusal(request(rows))
})

it("a second instance's fresh exact-alias binding conflicts even if its route is unhealthy", () => {
  const base = candidate()
  const duplicateAlias = edit(base, (row) => {
    row.route.id += '-second-instance'
    row.host.identity.instance = row.route.instance = 'fixture-second-flash-instance'
    row.measurement!.residentGeneration = row.host.identity.resident_generation = 'fixture-second-resident'
  })
  expectTarget(request([duplicateAlias]), duplicateAlias, 'only-eligible')
  duplicateAlias.route.healthy = false
  expectRefusal(request([duplicateAlias]))
  for (const rows of permutations([base, duplicateAlias])) expectRefusal(request(rows))
})

it('a duplicate node name poisons both node identities, not an independent third node', () => {
  const local = candidate(FLASH)
  const remote = candidate(FLASH, { ...PEER, name: LOCAL.name })
  const sameAmbiguousNode = candidate(TWENTY_SEVEN)
  const third = { id: 'fixture-third', name: 'fixture-gamma', bootGeneration: 'fixture-boot-gamma-1' }
  const unrelated = candidate(TWENTY_SEVEN, third)
  const overrides = { allowedNodeIds: [LOCAL.id, PEER.id, third.id], preferredVariantIds: [FLASH.variantId] }
  expect(local.route.alias).toBe(remote.route.alias)
  expectTarget(request([local], overrides), local, 'only-eligible')
  expectTarget(request([remote], overrides), remote, 'only-eligible')
  for (const rows of permutations([local, remote, sameAmbiguousNode])) expectRefusal(request(rows, overrides))
  for (const rows of permutations([local, remote, sameAmbiguousNode, unrelated])) {
    expectTarget(request(rows, overrides), unrelated, 'only-eligible')
  }
})

it('contradictory measurements for the same fresh residency do not pool proof', () => {
  const valid = candidate(FLASH)
  const invalid = edit(valid, (row) => {
    row.route.id += '-other-proof'
    row.measurement!.renderedPromptTokens = 4096
  })
  expectTarget(request([valid]), valid, 'only-eligible')
  expectRefusal(request([invalid]))
  for (const rows of permutations([valid, invalid])) expectRefusal(request(rows))
})

it('reserved automatic placement names never qualify as strict aliases', () => {
  for (const name of ['auto', 'Auto', 'AUTO']) {
    const row = candidate(FLASH, { ...LOCAL, name })
    expectRefusal(request([row]))
  }
})

for (const [field, value] of [
  ['healthy', false], ['noRetry', false], ['noFallback', false], ['contract', 'kinf.resident-only.v2'],
] as const) {
  it(`conflicting fresh ${field} proof for the same route ID cannot select that route`, () => {
    const base = candidate(FLASH)
    const contradicts = edit(base, row => Object.assign(row.route, { [field]: value }))
    for (const rows of permutations([base, contradicts])) expectRefusal(request(rows))
    const separateRoute = edit(base, (row) => { row.route.id += '-separate' })
    expectTarget(request([base, contradicts, separateRoute]), separateRoute, 'only-eligible')
  })
}

it('an unusable but still loaded claim contradicts an exact same-instance resident', () => {
  const base = candidate(FLASH)
  const unusable = edit(base, (row) => {
    row.host.usable = false
    row.host.identity.artifact_revision = 'different-loaded-artifact-revision'
  })
  for (const rows of permutations([base, unusable])) expectRefusal(request(rows))
})

it('one full resident tuple cannot advertise two conflicting canonical logical models', () => {
  const base = candidate(FLASH)
  const alias = edit(base, (row) => {
    row.model.logicalModel = row.route.logicalModel = 'conflicting-logical-model'
    row.route.id += '-other'
    row.route.alias = `${row.route.logicalModel}@${row.node.name}`
  })
  for (const rows of permutations([base, alias])) expectRefusal(request(rows))
})

for (const field of ['registered', 'authorized'] as const) {
  it(`conflicting fresh ${field} records deny only their node identity`, () => {
    const base = candidate(FLASH)
    const unauthorized = edit(base, (row) => { row.node[field] = false })
    const other = candidate(TWENTY_SEVEN, PEER)
    for (const rows of permutations([base, unauthorized])) expectRefusal(request(rows))
    expectTarget(request([base, unauthorized, other], { allowedNodeIds: [LOCAL.id, PEER.id] }), other, 'only-eligible')
  })
}

it('the returned native tuple is an immutable copy of exactly the v1 fields', () => {
  const row = candidate(FLASH)
  Object.assign(row.host.identity, { unrelated: 'do-not-forward' })
  const result = selectResidentSummary(request([row]))
  if (!result.ok) throw new Error('expected a selection')
  const { target } = result
  const former = target.precondition.resident_generation
  row.host.identity.resident_generation = 'next-residency'
  expect(target.precondition.resident_generation).toBe(former)
  expect(Object.isFrozen(target.precondition)).toBe(true)
  expect(Object.hasOwn(target.precondition, 'unrelated')).toBe(false)
  expect(Object.hasOwn(target.precondition, 'node_id')).toBe(false)
})
