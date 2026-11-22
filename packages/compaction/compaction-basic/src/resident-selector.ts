/**
 * Pure resident summary selection over already verified observations.
 *
 * A trusted observation source normalizes and verifies native host, node
 * registry, gateway route and request measurement records before calling this
 * policy. The selector does not query, route, tokenize, cancel, retry or
 * mutate a session or model.
 *
 * @module @deepseek-ai/dsh-compaction-basic/resident-selector
 */
export const RESIDENT_ONLY_V1 = 'kinf.resident-only.v1' as const

export interface ResidentPrecondition {
  readonly schema: typeof RESIDENT_ONLY_V1
  readonly instance: string
  readonly host_generation: string
  readonly serving_variant_id: string
  readonly serving_variant_revision: string
  readonly model_artifact: string
  readonly artifact_revision: string
  readonly target_config_revision: string
  readonly resident_generation: string
}

const identityFields = [
  'instance', 'host_generation', 'serving_variant_id', 'serving_variant_revision',
  'model_artifact', 'artifact_revision', 'target_config_revision', 'resident_generation',
] as const satisfies readonly (keyof ResidentPrecondition)[]

type Budget = Readonly<{
  contextTokens: number
  maxOutputTokens: number
  maxInputTokensAtMaxOutput: number
}>
type Size = Readonly<{
  group: string
  units: 'parameters'
  value: number
  source: 'authoritative' | 'unknown'
}>

export interface Candidate {
  readonly node: Readonly<{
    id: string
    name: string
    bootGeneration: string
    registered: boolean
    authorized: boolean
    freshness: 'fresh' | 'stale' | 'unavailable'
    observedAtMs: number
  }>
  readonly host: Readonly<{
    identity: ResidentPrecondition
    mode: 'mservice' | 'service'
    state: 'ready' | 'busy' | 'unloaded' | 'loading' | 'draining' | 'stopping' | 'recovery_required'
    enabled: boolean
    usable: boolean
    admissionOpen: boolean
  }>
  readonly model: Readonly<{
    logicalModel: string
    servingVariantId: string
    servingVariantRevision: string
    artifactId: string
    targetConfigRevision: string
    loaded: boolean
    ready: boolean
    configurationValidated: boolean
    operations: readonly string[]
    capabilitySource: 'runtime' | 'configuration' | 'unknown'
    capabilities: readonly string[]
    budget: Budget | null
    size?: Size
  }>
  readonly route: Readonly<{
    id: string
    gatewayId: string
    nodeId: string
    bootGeneration: string
    instance: string
    logicalModel: string
    servingVariantId: string
    alias: string
    healthy: boolean
    contract: string
    noRetry: boolean
    noFallback: boolean
    verifiedAtMs: number
  }>
  readonly queue?: Readonly<{ pending?: number; active?: number; maxPending?: number }>
  readonly measurement?: Readonly<{
    projectionId: string
    hostGeneration: string
    residentGeneration: string
    targetConfigRevision: string
    renderedPromptTokens: number
  }>
}

export interface SelectionInput {
  readonly nowMs: number
  readonly maxAgeMs: number
  readonly gatewayId: string
  readonly pinnedNodeId?: string
  /** When supplied this is the whole compaction scope, even if empty. */
  readonly allowedNodeIds?: readonly string[]
  readonly preferredVariantIds?: readonly string[]
  /** Opaque identity of this already projected summary envelope, not raw history. */
  readonly projectionId: string
  readonly outputTokens: number
  readonly reserveTokens: number
  readonly candidates: readonly Candidate[]
}

export type SelectionResult = Readonly<{
  ok: true
  target: Readonly<{
    nodeId: string
    gatewayId: string
    routeId: string
    alias: string
    precondition: ResidentPrecondition
  }>
  reason: 'only-eligible' | 'configured-preference' | 'comparable-size' | 'queue-pressure' | 'stable-identity'
}> | Readonly<{
  ok: false
  code: 'resident_summary_invalid_input' | 'resident_summary_scope_required' | 'resident_summary_no_eligible'
}>

type SelectionReason = Extract<SelectionResult, { ok: true }>['reason']

function count(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}
function positive(value: unknown): value is number {
  return count(value) && value > 0
}
function token(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0 && value.length <= 128 &&
    !value.startsWith('/') && !value.includes('..') && !/[^A-Za-z0-9_./:+-]/u.test(value)
}
function nodeName(value: unknown): value is string {
  // Never interpret a reserved automatic-placement selector as strict.
  return token(value) && /^[A-Za-z0-9][A-Za-z0-9_-]*$/u.test(value) && value.toLowerCase() !== 'auto'
}
function listOfTokens(value: unknown, bound: number): boolean {
  return Array.isArray(value) && value.length <= bound && Array.from(value).every(token)
}
function fresh(timestamp: unknown, input: SelectionInput): boolean {
  return count(timestamp) && timestamp <= input.nowMs && input.nowMs - timestamp <= input.maxAgeMs
}
function checkedIdentity(value: ResidentPrecondition | undefined): value is ResidentPrecondition {
  return value?.schema === RESIDENT_ONLY_V1 && identityFields.every(field => token(value[field]))
}
function identityKey(value: ResidentPrecondition): string {
  return JSON.stringify(identityFields.map(field => value[field]))
}
function freezeIdentity(value: ResidentPrecondition): ResidentPrecondition {
  // Do not forward arbitrary extensions from normalized input. v1 accepts
  // exactly this object, and this copy cannot follow later observation edits.
  return Object.freeze({
    schema: RESIDENT_ONLY_V1, instance: value.instance,
    host_generation: value.host_generation,
    serving_variant_id: value.serving_variant_id,
    serving_variant_revision: value.serving_variant_revision,
    model_artifact: value.model_artifact, artifact_revision: value.artifact_revision,
    target_config_revision: value.target_config_revision, resident_generation: value.resident_generation,
  })
}
function freshNode(node: Candidate['node'] | undefined, input: SelectionInput): node is Candidate['node'] {
  return node?.freshness === 'fresh' && token(node.id) && nodeName(node.name) &&
    token(node.bootGeneration) && fresh(node.observedAtMs, input)
}
function currentNode(node: Candidate['node'] | undefined, input: SelectionInput): node is Candidate['node'] {
  return freshNode(node, input) && node.registered === true && node.authorized === true
}
function pinnedMapping(route: Candidate['route'] | undefined, node: Candidate['node'], input: SelectionInput): route is Candidate['route'] {
  return route?.gatewayId === input.gatewayId && route.nodeId === node.id &&
    route.bootGeneration === node.bootGeneration && token(route.id) && token(route.instance) &&
    token(route.servingVariantId) && token(route.logicalModel) &&
    route.alias === `${route.logicalModel}@${node.name}`
}
function residentForwarding(route: Candidate['route']): boolean {
  return route.healthy === true && route.contract === RESIDENT_ONLY_V1 && route.noRetry === true && route.noFallback === true
}
function modelMatchesIdentity(model: Candidate['model'] | undefined, id: ResidentPrecondition): boolean {
  return model?.servingVariantId === id.serving_variant_id && model.servingVariantRevision === id.serving_variant_revision &&
    model.artifactId === id.model_artifact && model.targetConfigRevision === id.target_config_revision
}
function usableModel(row: Candidate): boolean {
  const { host, model, route } = row
  const id = host.identity
  return host.mode === 'mservice' && (host.state === 'ready' || host.state === 'busy') &&
    host.enabled === true && host.usable === true && host.admissionOpen === true &&
    model?.ready === true && model.loaded === true && model.configurationValidated === true &&
    Array.isArray(model.operations) && model.operations.length === 1 && model.operations[0] === 'chat' &&
    model.capabilitySource === 'runtime' && Array.isArray(model.capabilities) && model.capabilities.includes('text') &&
    model.logicalModel === route.logicalModel && id.instance === route.instance &&
    id.serving_variant_id === route.servingVariantId && modelMatchesIdentity(model, id)
}
function fits(row: Candidate, input: SelectionInput): boolean {
  const { measurement, model: { budget }, host: { identity } } = row
  if (!budget || !positive(budget.contextTokens) || !positive(budget.maxOutputTokens) ||
      !positive(budget.maxInputTokensAtMaxOutput) ||
      budget.contextTokens - budget.maxOutputTokens !== budget.maxInputTokensAtMaxOutput ||
      input.outputTokens > budget.maxOutputTokens) return false
  if (measurement?.projectionId !== input.projectionId || !positive(measurement.renderedPromptTokens) ||
      measurement.hostGeneration !== identity.host_generation || measurement.residentGeneration !== identity.resident_generation ||
      measurement.targetConfigRevision !== identity.target_config_revision) return false
  // Subtraction avoids overflowing a JS safe integer with hostile counts.
  const inputAllowance = budget.contextTokens - input.outputTokens
  return input.reserveTokens <= inputAllowance && measurement.renderedPromptTokens <= inputAllowance - input.reserveTokens
}
function queueAllows(row: Candidate): boolean {
  const { pending, maxPending } = row.queue ?? {}
  if ((pending !== undefined && !count(pending)) || (maxPending !== undefined && !count(maxPending))) return false
  if (count(pending) && count(maxPending)) return pending < maxPending
  return row.host.state === 'ready' && maxPending !== 0
}
function noteClaim(claims: Map<string, string | null>, key: string, value: string): void {
  const prior = claims.get(key)
  if (prior === undefined) claims.set(key, value)
  else if (prior !== value) claims.set(key, null)
}
function instanceKey(nodeId: string, instance: string): string {
  return JSON.stringify([nodeId, instance])
}
function staticBindingKey(row: Candidate): string {
  const { node, route } = row
  return JSON.stringify([node.id, node.bootGeneration, route.instance, route.logicalModel, route.servingVariantId])
}
function residentKey(row: Candidate): string {
  return JSON.stringify([row.node.id, identityKey(row.host.identity)])
}
function selectionFacts(row: Candidate, input: SelectionInput): string {
  const size = row.model.size
  return JSON.stringify([
    usableModel(row), queueAllows(row), fits(row, input),
    pressure(row.queue?.pending), pressure(row.queue?.active),
    knownSize(size) ? [size.group, size.value] : null,
  ])
}
function ordinal(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0
}
function preferredIndex(input: SelectionInput, row: Candidate): number {
  const index = input.preferredVariantIds?.indexOf(row.model.servingVariantId) ?? -1
  return index < 0 ? Number.POSITIVE_INFINITY : index
}
function knownSize(value: Size | undefined): value is Size {
  return value?.source === 'authoritative' && value.units === 'parameters' && token(value.group) && positive(value.value)
}
function minTier(rows: readonly Candidate[], value: (row: Candidate) => number): Candidate[] {
  const best = Math.min(...rows.map(value))
  return rows.filter(row => value(row) === best)
}
function pressure(value: unknown): number {
  return count(value) ? value : Number.POSITIVE_INFINITY
}
function stableKey(row: Candidate): string {
  return JSON.stringify([row.node.id, row.route.instance, row.model.servingVariantId, row.route.alias, row.route.id])
}
function first<Value>(values: readonly Value[]): Value {
  const [value] = values
  if (value === undefined) throw new Error('resident selector: empty tier')
  return value
}

function selected(row: Candidate, reason: SelectionReason): SelectionResult {
  return Object.freeze({
    ok: true, reason,
    target: Object.freeze({
      nodeId: row.node.id, gatewayId: row.route.gatewayId, routeId: row.route.id,
      alias: row.route.alias, precondition: freezeIdentity(row.host.identity),
    }),
  })
}

export function selectResidentSummary(input: SelectionInput): SelectionResult {
  if (!input || !count(input.nowMs) || !positive(input.maxAgeMs) || input.maxAgeMs > 60_000 ||
      !token(input.gatewayId) || !token(input.projectionId) || !positive(input.outputTokens) || !count(input.reserveTokens) ||
      !Array.isArray(input.candidates) || input.candidates.length > 256 ||
      (input.pinnedNodeId !== undefined && !token(input.pinnedNodeId)) ||
      (input.allowedNodeIds !== undefined && !listOfTokens(input.allowedNodeIds, 32)) ||
      (input.preferredVariantIds !== undefined && !listOfTokens(input.preferredVariantIds, 256)))
    return { ok: false, code: 'resident_summary_invalid_input' }
  const ids = input.allowedNodeIds ?? (input.pinnedNodeId === undefined ? undefined : [input.pinnedNodeId])
  if (!ids) return { ok: false, code: 'resident_summary_scope_required' }
  const allowed = new Set(ids)
  const nodeById = new Map<string, string | null>()
  const nodeByName = new Map<string, string | null>()
  const authorityByNode = new Map<string, string | null>()
  const residentByInstance = new Map<string, string | null>()
  const modelByResident = new Map<string, string | null>()
  const factsByResident = new Map<string, string | null>()
  const bindingByAlias = new Map<string, string | null>()
  const bindingByRoute = new Map<string, string | null>()
  const forwardingByRoute = new Map<string, string | null>()

  // Contradictory fresh evidence must not become safe just because one row
  // loses the budget/health filter. Do not combine capability, residency, or
  // transport proof across rows. Out-of-scope routes can still make a shared
  // gateway alias ambiguous; explicitly unauthorized rows have no authority.
  for (const row of input.candidates) {
    if (!freshNode(row?.node, input)) continue
    const { node, host, route } = row
    const authorized = currentNode(node, input)
    noteClaim(authorityByNode, node.id, authorized ? 'authorized' : 'unverified')
    if (!authorized) continue
    noteClaim(nodeById, node.id, JSON.stringify([node.name, node.bootGeneration]))
    noteClaim(nodeByName, node.name, node.id)
    if (host?.mode === 'mservice' && (host.usable === true || row.model?.loaded === true) && checkedIdentity(host.identity))
      noteClaim(residentByInstance, instanceKey(node.id, host.identity.instance), identityKey(host.identity))
    if (pinnedMapping(route, node, input) && fresh(route.verifiedAtMs, input)) {
      const binding = staticBindingKey(row)
      noteClaim(bindingByAlias, route.alias, binding)
      noteClaim(bindingByRoute, route.id, JSON.stringify([route.alias, binding]))
      noteClaim(forwardingByRoute, route.id, residentForwarding(route) ? 'verified' : 'unverified')
      if (host?.mode === 'mservice' && checkedIdentity(host.identity) && host.identity.instance === route.instance &&
          host.identity.serving_variant_id === route.servingVariantId && modelMatchesIdentity(row.model, host.identity)) {
        const key = residentKey(row)
        noteClaim(modelByResident, key, route.logicalModel)
        noteClaim(factsByResident, key, selectionFacts(row, input))
      }
    }
  }

  const deduped = new Map<string, Candidate>()
  for (const row of input.candidates) {
    if (!currentNode(row?.node, input) || !allowed.has(row.node.id) || !checkedIdentity(row.host?.identity) ||
        !pinnedMapping(row.route, row.node, input)) continue
    const { node, host, route } = row
    const binding = staticBindingKey(row)
    const key = residentKey(row)
    if (authorityByNode.get(node.id) !== 'authorized' || nodeById.get(node.id) === null || nodeByName.get(node.name) === null ||
        residentByInstance.get(instanceKey(node.id, host.identity.instance)) !== identityKey(host.identity) ||
        modelByResident.get(key) !== route.logicalModel || factsByResident.get(key) === null ||
        bindingByAlias.get(route.alias) !== binding ||
        bindingByRoute.get(route.id) !== JSON.stringify([route.alias, binding]) || forwardingByRoute.get(route.id) !== 'verified' ||
        !residentForwarding(route) ||
        !fresh(route.verifiedAtMs, input) || !usableModel(row) || !queueAllows(row) || !fits(row, input)) continue
    const prior = deduped.get(key)
    if (!prior || ordinal(stableKey(row), stableKey(prior)) < 0) deduped.set(key, row)
  }

  const eligible = [...deduped.values()]
  if (eligible.length === 0) return { ok: false, code: 'resident_summary_no_eligible' }
  if (eligible.length === 1) return selected(first(eligible), 'only-eligible')
  let tier = minTier(eligible, row => preferredIndex(input, row))
  if (tier.length === 1) return selected(first(tier), 'configured-preference')
  const size = tier[0]?.model.size
  // Pairwise comparisons across mixed/unknown groups can produce cycles. Only
  // use size when the entire tied tier is known and directly comparable.
  if (knownSize(size) && tier.every(row => knownSize(row.model.size) && row.model.size.group === size.group)) {
    tier = minTier(tier, (row) => {
      const size = row.model.size
      return knownSize(size) ? -size.value : 0
    })
    if (tier.length === 1) return selected(first(tier), 'comparable-size')
  }
  tier = minTier(tier, row => pressure(row.queue?.pending))
  tier = minTier(tier, row => pressure(row.queue?.active))
  if (tier.length === 1) return selected(first(tier), 'queue-pressure')
  tier.sort((a, b) => ordinal(stableKey(a), stableKey(b)))
  return selected(first(tier), 'stable-identity')
}
