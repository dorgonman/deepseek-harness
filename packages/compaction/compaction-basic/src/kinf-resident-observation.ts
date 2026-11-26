/**
 * Read passive KINF observations from explicitly bound local managed listeners.
 *
 * @module @deepseek-ai/dsh-compaction-basic/kinf-resident-observation
 */

import type { RequestMessage } from '@deepseek-ai/dsh-llm'
import type { ResidentObservationRequest, ResidentObservationSource } from './resident-observation.ts'
import { RESIDENT_ONLY_V1, type Candidate, type ResidentPrecondition } from './resident-selector.ts'

const MAX_ROUTES = 32
const MAX_MODEL_ROWS = 128
const MAX_OBSERVATION_BYTES = 1024 * 1024
const OBSERVATION_TIMEOUT_MS = 5_000
const TOKEN = /^(?!\/)(?!.*\.\.)[A-Za-z0-9_.:+/-]{1,128}$/u
const NODE_NAME = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/u
const NATIVE_BOUND_NAME = /^[A-Za-z0-9_-]{1,100}$/u

/** Explicit local managed listener and its DSH model route. */
export interface KinfResidentObservationRoute {
  /** Unique route identifier recorded in the resident selection receipt. */
  readonly id: string
  /** Configured node identity used by the preset's pinned node or allowlist. */
  readonly nodeId: string
  /** Node name appended to the logical model to form the DSH alias. */
  readonly nodeName: string
  /** Exact KINF managed instance served by this loopback endpoint. */
  readonly instance: string
  /** KINF logical model sent as the native service binding and chat model. */
  readonly logicalModel: string
  /** Loopback provider URL shared by passive observation and resident dispatch. */
  readonly baseUrl: string
}

/** Prompt accounting dependencies supplied by the mount. */
export interface KinfResidentObservationDependencies {
  readonly measure: (messages: readonly RequestMessage[]) => number
  readonly fetch?: typeof fetch
}

type JsonRecord = Readonly<Record<string, unknown>>
type LocalRoute = Readonly<{ binding: KinfResidentObservationRoute; endpoint: string; alias: string }>

function record(value: unknown): JsonRecord | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as JsonRecord : undefined
}

function array(value: unknown): value is readonly unknown[] {
  return Array.isArray(value)
}

function token(value: unknown): value is string {
  return typeof value === 'string' && TOKEN.test(value)
}

function count(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}

function positive(value: unknown): value is number {
  return count(value) && value > 0
}

function localEndpoint(raw: string): string | undefined {
  try {
    const url = new URL(raw)
    if ((url.protocol !== 'http:' && url.protocol !== 'https:')
      || (url.hostname !== '127.0.0.1' && url.hostname !== '[::1]')
      || url.username !== '' || url.password !== '' || url.search !== '' || url.hash !== ''
      || !['/', '/v1', '/v1/'].includes(url.pathname)) return undefined
    return new URL('/v1/observations', url.origin).href
  } catch {
    return undefined
  }
}

function localRoutes(routes: readonly KinfResidentObservationRoute[]): LocalRoute[] | undefined {
  if (!array(routes) || routes.length === 0 || routes.length > MAX_ROUTES) return undefined
  const ids = new Set<string>()
  const aliases = new Set<string>()
  const names = new Map<string, string>()
  const nodes = new Map<string, string>()
  const endpoints = new Map<string, string>()
  const instances = new Map<string, string>()
  const result: LocalRoute[] = []
  for (const route of routes) {
    if (record(route) === undefined || !token(route.id) || !token(route.nodeId)
      || !token(route.instance) || !NATIVE_BOUND_NAME.test(route.instance)
      || !token(route.logicalModel) || !NATIVE_BOUND_NAME.test(route.logicalModel)
      || !token(route.nodeName) || !NODE_NAME.test(route.nodeName) || route.nodeName.toLowerCase() === 'auto') return undefined
    const endpoint = localEndpoint(route.baseUrl)
    if (endpoint === undefined) return undefined
    const alias = `${route.logicalModel}@${route.nodeName}`
    const name = route.nodeName.toLowerCase()
    const instance = JSON.stringify([route.nodeId, route.instance])
    if (ids.has(route.id) || aliases.has(alias)
      || (names.has(name) && names.get(name) !== route.nodeId)
      || (nodes.has(route.nodeId) && nodes.get(route.nodeId) !== route.nodeName)
      || (endpoints.has(endpoint) && endpoints.get(endpoint) !== instance)
      || (instances.has(instance) && instances.get(instance) !== endpoint)) return undefined
    ids.add(route.id)
    aliases.add(alias)
    names.set(name, route.nodeId)
    nodes.set(route.nodeId, route.nodeName)
    endpoints.set(endpoint, instance)
    instances.set(instance, endpoint)
    result.push({ binding: Object.freeze({ ...route }), endpoint, alias })
  }
  return result
}

async function observationBody(response: Response): Promise<unknown> {
  const body = response.body
  if (body === null) return undefined
  if (!response.ok || !/^application\/json(?:\s*;|$)/iu.test(response.headers.get('content-type') ?? '')) {
    await body.cancel()
    return undefined
  }
  const decoder = new TextDecoder('utf-8', { fatal: true })
  const reader = body.getReader()
  const chunks: string[] = []
  let bytes = 0
  try {
    for (;;) {
      const next = await reader.read()
      if (next.done) break
      bytes += next.value.byteLength
      if (bytes > MAX_OBSERVATION_BYTES) return undefined
      chunks.push(decoder.decode(next.value, { stream: true }))
    }
    chunks.push(decoder.decode())
    const observation: unknown = JSON.parse(chunks.join(''))
    return observation
  } finally {
    await reader.cancel()
    reader.releaseLock()
  }
}

function residentIdentity(host: JsonRecord, route: KinfResidentObservationRoute): ResidentPrecondition | undefined {
  const {
    host_generation, serving_variant_id, serving_variant_revision, model_artifact, artifact_revision,
    target_config_revision, resident_generation,
  } = host
  if (host['instance'] !== route.instance || host['logical_model'] !== route.logicalModel
    || !token(host_generation) || !token(serving_variant_id) || !token(serving_variant_revision)
    || !token(model_artifact) || !token(artifact_revision) || !token(target_config_revision) || !token(resident_generation)
    || host['engine_generation'] !== resident_generation) return undefined
  return {
    schema: RESIDENT_ONLY_V1,
    instance: route.instance,
    host_generation,
    serving_variant_id,
    serving_variant_revision,
    model_artifact,
    artifact_revision,
    target_config_revision,
    resident_generation,
  }
}

function bootOf(host: JsonRecord, route: KinfResidentObservationRoute): string | undefined {
  const nodeId = host['node_id']
  const boot = host['boot_generation']
  // Independent loopback listeners can share one explicit local node binding.
  // The node marker correlates that binding; each dispatch still fences its
  // own native host and resident generations.
  if (nodeId === null && boot === null && host['route'] === null) return route.nodeId
  return nodeId === route.nodeId && token(boot) ? boot : undefined
}

function chatBudget(value: unknown): Candidate['model']['budget'] {
  const raw = record(value)
  const contextTokens = raw?.['context_tokens']
  const maxOutputTokens = raw?.['max_output_tokens']
  const maxInputTokensAtMaxOutput = raw?.['max_input_tokens_at_max_output']
  if (!positive(contextTokens) || !positive(maxOutputTokens) || !positive(maxInputTokensAtMaxOutput)
    || contextTokens - maxOutputTokens !== maxInputTokensAtMaxOutput) return null
  return { contextTokens, maxOutputTokens, maxInputTokensAtMaxOutput }
}

function matchingModel(
  value: unknown,
  route: KinfResidentObservationRoute,
  identity: ResidentPrecondition,
): Candidate['model'] | undefined {
  if (!array(value) || value.length > MAX_MODEL_ROWS) return undefined
  const matches = value.map(record).filter(row => row?.['logical_model'] === route.logicalModel)
  const [row] = matches
  if (matches.length !== 1 || row === undefined || row['instance'] !== identity.instance
    || row['loaded'] !== true || row['ready'] !== true || row['configuration_validated'] !== true
    || row['serving_variant_id'] !== identity.serving_variant_id || row['serving_variant_revision'] !== identity.serving_variant_revision
    || row['model_artifact'] !== identity.model_artifact || row['target_config_revision'] !== identity.target_config_revision
    || row['capability_source'] !== 'runtime') return undefined
  const operations = row['operations']
  const capabilities = row['capabilities']
  const budget = chatBudget(row['resident_only_budget'])
  if (!array(operations) || operations.length !== 1 || operations[0] !== 'chat' || !array(capabilities)
    || capabilities.length > 6 || !capabilities.every(token) || !capabilities.includes('text') || budget === null) return undefined
  return {
    logicalModel: route.logicalModel,
    servingVariantId: identity.serving_variant_id,
    servingVariantRevision: identity.serving_variant_revision,
    artifactId: identity.model_artifact,
    targetConfigRevision: identity.target_config_revision,
    loaded: true, ready: true, configurationValidated: true,
    operations: ['chat'], capabilitySource: 'runtime', capabilities, budget,
  }
}

function normalize(
  value: unknown,
  local: LocalRoute,
  request: ResidentObservationRequest,
  measure: KinfResidentObservationDependencies['measure'],
): Candidate | undefined {
  const { binding: route, alias } = local
  const snapshot = record(value)
  const host = record(snapshot?.['host'])
  const seen = snapshot?.['observed_at_unix_ms']
  const state = host?.['state']
  const projection = request.projection
  if (snapshot?.['schema'] !== 'kinf.host-observation.v1' || snapshot['complete'] !== true || !positive(seen)
    || host === undefined || (state !== 'ready' && state !== 'busy') || host['mode'] !== 'mservice'
    || host['enabled'] !== true || host['engine_usable'] !== true || host['admission_open'] !== true
    || host['resident_only_admission_open'] !== true || host['resident_only_contract'] !== RESIDENT_ONLY_V1
    || projection === undefined) return undefined
  const identity = residentIdentity(host, route)
  if (identity === undefined) return undefined
  const bootGeneration = bootOf(host, route)
  const model = matchingModel(snapshot['models'], route, identity)
  if (bootGeneration === undefined || model === undefined) return undefined

  const queue = record(snapshot['queue'])
  const pending = queue?.['pending_requests']
  const active = queue?.['active_requests']
  const maxPending = record(queue?.['configured'])?.['max_pending_requests']
  if (!count(pending) || !count(active) || !positive(maxPending) || pending >= maxPending
    || pending !== host['pending_requests'] || active !== host['active_requests']) return undefined
  const promptTokens = measure(projection.messages)
  if (!positive(promptTokens)) return undefined

  return {
    node: { id: route.nodeId, name: route.nodeName, bootGeneration, registered: true, authorized: true, freshness: 'fresh', observedAtMs: seen },
    host: { identity, mode: 'mservice', state, enabled: true, usable: true, admissionOpen: true },
    model,
    route: {
      id: route.id, gatewayId: request.gatewayId, nodeId: route.nodeId, bootGeneration, instance: route.instance,
      logicalModel: route.logicalModel, servingVariantId: identity.serving_variant_id, alias,
      healthy: true, contract: RESIDENT_ONLY_V1, noRetry: true, noFallback: true, verifiedAtMs: seen,
    },
    queue: { pending, active, maxPending },
    measurement: {
      projectionId: projection.id, hostGeneration: identity.host_generation, residentGeneration: identity.resident_generation,
      targetConfigRevision: identity.target_config_revision, promptTokens,
    },
  }
}

async function observeLocal(
  local: LocalRoute,
  request: ResidentObservationRequest,
  dependencies: KinfResidentObservationDependencies,
): Promise<Candidate | undefined> {
  const route = local.binding
  const timeout = AbortSignal.timeout(OBSERVATION_TIMEOUT_MS)
  const signal = request.signal === undefined ? timeout : AbortSignal.any([request.signal, timeout])
  try {
    signal.throwIfAborted()
    const headers = {
      accept: 'application/json',
      'x-kinf-mode': 'mservice',
      'x-kinf-instance': route.instance,
      'x-kinf-service': route.logicalModel,
    }
    const response = await (dependencies.fetch ?? fetch)(local.endpoint, { method: 'GET', headers, redirect: 'error', signal })
    const value = await observationBody(response)
    signal.throwIfAborted()
    return normalize(value, local, request, dependencies.measure)
  } catch {
    request.signal?.throwIfAborted()
    return undefined
  }
}

/**
 * Create a passive source for explicit local provider bindings without forwarding or fallback.
 * @param gatewayId - DSH's route group identity shared by the configured listeners.
 * @param routes - exact local listener and model bindings eligible for observation.
 * @param dependencies - prompt accounting and optional controlled HTTP.
 * @returns a source that observes only the configured local bindings in scope.
 * @throws Error for invalid or ambiguous route configuration.
 */
export function createKinfResidentObservationSource(
  gatewayId: string,
  routes: readonly KinfResidentObservationRoute[],
  dependencies: KinfResidentObservationDependencies,
): ResidentObservationSource {
  const local = token(gatewayId) ? localRoutes(routes) : undefined
  if (local === undefined) throw new Error('compaction: invalid local KINF resident observation bindings')
  return {
    async observe(request) {
      request.signal?.throwIfAborted()
      if (request.gatewayId !== gatewayId || request.projection === undefined || request.nodeIds.length > MAX_ROUTES) return []
      const scope = new Set(request.nodeIds)
      const rows = await Promise.all(local.filter(route => scope.has(route.binding.nodeId))
        .map(route => observeLocal(route, request, dependencies)))
      request.signal?.throwIfAborted()
      return rows.filter(row => row !== undefined)
    },
  }
}
