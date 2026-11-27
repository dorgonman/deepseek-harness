import { describe, expect, it, vi } from 'vitest'
import {
  createKinfResidentObservationSource,
  type KinfResidentObservationRoute,
} from '../src/kinf-resident-observation.ts'
import { selectResidentFromSource, type ResidentObservationRequest } from '../src/resident-observation.ts'

const gatewayId = 'kinf-local'
const route: KinfResidentObservationRoute = {
  id: 'local-27b',
  nodeId: 'local-node',
  nodeName: 'local',
  instance: 'default',
  logicalModel: 'chat-qwen-27b-uncensored-exl3-6bpw',
  baseUrl: 'http://127.0.0.1:49362/v1',
}
const projection = {
  id: 'summary-envelope',
  messages: [{ role: 'user', content: [{ type: 'text', text: 'Summarize this history.' }] }],
  outputTokens: 32,
} as const
const request: ResidentObservationRequest = { gatewayId, nodeIds: [route.nodeId], projection }
const seen = 1_795_662_507_115

function snapshot() {
  return {
    schema: 'kinf.host-observation.v1',
    complete: true,
    observed_at_unix_ms: seen,
    host: {
      node_id: null as string | null,
      boot_generation: null as string | null,
      route: null as { route_id: string } | null,
      instance: route.instance,
      logical_model: route.logicalModel,
      mode: 'mservice',
      state: 'ready',
      enabled: true,
      engine_usable: true,
      admission_open: true,
      resident_only_contract: 'kinf.resident-only.v1',
      resident_only_admission_open: true,
      host_generation: 'host-1',
      engine_generation: 'resident-1',
      resident_generation: 'resident-1',
      model_artifact: 'qwen27-artifact',
      artifact_revision: 'artifact-1',
      serving_variant_id: 'qwen27-uncensored-native-mtp',
      serving_variant_revision: 'chat-v1',
      target_config_revision: 'config-1',
      active_requests: 0,
      pending_requests: 0,
    },
    models: [{
      logical_model: route.logicalModel,
      instance: route.instance,
      ready: true,
      loaded: true,
      configuration_validated: true,
      capability_source: 'runtime',
      serving_variant_id: 'qwen27-uncensored-native-mtp',
      serving_variant_revision: 'chat-v1',
      model_artifact: 'qwen27-artifact',
      target_config_revision: 'config-1',
      operations: ['chat'],
      capabilities: ['text'],
      resident_only_budget: { context_tokens: 4096, max_output_tokens: 256, max_input_tokens_at_max_output: 3840 },
    }],
    queue: { active_requests: 0, pending_requests: 0, configured: { max_pending_requests: 4 } },
  }
}

function sourceFor(data: unknown) {
  const fetcher = vi.fn<typeof fetch>().mockImplementation(async () => Response.json(data))
  const measure = vi.fn(() => 128)
  const source = createKinfResidentObservationSource(gatewayId, [route], { fetch: fetcher, measure })
  return { source, fetcher, measure }
}

describe('KINF resident observations', () => {
  it('uses only a passive bound GET and joins the exact resident identity and capacity', async () => {
    const { source, fetcher, measure } = sourceFor(snapshot())
    const rows = await source.observe(request)
    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({
      node: { id: route.nodeId, name: route.nodeName, bootGeneration: route.nodeId, observedAtMs: seen },
      host: {
        identity: {
          schema: 'kinf.resident-only.v1', instance: 'default', host_generation: 'host-1',
          resident_generation: 'resident-1', model_artifact: 'qwen27-artifact', artifact_revision: 'artifact-1',
          serving_variant_id: 'qwen27-uncensored-native-mtp', serving_variant_revision: 'chat-v1',
          target_config_revision: 'config-1',
        },
      },
      route: { alias: `${route.logicalModel}@local`, id: route.id, gatewayId, verifiedAtMs: seen },
      model: { budget: { contextTokens: 4096, maxOutputTokens: 256, maxInputTokensAtMaxOutput: 3840 } },
      queue: { active: 0, pending: 0, maxPending: 4 },
      measurement: { projectionId: projection.id, hostGeneration: 'host-1', residentGeneration: 'resident-1', promptTokens: 128 },
    })
    expect(fetcher).toHaveBeenCalledTimes(1)
    const [url, init] = fetcher.mock.calls[0] ?? []
    expect(url).toBe('http://127.0.0.1:49362/v1/observations')
    expect(init).toMatchObject({ method: 'GET', redirect: 'error' })
    expect(init?.body).toBeUndefined()
    const headers = new Headers(init?.headers)
    expect(Object.fromEntries(headers)).toEqual({
      accept: 'application/json',
      'x-kinf-instance': 'default', 'x-kinf-mode': 'mservice', 'x-kinf-service': route.logicalModel,
    })
    expect(measure).toHaveBeenCalledExactlyOnceWith(projection.messages)
  })

  it('preserves a matching native node boot and refuses a mismatched node', async () => {
    const data = snapshot()
    data.host.node_id = route.nodeId
    data.host.boot_generation = 'native-boot'
    const { source } = sourceFor(data)
    expect(await source.observe(request)).toMatchObject([{ node: { bootGeneration: 'native-boot' } }])
    data.host.node_id = 'other-node'
    expect(await source.observe(request)).toEqual([])
  })

  it('keeps an admitted busy resident when the bounded queue has room', async () => {
    const data = snapshot()
    data.host.state = 'busy'
    data.host.active_requests = data.queue.active_requests = 1
    data.host.pending_requests = data.queue.pending_requests = 3
    const { source } = sourceFor(data)
    expect(await source.observe(request)).toMatchObject([{ host: { state: 'busy' }, queue: { active: 1, pending: 3 } }])
  })

  it('selects across independently bound managed instances on the same local node', async () => {
    const flash: KinfResidentObservationRoute = {
      ...route,
      id: 'local-flash',
      instance: 'daily',
      logicalModel: 'chat-qwen-flash-next',
      baseUrl: 'http://127.0.0.1:49363/v1',
    }
    const flashState = snapshot()
    Object.assign(flashState.host, {
      instance: flash.instance,
      logical_model: flash.logicalModel,
      host_generation: 'flash-host',
      engine_generation: 'flash-resident',
      resident_generation: 'flash-resident',
      serving_variant_id: 'flash-native-mtp',
    })
    Object.assign(flashState.models[0] ?? {}, {
      instance: flash.instance,
      logical_model: flash.logicalModel,
      serving_variant_id: 'flash-native-mtp',
    })
    const fetcher = vi.fn<typeof fetch>().mockImplementation(async url =>
      Response.json(typeof url === 'string' && url.includes(':49363/') ? flashState : snapshot()))
    const source = createKinfResidentObservationSource(gatewayId, [flash, route], {
      fetch: fetcher, measure: () => 128,
    })
    const selected = await selectResidentFromSource({
      gatewayId, pinnedNodeId: route.nodeId, projectionId: projection.id,
      preferredVariantIds: ['qwen27-uncensored-native-mtp', 'flash-native-mtp'],
      projectionMessages: projection.messages, outputTokens: 32, reserveTokens: 0, maxAgeMs: 5000,
    }, source, { now: () => seen })
    expect(selected).toMatchObject({
      ok: true, receipt: { alias: `${route.logicalModel}@local`, instance: route.instance, reason: 'configured-preference' },
    })
    expect(fetcher).toHaveBeenCalledTimes(2)
  })

  it('rejects one endpoint bound to two managed instances', () => {
    const other: KinfResidentObservationRoute = {
      ...route, id: 'other', instance: 'other', logicalModel: 'chat-qwen-flash-next',
    }
    const fetcher = vi.fn<typeof fetch>()
    expect(() => createKinfResidentObservationSource(gatewayId, [route, other], {
      fetch: fetcher, measure: () => 128,
    })).toThrow('compaction: invalid local KINF resident observation bindings')
    expect(fetcher).not.toHaveBeenCalled()
  })

  it('reads a new generation on the next observation after a refusal', async () => {
    const data = snapshot()
    const { source, fetcher } = sourceFor(data)
    expect(await source.observe(request)).toMatchObject([{ host: { identity: { resident_generation: 'resident-1' } } }])
    data.host.engine_generation = data.host.resident_generation = 'resident-2'
    expect(await source.observe(request)).toMatchObject([{ host: { identity: { resident_generation: 'resident-2' } } }])
    expect(fetcher).toHaveBeenCalledTimes(2)
  })

  it.each([
    ['an unloaded host', (data: ReturnType<typeof snapshot>) => { data.host.state = 'unloaded' }],
    ['closed resident admission', (data: ReturnType<typeof snapshot>) => { data.host.resident_only_admission_open = false }],
    ['unsupported contract', (data: ReturnType<typeof snapshot>) => { data.host.resident_only_contract = 'kinf.resident-only.v2' }],
    ['partial observation', (data: ReturnType<typeof snapshot>) => { data.complete = false }],
    ['stale config', (data: ReturnType<typeof snapshot>) => { data.host.target_config_revision = 'config-2' }],
    ['different configured model', (data: ReturnType<typeof snapshot>) => { data.host.logical_model = 'another-chat' }],
    ['unproven local registration', (data: ReturnType<typeof snapshot>) => { data.host.route = { route_id: 'unknown' } }],
    ['full queue', (data: ReturnType<typeof snapshot>) => { data.host.pending_requests = data.queue.pending_requests = 4 }],
    ['contradictory queue', (data: ReturnType<typeof snapshot>) => { data.host.pending_requests = 1 }],
    ['unknown budget', (data: ReturnType<typeof snapshot>) => { Object.assign(data.models[0] ?? {}, { resident_only_budget: null }) }],
    ['non-chat model', (data: ReturnType<typeof snapshot>) => { Object.assign(data.models[0] ?? {}, { operations: ['decision'] }) }],
    ['configured-only capability', (data: ReturnType<typeof snapshot>) => { Object.assign(data.models[0] ?? {}, { capability_source: 'configuration' }) }],
    ['duplicate matching model', (data: ReturnType<typeof snapshot>) => { data.models = [...data.models, ...data.models] }],
  ])('refuses %s before measuring prompt capacity', async (_label, edit) => {
    const data = snapshot()
    edit(data)
    const { source, measure } = sourceFor(data)
    expect(await source.observe(request)).toEqual([])
    expect(measure).not.toHaveBeenCalled()
  })

  it('returns no eligible model from the selector when the observation is stale', async () => {
    const { source } = sourceFor(snapshot())
    expect(await selectResidentFromSource(
      {
        gatewayId, pinnedNodeId: route.nodeId, projectionId: projection.id,
        projectionMessages: projection.messages, outputTokens: 32, reserveTokens: 0, maxAgeMs: 5000,
      },
      source, { now: () => seen + 5001 },
    )).toEqual({ ok: false, code: 'resident_summary_no_eligible' })
  })

  it('skips unrelated scope and missing projection before making requests', async () => {
    const { source, fetcher } = sourceFor(snapshot())
    expect(await source.observe({ ...request, gatewayId: 'other-gateway' })).toEqual([])
    expect(await source.observe({ ...request, nodeIds: ['other-node'] })).toEqual([])
    expect(await source.observe({ gatewayId, nodeIds: [route.nodeId] })).toEqual([])
    expect(fetcher).not.toHaveBeenCalled()
  })

  it.each([
    { ...route, baseUrl: 'https://remote.invalid/v1' },
    { ...route, baseUrl: 'http://127.0.0.1:49362/another-provider/v1' },
    { ...route, baseUrl: 'http://[' },
    { ...route, nodeName: 'auto' },
    { ...route, logicalModel: 'another/model' },
  ])('rejects a route outside direct local managed scope', (invalidRoute) => {
    const fetcher = vi.fn<typeof fetch>()
    expect(() => createKinfResidentObservationSource(gatewayId, [invalidRoute], {
      fetch: fetcher, measure: () => 128,
    })).toThrow('compaction: invalid local KINF resident observation bindings')
    expect(fetcher).not.toHaveBeenCalled()
  })

  it('rejects an invalid route group identity or an empty binding table', () => {
    const fetcher = vi.fn<typeof fetch>()
    const dependencies = { fetch: fetcher, measure: () => 128 }
    expect(() => createKinfResidentObservationSource('/bad-group', [route], dependencies))
      .toThrow('compaction: invalid local KINF resident observation bindings')
    expect(() => createKinfResidentObservationSource(gatewayId, [], dependencies))
      .toThrow('compaction: invalid local KINF resident observation bindings')
    expect(fetcher).not.toHaveBeenCalled()
  })

  it('rejects a duplicate alias before trying its routes', () => {
    const fetcher = vi.fn<typeof fetch>()
    expect(() => createKinfResidentObservationSource(gatewayId, [route, { ...route, id: 'other-route' }], {
      fetch: fetcher, measure: () => 128,
    })).toThrow('compaction: invalid local KINF resident observation bindings')
    expect(fetcher).not.toHaveBeenCalled()
  })

  it('rejects distinct endpoints for one managed instance', () => {
    const fetcher = vi.fn<typeof fetch>()
    const other = {
      ...route,
      id: 'alternate-binding',
      logicalModel: 'other-model',
      baseUrl: 'http://127.0.0.1:49363/v1',
    }
    expect(() => createKinfResidentObservationSource(gatewayId, [route, other], {
      fetch: fetcher, measure: () => 128,
    })).toThrow('compaction: invalid local KINF resident observation bindings')
    expect(fetcher).not.toHaveBeenCalled()
  })

  it.each([
    new Response(null, { status: 204 }),
    new Response(new Uint8Array([0x7b, 0x7d])),
    new Response('', { status: 401 }),
    new Response('', { status: 302, headers: { location: 'http://127.0.0.1:49363/v1/observations' } }),
    new Response('{not-json', { headers: { 'content-type': 'application/json' } }),
    Response.json({ ignored: 'x'.repeat(1024 * 1024) }),
  ])('refuses observation HTTP and body failures', async (response) => {
    const { source, fetcher, measure } = sourceFor(snapshot())
    fetcher.mockResolvedValueOnce(response)
    expect(await source.observe(request)).toEqual([])
    expect(measure).not.toHaveBeenCalled()
  })

  it.each([
    ['an invalid model collection', {}],
    ['too many model rows', Array.from({ length: 129 }, () => null)],
  ])('refuses %s before measuring the prompt', async (_scenario, models) => {
    const { source, measure } = sourceFor({ ...snapshot(), models })
    expect(await source.observe(request)).toEqual([])
    expect(measure).not.toHaveBeenCalled()
  })

  it('refuses a resident when the projected prompt has no usable token count', async () => {
    const { source, measure } = sourceFor(snapshot())
    measure.mockReturnValue(0)
    expect(await source.observe(request)).toEqual([])
    expect(measure).toHaveBeenCalledExactlyOnceWith(projection.messages)
  })

  it('withholds a receipt when the matching route has no budget', async () => {
    const { source } = sourceFor(snapshot())
    const rows = await source.observe(request)
    expect(rows).toHaveLength(1)
    const current = rows[0]!
    const stale = {
      ...current,
      node: { ...current.node, observedAtMs: seen - 501 },
      model: { ...current.model, budget: null },
    }
    await expect(selectResidentFromSource({
      gatewayId, pinnedNodeId: route.nodeId,
      projectionId: projection.id, outputTokens: projection.outputTokens, reserveTokens: 16, maxAgeMs: 500,
    }, {
      observe: async () => [stale, current],
    }, { now: () => seen })).resolves.toEqual({ ok: false, code: 'resident_summary_no_eligible' })
  })

  it('keeps caller cancellation distinct from an unavailable observation', async () => {
    const { source, fetcher } = sourceFor(snapshot())
    const controller = new AbortController()
    fetcher.mockImplementationOnce(async (_url, init) => {
      controller.abort(new Error('cancelled observation'))
      init?.signal?.throwIfAborted()
      return Response.json(snapshot())
    })
    await expect(source.observe({ ...request, signal: controller.signal })).rejects.toThrow('cancelled observation')
  })
})
