import { describe, expect, it } from 'vitest'
import { assertServiceable, Config, resolveProfiles, type Options } from '../src/config.ts'

/** Validate one hand-declared route, with the caller's fields layered onto it. */
const routeWith = (profile: Record<string, unknown>): (() => unknown) =>
  () => ({ providers: Config({
    providers: {
      'acme-gateway': {
        api: 'openai-completions',
        baseURL: 'https://acme.test',
        models: [{ id: 'm' }],
        ...profile,
      },
    },
  }).providers.get() })

/** Validate that route with the caller's fields on its single model entry. */
const configWith = (model: Record<string, unknown>): (() => unknown) =>
  routeWith({ models: [{ id: 'm', ...model }] })

describe('model request routing', () => {
  it('leaves routing fields absent on existing stored model rows', () => {
    const materialized = routeWith({})() as Options
    const model = materialized.providers?.['acme-gateway']?.models?.[0]
    expect(model).not.toHaveProperty('baseURL')
    expect(model).not.toHaveProperty('headers')
    expect(model).not.toHaveProperty('requestModel')
  })

  it('accepts explicit model endpoints and binding headers without a provider endpoint', () => {
    const materialized = routeWith({
      baseURL: undefined,
      models: [{ id: 'm@local', requestModel: 'm', baseURL: 'http://127.0.0.1:5000/v1', headers: { 'X-Route': 'local' } }],
    })() as Options
    expect(() => { assertServiceable(materialized) }).not.toThrow()
  })

  it.each(['', '/relative', 'file:///model', 'http://user:password@localhost/v1'])(
    'rejects an invalid model endpoint %j before settings can be persisted', (baseURL) => {
      expect(() => { assertServiceable(configWith({ baseURL })() as Options) })
        .toThrow(/baseURL must be an absolute HTTP\(S\) URL without credentials/)
    },
  )

  it('rejects an empty wire model id', () => {
    expect(() => { assertServiceable(configWith({ requestModel: '' })() as Options) })
      .toThrow(/requestModel must be non-empty/)
  })

  it.each([
    ['bad header name', 'value'],
    ['x-route', 'line\nbreak'],
    ['x-route', '部署'],
  ])('rejects model header %j when Fetch cannot represent it', (name, value) => {
    expect(() => { assertServiceable(configWith({ headers: { [name]: value } })() as Options) })
      .toThrow(/header .* is not valid for Fetch/)
  })

  it('retains an invalid stored model diagnostic beside a usable sibling', () => {
    const profile = resolveProfiles({ gateway: {
      api: 'openai-completions',
      models: [
        { id: 'invalid', baseURL: 'file:///model' },
        { id: 'usable', baseURL: 'http://127.0.0.1:5000/v1' },
      ],
    } }, 'deferred').get('gateway')
    expect(profile?.modelErrors.get('invalid')).toMatch(/absolute HTTP/)
    expect(profile?.piProvider?.getModels().map(model => model.id)).toEqual(['usable'])
  })
})

describe('reasoning schema boundary', () => {
  it('accepts an empty provider section and propagates unexpected catalog failures', () => {
    expect(() => { assertServiceable({}) }).not.toThrow()
    const failure = new TypeError('model metadata lookup failed')
    expect(() => resolveProfiles({ openrouter: { models: [{
      id: '111',
      get name(): string { throw failure },
    }], api: 'openai-completions' } }, 'deferred')).toThrow(failure)
  })

  it('rejects a level pi-ai does not know at the write that produced it', () => {
    expect(configWith({ reasoningEfforts: { ultra: 'x' } })).toThrow(/"off"/)
    expect(configWith({ reasoningEfforts: { high: 42 } })).toThrow()
  })

  it('keeps false distinguishable from an absent declaration', () => {
    type Materialized = { providers: Record<string, { models?: { reasoningEfforts?: unknown }[] }> }
    const withFalse = configWith({ reasoningEfforts: false })() as Materialized
    expect(withFalse.providers['acme-gateway']?.models?.[0]?.reasoningEfforts).toBe(false)
    const absent = configWith({})() as Materialized
    expect(absent.providers['acme-gateway']?.models?.[0]?.reasoningEfforts).toBeUndefined()
  })

  it('rejects a thinking format outside the offered set', () => {
    expect(configWith({ compat: { thinkingFormat: 'quantum' } })).toThrow(/expected/)
  })

  it('accepts Baseten template arguments and completion controls', () => {
    expect(configWith({
      compat: {
        supportsFinishReason: false,
        thinkingFormat: 'baseten',
        chatTemplateArgs: { enable_thinking: { $var: 'thinking.enabled' } },
        supportsThinkingTokenBudget: true,
      },
    })).not.toThrow()
  })
})

describe('modality schema boundary', () => {
  it('rejects a modality pi-ai does not know, at either level', () => {
    expect(configWith({ input: ['audio'] })).toThrow(/expected/)
    expect(routeWith({ defaultInput: ['text', 'audio'] })).toThrow(/expected/)
  })

  it('refuses a route whose models could accept nothing', () => {
    // The pair the settings seam runs: the schema accepts the empty list as
    // well-typed, and the namespace validator is what refuses it. Asserting
    // only the schema would report this route as writable.
    expect(routeWith({ defaultInput: [] })).not.toThrow()
    expect(() => { assertServiceable(routeWith({ defaultInput: [] })() as Options) })
      .toThrow(/defaultInput must name at least one modality/)
  })

  type Materialized = {
    providers: Record<string, { defaultInput?: unknown; models?: { input?: unknown }[] }>
  }

  it('materializes an absent entry list as empty and an absent route list as text', () => {
    // The empty-list inheritance rule exists because of exactly this: an entry
    // that declares nothing reaches resolution as `[]`, not as `undefined`.
    const absent = configWith({})() as Materialized
    expect(absent.providers['acme-gateway']?.models?.[0]?.input).toEqual([])
    expect(absent.providers['acme-gateway']?.defaultInput).toEqual(['text'])
  })
})

describe('request image policy bounds', () => {
  it.each([
    ['requestImagePixelBudget', 0, /requestImagePixelBudget must be a positive safe integer/],
    ['requestImagePixelBudget', Number.MAX_SAFE_INTEGER + 1, /requestImagePixelBudget must be a positive safe integer/],
    ['requestImageMaxBytes', 0, /requestImageMaxBytes must be a positive safe integer/],
    ['requestImageMaxBytes', 1.5, /requestImageMaxBytes must be a positive safe integer/],
  ] as const)('rejects %s=%s at service resolution', (field, value, message) => {
    const programmatic = {
      providers: {
        'acme-gateway': {
          api: 'openai-completions',
          baseURL: 'https://acme.test',
          models: [{ id: 'm' }],
          [field]: value,
        },
      },
    } as Options
    expect(() => {
      assertServiceable(programmatic)
    }).toThrow(message)
  })
})
