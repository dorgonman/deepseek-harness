import { describe, expect, it } from 'vitest'
import Schema from '@deepseek-ai/schemastery'
import { MAX_TIMER_DELAY_MS } from '@deepseek-ai/dsh-timeout'
import { getBuiltinModels } from '@earendil-works/pi-ai/providers/all'
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

/** Materialize raw override data and detach the readonly Config snapshot for the resolver. */
function overrideWith(streamIdleTimeoutMs: unknown): Options {
  const [target] = getBuiltinModels('deepseek')
  if (target === undefined) throw new Error('the installed catalog ships no deepseek model')
  const rawInput: unknown = { providers: {
    deepseek: { modelOverrides: { [target.id]: { streamIdleTimeoutMs } } },
  } }
  const materialized = Schema.resolve(rawInput, Config, {})[0] as ReturnType<typeof Config>
  const normalized: Options = { providers: structuredClone(materialized.providers.get()) } as Options
  return normalized
}

describe('model request routing', () => {
  it('leaves routing fields absent on existing stored model rows', () => {
    const materialized = routeWith({})() as Options
    const model = materialized.providers?.['acme-gateway']?.models?.[0]
    expect(model).not.toHaveProperty('baseURL')
    expect(model).not.toHaveProperty('headers')
    expect(model).not.toHaveProperty('requestModel')
    expect(model).not.toHaveProperty('requestProtocol')
    expect(model).not.toHaveProperty('nativeTools')
    expect(model).not.toHaveProperty('streamIdleTimeoutMs')
    const resolved = resolveProfiles(materialized.providers ?? {}).get('acme-gateway')
    expect(resolved?.requestProtocols.size).toBe(0)
    expect(resolved?.nativeTools.size).toBe(0)
    expect(resolved?.modelStreamIdleTimeouts.size).toBe(0)
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

describe('model stream idle budget schema', () => {
  it.each([1, 25.5, MAX_TIMER_DELAY_MS])('retains an explicit model budget of %s milliseconds', (streamIdleTimeoutMs) => {
    const materialized = configWith({ streamIdleTimeoutMs })() as Options
    expect(materialized.providers?.['acme-gateway']?.models?.[0]?.streamIdleTimeoutMs).toBe(streamIdleTimeoutMs)
    const profile = resolveProfiles(materialized.providers ?? {}).get('acme-gateway')
    expect(profile?.modelStreamIdleTimeouts).toEqual(new Map([['m', streamIdleTimeoutMs]]))
    expect(profile?.streamIdleTimeoutMs).toBe(300_000)
  })

  it('materializes a valid raw catalog override before serviceability validation', () => {
    const options = overrideWith(25.5)
    expect(() => { assertServiceable(options) }).not.toThrow()
    const profile = resolveProfiles(options.providers).get('deepseek')
    expect([...(profile?.modelStreamIdleTimeouts.values() ?? [])]).toEqual([25.5])
  })

  it.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY, null, '25', MAX_TIMER_DELAY_MS + 1])(
    'rejects a malformed budget %j on model declarations and overrides before persistence', (streamIdleTimeoutMs) => {
      expect(() => { assertServiceable(configWith({ streamIdleTimeoutMs })() as Options) }).toThrow()

      expect(() => { assertServiceable(overrideWith(streamIdleTimeoutMs)) }).toThrow()
    },
  )
})

describe('native tool capability schema boundary', () => {
  it.each([false, true])('retains an explicit nativeTools=%s declaration', (nativeTools) => {
    const materialized = configWith({ nativeTools })() as Options
    expect(materialized.providers?.['acme-gateway']?.models?.[0]?.nativeTools).toBe(nativeTools)
    expect(resolveProfiles(materialized.providers ?? {}).get('acme-gateway')?.nativeTools.get('m')).toBe(nativeTools)
  })

  it.each(['false', 0, {}, []])('rejects malformed nativeTools=%j before persistence', (nativeTools) => {
    expect(configWith({ nativeTools })).toThrow()
  })

  it('rejects a valueless YAML capability instead of treating it as absent', () => {
    expect(() => { assertServiceable(configWith({ nativeTools: null })() as Options) })
      .toThrow(/nativeTools must be a boolean/)
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
