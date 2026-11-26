import { runInNewContext } from 'node:vm'
import { Context } from '@deepseek-ai/cordis'
import { describe, expect, it, onTestFinished, vi } from 'vitest'
import * as configuration from '../src/config.ts'
import {
  CompactionPolicyControl, inspectPresetCompactionPolicy, restorePresetCompactionPolicy, updatePresetCompactionPolicy,
} from '../src/policy-control.ts'

function declaration(): Record<string, unknown> {
  return {
    id: 'standard', order: 1,
    plugins: [
      { id: 'persona', name: 'persona', config: { text: 'keep this persona' } },
      { id: 'compaction', name: 'cordis:group', group: true, isolate: { compaction: true }, config: [
        { id: 'compaction-basic', name: '@deepseek-ai/dsh-compaction-basic', config: {
          thresholdRatio: 0.8, retainRatio: 0.16,
          summarizationProvider: 'previous', summarizationModel: 'previous-summary', maxTokens: 4096,
          modelPolicies: [
            { provider: 'conversation', model: 'owned', thresholdRatio: 0.7,
              summarizationProvider: 'previous', summarizationModel: 'previous-summary', maxTokens: 2048 },
            { provider: 'conversation', model: 'user-choice', headroomTokens: 2048,
              summarizationProvider: 'custom', summarizationModel: 'keep-custom', maxTokens: 512 },
          ],
        } },
        { id: 'command-compact', name: 'keep-command', config: { display: true } },
      ] },
    ],
  }
}

const target = { provider: 'local', model: 'new-summary', maxTokens: 1024 }

const minimal = {
  id: 'minimal', plugins: [{
    id: 'compaction', name: 'cordis:group', group: true, config: [
      { id: 'compaction-basic', name: '@deepseek-ai/dsh-compaction-basic' },
    ],
  }],
}

function basicConfig(current: Record<string, unknown>): Record<string, unknown> {
  const group = (current.plugins as Record<string, unknown>[])[1]!
  return (group.config as Record<string, unknown>[])[0]!.config as Record<string, unknown>
}

describe('preset summary-policy updates', () => {
  it('changes only default summary fields and explicitly owned route overrides', () => {
    const current = declaration()
    const before = structuredClone(current)
    const snapshot = inspectPresetCompactionPolicy('preset-standard', current)
    const result = updatePresetCompactionPolicy(current, {
      entryId: 'preset-standard', expectedRevision: snapshot.revision, target,
      modelPolicyRoutes: [{ provider: 'conversation', model: 'owned' }],
    })
    expect(result.policy).toEqual({ summarizationProvider: 'local', summarizationModel: 'new-summary', summarizationFallback: 'error', maxTokens: 1024 })
    expect(result.modelPolicies[0]?.policy).toEqual(result.policy)
    expect(result.modelPolicies[1]).toEqual(snapshot.modelPolicies[1])
    expect(current.order).toBe(1)
    const plugins = current.plugins as Record<string, unknown>[]
    const previous = before.plugins as Record<string, unknown>[]
    expect(plugins.map(item => item.id)).toEqual(previous.map(item => item.id))
    expect(plugins[0]).toEqual(previous[0])
    expect(plugins[1]?.isolate).toEqual(previous[1]?.isolate)
    const children = plugins[1]?.config as Record<string, unknown>[]
    const oldChildren = previous[1]?.config as Record<string, unknown>[]
    expect(children[1]).toEqual(oldChildren[1])
    expect(children[0]?.config).toMatchObject({ thresholdRatio: 0.8, retainRatio: 0.16,
      modelPolicies: [{ thresholdRatio: 0.7 }, { headroomTokens: 2048 }] })
    expect(result.revision).not.toBe(snapshot.revision)
  })

  it('preserves absent inherited values and makes same-value retries explicit', () => {
    const current = structuredClone(minimal)
    const snapshot = inspectPresetCompactionPolicy('preset-minimal', current)
    expect(snapshot.policy).toEqual({})
    const first = updatePresetCompactionPolicy(current, {
      entryId: 'preset-minimal', expectedRevision: snapshot.revision, target, modelPolicyRoutes: [],
    })
    expect(() => updatePresetCompactionPolicy(current, {
      entryId: 'preset-minimal', expectedRevision: snapshot.revision, target, modelPolicyRoutes: [],
    })).toThrow(expect.objectContaining({ code: 'compaction-policy/conflict' }))
    expect(updatePresetCompactionPolicy(current, {
      entryId: 'preset-minimal', expectedRevision: first.revision, target, modelPolicyRoutes: [],
    })).toEqual(first)
  })

  it('refuses stale policy without changing the editor clone', () => {
    const current = declaration()
    const before = structuredClone(current)
    expect(() => updatePresetCompactionPolicy(current, {
      entryId: 'preset-standard', expectedRevision: 'stale', target, modelPolicyRoutes: [],
    })).toThrow(expect.objectContaining({ code: 'compaction-policy/conflict' }))
    expect(current).toEqual(before)
  })

  it('refuses a replaced preset identity even when its summary fields match', () => {
    const current = declaration()
    const snapshot = inspectPresetCompactionPolicy('preset-standard', current)
    current.id = 'replacement-preset'
    const before = structuredClone(current)
    expect(() => updatePresetCompactionPolicy(current, {
      entryId: 'preset-standard', expectedRevision: snapshot.revision, target, modelPolicyRoutes: [],
    })).toThrow(expect.objectContaining({ code: 'compaction-policy/conflict' }))
    expect(current).toEqual(before)
    expect(inspectPresetCompactionPolicy('preset-other', declaration()).revision).not.toBe(snapshot.revision)
  })

  it.each([0, -1, 0.5])('refuses invalid output budget %s', (maxTokens) => {
    const current = declaration()
    expect(() => updatePresetCompactionPolicy(current, {
      entryId: 'preset-standard', expectedRevision: inspectPresetCompactionPolicy('preset-standard', current).revision,
      target: { ...target, maxTokens }, modelPolicyRoutes: [],
    })).toThrow(expect.objectContaining({ code: 'compaction-policy/invalid' }))
  })

  it('refuses missing and duplicate selected override keys', () => {
    for (const routes of [[{ provider: 'absent', model: 'absent' }],
      [{ provider: 'conversation', model: 'owned' }, { provider: 'conversation', model: 'owned' }]]) {
      const current = declaration()
      expect(() => updatePresetCompactionPolicy(current, {
        entryId: 'preset-standard', expectedRevision: inspectPresetCompactionPolicy('preset-standard', current).revision,
        target, modelPolicyRoutes: routes,
      })).toThrow(expect.objectContaining({ code: 'compaction-policy/invalid' }))
    }
  })

  it('refuses ambiguous plugin rows instead of selecting one', () => {
    expect(() => inspectPresetCompactionPolicy('preset-bad', { id: 'bad', plugins: [
      { id: 'compaction', name: 'cordis:group', group: true, config: [] },
      { id: 'compaction', name: 'cordis:group', group: true, config: [] },
    ] })).toThrow(expect.objectContaining({ code: 'compaction-policy/invalid' }))
  })

  it.each(['group', 'backend'])('refuses a disabled %s instead of claiming activation', (disabled) => {
    const current = declaration()
    const group = (current.plugins as Record<string, unknown>[])[1]!
    if (disabled === 'group') group.disabled = true
    else (group.config as Record<string, unknown>[])[0]!.disabled = true
    expect(() => inspectPresetCompactionPolicy('preset-standard', current))
      .toThrow(expect.objectContaining({ code: 'compaction-policy/invalid' }))
  })

  it.each([
    [{ ...minimal, id: ' ' }, 'The declaration has no preset id'],
    [{ ...minimal, plugins: {} }, 'The compaction parent must contain plugin rows'],
    [{ ...minimal, plugins: [{ ...minimal.plugins[0], group: false }] }, 'The compaction row must be a Cordis group'],
    [{ ...minimal, plugins: [{ ...minimal.plugins[0], config: [
      { ...minimal.plugins[0]!.config[0], config: ['not a mapping'] },
    ] }] }, 'compaction-basic config must be a literal object'],
    [{ ...minimal, plugins: [{ ...minimal.plugins[0], config: [
      { ...minimal.plugins[0]!.config[0], config: { thresholdRatio: 0 } },
    ] }] }, 'BasicCompactionConfig.thresholdRatio (0) must be a number in (0, 1]'],
  ] as const)('reports malformed preset declarations %#', (current, reason) => {
    expect(() => inspectPresetCompactionPolicy('preset-minimal', current)).toThrow(expect.objectContaining({
      code: 'compaction-policy/invalid',
      details: { entryId: 'preset-minimal', reason },
    }))
  })

  it('refuses a requested override when the declaration never configured model policies', () => {
    const entryId = 'preset-minimal'
    const current = structuredClone(minimal)
    const snapshot = inspectPresetCompactionPolicy(entryId, current)
    const requested = { expectedRevision: snapshot.revision, modelPolicyRoutes: [{ provider: 'conversation', model: 'missing' }] }
    expect(() => updatePresetCompactionPolicy(current, { ...requested, entryId, target })).toThrow(expect.objectContaining({
      code: 'compaction-policy/invalid', message: 'Requested model-policy route is absent or ambiguous',
    }))
    expect(() => restorePresetCompactionPolicy(current, { ...requested, snapshot })).toThrow(expect.objectContaining({
      code: 'compaction-policy/invalid',
      message: 'Requested model-policy route is absent or ambiguous in the current policy or captured snapshot',
    }))
    expect(current).toEqual(minimal)
  })
})

async function controlFixture(metadataAvailable = true) {
  const ctx = new Context()
  const entryId = 'preset-standard'
  const entry = { options: { id: entryId, name: '@deepseek-ai/dsh-agent-preset', config: declaration() } }
  const entries = vi.fn(() => [entry])
  const edit = vi.fn<Context['configEditor']['edit']>()
  const verifyActivation = vi.fn<ConstructorParameters<typeof CompactionPolicyControl>[1]['verifyActivation']>()
  const resolveModelInfo = vi.fn<Context['llm']['resolveModelInfo']>().mockResolvedValue({
    provider: target.provider, id: target.model, name: 'Summary', context: { contextWindow: 8192 },
  })
  if (metadataAvailable) ctx.provide('llm', { resolveModelInfo })
  ctx.provide('configEditor', { entries, edit })
  onTestFinished(async () => { await ctx.fiber.dispose() })
  await ctx.plugin(CompactionPolicyControl, { verifyActivation })
  const control = ctx.compactionPolicyControl
  const request = { entryId, expectedRevision: control.read(entryId).revision, target, modelPolicyRoutes: [] }
  return { ctx, control, entry, entries, edit, request, verifyActivation, resolveModelInfo }
}

describe('verified summary-policy writes', () => {
  it('requires the LLM metadata service and text input support before editing', async () => {
    const { ctx, control, edit, request, resolveModelInfo } = await controlFixture(false)
    await expect(control.setSummaryTarget(request)).rejects.toMatchObject({
      code: 'compaction-policy/unverified',
      cause: { message: 'The LLM service is not available' },
    })

    resolveModelInfo.mockResolvedValueOnce({
      provider: target.provider, id: target.model, name: 'Images only',
      context: { contextWindow: 8192 }, inputModalities: ['image'],
    })
    ctx.provide('llm', { resolveModelInfo })
    await expect(control.setSummaryTarget(request)).rejects.toMatchObject({
      code: 'compaction-policy/unverified',
      cause: { message: 'The summary target must be ordinary Chat with context remaining after its output budget' },
    })
    expect(resolveModelInfo).toHaveBeenCalledWith(target.provider, target.model, undefined)
    expect(edit).not.toHaveBeenCalled()
  })

  it('refuses a replaced declaration when the editor acquires the write', async () => {
    const { control, entry, entries, edit, request, verifyActivation } = await controlFixture()
    const replacement = structuredClone(entry)
    edit.mockImplementationOnce(async (_entry, change) => {
      entries.mockReturnValue([replacement])
      change(structuredClone(replacement.options.config), {})
    })

    await expect(control.setSummaryTarget(request)).rejects.toMatchObject({
      code: 'compaction-policy/rejected', cause: { message: 'Preset declaration changed before the policy edit' },
    })
    expect(replacement.options.config).toEqual(entry.options.config)
    expect(verifyActivation).not.toHaveBeenCalled()
  })

  it('refuses success when the activated engine and effective declaration disagree', async () => {
    const { control, entry, edit, request, verifyActivation } = await controlFixture()
    edit.mockImplementationOnce(async (_entry, change, verify) => {
      const proposed = change(structuredClone(entry.options.config), {})
      verifyActivation.mockImplementationOnce(async (_presetId, inspect) => { inspect(configuration.resolveConfig(basicConfig(proposed))) })
      await verify?.()
    })

    await expect(control.setSummaryTarget(request)).rejects.toMatchObject({
      code: 'compaction-policy/rejected', cause: { message: 'Activated summary policy differs from the persisted request' },
    })
    expect(control.read(request.entryId).revision).toBe(request.expectedRevision)
  })

  it('refuses an editor that verifies before deriving the policy', async () => {
    const { control, edit, request, verifyActivation } = await controlFixture()
    edit.mockImplementationOnce(async (_entry, _change, verify) => { await verify?.() })

    await expect(control.setSummaryTarget(request)).rejects.toMatchObject({
      code: 'compaction-policy/rejected', cause: { message: 'Preset summary policy was not derived' },
    })
    expect(verifyActivation).not.toHaveBeenCalled()
    expect(control.read(request.entryId).revision).toBe(request.expectedRevision)
  })

  it('refuses a completed editor call without a derived policy', async () => {
    const { control, edit, request, verifyActivation } = await controlFixture()
    edit.mockResolvedValueOnce(undefined)

    await expect(control.setSummaryTarget(request)).rejects.toThrow('Preset summary policy result is unavailable')
    expect(verifyActivation).not.toHaveBeenCalled()
    expect(control.read(request.entryId).revision).toBe(request.expectedRevision)
  })
})

describe('preset summary-policy restores', () => {
  it('retains cross-realm validation diagnostics without changing the declaration', () => {
    const current = declaration()
    const snapshot = inspectPresetCompactionPolicy('preset-standard', current)
    const updated = updatePresetCompactionPolicy(current, {
      entryId: snapshot.entryId, expectedRevision: snapshot.revision, target, modelPolicyRoutes: [],
    })
    const before = structuredClone(current)
    const original = configuration.resolveConfig
    const validator = vi.spyOn(configuration, 'resolveConfig')
    onTestFinished(() => { validator.mockRestore() })
    const failure = runInNewContext('new Error("Config validation failed")') as Error
    const failValidation = (): never => { throw failure }
    const details = { entryId: snapshot.entryId, reason: 'Error: Config validation failed' }

    validator.mockImplementationOnce(failValidation)
    expect(() => inspectPresetCompactionPolicy(snapshot.entryId, current)).toThrow(expect.objectContaining({
      code: 'compaction-policy/invalid', details,
    }))

    validator.mockImplementationOnce(original).mockImplementationOnce(original).mockImplementationOnce(failValidation)
    expect(() => restorePresetCompactionPolicy(current, {
      expectedRevision: updated.revision, snapshot, modelPolicyRoutes: [],
    })).toThrow(expect.objectContaining({ code: 'compaction-policy/invalid', details }))
    expect(current).toEqual(before)
  })

  it('deletes previously absent default leaves while preserving fresh pressure settings', () => {
    const current = declaration()
    const config = basicConfig(current)
    delete config.summarizationProvider
    delete config.summarizationModel
    delete config.maxTokens
    const snapshot = inspectPresetCompactionPolicy('preset-standard', current)
    expect(snapshot.policy).toEqual({})
    const updated = updatePresetCompactionPolicy(current, {
      entryId: snapshot.entryId, expectedRevision: snapshot.revision, target, modelPolicyRoutes: [],
    })
    config.thresholdRatio = 0.9
    config.headroomTokens = 8192
    const before = structuredClone(current)
    const result = restorePresetCompactionPolicy(current, {
      expectedRevision: updated.revision, snapshot, modelPolicyRoutes: [],
    })

    expect(result).toEqual(snapshot)
    const restored = basicConfig(current)
    expect(restored).not.toHaveProperty('summarizationProvider')
    expect(restored).not.toHaveProperty('summarizationModel')
    expect(restored).not.toHaveProperty('summarizationFallback')
    expect(restored).not.toHaveProperty('maxTokens')
    expect(restored).toMatchObject({ thresholdRatio: 0.9, retainRatio: 0.16, headroomTokens: 8192 })
    expect(restored.modelPolicies).toEqual(basicConfig(before).modelPolicies)
    expect((current.plugins as Record<string, unknown>[])[0]).toEqual((before.plugins as Record<string, unknown>[])[0])
  })

  it('restores heterogeneous partial routes and latest-route without overwriting fresh siblings', () => {
    const current = declaration()
    const config = basicConfig(current)
    config.summarizationFallback = 'latest-route'
    config.modelPolicies = [
      { provider: 'conversation', model: 'owned', thresholdRatio: 0.7, summarizationFallback: 'latest-route' },
      { provider: 'conversation', model: 'paired', headroomTokens: 1024,
        summarizationProvider: 'archived', summarizationModel: 'archived-summary' },
      { provider: 'conversation', model: 'capped', retainRatio: 0.1, maxTokens: 333 },
      { provider: 'conversation', model: 'user-choice', headroomTokens: 2048,
        summarizationProvider: 'custom', summarizationModel: 'keep-custom', maxTokens: 512 },
    ]
    const snapshot = inspectPresetCompactionPolicy('preset-standard', current)
    const captured = structuredClone(snapshot)
    const selectedRoutes = ['owned', 'paired', 'capped'].map(model => ({ provider: 'conversation', model }))
    updatePresetCompactionPolicy(current, {
      entryId: snapshot.entryId, expectedRevision: snapshot.revision, target, modelPolicyRoutes: selectedRoutes,
    })
    config.thresholdRatio = 0.9
    config.headroomTokens = 8192
    const routes = config.modelPolicies as Record<string, unknown>[]
    routes[0]!.thresholdRatio = 0.65
    routes[1]!.headroomTokens = 3333
    routes[2]!.retainRatio = 0.05
    routes[3]!.summarizationModel = 'fresh-user-choice'
    routes.push({ provider: 'conversation', model: 'new-route', maxTokens: 222 })
    const before = structuredClone(current)
    const fresh = inspectPresetCompactionPolicy(snapshot.entryId, current)
    const result = restorePresetCompactionPolicy(current, {
      expectedRevision: fresh.revision, snapshot, modelPolicyRoutes: selectedRoutes,
    })

    expect(result.policy).toEqual(snapshot.policy)
    expect(result.modelPolicies.slice(0, 3)).toEqual(snapshot.modelPolicies.slice(0, 3))
    expect(result.modelPolicies.slice(3)).toEqual(fresh.modelPolicies.slice(3))
    expect(snapshot).toEqual(captured)
    const restored = basicConfig(current)
    expect(restored).toMatchObject({ thresholdRatio: 0.9, headroomTokens: 8192, retainRatio: 0.16 })
    expect(restored.modelPolicies).toEqual([
      { provider: 'conversation', model: 'owned', thresholdRatio: 0.65, summarizationFallback: 'latest-route' },
      { provider: 'conversation', model: 'paired', headroomTokens: 3333,
        summarizationProvider: 'archived', summarizationModel: 'archived-summary' },
      { provider: 'conversation', model: 'capped', retainRatio: 0.05, maxTokens: 333 },
      ...((basicConfig(before).modelPolicies as Record<string, unknown>[]).slice(3)),
    ])
    const plugins = current.plugins as Record<string, unknown>[]
    const previousPlugins = before.plugins as Record<string, unknown>[]
    expect(plugins[0]).toEqual(previousPlugins[0])
    expect(plugins[1]?.isolate).toEqual(previousPlugins[1]?.isolate)
    expect((plugins[1]?.config as Record<string, unknown>[])[1]).toEqual((previousPlugins[1]?.config as Record<string, unknown>[])[1])
    expect(current.order).toBe(before.order)
  })

  it('accepts reordered wire fields without changing the captured revision', () => {
    const current = declaration()
    basicConfig(current).summarizationFallback = 'latest-route'
    const captured = inspectPresetCompactionPolicy('preset-standard', current)
    const snapshot = { ...captured, policy: {
      maxTokens: captured.policy.maxTokens!,
      summarizationFallback: captured.policy.summarizationFallback!,
      summarizationModel: captured.policy.summarizationModel!,
      summarizationProvider: captured.policy.summarizationProvider!,
    } }
    const updated = updatePresetCompactionPolicy(current, {
      entryId: snapshot.entryId, expectedRevision: snapshot.revision, target, modelPolicyRoutes: [],
    })
    expect(restorePresetCompactionPolicy(current, {
      expectedRevision: updated.revision, snapshot, modelPolicyRoutes: [],
    })).toEqual(captured)
  })

  it('refuses a stale current revision without changing the editor clone', () => {
    const current = declaration()
    const snapshot = inspectPresetCompactionPolicy('preset-standard', current)
    updatePresetCompactionPolicy(current, {
      entryId: snapshot.entryId, expectedRevision: snapshot.revision, target, modelPolicyRoutes: [],
    })
    const before = structuredClone(current)
    expect(() => restorePresetCompactionPolicy(current, {
      expectedRevision: snapshot.revision, snapshot, modelPolicyRoutes: [],
    })).toThrow(expect.objectContaining({ code: 'compaction-policy/conflict' }))
    expect(current).toEqual(before)
  })

  it('refuses a restored default that is incompatible with fresh pressure settings without mutation', () => {
    const current = declaration()
    delete basicConfig(current).maxTokens
    const snapshot = inspectPresetCompactionPolicy('preset-standard', current)
    updatePresetCompactionPolicy(current, {
      entryId: snapshot.entryId, expectedRevision: snapshot.revision, target, modelPolicyRoutes: [],
    })
    basicConfig(current).headroomTokens = 0
    const fresh = inspectPresetCompactionPolicy(snapshot.entryId, current)
    const before = structuredClone(current)

    expect(() => restorePresetCompactionPolicy(current, {
      expectedRevision: fresh.revision, snapshot, modelPolicyRoutes: [],
    })).toThrow(expect.objectContaining({ code: 'compaction-policy/invalid' }))
    expect(current).toEqual(before)
  })

  it.each(['default', 'unselected route', 'revision', 'undefined leaf'])(
    'refuses a modified captured %s without changing the editor clone', (modified) => {
      const current = declaration()
      const captured = inspectPresetCompactionPolicy('preset-standard', current)
      const updated = updatePresetCompactionPolicy(current, {
        entryId: captured.entryId, expectedRevision: captured.revision, target,
        modelPolicyRoutes: [{ provider: 'conversation', model: 'owned' }],
      })
      let snapshot = structuredClone(captured)
      if (modified === 'default') snapshot = { ...snapshot, policy: { ...snapshot.policy, maxTokens: 1 } }
      else if (modified === 'unselected route') snapshot = { ...snapshot, modelPolicies: snapshot.modelPolicies.map((route, index) => (
        index === 1 ? { ...route, policy: { ...route.policy, maxTokens: 1 } } : route
      )) }
      else if (modified === 'revision') snapshot = { ...snapshot, revision: 'modified' }
      else Object.defineProperty(snapshot.policy, 'summarizationFallback', { value: undefined, enumerable: true })
      const before = structuredClone(current)

      expect(() => restorePresetCompactionPolicy(current, {
        expectedRevision: updated.revision, snapshot,
        modelPolicyRoutes: [{ provider: 'conversation', model: 'owned' }],
      })).toThrow(expect.objectContaining({ code: 'compaction-policy/invalid' }))
      expect(current).toEqual(before)
    },
  )

  it.each(['entry', 'preset'])('refuses a changed %s identity with an otherwise valid snapshot', (identity) => {
    const current = declaration()
    const snapshot = inspectPresetCompactionPolicy('preset-standard', current)
    if (identity === 'preset') current.id = 'replacement'
    const expectedRevision = inspectPresetCompactionPolicy(identity === 'entry' ? 'preset-other' : snapshot.entryId, current).revision
    const before = structuredClone(current)

    expect(() => restorePresetCompactionPolicy(current, {
      expectedRevision, snapshot, modelPolicyRoutes: [],
    })).toThrow(expect.objectContaining({ code: 'compaction-policy/conflict' }))
    expect(current).toEqual(before)
  })

  it.each(['current route', 'captured route', 'requested route', 'duplicate route'])(
    'refuses a missing or duplicate %s before changing any policy', (missing) => {
      const prior = declaration()
      if (missing === 'captured route') (basicConfig(prior).modelPolicies as Record<string, unknown>[]).shift()
      const snapshot = inspectPresetCompactionPolicy('preset-standard', prior)
      const current = declaration()
      updatePresetCompactionPolicy(current, {
        entryId: snapshot.entryId, expectedRevision: inspectPresetCompactionPolicy(snapshot.entryId, current).revision,
        target, modelPolicyRoutes: [],
      })
      if (missing === 'current route') (basicConfig(current).modelPolicies as Record<string, unknown>[]).shift()
      const routes = [{ provider: 'conversation', model: 'owned' }]
      if (missing === 'requested route') routes.push({ provider: 'absent', model: 'absent' })
      if (missing === 'duplicate route') routes.push({ provider: 'conversation', model: 'owned' })
      const before = structuredClone(current)

      expect(() => restorePresetCompactionPolicy(current, {
        expectedRevision: inspectPresetCompactionPolicy(snapshot.entryId, current).revision,
        snapshot, modelPolicyRoutes: routes,
      })).toThrow(expect.objectContaining({ code: 'compaction-policy/invalid' }))
      expect(current).toEqual(before)
    },
  )
})
