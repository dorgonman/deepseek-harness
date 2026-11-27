/** Durable policy edits through the real profile, Loader, preset and Remote owners. */
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import type { EntryOptions, ModuleLoaderV2 } from '@deepseek-ai/cordis-plugin-loader'
import type { PatchOptions } from '@deepseek-ai/cordis-plugin-include'
import z from '@deepseek-ai/schemastery'
import { boot, initProfile, readProfilePatches, reconcileProfilePatches, type ProfileContext } from '@deepseek-ai/dsh-app-boot'
import ConfigEditor from '@deepseek-ai/dsh-config-editor'
import LlmRuntime, { LlmAdapter, type LlmResolvedModelInfo, type StreamChunk } from '@deepseek-ai/dsh-llm'
import SessionStore, { SessionId } from '@deepseek-ai/dsh-session'
import SessionProjectionRegistry from '@deepseek-ai/dsh-session-projection'
import TokenMeter from '@deepseek-ai/dsh-token-meter'
import AgentRegistry, { type Agent } from '@deepseek-ai/dsh-agent'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import * as NativeToolExecutor from '@deepseek-ai/dsh-tool-executor-native'
import AgentPreset from '@deepseek-ai/dsh-agent-preset'
import BasicCompactionEngine, { CompactionPolicyControl, type BasicCompactionConfig } from '@deepseek-ai/dsh-compaction-basic'
import type { CompactionPolicyRoute, PresetCompactionPolicy, PresetCompactionPolicyRestore, PresetCompactionPolicyUpdate } from '@deepseek-ai/dsh-compaction-basic/types'
import TypertRegistry from '@deepseek-ai/dsh-typert-registry'
import RemoteAccessService from '@deepseek-ai/dsh-api-remote-access'
import TypertGatewayService from '@deepseek-ai/dsh-api-gateway'
import { Peer } from '@deepseek-ai/dsh-client-connection'
import type { Principal } from '@deepseek-ai/dsh-typert-protocol'
import { describe, expect, it, onTestFinished, vi } from 'vitest'
import AgentPresets, { type PresetDefinition } from '../src/index.ts'

declare module '@deepseek-ai/dsh-typert-protocol' {
  interface PrincipalMap {
    'policy-control-guest': { readonly kind: 'policy-control-guest' }
  }
}

const PRESETS = ['standard', 'ptc', 'minimal', 'cordis'] as const
const originalPolicy: BasicCompactionConfig = {
  thresholdRatio: 0.75,
  headroomTokens: 256,
  retainTokens: 32,
  summarizationProvider: 'metadata',
  summarizationModel: 'original',
  summarizationFallback: 'latest-route',
  maxTokens: 64,
  compactionRetries: 2,
  auto: false,
  modelPolicies: [
    { provider: 'conversation', model: 'owned', headroomTokens: 128, summarizationProvider: 'metadata', summarizationModel: 'owned-original', maxTokens: 32 },
    { provider: 'conversation', model: 'exception', summarizationProvider: 'exception', summarizationModel: 'user-choice', summarizationFallback: 'error', maxTokens: 48 },
  ],
}

function declaration(id: string): PresetDefinition {
  return {
    id,
    name: `Fixture ${id}`,
    description: 'Unrelated declaration metadata',
    order: PRESETS.findIndex(value => value === id),
    plugins: [
      { id: 'sibling', name: 'cordis:probe', config: { marker: id, nested: { kept: true }, values: [3, 1, 4] } },
      {
        id: 'compaction', name: 'cordis:group', group: true, isolate: { compaction: true },
        config: [
          { id: 'compaction-basic', name: '@deepseek-ai/dsh-compaction-basic', config: structuredClone(originalPolicy) },
          { id: 'activation-probe', name: 'cordis:activation' },
        ],
      },
    ],
  }
}

function compactionConfig(value: PresetDefinition): BasicCompactionConfig {
  const group = value.plugins.find(row => row.id === 'compaction')!
  return (group.config as EntryOptions[]).find(row => row.id === 'compaction-basic')!.config as BasicCompactionConfig
}

function barrier() {
  const state = Promise.withResolvers<undefined>()
  return { promise: state.promise, resolve: () => { state.resolve(undefined) } }
}

/** Only the external model metadata seam is synthetic; inference is forbidden. */
class MetadataAdapter extends LlmAdapter {
  block: { model: string; lookup?: number; entered: ReturnType<typeof barrier>; release: ReturnType<typeof barrier> } | undefined
  readonly lookups = new Map<string, number>()
  readonly availability = new Map<string, 'decision' | 'unavailable'>()
  inferenceCalls = 0

  override async resolveModel(provider: string, model: string, signal?: AbortSignal): Promise<LlmResolvedModelInfo> {
    const lookup = (this.lookups.get(model) ?? 0) + 1
    this.lookups.set(model, lookup)
    if (this.block?.model === model && (this.block.lookup === undefined || this.block.lookup === lookup)) {
      this.block.entered.resolve()
      await this.block.release.promise
    }
    signal?.throwIfAborted()
    const availability = this.availability.get(model)
    if (availability === 'unavailable') throw new Error('Synthetic target metadata became unavailable')
    return {
      provider, id: model, name: model, nativeTools: false,
      ...model === 'missing-capacity' ? {} : { context: { contextWindow: 8192 } },
      ...model === 'decision' || availability === 'decision' ? { requestProtocol: 'kinf-decision-v1' as const } : {},
    }
  }

  override async * stream(): AsyncIterable<StreamChunk> {
    this.inferenceCalls++
    throw new Error('This policy fixture must never perform inference')
  }
}

async function fixture(options: {
  declarations?: EntryOptions[]
  overlays?: readonly PatchOptions[]
  editorDisabled?: boolean
  incompatibleBackend?: boolean
} = {}) {
  const home = realpathSync(mkdtempSync(join(tmpdir(), 'preset-policy-')))
  const contexts: Context[] = []
  const adapter = new MetadataAdapter()
  const activation: {
    rejectModel?: string
    block?: { model: string; entered: ReturnType<typeof barrier>; release: ReturnType<typeof barrier> }
    refused: number
  } = { refused: 0 }
  onTestFinished(async () => {
    try {
      adapter.block?.release.resolve()
      activation.block?.release.resolve()
      for (const ctx of contexts.reverse()) await ctx.fiber.dispose()
      expect(adapter.inferenceCalls).toBe(0)
    } finally { rmSync(home, { recursive: true, force: true }) }
  })
  const dir = join(home, 'profiles', 'test')
  initProfile(dir, ['policy-fixture'])
  const bundle = join(dir, 'node_modules', 'policy-fixture')
  mkdirSync(bundle, { recursive: true })
  writeFileSync(join(home, 'package.json'), '{"name":"preset-policy-fixture"}\n')
  writeFileSync(join(bundle, 'package.json'), JSON.stringify({ name: 'policy-fixture', version: '1.0.0', dsh: { bundle: { patch: 'cordis.patch.yml' } } }))
  const rows: EntryOptions[] = [
    { id: 'llm', name: '@deepseek-ai/dsh-llm' },
    { id: 'metadata', name: 'cordis:metadata' },
    { id: 'sessions', name: '@deepseek-ai/dsh-session' },
    { id: 'projections', name: '@deepseek-ai/dsh-session-projection' },
    { id: 'meter', name: '@deepseek-ai/dsh-token-meter' },
    { id: 'prompt', name: '@deepseek-ai/dsh-system-prompt', config: { personaPrefix: 'Unrelated global prompt' } },
    { id: 'tools', name: '@deepseek-ai/dsh-tools' },
    { id: 'executor', name: '@deepseek-ai/dsh-tool-executor-native' },
    { id: 'agents', name: '@deepseek-ai/dsh-agent' },
    { id: 'loop', name: '@deepseek-ai/dsh-agent-loop', config: { agents: [] } },
    { id: 'config-editor', name: '@deepseek-ai/dsh-config-editor', disabled: options.editorDisabled ?? false },
    { id: 'typert', name: '@deepseek-ai/dsh-typert-registry' },
    { id: 'access', name: '@deepseek-ai/dsh-api-remote-access' },
    { id: 'gateway', name: '@deepseek-ai/dsh-api-gateway' },
    { id: 'presets', name: '@deepseek-ai/dsh-agent-preset-registry', config: { default: 'standard' } },
    ...(options.declarations ?? PRESETS.map(id => ({ id: `preset-${id}`, name: '@deepseek-ai/dsh-agent-preset', config: declaration(id) }))),
  ]
  writeFileSync(join(bundle, 'cordis.patch.yml'), JSON.stringify([{ insert: rows }]))
  writeFileSync(join(dir, 'cordis.yml'), '[]\n')
  if (options.incompatibleBackend === true) {
    const backend = join(dir, 'node_modules', '@deepseek-ai', 'dsh-compaction-basic')
    mkdirSync(backend, { recursive: true })
    writeFileSync(join(backend, 'package.json'), JSON.stringify({
      name: '@deepseek-ai/dsh-compaction-basic', version: '0.0.0', engines: { dsh: '>=999.0.0' },
    }))
  }
  const profile: ProfileContext = {
    name: 'test', startedBundles: ['policy-fixture'], dir, patchPath: join(dir, 'cordis.patch.yml'),
    installAnchor: join(home, 'package.json'), cwd: home, home,
    overlays: options.overlays ?? [], telemetryDisabledEnv: undefined,
  }
  const modules = new Map<string, unknown>([
    ['@deepseek-ai/dsh-llm', LlmRuntime],
    ['@deepseek-ai/dsh-session', SessionStore],
    ['@deepseek-ai/dsh-session-projection', SessionProjectionRegistry],
    ['@deepseek-ai/dsh-token-meter', TokenMeter],
    ['@deepseek-ai/dsh-system-prompt', SystemPrompt],
    ['@deepseek-ai/dsh-tools', ToolRuntime],
    ['@deepseek-ai/dsh-tool-executor-native', NativeToolExecutor],
    ['@deepseek-ai/dsh-agent', AgentRegistry],
    ['@deepseek-ai/dsh-agent-loop', AgentLoop],
    ['@deepseek-ai/dsh-config-editor', ConfigEditor],
    ['@deepseek-ai/dsh-typert-registry', TypertRegistry],
    ['@deepseek-ai/dsh-api-remote-access', RemoteAccessService],
    ['@deepseek-ai/dsh-api-gateway', TypertGatewayService],
    ['@deepseek-ai/dsh-agent-preset-registry', AgentPresets],
    ['@deepseek-ai/dsh-agent-preset', AgentPreset],
    ['@deepseek-ai/dsh-compaction-basic', BasicCompactionEngine],
  ])
  const start = async (): Promise<Context> => {
    const logs: string[] = []
    const ctx = await boot('test', join(dir, 'cordis.yml'), readProfilePatches('test', profile), (owner) => {
      contexts.push(owner)
      owner.logger.exporter({ levels: { default: 3 }, export: ({ args }) => {
        logs.push(args.map(value => value instanceof Error ? value.stack : String(value)).join(' '))
      } })
      owner.provide('profileContext', profile)
      Object.assign(owner.loader.builtins, {
        probe: { Config: z.object({ marker: z.string(), nested: z.any(), values: z.any() }), apply() {} },
        metadata: {
          inject: ['llm'],
          apply(child: Context) { child.effect(() => child.llm.registerAdapter(['metadata'], adapter)) },
        },
        activation: {
          inject: ['compaction'],
          async apply(child: Context) {
            const engine = child.compaction
            if (!(engine instanceof BasicCompactionEngine)) throw new Error('Expected the real basic compaction engine')
            if (activation.block?.model === engine.config.summarizationModel) {
              activation.block.entered.resolve()
              await activation.block.release.promise
            }
            if (activation.rejectModel !== undefined && engine.config.summarizationModel === activation.rejectModel) {
              activation.refused++
              throw new Error('Owned activation refusal')
            }
          },
        },
      })
      const internal: ModuleLoaderV2 = {
        version: 'v2', loadCache: new Map(),
        import: (specifier: string) => {
          if (!modules.has(specifier)) throw new Error(`Unexpected Loader import: ${specifier}`)
          return Promise.resolve(modules.get(specifier))
        },
        register(): never { throw new Error('Unexpected module hook registration') },
        getOrCreateModuleJob(): never { throw new Error('Unexpected module job creation') },
        resolveSync(): never { throw new Error('Unexpected synchronous module resolution') },
        load(): never { throw new Error('Unexpected module load') },
      }
      owner.loader.internal = internal
    })
    if (options.editorDisabled !== true) expect(ctx.get('compactionPolicyControl'), logs.join('\n')).toBeInstanceOf(CompactionPolicyControl)
    return ctx
  }
  return { ctx: await start(), start, profile, adapter, activation }
}

function request(ctx: Context, id = 'standard', model = 'replacement'): PresetCompactionPolicyUpdate {
  const entryId = `preset-${id}`
  return {
    entryId, expectedRevision: ctx.compactionPolicyControl.read(entryId).revision,
    target: { provider: 'metadata', model, maxTokens: 96 },
    modelPolicyRoutes: [{ provider: 'conversation', model: 'owned' }],
  }
}

function restoration(
  ctx: Context,
  snapshot: PresetCompactionPolicy,
  modelPolicyRoutes: readonly CompactionPolicyRoute[] = [{ provider: 'conversation', model: 'owned' }],
): PresetCompactionPolicyRestore {
  return { snapshot, expectedRevision: ctx.compactionPolicyControl.read(snapshot.entryId).revision, modelPolicyRoutes }
}

async function agent(ctx: Context, id: string, presetId = 'standard'): Promise<Agent> {
  const handle = await ctx.agents.create({
    sessionId: SessionId(id),
    setup: async (child) => { await ctx.agentPresets.mount(child, presetId) },
  })
  return handle.agent
}

function engine(ctx: Context, subject: Agent): BasicCompactionEngine {
  const result = ctx.agentPresets.serviceFor(subject, 'compaction')
  if (!(result instanceof BasicCompactionEngine)) throw new Error('Agent has no real basic compaction engine')
  return result
}

describe('profile-owned compaction policy control', () => {
  it('persists and activates four independent presets while retaining unrelated config and explicit exceptions', async () => {
    const { ctx, start, profile } = await fixture()
    const before = ctx.configEditor.entries().map((row) => {
      const config: unknown = row.options.config
      return { id: row.options.id, config: structuredClone(config) }
    })
    const receipts = []
    for (const id of PRESETS) {
      const update = request(ctx, id)
      const receipt = await ctx.compactionPolicyControl.setSummaryTarget(update)
      receipts.push(receipt)
      expect(receipt).toMatchObject({ entryId: `preset-${id}`, presetId: id, persisted: true, activated: true,
        policy: { summarizationProvider: 'metadata', summarizationModel: 'replacement', summarizationFallback: 'error', maxTokens: 96 } })
      expect(receipt.revision).not.toBe(update.expectedRevision)
      expect(await ctx.agentPresets.resolve(id)).toEqual({ id })
      for (const pending of PRESETS.slice(PRESETS.indexOf(id) + 1)) {
        expect(ctx.compactionPolicyControl.read(`preset-${pending}`).policy.summarizationModel).toBe('original')
      }
    }
    const expectedPolicy: BasicCompactionConfig = {
      ...structuredClone(originalPolicy), summarizationProvider: 'metadata', summarizationModel: 'replacement', summarizationFallback: 'error', maxTokens: 96,
      modelPolicies: originalPolicy.modelPolicies!.map(row => row.model === 'owned'
        ? { ...row, summarizationProvider: 'metadata', summarizationModel: 'replacement', summarizationFallback: 'error', maxTokens: 96 }
        : row),
    }
    for (const row of before) {
      const actual: unknown = ctx.configEditor.entries().find(entry => entry.options.id === row.id)?.options.config
      if (!row.id.startsWith('preset-')) { expect(actual).toEqual(row.config); continue }
      const expected = declaration(row.id.slice('preset-'.length))
      const group = expected.plugins.find(plugin => plugin.id === 'compaction')!
      const children = group.config as EntryOptions[]
      children[0]!.config = expectedPolicy
      expect(actual).toEqual(expected)
    }
    const bytes = readFileSync(profile.patchPath, 'utf8')
    expect(PRESETS.every(id => bytes.includes(`preset-${id}`))).toBe(true)
    await ctx.fiber.dispose()
    const restarted = await start()
    for (const receipt of receipts) {
      expect(restarted.compactionPolicyControl.read(receipt.entryId)).toEqual({
        entryId: receipt.entryId, presetId: receipt.presetId, revision: receipt.revision,
        policy: receipt.policy, modelPolicies: receipt.modelPolicies,
      })
      expect(await restarted.agentPresets.resolve(receipt.presetId)).toEqual({ id: receipt.presetId })
    }
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
  })

  it('gives fresh Agents a new engine generation while an existing Agent retains its old policy', async () => {
    const { ctx } = await fixture()
    const old = await agent(ctx, 'policy-old-agent')
    const previous = engine(ctx, old)
    await using previousScope = await ctx.agentPresets.acquireScope('standard')
    await ctx.compactionPolicyControl.setSummaryTarget(request(ctx))
    await using nextScope = await ctx.agentPresets.acquireScope('standard')
    const fresh = await agent(ctx, 'policy-fresh-agent')
    expect(nextScope.key).not.toBe(previousScope.key)
    expect(engine(ctx, old).config).toBe(previous.config)
    expect(previous.config.summarizationModel).toBe('original')
    expect(engine(ctx, fresh).config).not.toBe(previous.config)
    expect(engine(ctx, fresh).config).toMatchObject({ summarizationModel: 'replacement', summarizationFallback: 'error', maxTokens: 96 })
  })

  it('rejects a stale revision and an overlapping write against the fresh locked policy', async () => {
    const { ctx, adapter, profile } = await fixture()
    const first = request(ctx, 'standard', 'delayed')
    const winner = request(ctx, 'standard', 'winner')
    const entered = barrier()
    const release = barrier()
    adapter.block = { model: 'delayed', entered, release }
    const delayed = ctx.compactionPolicyControl.setSummaryTarget(first)
    const rejection = expect(delayed).rejects.toMatchObject({ code: 'compaction-policy/conflict' })
    await entered.promise
    try { await ctx.compactionPolicyControl.setSummaryTarget(winner) }
    finally { release.resolve() }
    await rejection
    const bytes = readFileSync(profile.patchPath, 'utf8')
    await expect(ctx.compactionPolicyControl.setSummaryTarget(first)).rejects.toMatchObject({ code: 'compaction-policy/conflict' })
    expect(ctx.compactionPolicyControl.read('preset-standard').policy.summarizationModel).toBe('winner')
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
  })

  it('refreshes an external durable edit before comparing the expected policy revision', async () => {
    const { ctx, adapter, profile } = await fixture()
    const update = request(ctx, 'standard', 'delayed')
    const entered = barrier()
    const release = barrier()
    adapter.block = { model: 'delayed', entered, release }
    const pending = ctx.compactionPolicyControl.setSummaryTarget(update)
    const rejection = expect(pending).rejects.toMatchObject({ code: 'compaction-policy/conflict' })
    await entered.promise
    const external = declaration('standard')
    const group = external.plugins.find(row => row.id === 'compaction')!
    const children = group.config as EntryOptions[]
    children[0]!.config = { ...originalPolicy, summarizationModel: 'external' }
    const bytes = JSON.stringify([{ id: 'preset-standard', config: external }]) + '\n'
    writeFileSync(profile.patchPath, bytes)
    release.resolve()
    await rejection
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
    expect(ctx.compactionPolicyControl.read(update.entryId).policy.summarizationModel).toBe('external')
    expect(engine(ctx, await agent(ctx, 'policy-external-fresh')).config.summarizationModel).toBe('external')
  })

  it('conflicts when a fresh external edit changes only the preset identity', async () => {
    const { ctx, profile } = await fixture()
    const before = ctx.compactionPolicyControl.read('preset-standard')
    const update = request(ctx)
    const external = { ...declaration('standard'), id: 'replacement-identity' }
    const bytes = JSON.stringify([{ id: 'preset-standard', config: external }]) + '\n'
    writeFileSync(profile.patchPath, bytes)
    await expect(ctx.compactionPolicyControl.setSummaryTarget(update)).rejects.toMatchObject({ code: 'compaction-policy/conflict' })
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
    const actual = ctx.compactionPolicyControl.read('preset-standard')
    expect(actual.presetId).toBe('replacement-identity')
    expect(actual.revision).not.toBe(before.revision)
    expect(actual.policy).toEqual(before.policy)
    expect(actual.modelPolicies).toEqual(before.modelPolicies)
    await expect(ctx.agentPresets.resolve('standard')).rejects.toMatchObject({ code: 'agent-preset/not-found' })
    expect(await ctx.agentPresets.resolve('replacement-identity')).toEqual({ id: 'replacement-identity' })
    const fresh = await agent(ctx, 'policy-replaced-identity', 'replacement-identity')
    expect(engine(ctx, fresh).config.summarizationModel).toBe('original')
  })

  it('refuses a declaration withdrawn while target metadata is being resolved', async () => {
    const { ctx, adapter, profile } = await fixture()
    const old = await agent(ctx, 'policy-withdrawn-old')
    const entered = barrier()
    const release = barrier()
    adapter.block = { model: 'delayed', entered, release }
    const pending = ctx.compactionPolicyControl.setSummaryTarget(request(ctx, 'standard', 'delayed'))
    const rejection = expect(pending).rejects.toMatchObject({ code: 'compaction-policy/rejected' })
    await entered.promise
    const bytes = '- id: preset-standard\n  disabled: true\n'
    try {
      writeFileSync(profile.patchPath, bytes)
      await reconcileProfilePatches(ctx, readProfilePatches('test', profile), 'test')
    } finally { release.resolve() }
    await rejection
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
    await expect(ctx.agentPresets.resolve('standard')).rejects.toMatchObject({ code: 'agent-preset/not-found' })
    expect(engine(ctx, old).config.summarizationModel).toBe('original')
  })

  it('restores exact previous patch bytes and usable runtime when a nested child fails activation', async () => {
    const { ctx, activation, profile } = await fixture()
    const old = await agent(ctx, 'policy-rollback-old')
    const earlier = await ctx.compactionPolicyControl.setSummaryTarget(request(ctx, 'ptc', 'earlier-success'))
    const before = ctx.compactionPolicyControl.read('preset-standard')
    const bytes = readFileSync(profile.patchPath, 'utf8')
    activation.rejectModel = 'refused'
    await expect(ctx.compactionPolicyControl.setSummaryTarget(request(ctx, 'standard', 'refused'))).rejects.toMatchObject({ code: 'compaction-policy/rejected' })
    expect(activation.refused).toBeGreaterThan(0)
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
    expect(ctx.compactionPolicyControl.read('preset-standard')).toEqual(before)
    expect(ctx.compactionPolicyControl.read('preset-ptc').revision).toBe(earlier.revision)
    expect(await ctx.agentPresets.resolve('standard')).toEqual({ id: 'standard' })
    const fresh = await agent(ctx, 'policy-rollback-fresh')
    expect(engine(ctx, old).config.summarizationModel).toBe('original')
    expect(engine(ctx, fresh).config.summarizationModel).toBe('original')
  })

  it.each(['decision', 'unavailable'] as const)('rolls back when target metadata becomes %s after preflight', async (availability) => {
    const { ctx, adapter, profile } = await fixture()
    const old = await agent(ctx, `policy-metadata-${availability}-old`)
    const previous = engine(ctx, old).config
    const before = ctx.compactionPolicyControl.read('preset-standard')
    const bytes = readFileSync(profile.patchPath, 'utf8')
    const entered = barrier()
    const release = barrier()
    adapter.block = { model: 'changing', lookup: 2, entered, release }
    const pending = ctx.compactionPolicyControl.setSummaryTarget(request(ctx, 'standard', 'changing'))
    const rejection = expect(pending).rejects.toMatchObject({ code: 'compaction-policy/rejected' })
    await entered.promise
    try {
      expect(adapter.lookups.get('changing')).toBe(2)
      expect(readFileSync(profile.patchPath, 'utf8')).not.toBe(bytes)
      expect(ctx.compactionPolicyControl.read('preset-standard').policy.summarizationModel).toBe('changing')
      adapter.availability.set('changing', availability)
    } finally { release.resolve() }
    await rejection
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
    expect(ctx.compactionPolicyControl.read('preset-standard')).toEqual(before)
    expect(await ctx.agentPresets.resolve('standard')).toEqual({ id: 'standard' })
    expect(engine(ctx, old).config).toBe(previous)
    expect(previous.summarizationModel).toBe('original')
    const fresh = await agent(ctx, `policy-metadata-${availability}-fresh`)
    expect(engine(ctx, fresh).config.summarizationModel).toBe('original')
  })

  it('refuses a higher overlay without writing a misleading durable policy', async () => {
    const { ctx, profile } = await fixture({ overlays: [{ id: 'preset-standard', config: declaration('standard') }] })
    const before = ctx.compactionPolicyControl.read('preset-standard')
    const bytes = readFileSync(profile.patchPath, 'utf8')
    await expect(ctx.compactionPolicyControl.setSummaryTarget(request(ctx))).rejects.toMatchObject({ code: 'compaction-policy/rejected' })
    expect(ctx.compactionPolicyControl.read('preset-standard')).toEqual(before)
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
  })

  it('rejects missing capacity, Decision targets and invalid output headroom before persistence', async () => {
    const { ctx, profile } = await fixture()
    const bytes = readFileSync(profile.patchPath, 'utf8')
    const before = ctx.compactionPolicyControl.read('preset-standard')
    for (const model of ['missing-capacity', 'decision']) {
      await expect(ctx.compactionPolicyControl.setSummaryTarget(request(ctx, 'standard', model))).rejects.toMatchObject({ code: 'compaction-policy/unverified' })
    }
    for (const maxTokens of [0, 8192, 8193]) {
      const update = request(ctx)
      await expect(ctx.compactionPolicyControl.setSummaryTarget({ ...update, target: { ...update.target, maxTokens } })).rejects.toMatchObject({ code: 'compaction-policy/unverified' })
    }
    expect(ctx.compactionPolicyControl.read('preset-standard')).toEqual(before)
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
  })

  it('refuses missing, ambiguous and replaced policy rows instead of guessing', async () => {
    const absent = declaration('standard')
    const duplicate = declaration('duplicate')
    const replaced = declaration('replaced')
    const groupOf = (value: PresetDefinition) => value.plugins.find(row => row.id === 'compaction')!
    groupOf(absent).config = []
    const duplicateRows = groupOf(duplicate).config as EntryOptions[]
    duplicateRows.push(structuredClone(duplicateRows[0]!))
    const replacedRows = groupOf(replaced).config as EntryOptions[]
    replacedRows[0]!.name = 'cordis:probe'
    replacedRows[0]!.config = { marker: 'replacement' }
    const { ctx, profile } = await fixture({ declarations: [
      { id: 'preset-standard', name: '@deepseek-ai/dsh-agent-preset', config: absent },
      { id: 'preset-duplicate', name: '@deepseek-ai/dsh-agent-preset', config: duplicate },
      { id: 'preset-replaced', name: '@deepseek-ai/dsh-agent-preset', config: replaced },
      { id: 'preset-other-owner', name: 'cordis:probe', config: { marker: 'different declaration owner' } },
    ] })
    const bytes = readFileSync(profile.patchPath, 'utf8')
    for (const id of ['preset-standard', 'preset-duplicate', 'preset-replaced']) {
      expect(() => ctx.compactionPolicyControl.read(id)).toThrow(expect.objectContaining({ code: 'compaction-policy/invalid' }))
    }
    for (const id of ['missing', 'preset-other-owner']) {
      expect(() => ctx.compactionPolicyControl.read(id)).toThrow(expect.objectContaining({ code: 'compaction-policy/not-found' }))
    }
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
  })

  it.each(['group', 'backend'] as const)('refuses a literally disabled %s policy owner', async (disabled) => {
    const config = declaration('standard')
    const group = config.plugins.find(row => row.id === 'compaction')!
    if (disabled === 'group') group.disabled = true
    else (group.config as EntryOptions[])[0]!.disabled = true
    const { ctx, profile } = await fixture({ declarations: [{ id: 'preset-standard', name: '@deepseek-ai/dsh-agent-preset', config }] })
    const bytes = readFileSync(profile.patchPath, 'utf8')
    expect(() => ctx.compactionPolicyControl.read('preset-standard')).toThrow(expect.objectContaining({ code: 'compaction-policy/invalid' }))
    await expect(ctx.compactionPolicyControl.setSummaryTarget({
      entryId: 'preset-standard', expectedRevision: 'unavailable',
      target: { provider: 'metadata', model: 'replacement', maxTokens: 96 }, modelPolicyRoutes: [],
    })).rejects.toMatchObject({ code: 'compaction-policy/invalid' })
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
  })

  it('refuses successful activation when compatibility admission leaves no actual engine', async () => {
    const config = declaration('standard')
    const group = config.plugins.find(row => row.id === 'compaction')!
    group.config = (group.config as EntryOptions[]).filter(row => row.id === 'compaction-basic')
    const { ctx, profile } = await fixture({
      incompatibleBackend: true,
      declarations: [{ id: 'preset-standard', name: '@deepseek-ai/dsh-agent-preset', config }],
    })
    const before = ctx.compactionPolicyControl.read('preset-standard')
    const bytes = readFileSync(profile.patchPath, 'utf8')
    // Raw declaration flags remain enabled; real compatibility preflight withdraws only the runtime row.
    expect((group.config as EntryOptions[])[0]!.disabled).toBeUndefined()
    const subject = await agent(ctx, 'policy-compatibility-absent')
    expect(ctx.agentPresets.serviceFor(subject, 'compaction')).toBeUndefined()
    await expect(ctx.compactionPolicyControl.setSummaryTarget(request(ctx))).rejects.toMatchObject({ code: 'compaction-policy/rejected' })
    expect(ctx.compactionPolicyControl.read('preset-standard')).toEqual(before)
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
    expect(ctx.agentPresets.serviceFor(await agent(ctx, 'policy-compatibility-after'), 'compaction')).toBeUndefined()
  })

  it('rolls back the policy and releases its scope lease when the leased mount is unavailable', async () => {
    const { ctx, profile } = await fixture()
    const before = ctx.compactionPolicyControl.read('preset-standard')
    const bytes = readFileSync(profile.patchPath, 'utf8')
    const release = vi.fn(() => Promise.resolve())
    const acquire = vi.spyOn(ctx.agentPresets, 'acquireScope').mockResolvedValueOnce({ key: {}, [Symbol.asyncDispose]: release })
    onTestFinished(() => { acquire.mockRestore() })

    await expect(ctx.compactionPolicyControl.setSummaryTarget(request(ctx))).rejects.toMatchObject({ code: 'compaction-policy/rejected' })
    expect(acquire).toHaveBeenCalledExactlyOnceWith('standard')
    expect(release).toHaveBeenCalledOnce()
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
    expect(ctx.compactionPolicyControl.read(before.entryId)).toEqual(before)
    expect(engine(ctx, await agent(ctx, 'policy-missing-mount')).config.summarizationModel).toBe('original')
  })

  it('withdraws and remounts the policy service with the real optional editor lifecycle', async () => {
    const { ctx, profile } = await fixture({ editorDisabled: true })
    expect(ctx.get('compactionPolicyControl')).toBeUndefined()
    writeFileSync(profile.patchPath, '- id: config-editor\n  disabled: false\n')
    await reconcileProfilePatches(ctx, readProfilePatches('test', profile), 'test')
    expect(ctx.compactionPolicyControl).toBeInstanceOf(CompactionPolicyControl)
    const previous = ctx.compactionPolicyControl
    const snapshot = previous.read('preset-standard')
    const restore = { snapshot, expectedRevision: snapshot.revision, modelPolicyRoutes: [] }
    const invokeRestore = () => ctx.typertGateway.invoke({
      peer: new Peer(ctx, { principal: { kind: 'owner' } }),
      namespace: 'compactionPolicy', method: 'restoreSummaryPolicy', args: { request: restore },
    })
    await expect(invokeRestore()).resolves.toMatchObject({ persisted: true, activated: true })
    writeFileSync(profile.patchPath, '- id: config-editor\n  disabled: true\n')
    await reconcileProfilePatches(ctx, readProfilePatches('test', profile), 'test')
    expect(ctx.get('compactionPolicyControl')).toBeUndefined()
    await expect(invokeRestore()).rejects.toMatchObject({ code: 'gateway/invocation-unavailable' })
    writeFileSync(profile.patchPath, '- id: config-editor\n  disabled: false\n')
    await reconcileProfilePatches(ctx, readProfilePatches('test', profile), 'test')
    expect(ctx.compactionPolicyControl).toBeInstanceOf(CompactionPolicyControl)
    expect(ctx.compactionPolicyControl).not.toBe(previous)
    expect(ctx.compactionPolicyControl.read('preset-standard').policy.summarizationModel).toBe('original')
    await expect(invokeRestore()).resolves.toMatchObject({ persisted: true, activated: true })
  })

  it('enforces owner authority through the actual Gateway for reads and writes', async () => {
    const { ctx, profile } = await fixture()
    const invoke = (principal: Principal, method: string, args: Record<string, unknown>) => ctx.typertGateway.invoke({
      peer: new Peer(ctx, { principal }), namespace: 'compactionPolicy', method, args,
    })
    const update = request(ctx)
    const snapshot = ctx.compactionPolicyControl.read(update.entryId)
    const bytes = readFileSync(profile.patchPath, 'utf8')
    const guest: Principal = { kind: 'policy-control-guest' }
    await expect(invoke(guest, 'read', { entryId: update.entryId })).rejects.toMatchObject({
      code: 'remote-access/denied', details: { action: 'compactionPolicy/read', reason: 'no-allow' },
    })
    await expect(invoke(guest, 'setSummaryTarget', { request: update })).rejects.toMatchObject({ code: 'remote-access/denied' })
    await expect(invoke(guest, 'restoreSummaryPolicy', { request: restoration(ctx, snapshot) })).rejects.toMatchObject({
      code: 'remote-access/denied', details: { action: 'compactionPolicy/restoreSummaryPolicy', reason: 'no-allow' },
    })
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
    await expect(invoke({ kind: 'owner' }, 'read', { entryId: update.entryId })).resolves.toEqual(ctx.compactionPolicyControl.read(update.entryId))
    await expect(invoke({ kind: 'owner' }, 'setSummaryTarget', { request: update })).resolves.toMatchObject({ persisted: true, activated: true })
    expect(ctx.compactionPolicyControl.read(update.entryId).policy.summarizationModel).toBe('replacement')
    await expect(invoke({ kind: 'owner' }, 'restoreSummaryPolicy', { request: restoration(ctx, snapshot) })).resolves.toMatchObject({
      ...snapshot, persisted: true, activated: true,
    })
    expect(ctx.compactionPolicyControl.read(update.entryId)).toEqual(snapshot)
  })
})

describe('restoring captured preset summary policies', () => {
  it.each(['ptc', 'minimal', 'cordis'])('persists absent %s default fields and restores their runtime inheritance', async (id) => {
    const config = declaration(id)
    const policy = compactionConfig(config)
    delete policy.summarizationProvider
    delete policy.summarizationModel
    delete policy.summarizationFallback
    delete policy.maxTokens
    const { ctx, start, profile } = await fixture({ declarations: [{
      id: `preset-${id}`, name: '@deepseek-ai/dsh-agent-preset', config,
    }] })
    const snapshot = ctx.compactionPolicyControl.read(`preset-${id}`)
    expect(snapshot.policy).toEqual({})
    await ctx.compactionPolicyControl.setSummaryTarget(request(ctx, id))
    const old = await agent(ctx, `restore-${id}-old`, id)
    const retained = engine(ctx, old).config
    const receipt = await ctx.compactionPolicyControl.restoreSummaryPolicy(restoration(ctx, snapshot))
    expect(receipt).toEqual({ ...snapshot, persisted: true, activated: true })
    const actual = ctx.configEditor.entries().find(row => row.options.id === snapshot.entryId)!.options.config as PresetDefinition
    for (const key of ['summarizationProvider', 'summarizationModel', 'summarizationFallback', 'maxTokens']) {
      expect(Object.hasOwn(compactionConfig(actual), key)).toBe(false)
    }
    expect(engine(ctx, old).config).toBe(retained)
    expect(retained.summarizationModel).toBe('replacement')
    expect(engine(ctx, await agent(ctx, `restore-${id}-fresh`, id)).config).toMatchObject({
      summarizationProvider: '', summarizationModel: '', summarizationFallback: 'latest-route',
    })
    const bytes = readFileSync(profile.patchPath, 'utf8')
    await ctx.fiber.dispose()
    const restarted = await start()
    expect(restarted.compactionPolicyControl.read(snapshot.entryId)).toEqual(snapshot)
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
  })

  it('restores heterogeneous raw leaves without checking old target availability or replacing fresh pressure and routes', async () => {
    const { ctx, adapter, profile } = await fixture()
    const snapshot = ctx.compactionPolicyControl.read('preset-standard')
    await ctx.compactionPolicyControl.setSummaryTarget(request(ctx))
    const current = structuredClone(
      ctx.configEditor.entries().find(row => row.options.id === snapshot.entryId)!.options.config,
    ) as PresetDefinition
    const currentPolicy = compactionConfig(current)
    currentPolicy.headroomTokens = 777
    currentPolicy.retainTokens = 99
    currentPolicy.modelPolicies![0]!.headroomTokens = 512
    currentPolicy.modelPolicies![1]!.summarizationModel = 'newer-exception'
    currentPolicy.modelPolicies!.push({ provider: 'conversation', model: 'new-route', maxTokens: 20, summarizationFallback: 'latest-route' })
    writeFileSync(profile.patchPath, JSON.stringify([{ id: snapshot.entryId, config: current }]) + '\n')
    await reconcileProfilePatches(ctx, readProfilePatches('test', profile), 'test')
    const old = await agent(ctx, 'restore-pressure-old')
    const retained = engine(ctx, old).config
    await using previousScope = await ctx.agentPresets.acquireScope('standard')
    adapter.availability.set('original', 'unavailable')
    adapter.availability.set('owned-original', 'unavailable')
    const lookups = new Map(adapter.lookups)
    const receipt = await ctx.compactionPolicyControl.restoreSummaryPolicy(restoration(ctx, snapshot))
    expect(adapter.lookups).toEqual(lookups)
    expect(adapter.inferenceCalls).toBe(0)
    expect(receipt.policy).toEqual(snapshot.policy)
    expect(receipt.modelPolicies[0]).toEqual(snapshot.modelPolicies[0])
    expect(receipt.modelPolicies[1]?.policy.summarizationModel).toBe('newer-exception')
    expect(receipt.modelPolicies[2]?.policy).toEqual({ maxTokens: 20, summarizationFallback: 'latest-route' })
    const expected = structuredClone(current)
    const expectedPolicy = compactionConfig(expected)
    Object.assign(expectedPolicy, snapshot.policy)
    const selected = expectedPolicy.modelPolicies![0]!
    delete selected.summarizationFallback
    Object.assign(selected, snapshot.modelPolicies[0]!.policy)
    const actual: unknown = ctx.configEditor.entries().find(row => row.options.id === snapshot.entryId)!.options.config
    expect(actual).toEqual(expected)
    await using nextScope = await ctx.agentPresets.acquireScope('standard')
    expect(nextScope.key).not.toBe(previousScope.key)
    expect(engine(ctx, old).config).toBe(retained)
    expect(retained).toMatchObject({ summarizationModel: 'replacement', headroomTokens: 777 })
    expect(engine(ctx, await agent(ctx, 'restore-pressure-fresh')).config).toMatchObject({
      summarizationModel: 'original', summarizationFallback: 'latest-route', headroomTokens: 777, retainTokens: 99,
    })
  })

  it('serializes overlapping restores and rejects the stale current revision without losing the winner', async () => {
    const { ctx, activation, profile } = await fixture()
    const snapshot = ctx.compactionPolicyControl.read('preset-standard')
    await ctx.compactionPolicyControl.setSummaryTarget(request(ctx))
    const restore = restoration(ctx, snapshot)
    const entered = barrier()
    const release = barrier()
    activation.block = { model: 'original', entered, release }
    const first = ctx.compactionPolicyControl.restoreSummaryPolicy(restore)
    const pending = [first]
    onTestFinished(async () => { release.resolve(); await Promise.allSettled(pending) })
    await entered.promise
    const second = ctx.compactionPolicyControl.restoreSummaryPolicy(restore)
    pending.push(second)
    const rejection = expect(second).rejects.toMatchObject({ code: 'compaction-policy/conflict' })
    release.resolve()
    await expect(first).resolves.toMatchObject({ ...snapshot, persisted: true, activated: true })
    await rejection
    const bytes = readFileSync(profile.patchPath, 'utf8')
    await expect(ctx.compactionPolicyControl.restoreSummaryPolicy(restore)).rejects.toMatchObject({ code: 'compaction-policy/conflict' })
    expect(ctx.compactionPolicyControl.read(snapshot.entryId)).toEqual(snapshot)
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
  })

  it('rejects a tampered restore snapshot before changing the persisted policy or runtime', async () => {
    const { ctx, profile } = await fixture()
    const snapshot = ctx.compactionPolicyControl.read('preset-standard')
    await ctx.compactionPolicyControl.setSummaryTarget(request(ctx))
    const before = ctx.compactionPolicyControl.read(snapshot.entryId)
    const bytes = readFileSync(profile.patchPath, 'utf8')
    const tampered = { ...snapshot, policy: { ...snapshot.policy, maxTokens: 65 } }
    await expect(ctx.compactionPolicyControl.restoreSummaryPolicy(restoration(ctx, tampered))).rejects.toMatchObject({ code: 'compaction-policy/invalid' })
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
    expect(ctx.compactionPolicyControl.read(snapshot.entryId)).toEqual(before)
    expect(engine(ctx, await agent(ctx, 'restore-tampered-fresh')).config.summarizationModel).toBe('replacement')
  })

  it('refuses a higher overlay without claiming that the captured policy was restored', async () => {
    const original = await fixture()
    const snapshot = original.ctx.compactionPolicyControl.read('preset-standard')
    const overlay = declaration('standard')
    compactionConfig(overlay).summarizationModel = 'overlay-target'
    const { ctx, profile } = await fixture({ overlays: [{ id: snapshot.entryId, config: overlay }] })
    const before = ctx.compactionPolicyControl.read(snapshot.entryId)
    const bytes = readFileSync(profile.patchPath, 'utf8')
    await expect(ctx.compactionPolicyControl.restoreSummaryPolicy(restoration(ctx, snapshot))).rejects.toMatchObject({ code: 'compaction-policy/rejected' })
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
    expect(ctx.compactionPolicyControl.read(snapshot.entryId)).toEqual(before)
    expect(engine(ctx, await agent(ctx, 'restore-overlay-fresh')).config.summarizationModel).toBe('overlay-target')
  })

  it('rolls back exact prior bytes and runtime when the restored generation fails actual child activation', async () => {
    const { ctx, activation, profile } = await fixture()
    const snapshot = ctx.compactionPolicyControl.read('preset-standard')
    await ctx.compactionPolicyControl.setSummaryTarget(request(ctx))
    const old = await agent(ctx, 'restore-activation-old')
    const retained = engine(ctx, old).config
    const before = ctx.compactionPolicyControl.read(snapshot.entryId)
    const bytes = readFileSync(profile.patchPath, 'utf8')
    activation.rejectModel = 'original'
    await expect(ctx.compactionPolicyControl.restoreSummaryPolicy(restoration(ctx, snapshot))).rejects.toMatchObject({ code: 'compaction-policy/rejected' })
    expect(activation.refused).toBeGreaterThan(0)
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
    expect(ctx.compactionPolicyControl.read(snapshot.entryId)).toEqual(before)
    expect(engine(ctx, old).config).toBe(retained)
    expect(engine(ctx, await agent(ctx, 'restore-activation-fresh')).config.summarizationModel).toBe('replacement')
  })

  it('rolls back a caller cancellation that arrives during restored-generation activation', async () => {
    const { ctx, activation, profile } = await fixture()
    const snapshot = ctx.compactionPolicyControl.read('preset-standard')
    await ctx.compactionPolicyControl.setSummaryTarget(request(ctx))
    const old = await agent(ctx, 'restore-cancel-old')
    const retained = engine(ctx, old).config
    const before = ctx.compactionPolicyControl.read(snapshot.entryId)
    const bytes = readFileSync(profile.patchPath, 'utf8')
    const entered = barrier()
    const release = barrier()
    activation.block = { model: 'original', entered, release }
    const controller = new AbortController()
    const pending = ctx.compactionPolicyControl.restoreSummaryPolicy(restoration(ctx, snapshot), controller.signal)
    onTestFinished(async () => {
      controller.abort('Restore test cleanup')
      release.resolve()
      await Promise.allSettled([pending])
    })
    const rejection = expect(pending).rejects.toMatchObject({ code: 'compaction-policy/rejected' })
    await entered.promise
    try {
      expect(readFileSync(profile.patchPath, 'utf8')).not.toBe(bytes)
      expect(ctx.compactionPolicyControl.read(snapshot.entryId).policy.summarizationModel).toBe('original')
      controller.abort('Owned restore caller stopped')
    } finally { release.resolve() }
    await rejection
    expect(readFileSync(profile.patchPath, 'utf8')).toBe(bytes)
    expect(ctx.compactionPolicyControl.read(snapshot.entryId)).toEqual(before)
    expect(engine(ctx, old).config).toBe(retained)
    expect(engine(ctx, await agent(ctx, 'restore-cancel-fresh')).config.summarizationModel).toBe('replacement')
  })
})
