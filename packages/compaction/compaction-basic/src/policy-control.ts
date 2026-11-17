/** Revision-checked summary-policy edits through the profile configuration owner. */
import { createHash } from 'node:crypto'
import { isDeepStrictEqual } from 'node:util'
import { Context } from '@deepseek-ai/cordis'
import type { Entry } from '@deepseek-ai/cordis-plugin-loader'
import type {} from '@deepseek-ai/dsh-config-editor'
import type {} from '@deepseek-ai/dsh-llm'
import { Remote, RemoteError, TypertRemoteService } from '@deepseek-ai/dsh-typert-protocol'
import { resolveConfig } from './config.ts'
import type { BasicCompactionConfig, CompactionPolicyConfig, ResolvedConfig } from './types.ts'
import type {
  CompactionSummaryPolicy,
  PresetCompactionPolicy,
  PresetCompactionPolicyRestore,
  PresetCompactionPolicyUpdate,
  PresetCompactionPolicyWrite,
} from './policy-types.ts'

declare module '@deepseek-ai/cordis' {
  interface Context {
    /** Revision-checked edits to profile-owned preset summary policies. */
    compactionPolicyControl: CompactionPolicyControl
  }
}

function invalid(entryId: string, reason: string): never {
  throw new RemoteError('compaction-policy/invalid', reason, { entryId, reason })
}

function object(value: unknown, entryId: string, label: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) invalid(entryId, `${label} must be a literal object`)
  return value as Record<string, unknown>
}

function row(value: unknown, id: string, entryId: string): Record<string, unknown> {
  if (!Array.isArray(value)) invalid(entryId, `The ${id} parent must contain plugin rows`)
  const matches = value.filter((item: unknown) => typeof item === 'object' && item !== null
    && !Array.isArray(item) && 'id' in item && item.id === id)
  if (matches.length !== 1) invalid(entryId, `Expected one ${id} plugin row`)
  return object(matches[0], entryId, id)
}

function engine(current: Record<string, unknown>, entryId: string): {
  presetId: string
  node: Record<string, unknown>
  config: BasicCompactionConfig
} {
  if (typeof current.id !== 'string' || current.id.trim().length === 0) invalid(entryId, 'The declaration has no preset id')
  const group = row(current.plugins, 'compaction', entryId)
  if (group.group !== true || group.name !== 'cordis:group') invalid(entryId, 'The compaction row must be a Cordis group')
  if (group.disabled !== undefined && group.disabled !== false) invalid(entryId, 'The compaction group must be unconditionally enabled')
  const node = row(group.config, 'compaction-basic', entryId)
  if (node.name !== '@deepseek-ai/dsh-compaction-basic') invalid(entryId, 'The summary policy must belong to compaction-basic')
  if (node.disabled !== undefined && node.disabled !== false) invalid(entryId, 'The compaction backend must be unconditionally enabled')
  const config = object(node.config === undefined ? {} : node.config, entryId, 'compaction-basic config') as BasicCompactionConfig
  try { resolveConfig(config) }
  catch (error) {
    invalid(entryId, error instanceof Error ? error.message : String(error))
  }
  return { presetId: current.id, node, config }
}

const SUMMARY_FIELDS = ['summarizationProvider', 'summarizationModel', 'summarizationFallback', 'maxTokens'] as const

function policy(config: CompactionPolicyConfig, entryId: string): CompactionSummaryPolicy {
  for (const field of SUMMARY_FIELDS) {
    if (Object.hasOwn(config, field) && config[field] === undefined) {
      invalid(entryId, `Summary field ${field} must be absent or contain a literal value`)
    }
  }
  return {
    ...!Object.hasOwn(config, 'summarizationProvider') || config.summarizationProvider === undefined ? {} : { summarizationProvider: config.summarizationProvider },
    ...!Object.hasOwn(config, 'summarizationModel') || config.summarizationModel === undefined ? {} : { summarizationModel: config.summarizationModel },
    ...!Object.hasOwn(config, 'summarizationFallback') || config.summarizationFallback === undefined ? {} : { summarizationFallback: config.summarizationFallback },
    ...!Object.hasOwn(config, 'maxTokens') || config.maxTokens === undefined ? {} : { maxTokens: config.maxTokens },
  }
}

function revision(entryId: string, presetId: string, fields: Pick<PresetCompactionPolicy, 'policy' | 'modelPolicies'>): string {
  return createHash('sha256').update(JSON.stringify({ entryId, presetId, ...fields })).digest('hex')
}

/** Read only summary fields from a validated literal preset declaration.
 * @param entryId Profile row identifying this declaration.
 * @param current Detached raw declaration configuration.
 * @returns Raw default and exact-route leaves with their content revision; omitted leaves remain absent.
 * @throws RemoteError if the declaration is missing or ambiguous.
 */
export function inspectPresetCompactionPolicy(entryId: string, current: Record<string, unknown>): PresetCompactionPolicy {
  const selected = engine(current, entryId)
  const fields = {
    policy: policy(selected.config, entryId),
    modelPolicies: (selected.config.modelPolicies ?? []).map(item => ({
      provider: item.provider, model: item.model, policy: policy(item, entryId),
    })),
  }
  return {
    entryId, presetId: selected.presetId,
    revision: revision(entryId, selected.presetId, fields),
    ...fields,
  }
}

/** Change summary leaves on an editor-owned clone after checking its fresh revision.
 * @param current Fresh detached declaration supplied inside ConfigEditor's lock.
 * @param request Explicit target and existing route overrides the caller owns.
 * @returns The resulting summary fields; unrelated plugin and policy values remain unchanged.
 * @throws RemoteError on stale revision, invalid fields or missing route overrides.
 */
export function updatePresetCompactionPolicy(
  current: Record<string, unknown>, request: PresetCompactionPolicyUpdate,
): PresetCompactionPolicy {
  const before = inspectPresetCompactionPolicy(request.entryId, current)
  if (before.revision !== request.expectedRevision) {
    throw new RemoteError('compaction-policy/conflict', 'Preset summary policy changed; read it again before saving', {
      entryId: request.entryId, expectedRevision: request.expectedRevision, actualRevision: before.revision,
    })
  }
  if (request.target.provider.trim().length === 0 || request.target.model.trim().length === 0
    || !Number.isSafeInteger(request.target.maxTokens) || request.target.maxTokens < 1) {
    invalid(request.entryId, 'Summary target needs an exact provider/model and a positive integer output budget')
  }
  const selected = engine(current, request.entryId)
  const target = {
    summarizationProvider: request.target.provider,
    summarizationModel: request.target.model,
    summarizationFallback: 'error' as const,
    maxTokens: request.target.maxTokens,
  }
  const keys = new Set<string>()
  for (const route of request.modelPolicyRoutes) {
    const key = JSON.stringify([route.provider, route.model])
    if (keys.has(key)) invalid(request.entryId, 'Requested model-policy routes must be unique')
    keys.add(key)
    const matches = (selected.config.modelPolicies ?? []).filter(item => item.provider === route.provider && item.model === route.model)
    const match = matches[0]
    if (matches.length !== 1 || match === undefined) invalid(request.entryId, 'Requested model-policy route is absent or ambiguous')
    Object.assign(match, target)
  }
  Object.assign(selected.config, target)
  resolveConfig(selected.config)
  selected.node.config = selected.config
  return inspectPresetCompactionPolicy(request.entryId, current)
}

/** Restore captured summary leaves on a fresh editor clone without checking target availability.
 * @param current Fresh detached declaration supplied inside ConfigEditor's lock.
 * @param request Current revision, unmodified prior snapshot and explicitly owned route keys.
 * @returns Restored raw summary leaves; unrelated current configuration remains unchanged.
 * @throws RemoteError on stale revision, changed identity, invalid snapshot or missing route overrides.
 */
export function restorePresetCompactionPolicy(
  current: Record<string, unknown>, request: PresetCompactionPolicyRestore,
): PresetCompactionPolicy {
  const { snapshot } = request
  const before = inspectPresetCompactionPolicy(snapshot.entryId, current)
  if (before.revision !== request.expectedRevision || before.presetId !== snapshot.presetId) {
    throw new RemoteError('compaction-policy/conflict', 'Preset summary policy or identity changed; read it again before restoring', {
      entryId: snapshot.entryId, expectedRevision: request.expectedRevision, actualRevision: before.revision,
    })
  }
  const captured = {
    policy: policy(snapshot.policy, snapshot.entryId),
    modelPolicies: snapshot.modelPolicies.map(item => ({
      provider: item.provider, model: item.model, policy: policy(item.policy, snapshot.entryId),
    })),
  }
  if (revision(snapshot.entryId, snapshot.presetId, captured) !== snapshot.revision) {
    invalid(snapshot.entryId, 'The captured summary snapshot was modified')
  }
  const selected = engine(current, snapshot.entryId)
  const proposed = structuredClone(selected.config)
  const keys = new Set<string>()
  const overrides = request.modelPolicyRoutes.map((route) => {
    const key = JSON.stringify([route.provider, route.model])
    if (keys.has(key)) invalid(snapshot.entryId, 'Requested model-policy routes must be unique')
    keys.add(key)
    const matches = (proposed.modelPolicies ?? []).filter(item => item.provider === route.provider && item.model === route.model)
    const prior = captured.modelPolicies.filter(item => item.provider === route.provider && item.model === route.model)
    const match = matches[0]
    const previous = prior[0]
    if (matches.length !== 1 || prior.length !== 1 || match === undefined || previous === undefined) {
      invalid(snapshot.entryId, 'Requested model-policy route is absent or ambiguous in the current policy or captured snapshot')
    }
    return { current: match, prior: previous.policy }
  })
  const restore = (destination: CompactionPolicyConfig, source: CompactionSummaryPolicy): void => {
    delete destination.summarizationProvider
    delete destination.summarizationModel
    delete destination.summarizationFallback
    delete destination.maxTokens
    Object.assign(destination, source)
  }
  restore(proposed, captured.policy)
  for (const item of overrides) restore(item.current, item.prior)
  try { resolveConfig(proposed) }
  catch (error) { invalid(snapshot.entryId, error instanceof Error ? error.message : String(error)) }
  selected.node.config = proposed
  return inspectPresetCompactionPolicy(snapshot.entryId, current)
}

/** Host-only policy control mounted by the preset owner when persistent editing exists. */
export class CompactionPolicyControl extends TypertRemoteService {
  static inject = ['configEditor']

  /**
   * @param ctx Host context providing the persistent profile editor.
   * @param options Private preset-owner verification, awaited inside editor rollback.
   */
  constructor(ctx: Context, private readonly options: {
    verifyActivation: (presetId: string, inspect: (policy: ResolvedConfig | undefined) => void) => Promise<void>
  }) {
    super(ctx, 'compactionPolicyControl', { namespace: 'compactionPolicy' })
  }

  /** Read one uniquely profile-owned preset without exposing other plugin configuration.
   * @param entryId Exact declaration row id in the active profile.
   * @returns Current summary fields and their revision.
   * @throws RemoteError when the declaration cannot be addressed safely.
   */
  @Remote
  read(entryId: string): PresetCompactionPolicy {
    const entry = this.entry(entryId)
    return inspectPresetCompactionPolicy(entryId, object(entry.options.config, entryId, 'Preset config'))
  }

  /** Persist one explicit summary target and verify the replacement preset before committing success.
   * @param request Expected revision, target and explicitly owned existing overrides.
   * @param signal Optional cancellation; absence permits the operation to finish without caller cancellation.
   * @returns Persisted and activated policy for new preset bindings; existing Agents retain their revision.
   * @throws RemoteError on stale policy, unverifiable target, persistence failure or failed activation.
   *   Editor failures restore the previous patch.
   */
  @Remote
  async setSummaryTarget(request: PresetCompactionPolicyUpdate, signal?: AbortSignal): Promise<PresetCompactionPolicyWrite> {
    const entry = this.entry(request.entryId)
    signal?.throwIfAborted()
    await this.verifyTarget(request, signal)
    return this.editPolicy(entry, request.entryId, current => updatePresetCompactionPolicy(current, request), signal,
      () => this.verifyTarget(request, signal))
  }

  /** Restore captured raw summary leaves and verify the replacement preset before committing success.
   * @param request Current revision, prior read snapshot and explicitly owned existing overrides.
   * @param signal Optional cancellation; absence permits completion without caller cancellation.
   * @returns Persisted and activated restored policy for new bindings; existing Agents retain their revision.
   * @throws RemoteError on stale policy, changed identity, invalid snapshot, persistence or activation failure.
   *   Restore validates configuration without requiring the captured target to be available and performs no inference.
   */
  @Remote
  async restoreSummaryPolicy(request: PresetCompactionPolicyRestore, signal?: AbortSignal): Promise<PresetCompactionPolicyWrite> {
    const entry = this.entry(request.snapshot.entryId)
    signal?.throwIfAborted()
    return this.editPolicy(entry, request.snapshot.entryId, current => restorePresetCompactionPolicy(current, request), signal)
  }

  private async editPolicy(
    entry: Entry, entryId: string, change: (current: Record<string, unknown>) => PresetCompactionPolicy,
    signal?: AbortSignal, verifyTarget?: () => Promise<void>,
  ): Promise<PresetCompactionPolicyWrite> {
    let result: PresetCompactionPolicy | undefined
    let expectedPolicy: ResolvedConfig | undefined
    try {
      await this.ctx.configEditor.edit(entry, (current) => {
        signal?.throwIfAborted()
        if (this.entry(entryId) !== entry) throw new Error('Preset declaration changed before the policy edit')
        result = change(current)
        expectedPolicy = resolveConfig(engine(current, entryId).config)
        return current
      }, async () => {
        signal?.throwIfAborted()
        if (result === undefined) throw new Error('Preset summary policy was not derived')
        await this.options.verifyActivation(result.presetId, (actualPolicy) => {
          if (actualPolicy === undefined || !isDeepStrictEqual(actualPolicy, expectedPolicy)) {
            throw new Error('The replacement preset does not provide the requested active compaction policy')
          }
        })
        signal?.throwIfAborted()
        const actual = this.read(entryId)
        if (actual.revision !== result.revision) throw new Error('Activated summary policy differs from the persisted request')
        await verifyTarget?.()
      })
    } catch (error) {
      const code = typeof error === 'object' && error !== null && 'code' in error ? error.code : undefined
      if (code === 'compaction-policy/conflict' || code === 'compaction-policy/invalid' || code === 'compaction-policy/not-found') throw error
      throw new RemoteError('compaction-policy/rejected', 'Preset summary policy was not activated; inspect the profile before retrying', {
        entryId,
      }, { cause: error })
    }
    if (result === undefined) throw new Error('Preset summary policy result is unavailable')
    return { ...result, persisted: true, activated: true }
  }

  private async verifyTarget(request: PresetCompactionPolicyUpdate, signal?: AbortSignal): Promise<void> {
    const llm = this.ctx.get('llm')
    try {
      if (llm === undefined) throw new Error('The LLM service is not available')
      const info = await llm.resolveModelInfo(request.target.provider, request.target.model, signal)
      if (info.requestProtocol !== undefined || info.context === undefined
        || (info.inputModalities !== undefined && !info.inputModalities.includes('text'))
        || !Number.isSafeInteger(request.target.maxTokens) || request.target.maxTokens < 1
        || request.target.maxTokens >= info.context.contextWindow) {
        throw new Error('The summary target must be ordinary Chat with context remaining after its output budget')
      }
      signal?.throwIfAborted()
    } catch (error) {
      signal?.throwIfAborted()
      throw new RemoteError('compaction-policy/unverified', 'The requested summary target cannot be verified', {
        entryId: request.entryId, provider: request.target.provider, model: request.target.model,
      }, { cause: error })
    }
  }

  private entry(entryId: string): Entry {
    const entries = this.ctx.configEditor.entries().filter(item => item.options.id === entryId
      && item.options.name === '@deepseek-ai/dsh-agent-preset')
    const selected = entries[0]
    if (entries.length !== 1 || selected === undefined) {
      throw new RemoteError('compaction-policy/not-found', 'Preset declaration is absent or not uniquely profile-owned', { entryId })
    }
    return selected
  }
}
