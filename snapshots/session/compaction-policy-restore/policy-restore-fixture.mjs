/** Exercise four persisted preset-policy restores before the recorded model turn. */
import assert from 'node:assert/strict'
import { readFile, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { basename, dirname, isAbsolute, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { load } from 'js-yaml'

const PRESETS = ['standard', 'ptc', 'minimal', 'cordis']
const SUMMARY_FIELDS = ['summarizationProvider', 'summarizationModel', 'summarizationFallback', 'maxTokens']

/** Cordis plugin name. */
export const name = 'policy-restore-fixture'
/** Existing owners of the isolated profile, policy transaction, and scoped prompt. */
export const inject = ['configEditor', 'profileContext', 'compactionPolicyControl', 'systemPrompt', 'workingDirectory']

function declaration(id) {
  return {
    id: `preset-${id}`,
    name: '@deepseek-ai/dsh-agent-preset',
    config: {
      id,
      name: `Restore fixture ${id}`,
      description: 'Literal profile-owned restore fixture.',
      plugins: [{
        id: 'compaction',
        name: 'cordis:group',
        group: true,
        isolate: { compaction: true },
        config: [{
          id: 'compaction-basic',
          name: '@deepseek-ai/dsh-compaction-basic',
          config: { thresholdRatio: 0.75, headroomTokens: 256, retainTokens: 32, compactionRetries: 2, auto: false },
        }],
      }],
    },
  }
}

function rawDeclaration(ctx, entryId) {
  const entries = ctx.configEditor.entries().filter(entry => entry.options.id === entryId)
  assert.equal(entries.length, 1)
  assert.equal(entries[0].options.name, '@deepseek-ai/dsh-agent-preset')
  return entries[0].options.config
}

function assertAbsentSummary(declarationConfig) {
  const group = declarationConfig.plugins.find(row => row.id === 'compaction')
  assert.ok(group)
  const backend = group.config.find(row => row.id === 'compaction-basic')
  assert.ok(backend)
  for (const field of SUMMARY_FIELDS) assert.equal(Object.hasOwn(backend.config, field), false, field)
}

/**
 * Seed only an empty harness-owned profile, then use the real set and restore owners.
 * @param {import('@deepseek-ai/cordis').Context} ctx - owner of the profile and created-agent listener.
 */
export function apply(ctx) {
  let completed = false
  let modelCalls = 0
  ctx.on('agent/created', async ({ agent }) => {
    assert.equal(completed, false, 'The scenario creates exactly one root Agent')
    agent.ctx.systemPrompt.context({
      name: 'working-directory:current',
      order: ctx.systemPrompt.getContextOrder('WORKING_DIRECTORY'),
      required: true,
      interpolate: false,
      text: 'Current working directory: the controlled snapshot workspace.',
    })

    const profile = ctx.profileContext
    assert.match(basename(process.cwd()), /^dsh-log-snap-/u)
    const expectedHome = resolve(process.cwd(), '.dsh')
    const expectedProfile = join(expectedHome, 'profiles', 'headless')
    assert.equal(profile.name, 'headless')
    assert.equal(resolve(profile.home), expectedHome)
    assert.equal(resolve(profile.dir), expectedProfile)
    assert.equal(resolve(profile.patchPath), join(expectedProfile, 'cordis.patch.yml'))
    assert.ok(process.env.DSH_SNAPSHOT === 'refresh' || process.env.DSH_SNAPSHOT === 'replay')
    assert.deepEqual(load(await readFile(profile.patchPath, 'utf8')), [])

    assert.equal(isAbsolute(profile.installAnchor), true)
    assert.equal(basename(profile.installAnchor), 'package.json')
    assert.equal(JSON.parse(await readFile(profile.installAnchor, 'utf8')).name, '@deepseek-ai/dsh')
    const runtimeRequire = createRequire(profile.installAnchor)
    const appBootManifestPath = runtimeRequire.resolve('@deepseek-ai/dsh-app-boot/package.json')
    const appBootManifest = JSON.parse(await readFile(appBootManifestPath, 'utf8'))
    assert.equal(appBootManifest.name, '@deepseek-ai/dsh-app-boot')
    assert.equal(appBootManifest.exports['.'].default, './lib/index.js')
    const appBootEntry = runtimeRequire.resolve('@deepseek-ai/dsh-app-boot')
    assert.equal(appBootEntry, resolve(dirname(appBootManifestPath), appBootManifest.exports['.'].default))
    const { readProfilePatches, reconcileProfilePatches } = await import(pathToFileURL(appBootEntry).href)
    assert.equal(typeof readProfilePatches, 'function')
    assert.equal(typeof reconcileProfilePatches, 'function')

    const declarations = PRESETS.map(declaration)
    const initialDocument = [{ insert: declarations }]
    await writeFile(profile.patchPath, JSON.stringify(initialDocument, null, 2) + '\n', { mode: 0o600 })
    await reconcileProfilePatches(ctx.root, readProfilePatches('snapshot', profile), 'snapshot')

    const control = ctx.compactionPolicyControl
    for (const expected of declarations) {
      const beforeDeclaration = structuredClone(rawDeclaration(ctx, expected.id))
      assert.deepEqual(beforeDeclaration, expected.config)
      assertAbsentSummary(beforeDeclaration)
      const before = control.read(expected.id)
      assert.deepEqual(before.policy, {})
      assert.deepEqual(before.modelPolicies, [])
      const applied = await control.setSummaryTarget({
        entryId: expected.id,
        expectedRevision: before.revision,
        target: { provider: 'policy-restore', model: 'summary', maxTokens: 96 },
        modelPolicyRoutes: [],
      })
      assert.equal(applied.persisted, true)
      assert.equal(applied.activated, true)
      assert.notEqual(applied.revision, before.revision)
      assert.deepEqual(applied.policy, {
        summarizationProvider: 'policy-restore', summarizationModel: 'summary',
        summarizationFallback: 'error', maxTokens: 96,
      })
      const restored = await control.restoreSummaryPolicy({
        expectedRevision: applied.revision,
        snapshot: before,
        modelPolicyRoutes: [],
      })
      assert.equal(restored.persisted, true)
      assert.equal(restored.activated, true)
      assert.deepEqual(control.read(expected.id), before)
      const afterDeclaration = rawDeclaration(ctx, expected.id)
      assert.deepEqual(afterDeclaration, beforeDeclaration)
      assertAbsentSummary(afterDeclaration)
    }
    assert.deepEqual(load(await readFile(profile.patchPath, 'utf8')), initialDocument)
    completed = true
  })

  ctx.on('llm/stream', (options, next) => {
    assert.equal(completed, true, 'All four policy restores finish before the recorded model call')
    assert.equal(options.provider, 'policy-restore')
    assert.equal(options.model, 'conversation')
    modelCalls++
    return next()
  })
  ctx.effect(() => () => {
    assert.equal(completed, true)
    assert.equal(modelCalls, 1)
  })
}
