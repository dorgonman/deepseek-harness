import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { boot, initProfile, readProfilePatches, type ProfileContext } from '@deepseek-ai/dsh-app-boot'
import { expect, it, onTestFinished } from 'vitest'
import ConfigEditor from '../src/index.ts'

async function fixture() {
  const home = realpathSync(mkdtempSync(join(tmpdir(), 'config-editor-test-')))
  const scope: { ctx?: Context } = {}
  onTestFinished(async () => {
    try { await scope.ctx?.fiber.dispose() }
    finally { rmSync(home, { recursive: true, force: true }) }
  })
  const dir = join(home, 'profiles', 'test')
  initProfile(dir, ['test-bundle'])
  const bundle = join(dir, 'node_modules', 'test-bundle')
  mkdirSync(bundle, { recursive: true })
  writeFileSync(join(home, 'package.json'), '{"name":"config-editor-test"}\n')
  writeFileSync(join(bundle, 'package.json'), JSON.stringify({ name: 'test-bundle', version: '1.0.0', dsh: { bundle: { patch: 'cordis.patch.yml' } } }))
  writeFileSync(join(bundle, 'cordis.patch.yml'), JSON.stringify([{ insert: [
    { id: 'config-editor', name: 'cordis:editor' },
    { id: 'probe', name: 'cordis:probe', config: { value: 1 } },
  ] }]))
  writeFileSync(join(dir, 'cordis.yml'), '[]\n')
  const profile: ProfileContext = {
    name: 'test', startedBundles: ['test-bundle'], dir, patchPath: join(dir, 'cordis.patch.yml'),
    installAnchor: join(home, 'package.json'), cwd: home, home, overlays: [], telemetryDisabledEnv: undefined,
  }
  const observed: { config?: { value: number } } = {}
  const Probe = {
    Config: z.object({ value: z.number().required() }),
    apply(child: Context, config: { value: number }) {
      child.effect(() => {
        observed.config = config
        return () => { if (observed.config === config) delete observed.config }
      })
    },
  }
  const ctx = await boot('test', join(dir, 'cordis.yml'), readProfilePatches('test', profile), (owner) => {
    scope.ctx = owner
    owner.provide('profileContext', profile)
    Object.assign(owner.loader.builtins, { editor: ConfigEditor, probe: Probe })
  })
  const entry = ctx.configEditor.entries().find(row => row.options.id === 'probe')!
  return { ctx, entry, observed, profile }
}

it('awaits activation verification after the edited runtime is available', async () => {
  const { ctx, entry, observed, profile } = await fixture()
  let verified = false
  await ctx.configEditor.edit(entry, current => ({ ...current, value: 2 }), async () => {
    expect(observed.config?.value).toBe(2)
    expect(readFileSync(profile.patchPath, 'utf8')).toContain('value: 2')
    await Promise.resolve()
    verified = true
  })
  expect(verified).toBe(true)
  expect(observed.config?.value).toBe(2)
})

it('restores the prior profile bytes and runtime when verification rejects', async () => {
  const { ctx, entry, observed, profile } = await fixture()
  const before = readFileSync(profile.patchPath, 'utf8')
  const refusal = new Error('domain activation refused')
  await expect(ctx.configEditor.edit(entry, current => ({ ...current, value: 2 }), () => {
    expect(observed.config?.value).toBe(2)
    throw refusal
  })).rejects.toBe(refusal)
  expect(readFileSync(profile.patchPath, 'utf8')).toBe(before)
  expect(observed.config?.value).toBe(1)
  expect(entry.options.config).toEqual({ value: 1 })
})

it('holds the profile lock through verification before deriving the next edit', async () => {
  const { ctx, entry, observed } = await fixture()
  const entered = Promise.withResolvers<undefined>()
  const release = Promise.withResolvers<undefined>()
  const order: string[] = []
  const first = ctx.configEditor.edit(entry, current => ({ ...current, value: 2 }), async () => {
    order.push('first-entered')
    entered.resolve(undefined)
    await release.promise
    order.push('first-verified')
  })
  await entered.promise
  let secondValue: unknown
  const second = ctx.configEditor.edit(entry, (current) => {
    order.push('second-change')
    secondValue = current.value
    return { ...current, value: 3 }
  }, () => { order.push('second-verified') })
  try { expect(secondValue).toBeUndefined() }
  finally { release.resolve(undefined) }
  await Promise.all([first, second])
  expect(secondValue).toBe(2)
  expect(order).toEqual(['first-entered', 'first-verified', 'second-change', 'second-verified'])
  expect(observed.config?.value).toBe(3)
})
