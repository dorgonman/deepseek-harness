import { Context } from '@deepseek-ai/cordis'
import { describe, expect, it, onTestFinished, vi } from 'vitest'
import { TestRemote } from '@deepseek-ai/dsh-client-test-runtime'
import { apply, inject } from '../src/client/index.ts'
import { SettingsSchemaService } from '../src/client/schema.ts'
import { ConfigForms } from '../src/client/config-form.ts'

function bench() {
  const describeCall = vi.fn().mockResolvedValue({
    ok: true, value: { writable: true, hasDocument: true, namespaces: [] },
  })
  const ctx = new Context()
  const remote = new TestRemote(ctx, { settings: { describe: describeCall } })
  return { ctx, describeCall, remote, fiber: ctx.plugin({ inject: [...inject], apply }) }
}

describe('settings domain base plugin', () => {
  it('mounts the scope service under configForms and reads once eagerly', async () => {
    const { ctx, describeCall, fiber } = bench()
    await fiber.await()
    expect(ctx.get('configForms')).toBeInstanceOf(ConfigForms)
    expect(ctx.get('settingsSchema')).toBeInstanceOf(SettingsSchemaService)
    await vi.waitFor(() => { expect(describeCall).toHaveBeenCalledTimes(1) })
  })

  it('keeps non-loopback settings memory-only unless remote persistence is opted in', async () => {
    const { ctx, describeCall, remote, fiber } = bench()
    onTestFinished(async () => {
      await fiber.dispose()
      vi.unstubAllGlobals()
    })
    remote.$host = { home: undefined, isLoopback: false }
    vi.stubGlobal('localStorage', { getItem: vi.fn(() => null) })

    await fiber.await()
    await ctx.configForms.describe().ensure()

    expect(ctx.configForms.describe().getSnapshot().status).toBe('unavailable')
    expect(describeCall).not.toHaveBeenCalled()
  })

  it('keeps non-loopback settings memory-only when browser storage is blocked', async () => {
    const { ctx, describeCall, remote, fiber } = bench()
    onTestFinished(async () => {
      await fiber.dispose()
      vi.unstubAllGlobals()
    })
    remote.$host = { home: undefined, isLoopback: false }
    vi.stubGlobal('localStorage', {
      getItem: vi.fn(() => { throw new Error('storage is blocked') }),
    })

    await fiber.await()
    await ctx.configForms.describe().ensure()

    expect(ctx.configForms.describe().getSnapshot().status).toBe('unavailable')
    expect(describeCall).not.toHaveBeenCalled()
  })

  it('loads Host settings on a non-loopback origin with the explicit browser opt-in', async () => {
    const { ctx, describeCall, remote, fiber } = bench()
    onTestFinished(async () => {
      await fiber.dispose()
      vi.unstubAllGlobals()
    })
    remote.$host = { home: undefined, isLoopback: false }
    vi.stubGlobal('localStorage', {
      getItem: vi.fn((key: string) => key === 'dsh.settings.allowRemotePersistence' ? 'true' : null),
    })

    await fiber.await()
    await vi.waitFor(() => { expect(describeCall).toHaveBeenCalledOnce() })

    expect(ctx.configForms.describe().getSnapshot()).toMatchObject({ status: 'ready', view: { namespaces: [] } })
  })

  it('records explicit remote consent in this origin for the next page load', async () => {
    const { ctx, remote, fiber } = bench()
    onTestFinished(async () => {
      await fiber.dispose()
      vi.unstubAllGlobals()
    })
    remote.$host = { home: undefined, isLoopback: false }
    const setItem = vi.fn()
    vi.stubGlobal('localStorage', { getItem: vi.fn(() => null), setItem })

    await fiber.await()

    expect(ctx.configForms.enableRemotePersistence()).toBe(true)
    expect(setItem).toHaveBeenCalledWith('dsh.settings.allowRemotePersistence', 'true')
    expect(ctx.configForms.describe().getSnapshot().status).toBe('unavailable')
  })

  it('keeps remote consent disabled when browser storage refuses the write', async () => {
    const { ctx, remote, fiber } = bench()
    onTestFinished(async () => {
      await fiber.dispose()
      vi.unstubAllGlobals()
    })
    remote.$host = { home: undefined, isLoopback: false }
    vi.stubGlobal('localStorage', {
      getItem: vi.fn(() => null),
      setItem: vi.fn(() => { throw new Error('storage is blocked') }),
    })

    await fiber.await()

    expect(ctx.configForms.enableRemotePersistence()).toBe(false)
    expect(ctx.configForms.describe().getSnapshot().status).toBe('unavailable')
  })

  it('refreshes the mirror on document commits and connection resets, once each', async () => {
    const { ctx, describeCall, remote, fiber } = bench()
    await fiber.await()
    await vi.waitFor(() => { expect(describeCall).toHaveBeenCalledTimes(1) })
    remote.emit('settings/document-updated', ['ui-test', 0])
    await vi.waitFor(() => { expect(describeCall).toHaveBeenCalledTimes(2) })
    ctx.emit('connection/reset')
    await vi.waitFor(() => { expect(describeCall).toHaveBeenCalledTimes(3) })
  })

  it('fiber disposal retires the service and its invalidation subscriptions', async () => {
    const { ctx, describeCall, remote, fiber } = bench()
    await fiber.await()
    await vi.waitFor(() => { expect(describeCall).toHaveBeenCalledTimes(1) })
    await fiber.dispose()
    expect(ctx.get('configForms')).toBeUndefined()
    expect(ctx.get('settingsSchema')).toBeUndefined()
    remote.emit('settings/document-updated', ['ui-test', 0])
    ctx.emit('connection/reset')
    await Promise.resolve()
    expect(describeCall).toHaveBeenCalledTimes(1)
  })
})
