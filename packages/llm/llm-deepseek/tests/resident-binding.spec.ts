/** A resident dispatch binding is refused before this transport performs any I/O. */
import { expect, it, vi } from 'vitest'
import { adapter, chunks, options } from './helpers.ts'

it('refuses a resident dispatch binding before any outbound request', async () => {
  const fetchImpl = vi.fn<typeof fetch>()
  vi.stubGlobal('fetch', fetchImpl)
  const residentBinding = {
    schema: 'kinf.resident-only.v1', instance: 'fixture-instance', hostGeneration: 'host-1',
    servingVariantId: 'variant-1', servingVariantRevision: 'variant-rev-1',
    modelArtifact: 'artifact-1', artifactRevision: 'artifact-rev-1',
    targetConfigRevision: 'config-1', residentGeneration: 'resident-1', projectionId: 'projection-1',
  }
  await expect(chunks(adapter().stream({ ...options(), residentBinding })))
    .rejects.toMatchObject({ code: 'UNSUPPORTED_RESIDENT_TRANSPORT' })
  expect(fetchImpl).not.toHaveBeenCalled()
})
