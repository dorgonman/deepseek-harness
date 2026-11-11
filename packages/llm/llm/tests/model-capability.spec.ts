/** Direct and prepared requests enforce exact model capabilities before provider dispatch. */
import { afterEach, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { AttachmentId } from '@deepseek-ai/dsh-attachment'
import LlmRuntime, { LlmAdapter } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'

const contexts: Context[] = []
afterEach(async () => {
  for (const ctx of contexts.splice(0)) await ctx.fiber.dispose()
})

class CapabilityAdapter extends LlmAdapter {
  readonly requests: GenerateOptions[] = []
  override resolveModel(provider: string, model: string) {
    return Promise.resolve({ provider, id: model, name: model, inputModalities: ['text' as const],
      ...model === 'ordinary' ? {} : { nativeTools: false },
      ...model === 'decision' ? { inputProjectionPolicy: 'reject' as const } : {},
      ...model === 'history' ? { toolUpdate: 'addition-only' as const } : {},
    })
  }
  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.requests.push(options)
    yield { type: 'finish', reason: { kind: 'stop' } }
  }
}

const image: GenerateOptions['messages'] = [{ role: 'user', content: [{ type: 'image', attachment: {
  attachmentId: AttachmentId(`sha256:${'a'.repeat(64)}`), mediaType: 'image/png', bytes: 1, width: 1, height: 1,
} }] }]
const tools = [{ name: 'echo', description: 'Echo.', parameters: {} }]
const file: GenerateOptions['messages'] = [{ role: 'user', content: [{ type: 'file', attachment: {
  attachmentId: AttachmentId(`sha256:${'b'.repeat(64)}`), name: 'evidence.txt', bytes: 1,
} }] }]

async function chunks(stream: AsyncIterable<StreamChunk>) {
  const result: StreamChunk[] = []
  for await (const chunk of stream) result.push(chunk)
  return result
}

it.each(['direct', 'prepared'] as const)('%s Decision calls reject tools and images before dispatch', async (mode) => {
  const ctx = new Context()
  contexts.push(ctx)
  await ctx.plugin(LlmRuntime)
  const adapter = new CapabilityAdapter()
  ctx.llm.registerAdapter(['fixture'], adapter)
  const stream = async (input: Pick<GenerateOptions, 'messages' | 'tools' | 'toolHistory'>, model = 'decision') => {
    const config = { provider: 'fixture', model }
    if (mode === 'direct') return chunks(ctx.llm.stream({ ...config, ...input }))
    const prepared = await ctx.llm.prepareCall(config)
    return chunks(prepared.stream({ ...prepared.config, ...input }))
  }
  expect(await stream({ messages: [], tools })).toMatchObject([
    { type: 'finish', reason: { kind: 'error', failure: { code: 'UNSUPPORTED_TOOLS' } } },
  ])
  expect(await stream({ messages: image })).toMatchObject([
    { type: 'finish', reason: { kind: 'error', failure: { code: 'UNSUPPORTED_CONTENT' } } },
  ])
  expect(await stream({ messages: file })).toMatchObject([
    { type: 'finish', reason: { kind: 'error', failure: { code: 'UNSUPPORTED_CONTENT' } } },
  ])
  expect(await stream({ messages: [], toolHistory: { tools, updates: [] } }, 'history')).toMatchObject([
    { type: 'finish', reason: { kind: 'error', failure: { code: 'UNSUPPORTED_TOOLS' } } },
  ])
  expect(adapter.requests).toEqual([])
  expect(await chunks(ctx.llm.stream({ provider: 'fixture', model: 'ordinary', messages: image, tools })))
    .toEqual([{ type: 'finish', reason: { kind: 'stop' } }])
  expect(adapter.requests[0]?.tools).toEqual(tools)
  expect(adapter.requests[0]?.messages[0]?.content[0]?.type).toBe('text')
})
