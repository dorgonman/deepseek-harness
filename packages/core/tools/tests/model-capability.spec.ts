/** Exact model capability suppresses declarations, executor guidance, and execution together. */
import { afterEach, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import LlmRuntime, { LlmAdapter, ToolCallId } from '@deepseek-ai/dsh-llm'
import type { StreamChunk } from '@deepseek-ai/dsh-llm'
import { SessionId } from '@deepseek-ai/dsh-session'
import SystemPrompt, { renderPrompt } from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime, { defineTool } from '@deepseek-ai/dsh-tools'

const contexts: Context[] = []
afterEach(async () => {
  for (const ctx of contexts.splice(0)) await ctx.fiber.dispose()
})

class CapabilityAdapter extends LlmAdapter {
  override resolveModel(provider: string, model: string) {
    return Promise.resolve({ provider, id: model, name: model, ...model === 'decision' ? { nativeTools: false } : {} })
  }
  async *stream(): AsyncIterable<StreamChunk> { yield { type: 'finish', reason: { kind: 'stop' } } }
}

it('keeps ordinary tools while a no-tools model removes even an unconditional transport schema', async () => {
  const ctx = new Context()
  contexts.push(ctx)
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(LlmRuntime)
  ctx.llm.registerAdapter(['fixture'], new CapabilityAdapter())
  const tool = defineTool({
    name: 'echo', description: 'Echo.', parameters: {},
    output: { schema: { type: 'string' }, render: (_args, value) => [{ type: 'text', text: value }] },
    execute: () => Promise.resolve('done'),
  })
  ctx.tools.register(tool)
  let executed = false
  ctx.tools.setExecutor({
    sections: [{ name: 'transport', order: 1, text: 'Call run_code to reach tools.' }],
    describe: () => ({ schemas: [{ name: 'run_code', description: 'Run.', parameters: {} }] }),
    execute: () => { executed = true; return Promise.resolve({ results: [], additionalContexts: [], concluded: false }) },
  })
  const agent = { id: SessionId('model-capability') } as Agent
  const signal = new AbortController().signal
  const decision = await ctx.tools.resolve({ agent, signal, modelSelection: { provider: 'fixture', model: 'decision' } })
  expect(decision.nativeTools).toBe(false)
  expect(ctx.tools.schemas(agent, decision)).toEqual([])
  const hidden = await ctx.systemPrompt.assemble({ scope: agent, toolSet: decision })
  expect(hidden.tools).toEqual([])
  expect(renderPrompt(hidden)).not.toContain('run_code')
  await expect(ctx.tools.execute({
    agent, signal, toolSet: decision,
    calls: [{ type: 'tool-call', id: ToolCallId('forbidden'), name: 'run_code', arguments: '{}' }],
  })).rejects.toMatchObject({ code: 'UNSUPPORTED_TOOLS' })
  expect(executed).toBe(false)
  decision.dispose()
  const ordinary = await ctx.tools.resolve({ agent, signal, modelSelection: { provider: 'fixture', model: 'ordinary' } })
  expect(ordinary).not.toHaveProperty('nativeTools')
  expect(ctx.tools.schemas(agent, ordinary).map(schema => schema.name)).toEqual(['echo'])
  const visible = await ctx.systemPrompt.assemble({ scope: agent, toolSet: ordinary })
  expect(visible.tools.map(schema => schema.name)).toEqual(['run_code'])
  expect(renderPrompt(visible)).toContain('run_code')
  ordinary.dispose()
})
