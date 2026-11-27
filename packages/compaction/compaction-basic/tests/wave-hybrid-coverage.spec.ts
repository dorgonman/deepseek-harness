/** Standalone compaction coverage for admission and summarizer routing. */
import { Context } from '@deepseek-ai/cordis'
import { agentEvents, type Agent } from '@deepseek-ai/dsh-agent'
import LlmRuntime, { LlmAdapter, createUserMessage } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, LlmResolvedModelInfo, StreamChunk } from '@deepseek-ai/dsh-llm'
import { Session, SessionId } from '@deepseek-ai/dsh-session'
import { describe, expect, it, onTestFinished, vi } from 'vitest'
import { BasicCompactionEngine, type BasicCompactionConfig } from '../src/index.ts'
import type { SummarizationInput, SummaryResult } from '../src/summarizer.ts'

const provider = 'summary-provider'
const model = 'summary-model'
const message = createUserMessage({ content: [{ type: 'text', text: 'A decision from the conversation' }], source: { kind: 'user' } })

class ExposedBasicEngine extends BasicCompactionEngine {
  runSummary(input: SummarizationInput, agent: Agent, signal?: AbortSignal): Promise<SummaryResult> {
    return this.summarize(input, agent, signal)
  }
}

type RouteMetadata = Pick<LlmResolvedModelInfo, 'context' | 'defaultMaxTokens' | 'transportMaxTokens'>

class SummaryAdapter extends LlmAdapter {
  readonly requests: GenerateOptions[] = []

  constructor(private readonly metadata: RouteMetadata) {
    super()
  }

  override resolveModel(route: string, id: string): Promise<LlmResolvedModelInfo> {
    return Promise.resolve({ provider: route, id, name: id, ...this.metadata })
  }

  override async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.requests.push(options)
    yield { type: 'block-start', index: 0, blockType: 'text' }
    yield { type: 'text-delta', index: 0, text: 'checkpoint' }
    yield { type: 'finish', reason: { kind: 'stop' } }
  }
}

async function fixture(config: BasicCompactionConfig = {}, metadata: RouteMetadata = {}) {
  const ctx = new Context()
  onTestFinished(async () => { await ctx.fiber.dispose() })
  await ctx.plugin(LlmRuntime)
  const adapter = new SummaryAdapter(metadata)
  ctx.llm.registerAdapter([provider], adapter)
  const engine = new ExposedBasicEngine(ctx, { auto: false, ...config })
  const session = Session.create(SessionId('basic-index-coverage'))
  const agent = { session, options: { provider, model } } as Agent
  return { ctx, engine, agent, adapter }
}

describe('standalone Basic summarization', () => {
  it('retains supplied resident routing values in the resolved fixed config', async () => {
    const resident = { residentGatewayId: 'gateway-test', residentProvider: provider }
    const { engine } = await fixture(resident)

    expect(engine.config).toMatchObject({ summarizationMode: 'fixed', ...resident })
  })

  it('requires the Hybrid backend for resident-preferred summaries before dispatch', async () => {
    const { engine, agent, adapter } = await fixture({
      summarizationMode: 'resident-preferred', residentProvider: provider,
      residentGatewayId: 'gateway-test', residentPinnedNodeId: 'node-test',
    })

    await expect(engine.runSummary({ messages: [message] }, agent))
      .rejects.toThrow('resident-preferred summarization requires the Hybrid backend')
    expect(adapter.requests).toEqual([])
  })

  it.each([
    ['adapter default', { defaultMaxTokens: 1024, transportMaxTokens: 512 }, 896],
    ['transport fallback', { transportMaxTokens: 512 }, 960],
    ['unknown completion cap', {}, 1024],
  ] as const)('sizes the summary from capacity left by the %s', async (_source, reservation, maxTokens) => {
    const { engine, agent, adapter } = await fixture({}, { context: { contextWindow: 8192 }, ...reservation })

    await expect(engine.runSummary({ messages: [message] }, agent)).resolves.toMatchObject({
      provider, model, maxTokens, summary: [{ type: 'text', text: 'checkpoint' }],
    })
    expect(adapter.requests).toHaveLength(1)
    expect(adapter.requests[0]?.maxTokens).toBe(maxTokens)
  })

  it('preserves the configured cap if the optional sizing lookup fails', async () => {
    const { ctx, engine, agent, adapter } = await fixture()
    vi.spyOn(ctx.llm, 'resolveModelInfo').mockRejectedValueOnce(new Error('Sizing metadata unavailable'))

    await expect(engine.runSummary({ messages: [message] }, agent)).resolves.toMatchObject({ maxTokens: engine.config.maxTokens })
    expect(adapter.requests).toHaveLength(1)
    expect(adapter.requests[0]?.maxTokens).toBe(engine.config.maxTokens)
  })

  it('preserves cancellation when the optional sizing lookup fails', async () => {
    const { ctx, engine, agent, adapter } = await fixture()
    const controller = new AbortController()
    const reason = new Error('Summary cancelled during model sizing')
    vi.spyOn(ctx.llm, 'resolveModelInfo').mockImplementationOnce(async () => {
      controller.abort(reason)
      throw new Error('Sizing metadata unavailable')
    })

    await expect(engine.runSummary({ messages: [message] }, agent, controller.signal)).rejects.toBe(reason)
    expect(adapter.requests).toEqual([])
  })
})

describe('standalone Basic envelope preview', () => {
  it('skips cancelled or unpriced requests and otherwise prices messages without an adapter projection', async () => {
    const { ctx, agent } = await fixture({ auto: true })
    const estimateRequest = vi.fn<Context['tokenMeter']['estimateRequest']>().mockReturnValue(32)
    ctx.provide('tokenMeter', { estimateRequest })
    const session = agent.session
    session.append('turn/start', { turn: 1 })
    session.append('user/message', message, { surfaceOp: 'append' })
    const header = { config: { provider, model }, tools: [{ name: 'example', description: 'Example', parameters: {} }] }
    const view = vi.fn(() => session)
    const preview = { phase: 'preview', header, view, completionSource: 'unknown', signal: new AbortController().signal } as const
    const dispatcher = agentEvents(ctx, agent)

    await dispatcher.serial('agent/request-envelope', preview)
    await dispatcher.serial('agent/request-envelope', { ...preview, contextWindow: 64, signal: AbortSignal.abort() })
    expect(view).not.toHaveBeenCalled()
    expect(estimateRequest).not.toHaveBeenCalled()

    await dispatcher.serial('agent/request-envelope', { ...preview, contextWindow: 64 })
    expect(view).toHaveBeenCalledOnce()
    expect(estimateRequest).toHaveBeenCalledExactlyOnceWith(session.deriveMessages(), header.tools, undefined)
  })
})
