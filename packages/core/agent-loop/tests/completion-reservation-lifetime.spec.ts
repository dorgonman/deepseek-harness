import * as NativeToolExecutor from '@deepseek-ai/dsh-tool-executor-native'
import { expect, it, onTestFinished } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import AgentRegistry, { type Agent } from '@deepseek-ai/dsh-agent'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import LlmRuntime, { createUserMessage } from '@deepseek-ai/dsh-llm'
import SessionStore, { SessionId } from '@deepseek-ai/dsh-session'
import * as JsonlSessionPersistence from '@deepseek-ai/dsh-session-persistence-jsonl'
import SessionProjectionRegistry from '@deepseek-ai/dsh-session-projection'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import { MockAdapter, textResponse } from './mock-adapter.ts'

const route = { provider: 'mock', model: 'mock' }
const sessionId = SessionId('completion-reservation')

interface Fixture {
  root: string
  contexts: Context[]
}

async function mountProfile(fixture: Fixture): Promise<Context> {
  const ctx = new Context()
  fixture.contexts.push(ctx)
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(SessionStore)
  await ctx.plugin(SessionProjectionRegistry)
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(NativeToolExecutor)
  await ctx.plugin(AgentRegistry)
  await ctx.plugin(JsonlSessionPersistence, { root: fixture.root })
  return ctx
}

async function newFixture(): Promise<Fixture> {
  const root = await mkdtemp(join(tmpdir(), 'dsh-completion-reservation-'))
  const contexts: Context[] = []
  onTestFinished(async () => {
    try {
      for (const ctx of contexts) await ctx.fiber.dispose()
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  })
  return { root, contexts }
}

async function expectTurn(
  ctx: Context,
  agent: Agent,
  adapter: MockAdapter,
  text: string,
  maxTokens: number | undefined,
  source: 'explicit' | 'adapter-default' | 'unknown',
): Promise<void> {
  const priorCalls = adapter.requests.length
  const admissions: { maxTokens: number | undefined; source: string; calls: number }[] = []
  const disposeAdmission = ctx.on('agent/request-envelope', (event) => {
    if (event.agent === agent && event.phase === 'verify') {
      admissions.push({
        maxTokens: event.completionMaxTokens,
        source: event.completionSource,
        calls: adapter.requests.length,
      })
    }
  })
  const idle = Promise.withResolvers<undefined>()
  const disposeStatus = ctx.on('agent/status', ({ agent: subject, status }) => {
    if (subject === agent && status === 'idle') idle.resolve(undefined)
  })
  try {
    agent.followup(createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } }))
    await idle.promise
    expect(admissions).toEqual([{ maxTokens, source, calls: priorCalls }])
    expect(adapter.requests).toHaveLength(priorCalls + 1)
    expect(adapter.requests.at(-1)?.maxTokens).toBe(maxTokens)
    const header = agent.session.requestHeader()
    expect(header?.config.maxTokens).toBe(maxTokens)
    expect(header?.adapterDefaults?.maxTokens).toBe(source === 'adapter-default' ? true : undefined)
  } finally {
    disposeStatus()
    disposeAdmission()
  }
}

it.each([
  { scenario: 'persisted explicit', priorCap: 512, resumeCap: undefined, expectedCap: 512, source: 'explicit' },
  { scenario: 'unmarked header without a cap', priorCap: undefined, resumeCap: undefined, expectedCap: 2_048, source: 'adapter-default' },
  { scenario: 'new explicit resume option', priorCap: 512, resumeCap: 768, expectedCap: 768, source: 'explicit' },
] as const)('resolves $scenario through cold JSONL resume', async ({ priorCap, resumeCap, expectedCap, source }) => {
  const fixture = await newFixture()
  const first = await mountProfile(fixture)
  const firstAdapter = new MockAdapter([textResponse('stored')])
  first.llm.registerAdapter([route.provider], firstAdapter)
  await first.plugin(AgentLoop, { agents: [] })
  const handle = await first.agents.create({
    sessionId,
    agentOptions: { ...route, ...priorCap === undefined ? {} : { maxTokens: priorCap } },
  })
  await expectTurn(first, handle.agent, firstAdapter, 'first', priorCap, priorCap === undefined ? 'unknown' : 'explicit')
  await handle.dispose()
  await first.fiber.dispose()
  fixture.contexts.pop()

  const resumed = await mountProfile(fixture)
  const resumedAdapter = new MockAdapter([textResponse('resumed')], undefined, 2_048)
  resumed.llm.registerAdapter([route.provider], resumedAdapter)
  await resumed.plugin(AgentLoop, { agents: [] })
  const { agent } = await resumed.agents.resume({
    resumeSessionId: sessionId,
    ...resumeCap === undefined ? {} : { agentOptions: { maxTokens: resumeCap } },
  })
  await expectTurn(resumed, agent, resumedAdapter, 'continue', expectedCap, source)
  expect(agent.session.snapshotEvents().filter(event => event.type === 'request/header').map(event => event.data.reason))
    .toEqual(['initial', 'resume'])
})

it.each([
  { scenario: 'adapter default', priorCap: undefined, reloadedCap: undefined, before: 1_024, after: 2_048, source: 'adapter-default' },
  { scenario: 'persisted explicit', priorCap: 512, reloadedCap: undefined, before: 512, after: 512, source: 'explicit' },
  { scenario: 'new explicit configured cap', priorCap: 512, reloadedCap: 768, before: 512, after: 768, source: 'explicit' },
] as const)('resolves $scenario through configured AgentLoop reload', async ({ priorCap, reloadedCap, before, after, source }) => {
  const ctx = await mountProfile(await newFixture())
  const firstAdapter = new MockAdapter([textResponse('stored')], undefined, 1_024)
  const disposeAdapter = ctx.llm.registerAdapter([route.provider], firstAdapter)
  const configured = { id: 'main', sessionId, ...route }
  const firstLoop = await ctx.plugin(AgentLoop, {
    agents: [{ ...configured, ...priorCap === undefined ? {} : { maxTokens: priorCap } }],
  })
  await expect.poll(() => ctx.agents.get(sessionId)).toBeDefined()
  await expectTurn(ctx, ctx.agents.get(sessionId)!, firstAdapter, 'first', before, source)
  await firstLoop.dispose()
  disposeAdapter()

  const nextAdapter = new MockAdapter([textResponse('reloaded')], undefined, 2_048)
  ctx.llm.registerAdapter([route.provider], nextAdapter)
  await ctx.plugin(AgentLoop, {
    agents: [{ ...configured, ...reloadedCap === undefined ? {} : { maxTokens: reloadedCap } }],
  })
  await expect.poll(() => ctx.agents.get(sessionId)).toBeDefined()
  const agent = ctx.agents.get(sessionId)!
  await expectTurn(ctx, agent, nextAdapter, 'continue', after, source)
  expect(agent.session.snapshotEvents().filter(event => event.type === 'request/header').map(event => event.data.reason))
    .toEqual(['initial', 'resume'])
})
