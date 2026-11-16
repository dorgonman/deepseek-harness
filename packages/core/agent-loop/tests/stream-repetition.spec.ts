import { Context } from '@deepseek-ai/cordis'
import { createUserMessage, type StreamChunk } from '@deepseek-ai/dsh-llm'
import { SessionId } from '@deepseek-ai/dsh-session'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import { describe, expect, it } from 'vitest'
import { REPEATED_MODEL_OUTPUT, StreamRepetition } from '../src/stream-repetition.ts'
import { MockAdapter, textResponse } from './mock-adapter.ts'

const PERIOD = 'The checkpoint contains verified configuration, changed paths, known failures, and the remaining deployment task. '
const repeated = PERIOD.repeat(8)
type Channel = 'text-delta' | 'reasoning-delta'

function detectedAt(text: string, stride: number): number | undefined {
  const guard = new StreamRepetition()
  for (let start = 0; start < text.length; start += stride) {
    const offset = guard.push({ type: 'text-delta', index: 0, text: text.slice(start, start + stride) })
    if (offset !== undefined) return start + offset
  }
  return undefined
}

describe('exact periodic output', () => {
  it('detects the same prefix regardless of provider chunk boundaries', () => {
    const expected = detectedAt(repeated, repeated.length)
    expect(expected).toBeGreaterThanOrEqual(PERIOD.length * 6)
    expect(expected).toBeLessThan(PERIOD.length * 6 + 32)
    for (const stride of [1, 17, 131]) expect(detectedAt(repeated, stride)).toBe(expected)
  })

  it('leaves shorter repetition, formatting padding, and changing lines alone', () => {
    const changingLines = Array.from({ length: 100 }, (_, index) => `${index}: ${PERIOD}`).join('\n')
    for (const text of [PERIOD.repeat(5), ' '.repeat(65_536), 'x'.repeat(65_536), 'hello world '.repeat(100), changingLines]) {
      expect(detectedAt(text, 19)).toBeUndefined()
    }
  })

  it('does not combine different blocks or reasoning with visible text', () => {
    const guard = new StreamRepetition()
    const text = PERIOD.repeat(5)
    expect(guard.push({ type: 'text-delta', index: 0, text })).toBeUndefined()
    expect(guard.push({ type: 'reasoning-delta', index: 0, text })).toBeUndefined()
    expect(guard.push({ type: 'text-delta', index: 1, text })).toBeUndefined()
  })
})

function scriptedOutput(channel: Channel): StreamChunk[] {
  const type = channel === 'text-delta' ? 'text' : 'reasoning'
  return [
    { type: 'block-start', index: 0, blockType: type },
    { type: channel, index: 0, text: repeated },
    { type: 'block-end', index: 0, block: { type, text: repeated } },
    { type: 'finish', reason: { kind: 'stop' } },
  ]
}

describe('agent loop repeat stop', () => {
  it.each(['text-delta', 'reasoning-delta'] as const)('cancels %s and allows the next prompt', async (channel) => {
    const ctx = new Context()
    try {
      await mountAgentLoopTestDependencies(ctx)
      await ctx.plugin(AgentLoop, { agents: [], stopOnRepeatedOutput: true })
      const adapter = new MockAdapter([scriptedOutput(channel), textResponse('next prompt completed')])
      ctx.llm.registerAdapter(['mock'], adapter)
      let recoveries = 0
      ctx.on('agent/request-error', (_event, next) => { recoveries++; return next() })
      const agent = await ctx.agentLoop.create(SessionId('repeat-output'), { provider: 'mock', model: 'mock' })
      agent.followup(createUserMessage({ content: [{ type: 'text', text: 'first' }], source: { kind: 'user' } }))
      await agent.whenIdle()
      expect(adapter.requests).toHaveLength(1)
      expect(adapter.requests[0]?.signal?.aborted).toBe(true)
      expect(recoveries).toBe(0)
      expect(agent.session.snapshotEvents().findLast(event => event.type === 'turn/end')?.data.reason)
        .toEqual({ kind: 'aborted', reason: { kind: 'hook', reason: REPEATED_MODEL_OUTPUT } })
      const message = agent.session.snapshotEvents().findLast(event => event.type === 'assistant/message')
      expect(message?.data.interrupted).toBe(true)
      const partial = message?.data.message.content[0]
      expect(partial?.type === 'text' || partial?.type === 'reasoning' ? partial.text : '').toBe(
        repeated.slice(0, detectedAt(repeated, repeated.length)),
      )
      agent.followup(createUserMessage({ content: [{ type: 'text', text: 'second' }], source: { kind: 'user' } }))
      await agent.whenIdle()
      expect(adapter.requests).toHaveLength(2)
      expect(agent.session.snapshotEvents().findLast(event => event.type === 'turn/end')?.data.reason)
        .toEqual({ kind: 'completed' })
    } finally {
      await ctx.fiber.dispose()
    }
  })

  it('is disabled by default', async () => {
    const ctx = new Context()
    try {
      await mountAgentLoopTestDependencies(ctx)
      await ctx.plugin(AgentLoop, { agents: [] })
      const adapter = new MockAdapter([scriptedOutput('text-delta')])
      ctx.llm.registerAdapter(['mock'], adapter)
      const agent = await ctx.agentLoop.create(SessionId('repeat-disabled'), { provider: 'mock', model: 'mock' })
      agent.followup(createUserMessage({ content: [{ type: 'text', text: 'allowed repetition' }], source: { kind: 'user' } }))
      await agent.whenIdle()
      expect(adapter.requests[0]?.signal?.aborted).toBe(false)
      expect(agent.session.snapshotEvents().findLast(event => event.type === 'turn/end')?.data.reason)
        .toEqual({ kind: 'completed' })
    } finally {
      await ctx.fiber.dispose()
    }
  })
})
