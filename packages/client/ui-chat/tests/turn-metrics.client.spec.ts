// Assistant timing readings and statistics formatting.

import { describe, expect, it } from 'vitest'
import type {
  AssistantMessageNode,
} from '@deepseek-ai/dsh-client-ui-chat/client'
import { assistantStepReading } from '../src/client/contract/turn-metrics.ts'
import { formatTokensPerSecond } from '../src/client/chat/message-chrome.ts'
import { formatCacheHitPercent } from '../src/client/chat/token-format.ts'

interface StepSpec {
  seq: number
  turn: number
  step: number
  timing?: AssistantMessageNode['timing']
  usage?: unknown
}

const assistant = ({ seq, turn, step, timing, usage }: StepSpec): AssistantMessageNode => ({
  kind: 'assistant', seq, time: seq * 1_000, turn, step, blocks: [{ kind: 'text', text: `t${seq}` }],
  ...(timing === undefined ? {} : { timing }),
  ...(usage === undefined ? {} : { usage }),
})

describe('assistantStepReading', () => {
  it('includes ttft in the model-phase duration used for output rate', () => {
    const reading = assistantStepReading(assistant({
      seq: 2, turn: 1, step: 1,
      timing: { stepStartTime: 1_000, firstTokenTime: 1_800, completedTime: 6_800 },
      usage: { outputTokens: 200 },
    }))
    expect(reading).toEqual({ ttftMs: 800, decodeMs: 5_800, outputTokens: 200 })
  })

  it('measures a buffered response over the complete model phase', () => {
    const reading = assistantStepReading(assistant({
      seq: 2, turn: 1, step: 1,
      timing: { stepStartTime: 1_000, firstTokenTime: 36_000, completedTime: 36_002 },
      usage: { outputTokens: 351 },
    }))
    expect(reading).toEqual({ ttftMs: 35_000, decodeMs: 35_002, outputTokens: 351 })
    expect(formatTokensPerSecond(reading.outputTokens! / (reading.decodeMs! / 1_000))).toBe('10')
  })

  it('returns nulls when timing is absent', () => {
    const reading = assistantStepReading(assistant({ seq: 2, turn: 1, step: 1, usage: { outputTokens: 5 } }))
    expect(reading).toEqual({ ttftMs: null, decodeMs: null, outputTokens: 5 })
  })

  it('requires the model start for output rate and reads ttft independently', () => {
    expect(assistantStepReading(assistant({
      seq: 2, turn: 1, step: 1,
      timing: { stepStartTime: null, firstTokenTime: 1_800, completedTime: 6_800 },
    }))).toEqual({ ttftMs: null, decodeMs: null, outputTokens: null })
    expect(assistantStepReading(assistant({
      seq: 2, turn: 1, step: 1,
      timing: { stepStartTime: 1_000, firstTokenTime: null, completedTime: 6_800 },
    }))).toEqual({ ttftMs: null, decodeMs: 5_800, outputTokens: null })
    expect(assistantStepReading(assistant({
      seq: 2, turn: 1, step: 1,
      timing: { stepStartTime: 2_000, firstTokenTime: 1_500, completedTime: 1_200 },
    }))).toEqual({ ttftMs: 0, decodeMs: null, outputTokens: null })
  })

  it.each([
    { stepStartTime: 1_000, completedTime: 1_000 },
    { stepStartTime: 2_000, completedTime: 1_000 },
    { stepStartTime: Number.NaN, completedTime: 2_000 },
    { stepStartTime: 1_000, completedTime: Number.POSITIVE_INFINITY },
    { stepStartTime: Number.NEGATIVE_INFINITY, completedTime: 2_000 },
    { stepStartTime: -Number.MAX_VALUE, completedTime: Number.MAX_VALUE },
  ])('omits a nonpositive or non-finite model-phase duration: $stepStartTime → $completedTime', (timing) => {
    const reading = assistantStepReading(assistant({
      seq: 2, turn: 1, step: 1,
      timing: { ...timing, firstTokenTime: null },
      usage: { outputTokens: 351 },
    }))
    expect(reading).toEqual({ ttftMs: null, decodeMs: null, outputTokens: 351 })
  })

  it('rejects non-object, missing, and non-finite usage token counts', () => {
    const timing = { stepStartTime: 1_000, firstTokenTime: 1_500, completedTime: 2_000 }
    expect(assistantStepReading(assistant({ seq: 2, turn: 1, step: 1, timing, usage: 'weird' })).outputTokens).toBeNull()
    expect(assistantStepReading(assistant({ seq: 2, turn: 1, step: 1, timing, usage: {} })).outputTokens).toBeNull()
    const nan = assistant({ seq: 2, turn: 1, step: 1, timing, usage: { outputTokens: Number.NaN } })
    expect(assistantStepReading(nan).outputTokens).toBeNull()
    expect(assistantStepReading(assistant({ seq: 2, turn: 1, step: 1, timing, usage: { outputTokens: -3 } })).outputTokens).toBeNull()
  })
})

describe('footer figure formatters', () => {
  it('omits a redundant decimal zero in cache-hit percentages', () => {
    expect(formatCacheHitPercent(1, 2, 1)).toBe('50')
  })

  it('formats throughput with whole tokens from ten up and one decimal below', () => {
    expect(formatTokensPerSecond(34.4)).toBe('34')
    expect(formatTokensPerSecond(9.96)).toBe('10')
    expect(formatTokensPerSecond(3.14)).toBe('3.1')
    expect(formatTokensPerSecond(-1)).toBe('0')
  })
})
