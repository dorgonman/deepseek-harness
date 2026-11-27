import { describe, expect, it } from 'vitest'
import { MAX_TIMER_DELAY_MS } from '@deepseek-ai/dsh-timeout'
import {
  codePointLength,
  DEFAULT_ELISION_TEXT,
  DEFAULTS,
  isEffective,
  resolveConfig,
  truncateCodePoints,
} from '../src/config.ts'

describe('decision relevance configuration', () => {
  it('resolves detached immutable defaults and partial overrides', () => {
    const raw: { mode: 'on'; provider: string; model: string; retainThreshold: number } = {
      mode: 'on', provider: 'p', model: 'm', retainThreshold: 0.5,
    }
    const resolved = resolveConfig(raw)
    raw.retainThreshold = 1
    expect(resolved).toEqual({
      mode: 'on',
      provider: 'p',
      model: 'm',
      retainThreshold: 0.5,
      uncertaintyMargin: DEFAULTS.uncertaintyMargin,
      maxCandidatesPerRequest: DEFAULTS.maxCandidatesPerRequest,
      minCandidateChars: DEFAULTS.minCandidateChars,
      maxInputChars: DEFAULTS.maxInputChars,
      timeoutMs: DEFAULTS.timeoutMs,
      elisionText: DEFAULT_ELISION_TEXT,
    })
    expect(Object.isFrozen(resolved)).toBe(true)
    expect(Object.isFrozen(DEFAULTS)).toBe(true)
  })

  it('detaches nested values so a later caller mutation cannot reach them', () => {
    const resolved = resolveConfig({ mode: 'on', provider: 'p', model: 'm' })
    expect(() => {
      ;(resolved as { provider?: string }).provider = 'other'
    }).toThrow()
  })

  it('is inactive by default and requires a complete route', () => {
    expect(isEffective(resolveConfig())).toBe(false)
    expect(isEffective(resolveConfig({ mode: 'on' }))).toBe(false)
    expect(isEffective(resolveConfig({ mode: 'on', provider: 'p', model: 'm' }))).toBe(true)
    expect(isEffective(resolveConfig({ mode: 'off', provider: 'p', model: 'm' }))).toBe(false)
  })

  it('rejects stale keys, half routes, and out-of-range thresholds', () => {
    const bad = [
      [{ threshold: 1 }, /unknown key "threshold"/],
      [{ provider: 'p' }, /provider and model must be configured together/],
      [{ model: 'm' }, /provider and model must be configured together/],
      [{ provider: '' }, /provider must be a non-empty string/],
      [{ mode: 'sometimes' }, /mode .* must be "off" or "on"/],
      [{ retainThreshold: 1.5 }, /retainThreshold .* must be a number in \[0, 1\]/],
      [{ retainThreshold: 0.05, uncertaintyMargin: 0.5 }, /must stay within \[0, 1\]/],
      [{ maxCandidatesPerRequest: 0 }, /maxCandidatesPerRequest .* positive integer/],
      [{ minCandidateChars: -1 }, /minCandidateChars .* non-negative integer/],
      [{ timeoutMs: 0 }, /timeoutMs .* positive integer/],
      [{ elisionText: '   ' }, /elisionText must be a non-empty string/],
    ] as Array<[Record<string, unknown>, RegExp]>
    for (const [config, pattern] of bad) {
      expect(() => resolveConfig(config)).toThrow(pattern)
    }
  })

  it('bounds text on Unicode code-point boundaries', () => {
    expect(codePointLength('abc')).toBe(3)
    expect(codePointLength('😀😀')).toBe(2)
    expect(truncateCodePoints('abcdef', 10)).toBe('abcdef')
    expect(truncateCodePoints('abcdef', 3)).toBe('abc')
    // Cutting a surrogate pair in half would produce a lone surrogate.
    expect(truncateCodePoints('a😀b', 2)).toBe('a😀')
    expect(codePointLength(truncateCodePoints('a😀b', 2))).toBe(2)
  })

  it('rejects a deadline beyond the timer range', () => {
    expect(resolveConfig({ timeoutMs: MAX_TIMER_DELAY_MS }).timeoutMs).toBe(MAX_TIMER_DELAY_MS)
    expect(() => resolveConfig({ timeoutMs: MAX_TIMER_DELAY_MS + 1 })).toThrow(/exceeds the maximum timer delay/)
  })

  it.each([
    [null, 'null'],
    [true, 'true'],
    [{ privateValue: 'hidden' }, 'object'],
  ])('describes invalid threshold input as %j without dumping objects', (value, description) => {
    const config: Record<string, unknown> = { retainThreshold: value }
    expect(() => resolveConfig(config)).toThrow(`retainThreshold (${description}) must be a number in [0, 1]`)
  })
})
