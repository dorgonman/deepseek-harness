/** Bounded exact repetition detection for one model-stream attempt. */

import type { StreamChunk } from '@deepseek-ai/dsh-llm'

const MIN_PERIOD = 96
const MAX_PERIOD = 512
const COPIES = 6
const WINDOW = MAX_PERIOD * COPIES
const STRIDE = 32
const ANCHOR = 16
const MIN_DISTINCT_LETTERS = 12
const LETTER = /\p{L}/u

/** Durable hook reason used to distinguish a guard stop from a manual stop. */
export const REPEATED_MODEL_OUTPUT = 'agent-loop/repeated-model-output'

type TextDelta = Extract<StreamChunk, { type: 'text-delta' | 'reasoning-delta' }>

function hasDistinctLetters(text: string): boolean {
  const letters = new Set<string>()
  for (const character of text) {
    if (!LETTER.test(character)) continue
    letters.add(character.toLowerCase())
    if (letters.size >= MIN_DISTINCT_LETTERS) return true
  }
  return false
}

/** A periodic detector for consecutive deltas from a single content block. */
export class StreamRepetition {
  private channel: TextDelta['type'] | undefined
  private index: number | undefined
  private tail = ''
  private untilScan = STRIDE

  /**
   * Consume the delta and report the detection offset, if any. The fixed
   * scan cadence is independent of transport chunk boundaries.
   * @param chunk - normalized text or reasoning delta in stream order.
   * @returns UTF-16 offset immediately after the matching prefix.
   */
  push(chunk: TextDelta): number | undefined {
    if (chunk.type !== this.channel || chunk.index !== this.index) {
      this.channel = chunk.type
      this.index = chunk.index
      this.tail = ''
      this.untilScan = STRIDE
    }
    let offset = 0
    while (offset < chunk.text.length) {
      const count = Math.min(this.untilScan, chunk.text.length - offset)
      this.tail = (this.tail + chunk.text.slice(offset, offset + count)).slice(-WINDOW)
      offset += count
      this.untilScan -= count
      if (this.untilScan !== 0) continue
      this.untilScan = STRIDE
      if (this.repeatedTail()) return offset
    }
    return undefined
  }

  /** Match only exact adjacent repeats of a substantial span. */
  private repeatedTail(): boolean {
    const end = this.tail.length
    if (end < MIN_PERIOD * COPIES) return false
    // A matching period has at most MAX_PERIOD units. Padding or one-letter
    // runs can repeat at every candidate offset but cannot satisfy this gate.
    if (!hasDistinctLetters(this.tail.slice(-MAX_PERIOD))) return false
    const limit = Math.min(MAX_PERIOD, Math.floor(end / COPIES))
    const start = end - ANCHOR
    const anchor = this.tail.slice(start)
    let previous = this.tail.lastIndexOf(anchor, start - MIN_PERIOD)
    while (previous >= start - limit) {
      const length = start - previous
      const period = this.tail.slice(end - length)
      if (this.tail.startsWith(period, end - length * 2)
        && this.tail.endsWith(period.repeat(COPIES)) && hasDistinctLetters(period)) return true
      previous = this.tail.lastIndexOf(anchor, previous - 1)
    }
    return false
  }
}
