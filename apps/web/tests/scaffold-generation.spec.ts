import { afterEach, describe, expect, it } from 'vitest'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { parseSessionLog, prepareSessionSnapshotFixtureForComparison } from '@deepseek-ai/dsh-llm-replay'
import { createSystemMessage } from '@deepseek-ai/dsh-llm'
import { SESSION_FORMAT_VERSION } from '@deepseek-ai/dsh-session'
import {
  assertFixtureInventory,
  normalizeAria,
  parseSeedFixture,
  recordedSessionFixturePath,
  selectedSessionFixture,
} from './scaffold.ts'

const roots: string[] = []

afterEach(async () => {
  for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true })
})

describe('Web aria workspace normalization', () => {
  it.each([String.raw`D:\temp\dsh-web-e2e-ws-generated`, '/tmp/dsh-web-e2e-ws-generated'])('normalizes the workspace path and basename from %s', (workspaceCwd) => {
    const snapshot = `- text: dsh-web-e2e-ws-generated\n- text: ${workspaceCwd}`
    expect(normalizeAria(snapshot, workspaceCwd, false)).toBe('- text: {{workspace}}\n- text: {{cwd}}')
  })
})

describe('Web seed stream timing', () => {
  it('retains complete generated event timestamps and buffered first-token timing', () => {
    const startTime = 1_001
    const events = [
      { type: 'turn/start', time: 1_000, data: { turn: 1 } },
      { type: 'step/start', time: startTime, data: { turn: 1, step: 1 } },
      { type: 'system/message', time: 1_002, data: { turn: 1, step: 1, message: createSystemMessage('') }, surfaceOp: 'append' },
      {
        type: 'user/message', time: 1_003,
        data: { id: '00000000-0000-4000-9000-000000000001', role: 'user', content: [{ type: 'text', text: 'prompt' }], source: { kind: 'user' } },
        surfaceOp: 'append',
      },
      {
        type: 'assistant/message', time: startTime + 35_002,
        data: {
          turn: 1, step: 1, usage: { inputTokens: 0, outputTokens: 351 },
          stream: [{ type: 'text-chunks', time0: startTime + 35_000, index: 0, dt: [], texts: ['reply'] }],
          message: {
            id: '00000000-0000-4000-8000-000000000001', role: 'assistant', content: [{ type: 'text', text: 'reply' }],
            source: { kind: 'model', provider: 'snapshot', model: 'snapshot-replier' },
          },
        },
        surfaceOp: 'append',
      },
      { type: 'step/end', time: startTime + 35_003, data: { turn: 1, step: 1 } },
      { type: 'turn/end', time: startTime + 35_004, data: { turn: 1, reason: { kind: 'completed' } } },
    ]
    const fixture = [
      JSON.stringify({ type: 'session', version: SESSION_FORMAT_VERSION, id: 'recorded-timing', createdAt: 1_000, cwd: '/workspace', isSeeded: false, delegationDepth: 0 }),
      ...events.map((event, seq) => JSON.stringify({ ...event, seq })),
      '',
    ].join('\n')
    const parsed = parseSeedFixture(fixture, { timing: 'recorded' })
    expect(parsed.events.map(event => event.time)).toEqual(events.map(event => event.time))
    expect(parsed.events.find(event => event.type === 'assistant/message')?.data.stream[0])
      .toMatchObject({ time0: startTime + 35_000 })
  })

  it('requires complete event envelopes when preserving recorded timing', async () => {
    const fixture = await readFile(new URL('../../../snapshots/session/skill-load/session.v3.jsonl', import.meta.url), 'utf8')
    const projected = fixture.split(/\r?\n/).map((line, index) => {
      if (index === 0 || line.trim() === '') return line
      const { seq: _seq, time: _time, ...event } = JSON.parse(line)
      return JSON.stringify(event)
    }).join('\n')
    expect(() => parseSeedFixture(projected, { timing: 'recorded' }))
      .toThrow('recorded seed timing requires complete event envelopes')
  })

  it.each([2, 3])('preserves embedded timestamps from format v%i and its current projection', async (version) => {
    const fixture = await readFile(new URL(`../../../snapshots/session/skill-load/session.v${version}.jsonl`, import.meta.url), 'utf8')
    const current = prepareSessionSnapshotFixtureForComparison(fixture)
    const expected = parseSessionLog(current)

    expect(parseSeedFixture(fixture).events).toEqual(expected)
    expect(parseSeedFixture(current).events).toEqual(expected)
  })

  it.each(['session.jsonl', 'session.v1.jsonl'])('reconstructs positive stream intervals from %s', async (filename) => {
    const fixture = await readFile(new URL(`../../../snapshots/session/text-turn/${filename}`, import.meta.url), 'utf8')
    const messages = parseSeedFixture(fixture).events.filter(event => event.type === 'assistant/message')
    expect(messages).toHaveLength(1)
    const stream = messages[0]!.data.stream
    expect(stream.length).toBeGreaterThan(1)

    let previousEnd = -1
    for (const record of stream) {
      const start = 'time' in record ? record.time : record.time0
      expect(start).toBeGreaterThan(previousEnd)
      previousEnd = 'time' in record ? record.time : record.time0 + record.dt.reduce((total, delta) => total + delta, 0)
    }
    expect(previousEnd).toBeGreaterThan(0)
  })
})

describe('Web snapshot generation filenames', () => {
  it('selects the highest parent and child generations without counting retained inputs twice', async () => {
    const root = await mkdtemp(join(tmpdir(), 'dsh-web-fixture-generations-'))
    roots.push(root)
    for (const [name, version] of [
      ['session.jsonl', 0],
      ['session.v2.jsonl', 2],
      ['session.1.jsonl', 0],
      ['session.1.v1.jsonl', 1],
    ] as const) {
      await writeFile(join(root, name), `${JSON.stringify({
        type: 'session', version, id: '{{session:1}}', createdAt: 0, delegationDepth: 0,
      })}\n`)
    }

    await expect(selectedSessionFixture(join(root, 'session.jsonl')))
      .resolves.toBe(join(root, 'session.v2.jsonl'))
    await expect(selectedSessionFixture(join(root, 'session.1.jsonl')))
      .resolves.toBe(join(root, 'session.1.v1.jsonl'))
    await expect(selectedSessionFixture(join(root, 'replay.override.json')))
      .resolves.toBe(join(root, 'replay.override.json'))
  })

  it('leaves an absent override-only parent fixture unresolved', async () => {
    const root = await mkdtemp(join(tmpdir(), 'dsh-web-fixture-generations-'))
    roots.push(root)

    await expect(selectedSessionFixture(join(root, 'session.jsonl'), true))
      .resolves.toBe(join(root, 'session.jsonl'))
    await expect(selectedSessionFixture(join(root, 'session.jsonl')))
      .rejects.toThrow('missing parent session fixture')
  })

  it('records beside an older generation and preserves the parent or child role', () => {
    const fixtures = join('/', 'fixtures')
    expect(recordedSessionFixturePath(join(fixtures, 'session.jsonl'), 1))
      .toBe(join(fixtures, 'session.v1.jsonl'))
    expect(recordedSessionFixturePath(join(fixtures, 'session.2.jsonl'), 3))
      .toBe(join(fixtures, 'session.2.v3.jsonl'))
    expect(recordedSessionFixturePath(join(fixtures, 'session.v1.jsonl'), 1))
      .toBe(join(fixtures, 'session.v1.jsonl'))
    expect(() => recordedSessionFixturePath(join(fixtures, 'notes.jsonl'), 1))
      .toThrow('invalid Session fixture path')
  })

  it('treats retained generations as one exact inventory role', async () => {
    const root = await mkdtemp(join(tmpdir(), 'dsh-web-fixture-inventory-'))
    roots.push(root)
    await writeFile(join(root, 'session.jsonl'), `${JSON.stringify({
      type: 'session', version: 0, id: '{{session:1}}', createdAt: 0, delegationDepth: 0,
    })}\n`)
    await writeFile(join(root, 'session.v1.jsonl'), `${JSON.stringify({
      type: 'session', version: 1, id: '{{session:1}}', createdAt: 0, delegationDepth: 0,
    })}\n`)
    await writeFile(join(root, 'ui.expected.md'), 'stable\n')

    await expect(assertFixtureInventory(root, ['session.jsonl', 'ui.expected.md']))
      .resolves.toBeUndefined()
  })
})
