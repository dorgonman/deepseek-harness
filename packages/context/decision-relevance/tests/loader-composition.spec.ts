import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import Loader from '@deepseek-ai/cordis-plugin-loader'
import Include from '@deepseek-ai/cordis-plugin-include'
import LlmRuntime from '@deepseek-ai/dsh-llm'
import { SessionStore } from '@deepseek-ai/dsh-session'
import DecisionRelevanceFilter from '@deepseek-ai/dsh-context-decision-relevance'

let root: string | undefined
let context: Context | undefined

afterEach(async () => {
  await context?.fiber.dispose()
  context = undefined
  if (root !== undefined) await rm(root, { recursive: true, force: true })
  root = undefined
})

/** Boot the Loader over a flat YAML document with the real LLM runtime present. */
async function load(yaml: string[], label: string): Promise<Context> {
  root = await mkdtemp(join(tmpdir(), `dsh-decision-relevance-${label}-`))
  const configPath = join(root, 'cordis.yml')
  await writeFile(configPath, [
    "- name: '@deepseek-ai/dsh-llm'",
    "- name: '@deepseek-ai/dsh-session'",
    ...yaml,
    '',
  ].join('\n'))

  const ctx = new Context()
  context = ctx
  ctx.baseUrl = pathToFileURL(root).href + '/'
  await ctx.plugin(Loader)
  ctx.loader.builtins.include = Include
  ctx.loader.internal = {
    version: 'v2',
    async import(specifier: string) {
      if (specifier === '@deepseek-ai/dsh-llm') return LlmRuntime
      if (specifier === '@deepseek-ai/dsh-session') return SessionStore
      if (specifier === '@deepseek-ai/dsh-context-decision-relevance') return DecisionRelevanceFilter
      throw new Error(`unexpected Loader import: ${specifier}`)
    },
  } as unknown as NonNullable<typeof ctx.loader.internal>
  await ctx.loader.create({
    name: 'cordis:include',
    config: { path: pathToFileURL(configPath).href },
  })
  await ctx.loader.await()
  return ctx
}

describe('context decision-relevance real Loader composition', () => {
  it('loads and resolves the flat YAML plugin shape', async () => {
    const ctx = await load([
      "- name: '@deepseek-ai/dsh-context-decision-relevance'",
      '  config:',
      "    mode: 'on'",
      "    provider: 'decision-provider'",
      "    model: 'chat-qwen-flash-next-exl3-decision@local'",
      '    retainThreshold: 0.7',
      '    maxCandidatesPerRequest: 2',
    ], 'load')

    expect(ctx.get('decisionRelevance')).toBeInstanceOf(DecisionRelevanceFilter)
    const filter = ctx.decisionRelevance
    expect(filter.enabled).toBe(true)
    expect(filter.config).toEqual({
      mode: 'on',
      provider: 'decision-provider',
      model: 'chat-qwen-flash-next-exl3-decision@local',
      retainThreshold: 0.7,
      uncertaintyMargin: 0.1,
      maxCandidatesPerRequest: 2,
      minCandidateChars: 512,
      maxInputChars: 4096,
      timeoutMs: 15000,
      elisionText: '[tool result omitted for this request: judged irrelevant to the current goal]',
    })
  })

  it('stays inactive when the YAML omits the Decision route', async () => {
    const ctx = await load([
      "- name: '@deepseek-ai/dsh-context-decision-relevance'",
      '  config:',
      "    mode: 'on'",
    ], 'unset')

    // A half route is a configuration gap, not a fallback to the chat model.
    expect(ctx.decisionRelevance.enabled).toBe(false)
  })

  it('rejects stale config after plugin schema normalization', async () => {
    context = new Context()
    await context.plugin(LlmRuntime)
    await context.plugin(SessionStore)
    await expect(context.plugin(DecisionRelevanceFilter, {
      maxChars: 100,
    } as Record<string, unknown>)).rejects.toThrow(/unknown key "maxChars"/)
  })
})
