/** Typed KINF Decisions use the shipped SDK profile with either stock tool executor. */
import { createServer } from 'node:http'
import type { IncomingHttpHeaders } from 'node:http'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { DeepSeekHarness } from '@deepseek-ai/dsh-sdk-client'
import type { RunResult } from '@deepseek-ai/dsh-sdk-client'
import { expect, it, onTestFinished } from 'vitest'
import { profileEnvironment } from './fixtures/profile-environment.ts'

type JsonObject = Record<string, unknown>

const DECISION_MODELS = [
  'flash-native-mtp-decision',
  'flash-next-uncensored-native-mtp-decision',
  'qwen27-standard-bf16-decision',
  'qwen27-uncensored-native-mtp-decision',
] as const
const PROVIDER = 'kinf-profile-fixture'
const ORDINARY_MODEL = 'kinf-chat'
const INSTANCE = 'fixture-managed'
const PTC_PATCH = fileURLToPath(new URL('../../../config/examples/ptc.yml', import.meta.url))
const SOURCE_PATCH = fileURLToPath(new URL('../../../src/sdk-source.cordis.patch.yml', import.meta.url))
const SOURCE_BIN = fileURLToPath(new URL('../../../src/bin.ts', import.meta.url))
const SOURCE_TSCONFIG = fileURLToPath(new URL('../../../tsconfig.json', import.meta.url))

/** Native fixture fields follow KINF's local_chat_tests DecisionReply result. */
function decisionReply(model: string) {
  const decision = {
    schema: 'kinf.decision.v1', mode: 'label_scores', label: 'yes', token_position: 8,
    logits: { yes: 1.25, no: 1.25 }, label_scores: { yes: 0.5, no: 0.5 }, calibration: 'uncalibrated',
  }
  const servedIdentity = {
    mode: 'mservice', instance: INSTANCE, logical_model: model, requested_model: model,
    requested_serving_variant_id: model, serving_variant_id: model,
    serving_variant_revision: 'decision-v1', model_artifact: 'fixture-artifact', artifact_revision: 'fixture-revision',
    backend: 'exllamav3', runtime_profile_id: 'fixture-profile', runtime_generation: '1.6.4+cu132.torch2.13.0',
    host_generation: '1ab-2bc-3cd', engine_generation: '2ab-3bc-4cd', resident_generation: '2ab-3bc-4cd',
    worker_pid: 42, config_revision: 'a'.repeat(64), target_config_revision: 'b'.repeat(64),
  }
  const usage = { prompt_tokens: 8, completion_tokens: 1, total_tokens: 9 }
  return {
    model, choices: [{ index: 0, message: { role: 'assistant', content: 'yes' }, finish_reason: 'length' }],
    generation: servedIdentity.host_generation, engine_generation: servedIdentity.engine_generation,
    decision, served_identity: servedIdentity, usage,
  }
}

/** Extract the one durable request header published by a fresh profile conversation. */
function requestHeader(result: RunResult) {
  const event = result.events.find(entry => entry.type === 'request/header')
  if (event?.type !== 'request/header') throw new Error('SDK profile omitted its durable request header')
  return event.data.header
}

it.for(['native', 'ptc'] as const)('completes all typed Decisions and preserves ordinary tools in the %s SDK profile', { timeout: 120_000, retry: 0 }, async (mode, test) => {
  const root = await mkdtemp(join(tmpdir(), 'sdk-kinf-decision-'))
  const requests: { body: JsonObject; headers: IncomingHttpHeaders; path: string | undefined }[] = []
  let fixtureError: unknown
  const server = createServer((request, response) => {
    let body = ''
    request.setEncoding('utf8')
    request.on('data', (chunk: string) => { body += chunk })
    request.on('end', () => {
      try {
        const parsed: unknown = JSON.parse(body)
        if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('Model request is not an object')
        const value = parsed as JsonObject
        requests.push({ body: value, headers: request.headers, path: request.url })
        if (value['kinf_decision'] !== undefined) {
          if (typeof value['model'] !== 'string') throw new Error('Decision request omitted its model')
          response.writeHead(200, { 'content-type': 'application/json' })
          response.end(JSON.stringify(decisionReply(value['model'])))
        } else {
          response.writeHead(200, { 'content-type': 'text/event-stream' })
          response.end([
            { choices: [{ delta: { role: 'assistant', content: 'ordinary chat preserved' }, index: 0, finish_reason: null }] },
            { choices: [{ delta: {}, index: 0, finish_reason: 'stop' }], usage: { prompt_tokens: 8, completion_tokens: 3 } },
          ].map(value => `data: ${JSON.stringify(value)}\n\n`).join('') + 'data: [DONE]\n\n')
        }
      } catch (error: unknown) {
        fixtureError = error
        response.writeHead(500, { 'content-type': 'application/json' })
        response.end('{"error":{"message":"invalid fixture request"}}')
      }
    })
  })
  onTestFinished(async () => {
    try {
      server.closeAllConnections()
      await new Promise<void>((resolve, reject) => server.close((error) => {
        if (error === undefined) resolve()
        else reject(error)
      }))
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('Decision fixture has no TCP address')
  const baseURL = `http://127.0.0.1:${address.port}/v1`
  const patch = join(root, 'kinf-decision.patch.yml')
  await writeFile(patch, JSON.stringify([
    { id: 'llm-deepseek', disabled: true },
    { id: 'session-title-llm', disabled: true },
    { id: 'llm-pi-ai', config: { providers: { [PROVIDER]: {
      api: 'openai-completions', apiKeyEnv: 'DEEPSEEK_API_KEY', baseURL, retryPolicy: { mode: 'normal', maxRetries: 0 },
      compat: { maxTokensField: 'max_tokens', supportsUsageInStreaming: false, supportsStore: false },
      headers: { 'X-Kinf-Mode': 'mservice', 'X-Kinf-Instance': INSTANCE },
      models: [
        { id: ORDINARY_MODEL, requestModel: 'chat-qwen-flash-next', contextWindow: 262144, maxTokens: 64,
          input: ['text'], reasoningEfforts: false, headers: { 'X-Kinf-Service': 'chat-qwen-flash-next' } },
        ...DECISION_MODELS.map(model => ({
          id: model, requestModel: model, requestProtocol: 'kinf-decision-v1',
          contextWindow: model.startsWith('flash') ? 262144 : 65536, maxTokens: 1, input: ['text'],
          reasoningEfforts: false, headers: { 'X-Kinf-Service': model },
        })),
      ],
    } } } },
  ]))
  const env = profileEnvironment(root, baseURL)
  env.NODE_OPTIONS = `--import=${import.meta.resolve('tsx/esm')}`
  env.TSX_TSCONFIG_PATH = SOURCE_TSCONFIG
  const run = async (model: string): Promise<RunResult> => {
    const harness = new DeepSeekHarness({
      cwd: root, processCwd: root, dshHome: join(root, 'home'), env, provider: PROVIDER, model,
      dshBin: SOURCE_BIN, patches: [SOURCE_PATCH, ...mode === 'ptc' ? [PTC_PATCH] : [], patch],
      initializeTimeoutMs: 30_000,
    })
    const abort = (): void => { void harness.close() }
    test.signal.addEventListener('abort', abort, { once: true })
    try {
      return await harness.session(`fixture-${model}`).run('Is this evidence relevant?')
    } finally {
      await harness.close()
      test.signal.removeEventListener('abort', abort)
    }
  }

  for (const model of DECISION_MODELS) {
    const result = await run(model)
    const fixture = decisionReply(model)
    expect(fixtureError).toBeUndefined()
    expect(result.finishReason).toBe('completed')
    const visible: unknown = JSON.parse(result.finalResponse)
    expect(visible).toMatchObject({
      model, decision: fixture.decision, served_identity: fixture.served_identity, usage: fixture.usage,
    })
    if (model === DECISION_MODELS[0]) {
      if (visible === null || typeof visible !== 'object' || Array.isArray(visible)) throw new Error('Decision output is not JSON data')
      expect((visible as JsonObject)['decision']).toMatchInlineSnapshot(`
        {
          "calibration": "uncalibrated",
          "label": "yes",
          "label_scores": {
            "no": 0.5,
            "yes": 0.5,
          },
          "logits": {
            "no": 1.25,
            "yes": 1.25,
          },
          "mode": "label_scores",
          "schema": "kinf.decision.v1",
          "token_position": 8,
        }
      `)
    }
    expect(result.events.filter(event => event.type === 'assistant/message')).toHaveLength(1)
    const header = requestHeader(result)
    expect(header.config).toMatchObject({ provider: PROVIDER, model, maxTokens: 1 })
    expect(header.tools).toBeUndefined()
    const request = requests.at(-1)!
    expect(request.path).toBe('/v1/chat/completions')
    expect(request.body).toMatchObject({ model, kinf_decision: 'label_scores', stream: false, max_tokens: 1 })
    for (const field of ['tools', 'tool_choice', 'parallel_tool_calls', 'stream_options', 'max_completion_tokens']) {
      expect(request.body).not.toHaveProperty(field)
    }
    expect(request.headers['x-kinf-mode']).toBe('mservice')
    expect(request.headers['x-kinf-instance']).toBe(INSTANCE)
    expect(request.headers['x-kinf-service']).toBe(model)
    expect(JSON.stringify(request.body['messages'])).not.toContain('is the only tool you can call directly')
    expect(JSON.stringify(request.body['messages'])).not.toContain('tools.read:')
    expect(result.events.some(event => event.type === 'tool/call')).toBe(false)
  }

  const ordinary = await run(ORDINARY_MODEL)
  expect(ordinary.finishReason).toBe('completed')
  expect(ordinary.finalResponse).toBe('ordinary chat preserved')
  expect(requestHeader(ordinary).tools?.length).toBeGreaterThan(0)
  const request = requests.at(-1)!
  expect(request.body).toMatchObject({ model: 'chat-qwen-flash-next', stream: true, max_tokens: 64 })
  expect(request.body).not.toHaveProperty('kinf_decision')
  expect(request.headers['x-kinf-service']).toBe('chat-qwen-flash-next')
  const toolNames = (request.body['tools'] as { function: { name: string } }[]).map(tool => tool.function.name)
  expect(toolNames).toContain(mode === 'ptc' ? 'run_code' : 'read')
  expect(requests).toHaveLength(DECISION_MODELS.length + 1)
  expect(fixtureError).toBeUndefined()
})
