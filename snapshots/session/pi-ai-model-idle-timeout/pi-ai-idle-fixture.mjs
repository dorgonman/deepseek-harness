/** Keep one real pi-ai response pending until its model-specific idle deadline. */
import assert from 'node:assert/strict'
import { applyLoopbackServerEffect } from '../loopback-fixture-server.mjs'

const RECORDED_ENDPOINT = 'http://127.0.0.1:43119/v1/chat/completions'
const RECORDED_AUTHORITY = new URL(RECORDED_ENDPOINT).host

/** Cordis plugin name. */
export const name = 'pi-ai-idle-fixture'
/** Services owning scoped runtime context and completed-turn verification. */
export const inject = ['agents', 'systemPrompt', 'workingDirectory']

function requestUrl(input) {
  if (typeof input === 'string') return input
  if (input instanceof URL) return input.href
  if (input instanceof Request) return input.url
  return undefined
}

function transportInput(input, endpoint) {
  const url = requestUrl(input)
  if (url === RECORDED_ENDPOINT) {
    return input instanceof Request ? new Request(endpoint, input) : endpoint
  }
  if (url === undefined) return input
  let parsed
  try {
    parsed = new URL(url)
  } catch {
    return input
  }
  if (parsed.host === RECORDED_AUTHORITY) {
    throw new Error('pi-ai-idle-fixture: unexpected path for the owned authority')
  }
  return input
}

/**
 * Redirect only the fixture endpoint and verify client-owned cancellation.
 * @param {import('@deepseek-ai/cordis').Context} ctx - owner of the listener and fetch override.
 */
export async function apply(ctx) {
  ctx.on('agent/created', ({ agent }) => {
    agent.ctx.systemPrompt.context({
      name: 'working-directory:current',
      order: ctx.systemPrompt.getContextOrder('WORKING_DIRECTORY'),
      required: true,
      interpolate: false,
      text: 'Current working directory: the controlled snapshot workspace.',
    })
  })

  let requests = 0
  let closedResponses = 0
  let restoreFetch = () => {}
  let resolveClosed
  const responseClosed = new Promise(resolve => { resolveClosed = resolve })

  await applyLoopbackServerEffect(ctx, {
    label: 'pi-ai-idle-fixture',
    requestListener: (request, response) => {
      if (request.method !== 'POST' || request.url !== '/v1/chat/completions') {
        response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' })
        response.end('not found')
        return
      }
      const chunks = []
      request.on('data', chunk => { chunks.push(chunk) })
      request.on('end', () => {
        const body = JSON.parse(Buffer.concat(chunks).toString('utf8'))
        assert.equal(body.model, 'idle-wire-model')
        assert.equal(body.stream, true)
        assert.equal(body.tools, undefined)
        requests++
        response.on('close', () => {
          closedResponses++
          resolveClosed()
        })
        response.writeHead(200, { 'content-type': 'text/event-stream' })
        response.flushHeaders()
      })
    },
    onListening: (address) => {
      const endpoint = `http://127.0.0.1:${String(address.port)}/v1/chat/completions`
      const originalFetch = globalThis.fetch
      const fixtureFetch = (input, init) => originalFetch(transportInput(input, endpoint), init)
      globalThis.fetch = fixtureFetch
      restoreFetch = () => {
        if (globalThis.fetch !== fixtureFetch) {
          throw new Error('pi-ai-idle-fixture: global fetch owner changed before cleanup')
        }
        globalThis.fetch = originalFetch
      }
    },
    onCleanup: () => {
      restoreFetch()
      assert.equal(requests, 1)
      assert.equal(closedResponses, 1)
    },
  })

  ctx.on('agent/turn-stopping', async () => {
    assert.equal(requests, 1)
    let deadline
    try {
      await Promise.race([
        responseClosed,
        new Promise((_resolve, reject) => {
          deadline = setTimeout(() => reject(new Error('pi-ai-idle-fixture: client did not close the response')), 5000)
        }),
      ])
    } finally {
      clearTimeout(deadline)
    }
    assert.equal(closedResponses, 1)
  })
}
