import type { Agent } from '@deepseek-ai/dsh-agent'
import { createMessage, createToolResultMessage, createUserMessage, ToolCallId, type UserMessage } from '@deepseek-ai/dsh-llm'
import type { Session } from '@deepseek-ai/dsh-session'

/** Append one complete synthetic user, assistant, tool, and follow-up transaction. */
export function appendFixtureTurn(session: Session, turn: number, user: string | UserMessage['content'], result: string) {
  const callId = ToolCallId(`fixture-${turn}`)
  const model = { kind: 'model', provider: 'fixture', model: 'fixture' } as const
  session.append('turn/start', { turn })
  const input = session.append('user/message', createUserMessage({
    content: typeof user === 'string' ? [{ type: 'text', text: user }] : user,
    source: { kind: 'user' },
  }), { surfaceOp: 'append' })
  session.append('step/start', { turn, step: 1 })
  const assistant = session.append('assistant/message', {
    turn,
    step: 1,
    stream: [],
    message: createMessage({
      role: 'assistant',
      content: [{ type: 'tool-call', id: callId, name: 'fixture_read', arguments: '{}' }],
      source: model,
    }),
  }, { surfaceOp: 'append' })
  const call = session.append('tool/call', { turn, step: 1, callId, name: 'fixture_read', arguments: '{}' })
  const output = session.append('tool/result', {
    turn,
    step: 1,
    message: createToolResultMessage({ callId, content: [{ type: 'text', text: result }], isError: false }),
  }, { surfaceOp: 'append' })
  session.append('step/end', { turn, step: 1 })
  session.append('step/start', { turn, step: 2 })
  const followUp = session.append('assistant/message', {
    turn,
    step: 2,
    stream: [],
    message: createMessage({
      role: 'assistant',
      content: [{ type: 'text', text: 'Fixture result inspected.' }],
      source: model,
    }),
  }, { surfaceOp: 'append' })
  session.append('step/end', { turn, step: 2 })
  session.append('turn/end', { turn, reason: { kind: 'completed' } })
  return { input, assistant, call, output, followUp }
}

/** Supply the Session and options accessed by the isolated compaction fixture. */
export function fixtureAgent(session: Session): Agent {
  return { id: session.id, session, options: {} } as Agent
}
