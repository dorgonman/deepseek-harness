/** Fixed cwd stimulus for the portable Decision readout snapshot; execution keeps its real workspace. @module */
import type { Context } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-agent'
import type {} from '@deepseek-ai/dsh-system-prompt'

export const name = 'kinf-decision-context'
export const inject = ['systemPrompt', 'workingDirectory']

/** Register only fixed model-facing cwd prose in each snapshot Agent's scope. */
export function apply(ctx: Context): void {
  ctx.on('agent/created', ({ agent }) => {
    agent.ctx.systemPrompt.context({
      name: 'working-directory:current',
      order: ctx.systemPrompt.getContextOrder('WORKING_DIRECTORY'),
      required: true,
      interpolate: false,
      text: 'Current working directory: the controlled snapshot workspace.',
    })
  })
}
