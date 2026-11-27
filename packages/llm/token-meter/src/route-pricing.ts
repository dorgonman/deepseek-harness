/**
 * Request-projected surface pricing: replaces attachment-block heuristics with
 * the image and file representations sent to the routed model.
 *
 * @module @deepseek-ai/dsh-token-meter/route-pricing
 */

import type { ContentBlock, ImageBlock, LlmImageRequestPricing, RequestMessage, ToolSchema } from '@deepseek-ai/dsh-llm'
import { estimateContent, estimateMessage, estimateStructuralBlock, estimateToolSchemasTokens } from './estimate.ts'
import type { MeterSurfaceNode } from './surface-fold.ts'
import type { TokenSurfaceNode } from './types.ts'

type FileAttachmentRef = Extract<ContentBlock, { type: 'file' }>['attachment']

/** Price each image occurrence in request order, including its model-visible text. */
function priceImages(images: readonly ImageBlock[], pricing: LlmImageRequestPricing): number[] {
  const prices = pricing.priceImages(images)
  if (prices.length !== images.length) {
    throw new Error(
      `token meter: route image pricing answered ${prices.length} prices for ${images.length} occurrences`,
    )
  }
  return Array.from(prices, price => price.visualTokens + estimateContent([{ type: 'text', text: price.text }]))
}

/**
 * Estimate the projected messages and tools of an adapter request.
 * Files and substitutions are already represented by their projected text.
 * @param messages - the adapter-facing messages in request order.
 * @param tools - the declarations sent with the request.
 * @param pricing - image pricing bound to the prepared route, when declared.
 * @returns estimated input tokens, including declared visual tokens.
 */
export function estimateRequestTokens(
  messages: readonly RequestMessage[],
  tools: readonly ToolSchema[] | undefined,
  pricing?: LlmImageRequestPricing,
): number {
  const heuristic = messages.reduce((total, message) => total + estimateMessage(message), estimateToolSchemasTokens(tools))
  if (pricing === undefined) return heuristic
  const images = messages.flatMap(message => message.content.filter(block => block.type === 'image'))
  if (images.length === 0) return heuristic
  const visualAndText = priceImages(images, pricing).reduce((total, tokens) => total + tokens, 0)
  return images.reduce((total, image) => total - estimateStructuralBlock(image), heuristic + visualAndText)
}

/** One surface priced for a request route: public nodes plus their total. */
export interface PricedSurface {
  /** Positional nodes carrying both the route price and the fixed-heuristic price. */
  readonly nodes: TokenSurfaceNode[]
  /** Sum of the route prices across the surface. */
  readonly surfaceTokens: number
}

/**
 * Price one ordered surface under its model-request attachment projection.
 * @param nodes - the fold's current or snapshotted surface, in model-visible order.
 * @param pricing - the routed model's image pricing, or undefined to keep the fixed heuristic.
 * @param fileText - exact file handle projection used by the mounted LLM service.
 * @returns detached public nodes and their route-priced total.
 * @throws when the pricing answers a different occurrence count than it was
 *   asked — misalignment would silently misprice nodes, so it must fail loud.
 */
export function priceSurface(
  nodes: readonly MeterSurfaceNode[],
  pricing: LlmImageRequestPricing | undefined,
  fileText?: (ref: FileAttachmentRef) => string,
): PricedSurface {
  const images = pricing === undefined ? [] : nodes.flatMap(node => node.images)
  const hasFiles = fileText !== undefined && nodes.some(node => node.files.length > 0)
  if ((pricing === undefined || images.length === 0) && !hasFiles) {
    let surfaceTokens = 0
    const publicNodes = nodes.map((node) => {
      surfaceTokens += node.heuristicTokens
      return { seq: node.seq, tokens: node.heuristicTokens, heuristicTokens: node.heuristicTokens }
    })
    return { nodes: publicNodes, surfaceTokens }
  }
  const prices = pricing === undefined ? [] : priceImages(images, pricing)
  let cursor = 0
  let surfaceTokens = 0
  const publicNodes = nodes.map((node) => {
    let tokens = node.heuristicTokens
    if (fileText !== undefined && node.files.length > 0) {
      tokens -= node.fileStructuralTokens
      for (const file of node.files) {
        tokens += estimateContent([{ type: 'text', text: fileText(file) }])
      }
    }
    if (pricing !== undefined && node.images.length > 0) {
      tokens -= node.imageStructuralTokens
      for (let occurrence = 0; occurrence < node.images.length; occurrence += 1) {
        // oxlint-disable-next-line typescript/no-non-null-assertion -- length equality is asserted above
        tokens += prices[cursor]!
        cursor += 1
      }
    }
    surfaceTokens += tokens
    return { seq: node.seq, tokens, heuristicTokens: node.heuristicTokens }
  })
  return { nodes: publicNodes, surfaceTokens }
}
