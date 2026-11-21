/** Read-only PR snapshots and policy preflight/check workflow entry points. */

import fs from 'node:fs'
import process from 'node:process'

import config from './config.json' with { type: 'json' }
import { api, projectContext } from './github.mjs'
import {
  parseReferences,
  retainIssueReferences,
  requiresPullRequestPolicy,
  validatePullRequest,
} from './rules.mjs'

const projectRepository = `${config.organization}/${config.repository}`
const policyReadOptions = { redirect: 'error' }

function matchesRepository(repo, repository, repositoryId) {
  return repo?.id === repositoryId && typeof repo?.full_name === 'string'
    && repo.full_name.toLowerCase() === repository.toLowerCase()
}

function workflowPullRequest(event, workflow) {
  const { repository, repositoryId } = workflow
  const number = event?.pull_request?.number
  const id = Number(repositoryId)
  if (typeof repository !== 'string' || !/^[a-z\d_.-]+\/[a-z\d_.-]+$/i.test(repository)
    || repository.split('/').some(part => part === '.' || part === '..')
    || typeof repositoryId !== 'string' || !/^[1-9]\d*$/.test(repositoryId)
    || !Number.isSafeInteger(id) || !Number.isSafeInteger(number) || number <= 0) {
    throw new Error('Issue policy requires a valid workflow repository and PR number.')
  }
  if (!matchesRepository(event.repository, repository, id)
    || !matchesRepository(event.pull_request.base?.repo, repository, id)) {
    throw new Error('Issue policy workflow repository does not match the event repository and PR base.')
  }
  return { repository, repositoryId: id, number }
}

async function resolvingReferencesSnapshot(repository, number, pull, readOptions) {
  const references = parseReferences({ body: pull.body ?? '', repository })
  const issues = new Map()
  for (const issueNumber of references.all) {
    const issue = await api(`/repos/${repository}/issues/${issueNumber}`, readOptions)
    if (!issue.pull_request) issues.set(issueNumber, { priority: null })
  }
  return {
    number,
    references: retainIssueReferences(references, issues),
    issues,
  }
}

/**
 * Read current PR policy inputs, skipping references for exempt PRs.
 * @param {{repository: string, repositoryId: number, number: number}} scope Verified workflow PR.
 * @param {boolean} includeProject Read Project Priority for resolving Issues when true.
 * @returns {Promise<object>} Policy snapshot; rejects a failed read or identity mismatch and performs no writes.
 */
export async function pullRequestSnapshot({ repository, repositoryId, number }, includeProject = true) {
  const path = `/repos/${repository}/pulls/${number}`
  const pull = await api(path, policyReadOptions)
  if (pull?.number !== number || !matchesRepository(pull.base?.repo, repository, repositoryId)) {
    throw new Error('Issue policy REST pull request does not match the verified repository and number.')
  }
  const snapshot = {
    number,
    isDraft: pull.draft,
    authorType: pull.user?.type ?? 'User',
    reviewRequestCount: 0,
    reviewCount: 0,
    labels: pull.labels.map((label) => label.name),
    references: { all: [], resolving: [], related: [] },
    issues: new Map(),
  }
  if (snapshot.isDraft || ['Bot', 'App'].includes(snapshot.authorType)) return snapshot
  const [reviewRequests, reviews] = await Promise.all([
    api(`${path}/requested_reviewers`, policyReadOptions),
    api(`${path}/reviews?per_page=100`, policyReadOptions),
  ])
  snapshot.reviewRequestCount = reviewRequests.users.length + reviewRequests.teams.length
  snapshot.reviewCount = reviews.length
  if (!requiresPullRequestPolicy(snapshot)) return snapshot
  Object.assign(snapshot, await resolvingReferencesSnapshot(repository, number, pull, policyReadOptions))
  if (snapshot.references.resolving.length > 0 && repository.toLowerCase() !== projectRepository.toLowerCase()) {
    throw new Error('Issue policy resolving references require a configured Project authority for the workflow repository.')
  }
  if (includeProject) {
    for (const issueNumber of snapshot.references.resolving) {
      const context = await projectContext(issueNumber)
      snapshot.issues.get(issueNumber).priority = context.item?.priorityValue?.name ?? null
    }
  }
  return snapshot
}

/**
 * Read PR creation time and Issue references without review eligibility gating.
 * @param {number} number Pull-request number.
 * @returns {Promise<object>} Lifecycle snapshot without Project reads or writes.
 */
export async function lifecyclePullRequestSnapshot(number) {
  const pull = await api(`/repos/${projectRepository}/pulls/${number}`)
  return {
    ...(await resolvingReferencesSnapshot(projectRepository, number, pull)),
    createdAt: pull.created_at,
  }
}

const EXEMPT_MESSAGE =
  'Issue policy exempt：当前 PR 不在强制范围（Draft、Bot/App 或尚无 review request/review）。\n'

/**
 * Determine current policy eligibility and Project access needs without Project credentials.
 * @param {object} event GitHub event identifying the PR and repository.
 * @param {{repository?: string, repositoryId?: string}} workflow GitHub Actions repository identity.
 * @returns {Promise<{eligible: boolean, needsProject: boolean}>} Trusted workflow decisions.
 */
export async function runPullRequestPreflight(event, workflow) {
  const pull = await pullRequestSnapshot(workflowPullRequest(event, workflow), false)
  const eligible = requiresPullRequestPolicy(pull)
  const needsProject = eligible && pull.references.resolving.length > 0
  if (process.env.GITHUB_OUTPUT) {
    fs.appendFileSync(
      process.env.GITHUB_OUTPUT,
      `eligible=${eligible}\nexempt=${!eligible}\nneeds-project=${needsProject}\n`,
    )
  }
  process.stdout.write(eligible ? 'Issue policy applicable；执行完整校验。\n' : EXEMPT_MESSAGE)
  return { eligible, needsProject }
}

/**
 * Enforce all PR rules against current GitHub state, independently of preflight.
 * @param {object} event GitHub event identifying the PR and repository.
 * @param {{repository?: string, repositoryId?: string}} workflow GitHub Actions repository identity.
 * @returns {Promise<void>} Resolves on success or exemption; rejects policy failures.
 */
export async function runPullRequestCheck(event, workflow) {
  const pull = await pullRequestSnapshot(workflowPullRequest(event, workflow))
  const errors = validatePullRequest(pull)
  if (errors.length > 0) {
    for (const error of errors) process.stdout.write(`::error::${error}\n`)
    throw new Error(`Issue policy 未通过，共 ${errors.length} 项`)
  }
  process.stdout.write(
    requiresPullRequestPolicy(pull) ? 'Issue policy 通过。\n' : EXEMPT_MESSAGE,
  )
}
