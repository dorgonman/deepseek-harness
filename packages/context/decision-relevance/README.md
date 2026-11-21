---
description: "Model-judged tool-output relevance for deployments composing a Decision route: choosing whether irrelevant tool results should leave the request, or debugging why a tool result was elided."
kind: "package-reference"
---

# @deepseek-ai/dsh-context-decision-relevance

English | [中文](README.zh.md)

## Summary

`dsh-context-decision-relevance` keeps tool results that no longer serve the current goal from riding along on every request. For each oversized tool result in an outbound request, it asks a separately configured Decision model a one-token relevance question and replaces judged-irrelevant bodies with a short elision marker. The complete original result stays in the session log for exact replay and inspection; only the wire copy for that one request changes. Every evaluation is bounded, and anything uncertain, failed, or cancelled is retained, so the feature can only ever shrink a request, never lose history.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further Exploration](#further-exploration)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>
## Use this package

Mount this package when long-running sessions accumulate large tool results that stop mattering — old file reads, superseded search output, exhausted command logs — and you want the model to stop re-reading them. It is opt-in: with no configuration the package registers nothing and ordinary requests are untouched.

### Smallest working composition

The package needs the LLM runtime to reach the Decision route and the session store to record its evidence:

```yaml
- name: '@deepseek-ai/dsh-llm'
- name: '@deepseek-ai/dsh-session'
- name: '@deepseek-ai/dsh-context-decision-relevance'
  config:
    mode: 'on'
    provider: 'my-decision-provider'
    model: 'my-decision-model'
```

`provider` and `model` must be configured together; either one alone leaves the feature inactive regardless of `mode`. The route is resolved independently of the conversation's own model, so the chat model stays exactly what it was. You can verify success by checking that future requests show the elision marker where an irrelevant result used to be; the full original remains in the session log.

### What gets considered

Only tool results are candidates, and only some of them:

- the result must not be an error — a failed tool result stays visible, because its error is part of the record of what was tried;
- its text must reach `minCandidateChars`, so small results are never worth a judgment call;
- at most `maxCandidatesPerRequest` candidates are evaluated per request, in message order; the rest are retained.

Each candidate is judged once and its verdict is remembered for the session, so a long conversation does not re-ask about the same result on every turn.

### Setting the thresholds

All settings are optional; the generated [configuration catalog](../../../docs/config-catalog.md#deepseek-aidsh-context-decision-relevance) is the exhaustive source.

| Field | Default | Meaning |
|---|---|---|
| `mode` | `off` | `off` disables the feature; `on` applies validated selections to the outbound projection. |
| `provider` | unset | Canonical provider of the Decision route. |
| `model` | unset | Canonical model of the Decision route. |
| `retainThreshold` | `0.6` | Relevance score required to retain a tool result. |
| `uncertaintyMargin` | `0.1` | Half-width of the band around `retainThreshold` that counts as uncertain. |
| `maxCandidatesPerRequest` | `4` | Maximum candidates evaluated per request. |
| `minCandidateChars` | `512` | Minimum text size, in Unicode code points, before a result is eligible. |
| `maxInputChars` | `4096` | Maximum text size, in Unicode code points, sent to the Decision model. |
| `timeoutMs` | `15000` | End-to-end deadline for one evaluation. |
| `elisionText` | built-in | Replacement text for an elided body. |

A score below `retainThreshold - uncertaintyMargin` elides the result; a score at or above `retainThreshold + uncertaintyMargin` retains it; anything in between is uncertain and retained. An unknown setting rejects the plugin at construction.

### When judgments run

Evaluations run inside the LLM runtime's request-projection stage, after the outbound message list is resolved and before the adapter is dispatched. They are background calls on their own route: they never change the conversation's model, its sampling, or its tools.

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>Implementation internals — click to expand</summary>

This section explains the design decisions behind the filter; the observable behavior is fully covered in [Use this package](#use-this-package).

### Design philosophy

The filter is built on four commitments:

- **Outbound projection, never a durable rewrite.** The session log is append-only and the loop's request must stay a pure function of it. Selection therefore rewrites only the wire copy for one request, leaving every original message and its tool-call correlation intact.
- **An independent qualified route.** The Decision model is resolved through its own `prepareCall`, not the conversation's. Its served identity is validated against the request, so a route that silently serves something else is treated as a failure rather than a verdict.
- **Fail-open everywhere.** A transport error, a malformed readout, a served-model mismatch, an uncertainty-band score, or a cancellation all retain the content. The filter can never delete something it is unsure about.
- **Bounded durable evidence.** Each pass appends one `context/decision-relevance` event naming the conversation route, whether anything was elided, and one bounded record per candidate: candidate identity, requested and served canonical identity, score, disposition, and a short reason. No credentials and no history dumps.

### Evaluation mechanics

For each candidate the filter builds a plain-text prompt from the most recent user message and a code-point-bounded copy of the candidate, freezes it as a one-token, zero-temperature `label_scores` request with purpose `decision-relevance`, and assembles the adapter's response under a deadline fused with the caller's signal. The readout must carry the expected schema, mode, label, and finite scores in `[0, 1]`; anything else is a malformed readout and retains the content. Exact signatures are in [`src/index.ts`](src/index.ts).

### Source map

The npm package includes the source exported under `./src/*` and the generated declaration and JavaScript source maps.

| File | Role |
|---|---|
| [`src/index.ts`](src/index.ts) | Plugin entry: `DecisionRelevanceFilter` service, `project`, readout parsing |
| [`src/config.ts`](src/config.ts) | Defaults, sparse-config validation, code-point bounding |
| [`src/types.ts`](src/types.ts) | `DecisionRelevanceConfig`, `ResolvedConfig`, `RelevanceJudgment`, `ProjectionResult` |

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

Read these pages when the package-level contract is not enough; they move from the projection seam to the Decision protocol and the durable vocabulary.

- [LLM runtime](../../llm/llm/README.md) — the request-projection seam where selection is applied.
- [Pi AI provider](../../llm/llm-pi-ai/README.md) — the adapter that renders the Decision readout.
- [Compaction subsystem reference](../../../docs/subsystems/compaction.md) — the durable vocabulary this package's evidence joins.
- [Generated configuration catalog](../../../docs/config-catalog.md#deepseek-aidsh-context-decision-relevance) — every accepted config field and its source declaration.

-----

<a id="model-experience"></a>
## Model Experience

### Elided tool result

#### What the model sees

For one request, an irrelevant tool result's body is replaced by `elisionText`. The message keeps its identity, its `tool` role, and its tool-call correlation, so the assistant's earlier call still resolves; only the text is gone. Relevant, uncertain, and failed results are sent unchanged.

#### Token effect

Each elided result costs `elisionText` instead of its original body, and each evaluated candidate costs one bounded one-token Decision call on the Decision route. A request with no eligible candidate makes no Decision call at all.

#### KV Cache effect

Elision changes the prompt at the first elided message, so reuse from that point is lost for the request that first applies it. Because a verdict is remembered per session, later requests keep the same elision and reuse normally from there on.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>


These limits define when filtering is a poor fit or needs special care; they are the current package constraints.

- **Relevance is judged per result, not per span** — a result that is half still-relevant is elided or retained whole; partial selection does not ship.
- **The goal is the latest user message** — the prompt states the most recent user text as the goal, so a result that serves an older standing instruction may be judged irrelevant.
- **Character bounds are not token bounds** — `minCandidateChars` and `maxInputChars` are Unicode code points; provider token density varies, so the token meter remains the authority on request pressure.
- **No fallback route** — when the configured Decision route fails, content is retained; the filter never substitutes another Decision model.

<a id="dev-note"></a>
### Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

This Dev Note is working context for maintainers and is explicitly non-authoritative; shipped behavior lives in the sections above, the package code, and the linked Agent Notes.

- **Turn and step attribution, deferred** — the projection seam does not receive the loop's turn and step, so evidence records the conversation route without a position.
- **Per-candidate selection, undecided** — a verdict is cached by message identity for the life of the service; whether a changed goal should invalidate earlier verdicts is not yet decided.

</details>
