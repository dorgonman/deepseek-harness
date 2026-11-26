---
description: "Automatic conversation condensation for deployments choosing, tuning, or debugging how older history is summarized as token pressure builds."
kind: "package-reference"
---

# @deepseek-ai/dsh-compaction-basic

English | [中文](README.zh.md)

## Summary

This package keeps long agent conversations working near the model's context limit. As token pressure builds, it condenses the oldest history into a summary while preserving recent messages; after a context-overflow error, it condenses and retries. You can also request condensation with `/compact` and optionally trim oversized tool outputs first. Condensation uses one extra model request and retains only its summary text. It cannot reduce the system prompt, tools, or session prefix, or split one indivisible unit such as a single huge tool call.

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

Mount this package to get automatic conversation condensation in a composition that already provides an LLM, session storage, and token measurement. The shipped `dsh` base enables it by default; mount it explicitly to control when condensation starts.

<a id="opt-in-hybrid-provider"></a>
### Opt-in Hybrid provider

Select `@deepseek-ai/dsh-compaction-basic/hybrid` in place of the Basic provider in a composition that also mounts Session Query. It uses the same compaction settings and entry points. Configure a text summarizer with a known context window and give the destination route a known completion cap.

Hybrid traces the selected Surface span back to original Session events. Previous checkpoint text stays out of summarizer requests. It quotes each history group as data and sends balanced groups in bounded chunks. When a tool result prevents its group from fitting, the request cites its original event so Session Query can recover the exact result. Each request reserves the configured output tokens and prices the actual formatted input with the existing token estimator.

Hybrid combines chunk results in source order. During a model switch it prices the checkpoint with the pending prompt, accepted input, runtime context, and tools under the resolved destination envelope. If that request exceeds the budget, the switch fails before the checkpoint commits. Hybrid uses the existing transaction and flush paths. An interrupted physical write can leave a partial durable log. If Hybrid cannot confirm the write, it stops the pressure step or refuses the overflow retry.

Cold Session replay restores the committed checkpoint and original history. Hybrid records each successful chunk in the summary record so adapter replay can reproduce its output in call order. The source test suite uses synthetic Sessions and fake adapters.

The [resident compaction example](../../../apps/cli/config/examples/kinf-resident-compaction.cordis.yml) opts Hybrid into an existing local KINF managed listener. It ranks the configured 27B variant ahead of Flash-Next when both are eligible. Set the order with `residentPreferredVariantIds`. Match each observation route to its pi-ai model row, including the listener and logical model. Enable `residentTransport: true` on those rows.

Hybrid checks fresh observations for loaded text Chat deployments and compares the projected history estimate against the resident's reported runtime context and output budgets. KINF owns final admission. Automatic, manual, overflow, and model-switch compaction share the selector. Hybrid prices each packed chunk against the chosen window before dispatch and records the resident identity, selection reason, and each dispatched projection digest in the committed summary.

The resident contract prohibits cold loads and evictions. A definite stale refusal before first admission permits one refresh and reselect. Missing eligible capacity ends with a typed `no-eligible` cause. An accepted request or uncertain delivery cannot trigger a reselect. The example uses an explicitly configured loopback listener. A conversation alias with a node suffix scopes selection to that configured node unless an explicit compaction allowlist is set.

### What you get

With the default settings you get four behaviors: automatic condensation as the conversation grows toward the model's context limit; recovery after a confirmed context-overflow error, where the conversation condenses and the request retries; on-demand condensation through the `/compact` command; and — when the pruner is mounted — trimming of oversized tool outputs before condensation.

### Smallest working composition

Mount session storage, token measurement, the optional pruner, this backend, and optionally the on-demand command:

```yaml
- name: '@deepseek-ai/dsh-session'
- name: '@deepseek-ai/dsh-token-meter'
- name: '@deepseek-ai/dsh-compaction-tool-result-pruner'
- name: '@deepseek-ai/dsh-compaction-basic'
- name: '@deepseek-ai/dsh-command-compact'
```

You can verify success by watching the conversation continue past the point where it would otherwise overflow, and by running `/compact` for an immediate condensation. If the composition lacks an LLM, session storage, or token measurement, the plugin fails to load. One backend can serve models with different context sizes; give each route its own threshold and retention with a per-model override:

```yaml
- name: '@deepseek-ai/dsh-compaction-basic'
  config:
    thresholdRatio: 0.8
    retainRatio: 0.16
    modelPolicies:
      - provider: local
        model: small-context
        thresholdRatio: 0.7
        headroomTokens: 4096
        maxTokens: 4096
        retainTokens: 2048
```

### Tuning when condensation starts

All settings are optional. With context window `W`, effective request output cap `O`, and headroom `B`, the default trigger is `floor(min(W × 0.8, W − O − B))`, where `B` is 65,536 tokens unless that leaves no pressure threshold above the retained history. In that case omitted headroom is one eighth of `W − O`, rounded down. Retention keeps the newest 16% of `W − O` verbatim. The table below lists every setting; the generated [configuration catalog](../../../docs/config-catalog.md#deepseek-aidsh-compaction-basic) also includes their types.

| Field | Default | Meaning |
|---|---|---|
| `thresholdRatio` | `0.8` | Window fraction used in `floor(min(W × thresholdRatio, W − O − headroomTokens))`. |
| `headroomTokens` | `65536` | Additional pressure headroom beyond the routed output reservation; a non-negative integer. |
| `retainRatio` | `0.16` | Recent conversation kept verbatim as a fraction of `W − O`; mutually exclusive with `retainTokens`. |
| `retainTokens` | — | Absolute recent-conversation budget kept verbatim; mutually exclusive with `retainRatio` and must be below the resolved threshold. |
| `summarizationProvider` | `''` | Set together with `summarizationModel`; an empty pair uses the latest routed request target, then the `AgentOptions` pair. |
| `summarizationModel` | `''` | Set together with `summarizationProvider`; an empty pair uses the latest routed request target, then the `AgentOptions` pair. |
| `summarizationFallback` | `latest-route` | `error` fails an empty pair with `SUMMARY_TARGET_UNCONFIGURED` instead of using the latest routed request or `AgentOptions`; use it when a policy must never summarize through an unconfigured or depleted conversation provider. |
| `maxTokens` | `headroomTokens` (`65536`) | Positive summary output cap, including any provider-counted reasoning tokens. Explicit per-model caps override explicit global caps; otherwise the cap follows the resolved headroom. |
| `compactionRetries` | `1` | Extra condensation attempts after the first when pressure remains above threshold. |
| `maxOverflowRetries` | `1` | Maximum retries after a confirmed context-window overflow; `0` disables recovery only. |
| `modelPolicies` | `[]` | Exact `{ provider, model, ...partialPolicy }` overrides for individual model routes. |
| `auto` | `true` | Enable automatic condensation and overflow recovery; set `false` for manual-only operation. |
| `incremental` | `false` | Hybrid only: record append-origin coverage on each summary and later quote only RAW events above the committed watermark. A carried checkpoint is assembled locally and never sent to a summarizer. |
| `factLedger` | `false` | With `incremental`, extract typed sourced facts per chunk, commit the merged ledger, and compose the checkpoint from the bounded fact view. Requires `incremental: true`. |
| `factMaxClaims` | `64` | Maximum facts accepted from one chunk response. |
| `factMaxClaimChars` | `2000` | Maximum characters in one extracted claim. |
| `summarizationMode` | `fixed` | `resident-preferred` resolves one already-loaded resident deployment per compaction through the mounted observation source and binds its identity to the dispatch; the transport must enforce that binding, and an unsupported transport refuses with `UNSUPPORTED_RESIDENT_TRANSPORT` instead of falling back. `fixed` keeps the configured or routed target. |
| `residentGatewayId` | — | Authorized gateway identity; required for resident-preferred. |
| `residentPinnedNodeId` | — | Conversation-pinned node used when no allowlist is supplied. |
| `residentAllowedNodeIds` | — | Explicit compaction node allowlist; replaces the pin. |
| `residentPreferredVariantIds` | — | Configured quality preference, most preferred first. |
| `residentProvider` | — | DSH LLM provider route that fronts the gateway. |
| `residentMaxAgeMs` | `60000` | Maximum accepted observation age, 1–60000 ms. |
| `residentObservationRoutes` | — | Hybrid KINF loopback bindings with node ID, node name, instance, logical model, and URL. |

Misconfiguration fails fast: unknown settings, duplicate per-model overrides, invalid token counts, both retention forms together, or a retention ratio at least as large as the threshold ratio reject the plugin at load. When the model is first used, `W − O − B` must be positive and the resolved retained budget must be below the trigger. Zero headroom requires an explicit positive `maxTokens`, globally or in that model policy. Explicit headroom must fit the window. An omitted summary cap follows the adjusted headroom on the conversation route when it declares a window, with a minimum of one token. Explicit summary targets keep their configured or inherited cap. Lower `thresholdRatio` to compact earlier.

Basic with `summarizationFallback: error` and the Hybrid backend verify the exact call before sending history. Missing or invalid context capacity fails with `SUMMARY_TARGET_UNVERIFIED`. A non-Chat protocol or declared input modalities that exclude text fail with `SUMMARY_TARGET_NOT_CHAT`. Undeclared modalities are allowed. Basic with `latest-route` still accepts an adapter that omits its context capacity.

When capacity is known, the full summary input and output cap must fit within the context window. If they exceed that window, compaction fails with `SUMMARY_CAPACITY_EXCEEDED`. The estimate includes transcript framing and the final instruction.

<a id="persist-preset-summary-policy"></a>
### Persist preset summary policy

Policy control addresses canonical `@deepseek-ai/dsh-agent-preset` declarations with enabled `compaction` and `compaction-basic` row IDs. Disabled or conditional compaction rows are refused.

When the profile mounts both the preset registry and configuration editor, the host exposes `compactionPolicy.read(entryId)` and `setSummaryTarget(request)`. Read the exact declaring row, then pass its revision, an explicit provider/model/output budget, and only the existing source-route overrides you own. Policy wire types are exported from `@deepseek-ai/dsh-compaction-basic/types`. The operation sets fallback to `error`; omitted overrides, conversation defaults and sibling plugin values remain unchanged. Missing or ambiguous rows, stale revisions and unverifiable Chat targets reject the operation.

Keep the unmodified read snapshot for explicit rollback. `restoreSummaryPolicy({ expectedRevision, snapshot, modelPolicyRoutes })` restores its four default summary leaves and only the selected existing route overrides. Omitted snapshot leaves become absent again, including inherited bindings and output budgets; heterogeneous route policies retain their captured values. Restore checks the current revision, declaration identity and snapshot integrity. It validates configuration and activation without checking captured target availability or running inference.

Each write edits one preset through the editor’s locked, freshly reconciled transaction. `setSummaryTarget` checks target metadata before editing and again after replacement-preset reconciliation. Activation failure, or metadata failure during target-setting, restores that call’s prior patch and runtime. Success reports persisted and activated policy for new bindings, while existing Agents retain their previous revision. This does not guarantee future provider availability. Separate preset writes are not atomic together; explicit restore is a separate operation. The editor stores the complete preset configuration, retaining its current plugin list as an explicit profile override.

### What happens when condensation runs

The oldest balanced span is replaced by one summary message and the recent tail stays verbatim; the conversation continues from the summary. The operation reports how many history items were condensed and the estimated tokens freed. If nothing can be condensed safely — for example the whole conversation is one indivisible unit — nothing changes and nothing is written to the session log. If no model is available to write the summary (no configured target and no routed request yet), condensation fails with a clear error telling you to configure the summarization provider and model or route one request.

### Switching models

Before a changed provider/model request is committed, Basic automatic compaction measures the existing history against the destination model's capacity, output reservation, and pressure policy. When condensation is needed, it uses the previous routed model and its summary settings; an explicitly configured summary target still takes precedence. The destination request proceeds only after history falls below its threshold. Failure reports `MODEL_SWITCH_COMPACTION_FAILED` with the previous route, destination route, effective summarizer, and failure stage, and leaves the previous request header intact; completed pruning or condensation remains in the session even when a later attempt fails. The check runs when the next request is prepared, and `auto: false` disables it.

The destination must have a registered adapter, declared context capacity, and a valid pressure budget even for short history. For example, an unconfigured 65,536-token destination that reserves 32,768 output tokens uses 4,096 headroom and a 28,672-token pressure threshold. Set `modelPolicies[].headroomTokens` to override that policy for an exact route. See the [upgrade guide](../../../docs/upgrade-guide/v0.2.1-alpha.4/model-switch-compaction/guide.md) for migration.

The opt-in Hybrid backend estimates the full pending request and checks it against the destination's declared context capacity and effective completion reservation. It skips automatic compaction for the switch when that request fits. If the request exceeds capacity, the destination pressure policy sizes the replacement. The transport limits a fallback completion cap to the capacity left after input.

With automatic compaction enabled and a declared context window, final admission checks the frozen adapter request after attachment and tool projection. It records the estimated input, output reservation, and window in `request/admission`. A persisted Session requires its own write handle to acknowledge the durable receipt before dispatch. A missing acknowledgement raises `CONTEXT_ADMISSION_REFUSED` and asks the caller to reopen the Session. A budget refusal or flush failure stops dispatch. An in-memory Session records the receipt in its live log.

### On-demand condensation with /compact

With `dsh-command-compact` mounted, type `/compact` in a chat UI to condense immediately, even below the pressure threshold. The command reports how many history items were condensed and the estimated tokens saved. While the agent is mid-turn or condensation is already running, `/compact` reports that condensation is unavailable; prompts you send while it runs are accepted and start after it finishes.

### Trimming oversized tool outputs

Mount `dsh-compaction-tool-result-pruner` before this package to trim oversized tool results as part of condensation. Trimming makes no model call and can remove the need to summarize at all: when the trimmed conversation fits within the threshold, condensation skips the summary. Trimming only runs after a condensation trigger qualifies — a below-pressure conversation is never touched.

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>Implementation internals — click to expand</summary>

This section explains the design decisions behind the backend; the observable behavior is fully covered in [Use this package](#use-this-package).

### Design philosophy

The backend is built on four commitments:

- **One measurement service prices every decision.** The singleton `ctx.tokenMeter` measures the latest canonical logged envelope and current surface at one consumed-log revision. When the routed adapter declares request-image pricing, the meter applies it to image history. Model-switch pressure, retention, and range selection use destination pricing; region shrink validation uses the last committed route; logged replacement shadow prices stay on the route-independent heuristic so pure projection folds remain consistent.
- **The log-recorded bracket is the transaction.** All entry points share one bracket-first region transaction: validate the range and live lock, append `compaction/start` synchronously, prepare and await the summary, revalidate, append `compaction/summary` plus the replacement, and make exactly one closing attempt. Automatic and explicit-region calls require a numeric open-turn owner and whole-surface stability; `compactNow()` reserves idle admission, uses `turn: null`, accepts append-only context outside its selected span, flushes every closed attempt, and releases admission in `finally`.
- **Summary input follows exact-route capability.** Ordinary Chat targets with explicit `nativeTools: false` receive an inert text transcript. Other ordinary Chat targets retain `conversation-prefix-v1`, replaying the system head, tools and selected history for warm-prefix reuse where projection permits.
- **`summarize()` is the sole subclass hook.** A template- or remote-summarizer subclass can override it while pressure, retention, cited source events, shrink validation, and shadowed-token accounting stay on the token meter.

### Automatic triggers and overflow recovery

The `agent/request` listener waits for routing listeners to select the destination, then checks its pressure before the new header is logged. It uses `compactPressure` directly because `compactIfNeeded` reads the committed header; both paths generate summaries through `summarize()`.

With `auto: true`, a serial `agent/pre-step` listener checks pressure before request derivation: it prices the latest durable routed request envelope through `ctx.tokenMeter`, and when pressure crosses the routed model's threshold it prunes, then summarizes the oldest balanced span while keeping a priced recent tail. Every selected range starts at the first surface node that is not a `system/message`, so a system prompt at surface node 0 is never shadowed; a later `system/message` appended by an in-history prompt update is ordinary history that the range may shadow, and the agent loop's projection then replaces node 0 with the current prompt when their text differs ([decision rule](../../core/agent-loop/README.md#understand-the-implementation)). The `agent/request-error` listener reacts to a provider-confirmed `CONTEXT_WINDOW_EXCEEDED`: it bypasses the normal threshold and retention policy, attempts one maximal balanced head reduction, and authorizes a retry only after the surface replacement generation advances. Cancellation stays authoritative throughout.

Pressure policy resolves capacity from the adapter that owns the durable route. Missing capacity, output plus headroom exhausting the window, or a retained budget at least as large as the threshold makes the manual pressure path throw a target-specific configuration error. The `agent/pre-step` listener warns once for that exact target and continues the turn; provider-confirmed overflow recovery remains available. Model-switch checks instead reject the request with `MODEL_SWITCH_COMPACTION_FAILED`, including when capacity or adapter lookup fails.

### Summarization mechanics

One direct `ctx.llm.stream()` call uses the resolved summary target and output cap without running the loop-only `agent/request` hook. `conversation-prefix-v1` replays the derived system head, selected messages and header tools, with `Session.toolHistory()` for deferred and retained declarations. The selected adapter resolves image references or rejects them. An empty system head stays outside the compacted range but contributes no message. The final compaction instruction is a frozen `RequestUserInput` without durable identity; source history and the checkpoint remain durable. The call uses purpose `compaction`; only text enters the checkpoint, reasoning stays excluded, and tool-call or image output fails explicitly. The replacement user message carries `<compacted-summary>` framing; the raw result remains on `compaction/summary`.

`text-transcript-v1` is selected only for an ordinary Chat target with explicit `nativeTools: false`. The auxiliary request quotes source roles, text/reasoning, tool names, raw arguments, call IDs, results/errors and attachment handle text in order. It sends no tool declarations/history, structured tool blocks or provider replay metadata. Unsupported content rejects instead of disappearing. This conversion does not rewrite the original Session or change region selection, commit or cancellation ownership.

New `compaction/summary` records declare `requestProjection` as `conversation-prefix-v1` or `text-transcript-v1`, so the auxiliary input choice is reconstructable from the recorded source events and code. Absence retains the historical conversation-prefix behavior.

### The region transaction

Failed summary requests dispatch synchronous `compaction/summary-error` after checking cancellation and selection stability. A recovery listener must record a durable input change before requesting retry. The backend re-derives the selected messages and refreshes their token prices and shrink baseline. The image-offload plugin owns image selection; its recorded omissions remain effective if the summary later fails or is cancelled.

The transaction validates the surface span and the durable lock, appends `compaction/start`, summarizes through the hook, revalidates stability (whole-surface for automatic calls, selected-span for manual calls), rejects a summary that does not shrink its source, appends `compaction/summary` plus the replacement `user/message`, and makes exactly one `compaction/end` attempt. A live unmatched start is the durable lock: an unmatched marker before a newer `session/end-seed` is stale evidence from a prior lifecycle and does not block; one after that boundary reports `busy`. A failed close deliberately leaves a blocking orphan. Cancellation remains authoritative after cleanup and durability.

### Config resolution

`resolveConfig` validates and detaches defaults, `resolveTargetPolicy` merges exact provider/model overrides, and `resolveCompactSpec` requires explicit adapter capacity and routed output reservation to resolve the trigger and retained budget. The effective envelope’s `maxTokens` supplies that reservation, falling back to the adapter default and then zero. Model discovery (`listModels()`) is never consulted for policy; capacity comes from the durable route or the proposed destination during a model switch.

### Source map

| File | Role |
|---|---|
| [`src/index.ts`](src/index.ts) | Plugin entry: `BasicCompactionEngine`, automatic listeners, entry-point dispatch |
| [`src/region.ts`](src/region.ts) | Retention selection and the shared bracket-first compaction transaction |
| [`src/summarizer.ts`](src/summarizer.ts) | Default `ctx.llm.stream()` summarization, checkpoint framing, safe-summary projection |
| [`src/config.ts`](src/config.ts) | Load-time validation and routed-model policy resolution |
| [`src/types.ts`](src/types.ts) | `BasicCompactionConfig` and resolved policy vocabulary |
| [`src/policy-control.ts`](src/policy-control.ts) | Revision-checked profile summary-policy edits |
| [`src/policy-types.ts`](src/policy-types.ts) | Policy control request and result fields |

The package root's type declarations add `compaction` to `LlmRequestPurposeMap`. Direct callers constructing compaction requests use `import type {} from '@deepseek-ai/dsh-compaction-basic'` so `GenerateOptions.purpose` includes that key.

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

Read these pages when the package-level contract is not enough; they move from the shared seam to the optional companions and the decision evidence.

- [Compaction seam](../compaction/README.md) — the condensation contract this backend implements.
- [Compaction subsystem reference](../../../docs/subsystems/compaction.md) — the condensation vocabulary, results, and service behavior.
- [Tool-result pruner](../compaction-tool-result-pruner/README.md) — the optional companion that trims oversized tool outputs first.
- [Human /compact command](../command-compact/README.md) — on-demand condensation without waiting for pressure.
- [Token meter](../../llm/token-meter/README.md) — the measurement service that decides when to condense.
- [Generated configuration catalog](../../../docs/config-catalog.md#deepseek-aidsh-compaction-basic) — every accepted config field and its source declaration.

-----

<a id="model-experience"></a>
## Model Experience

### Conversation history

#### What the model sees

After a successful step crosses the threshold, oversized tool results are first rewritten when the optional pruner is loaded. If summarization remains necessary, the next request receives the checkpoint preamble below, a blank line, `<compacted-summary>`, the data-dependent summary, and `</compacted-summary>`. Overflow recovery rebuilds the immediate retry from whatever replacement advanced the surface. A checkpoint replaces the selected older range and is followed by the retained recent units.

##### Conversation checkpoint preamble

```markdown
This is an automatically generated checkpoint condensing an earlier span of the conversation to free up context. Treat the captured context as established background and build on it without restating it. Continue the task directly from the messages that follow, without acknowledging this checkpoint.
```

#### Token effect

Model-free pruning can avoid the auxiliary call entirely; otherwise it reduces that call's transcript before the summary replaces an older range. The replacement reduces future input history rather than appending a second copy. A summary remains until a later compaction replaces it, while an indivisible non-tool unit can still exceed the budget.

#### KV Cache effect

Replacing rather than append-only. Each checkpoint invalidates reuse from the first replaced history token; the unchanged request prefix before that range remains reusable.

### Auxiliary summarizer request

#### What the model sees

The summarization model receives the system prompt and shadowed-region history, with tool declarations and developer updates projected for its route, followed by one final user message: the compaction instruction below. The conversation model never sees this private request or its reasoning; only returned text is stored.

##### Compaction instruction (final user message)

```markdown
You are now acting as a compaction engine for this AI coding assistant. Condense the conversation ABOVE into a structured checkpoint that lets another model resume the work with no loss of essential context.

Output EXACTLY the Markdown structure below: keep every section, in order. Use terse bullets, not prose paragraphs. Write "(none)" for an empty section — never drop a section.

## Primary Request and Intent
- [the user's original and evolving goals; quote verbatim where the exact wording matters]

## Key Technical Concepts
- [technologies, frameworks, patterns, and conventions in play]

## Files and Code
- [exact path: why it matters, key changes or snippets]

## Errors and Fixes
- [error: how it was resolved, plus any related user feedback]

## Pending Jobs
- [explicitly requested work not yet completed]

## Current Work
- [precisely what was in progress at this checkpoint]

## Next Step
- [the single next action, directly in line with the most recent request, or "(none)"]

## Critical Context
- [decisions and their rationale, constraints, user preferences, open questions, data needed to continue]

Rules:
- Write concise English engineering prose. Preserve exact file paths, commands, error strings, identifiers, numeric values, function signatures, and syntax fragments.
- Capture user feedback and explicit instructions faithfully, especially corrections.
- Do NOT mention this summarization request or that the context was compacted.
- Output only the checkpoint text: do not call any tool or take any other action.
- If the conversation already contains a <compacted-summary> block, it is a PRIOR checkpoint. Do not copy it forward verbatim: preserve still-true facts, drop stale ones, and merge newer information into a single consolidated summary under the same structure.
```

#### Token effect

This is a separate model call: the replayed conversation prefix plus the fixed instruction as input, with `maxTokens`-capped output. Convergence retries can pay this cost more than once.

#### KV Cache effect

`conversation-prefix-v1` can reuse the conversation’s byte-for-byte system, tool and selected-message prefix up to the trailing instruction. A different target or non-head range can lose that reuse. `text-transcript-v1` changes the prefix and adds framing tokens; it makes no warm-cache or speed promise.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>


These limits define when automatic condensation is a poor fit or needs special care; they are the current package constraints.

- **Meter accuracy follows the fixed heuristic** — missing reusable provider usage falls back to character count plus structural overhead rather than exact tokenization; image occurrences carry provider-exact visual tokens only on routes whose adapter declares request-image pricing.
- **Overflow classification is adapter-maintained** — provider wording can change; both DeepSeek adapters normalize recognized context-limit failures to `CONTEXT_WINDOW_EXCEEDED`.
- **Some indivisible-unit and envelope-only overflow remains outside surface compaction** — recovery cannot shrink system/tools/prefix, split an indivisible non-tool node, or repair a tool unit whose non-prunable remainder still exceeds the window. The optional pruner can shrink text-bearing tool-result bulk inside an otherwise indivisible pair.
- **`compactRegion` requires an open turn** — a manual call on a fully-closed session throws ("no open turn") rather than compacting.
- **Summarization failure preserves the latest durable surface** — pre-step pressure logs a warning and continues; model-switch pressure rejects the destination request. Completed pruning and earlier checkpoints survive either failure. Summarization truncation at `maxTokens`, which hidden reasoning tokens can consume, follows the same rule.
- **Incremental carry-forward grows the checkpoint** — with `incremental: true`, a later checkpoint carries the previous checkpoint text locally instead of re-summarizing it. Until the bounded-view work lands, repeated compaction can therefore reach the destination budget sooner; the switch still refuses rather than dropping history.
- **Cross-model image pricing can reject a useful summary** — switch range selection uses destination pricing, while shrink validation uses the committed route. Different image prices can make a summary fail shrink validation even when it would reduce destination pressure.
- **Preset saves retain the current plugin list** — a newly saved profile override captures inherited plugins; later bundle additions need explicit reconciliation. Existing Agents keep their old preset revision until released.

<a id="dev-note"></a>
### Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

This Dev Note is working context for maintainers and is explicitly non-authoritative; shipped behavior lives in the sections above, the package code, and the linked Agent Notes.

- **Default ratios, undecided** — `thresholdRatio: 0.8` and `retainRatio: 0.16` are fixed defaults; per-model tuning via `modelPolicies` exists, but no corpus-backed guidance on ideal values is recorded.
- **Tokenizer-accurate measurement, deferred** — the token meter's four-characters-per-token heuristic underprices CJK text and JSON Schema documents; exact tokenization remains an open direction for the measurement service.
- **Overflow recovery beyond canonical errors, undecided** — recovery triggers on `CONTEXT_WINDOW_EXCEEDED` only; other provider-side context failures are not classified.

</details>
