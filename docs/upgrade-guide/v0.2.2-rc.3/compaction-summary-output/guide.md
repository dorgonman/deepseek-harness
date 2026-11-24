---
kind: upgrade-guide
description: "Compaction verifies summary targets and rejects tool-call output instead of discarding it."
---

# Summarizers must return text without tool calls

English | [中文](guide.zh.md)

## Change

`compaction-basic` rejects summary output containing a `tool-call` block or ending with `tool-calls`, including a tool-calls finish without a block, with `UNSUPPORTED_CONTENT`. Earlier releases discarded tool-call blocks and kept any returned text. This affects summarization models and adapters that return text together with tool calls. Reasoning may accompany text, but it remains excluded from the checkpoint.

Summary calls also reject a selected route whose declared input modalities exclude text, with `SUMMARY_TARGET_NOT_CHAT`. With `summarizationFallback: error`, missing context capacity fails with `SUMMARY_TARGET_UNVERIFIED` before dispatch. Earlier versions could dispatch either case. Undeclared modalities remain ordinary Chat; the default best-effort policy still permits unknown capacity.

A rejected summary does not replace its selected history. Existing Session logs remain readable and need no rewrite.

## Migration

1. Check the model selected by `summarizationProvider` and `summarizationModel` and its adapter. It must accept text and answer compaction with text and no tool calls; do not rely on tool-call filtering. With fallback `error`, supply qualified `contextWindow` metadata or select a target that declares its capacity. Retain `error` when your deployment forbids fallback.
2. Run an owned synthetic compaction through `/compact` or automatic compaction. Confirm that valid text output produces a `compaction/summary` event and replacement checkpoint. Missing capacity under `error` or explicit text incompatibility must fail before a provider call; rejected metadata or tool-call output must preserve the selected history.

No profile keys change. The [basic backend reference](../../../../packages/compaction/compaction-basic/README.md) documents summary request projection and validation.
