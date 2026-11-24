---
kind: upgrade-guide
description: "Compaction verifies summarizer capacity and text support and rejects tool-call output."
---

# Summarizers must return text without tool calls

English | [中文](guide.zh.md)

## Change

`compaction-basic` rejects summary output containing a `tool-call` block or ending with `tool-calls`, including a tool-calls finish without a block, with `UNSUPPORTED_CONTENT`. Earlier releases discarded tool-call blocks and kept any returned text. This affects summarization models and adapters that return text together with tool calls. Reasoning may accompany text, but it remains excluded from the checkpoint.

Basic and Hybrid reject a selected route whose declared input modalities exclude text with `SUMMARY_TARGET_NOT_CHAT`. Basic with `summarizationFallback: error` and Hybrid also reject missing or invalid context capacity with `SUMMARY_TARGET_UNVERIFIED` before sending history. Basic with `latest-route` still allows undeclared capacity.

A rejected summary does not replace its selected history. Existing Session logs remain readable and need no rewrite.

## Migration

1. Check the model selected by `summarizationProvider` and `summarizationModel`. It must accept text and answer compaction with text without calling tools. For Hybrid or a Basic policy with `summarizationFallback: error`, configure `contextWindow` on the adapter model if it omits capacity.
2. Run an owned synthetic compaction through `/compact` or automatic compaction. Confirm that valid text output produces a `compaction/summary` event and a replacement checkpoint. A rejected target or response preserves the selected history.

No profile keys change. The [basic backend reference](../../../../packages/compaction/compaction-basic/README.md) documents summary request projection and validation.
