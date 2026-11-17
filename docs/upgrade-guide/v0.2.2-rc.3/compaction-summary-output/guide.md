---
kind: upgrade-guide
description: "Compaction summaries reject tool-call output instead of discarding it."
---

# Summarizers must return text without tool calls

English | [中文](guide.zh.md)

## Change

`compaction-basic` rejects summary output containing a `tool-call` block or ending with `tool-calls`, including a tool-calls finish without a block, with `UNSUPPORTED_CONTENT`. Earlier releases discarded tool-call blocks and kept any returned text. This affects summarization models and adapters that return text together with tool calls. Reasoning may accompany text, but it remains excluded from the checkpoint.

A rejected summary does not replace its selected history. Existing Session logs remain readable and need no rewrite.

## Migration

1. Check the model selected by `summarizationProvider` and `summarizationModel` and its adapter. It must answer compaction with text and no tool calls; do not rely on tool-call filtering.
2. Run an owned synthetic compaction through `/compact` or automatic compaction. Confirm that text output produces a `compaction/summary` event and replacement checkpoint, while tool-call output reports a summary failure and preserves the selected history.

No profile keys change. The [basic backend reference](../../../../packages/compaction/compaction-basic/README.md) documents summary request projection and validation.
