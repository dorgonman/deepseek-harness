---
description: "Records a persistence type transition and its compatibility acknowledgement."
kind: persistence-change
---

# 2026-11-22-incremental-raw-coverage

English | [中文](2026-11-22-incremental-raw-coverage.zh.md)

## Summary

Adds optional incremental RAW coverage to compaction/summary so repeated compaction can record which append-origin ranges a checkpoint already accounts for.

## Table of Contents

- [Declaration](#declaration)
- [Compatibility](#compatibility)
- [Verification](#verification)
- [Dev Note](#dev-note)

<a id="declaration"></a>
## Declaration

```yaml persistence-change
schemaVersion: 1
id: 2026-11-22-incremental-raw-coverage
baseline: false
changes:
  - root: "event:compaction/summary"
    previous: "2026-11-20-hybrid-compaction-replay"
    after: "ed393982b25498a87088b799e3cc6f41a9a3a3a431d3141cbe19582503057dfd"
    decision: same-version
```

<a id="compatibility"></a>
## Compatibility

Existing records remain valid and keep their full-surface behavior. Absence of the field never implies coverage. Older V4 readers retain the extra payload as data and restore the same Surface; current readers accept absence and validate the record before using it. Producers write it only for incremental compaction, in the same event as the checkpoint summary.

<a id="verification"></a>
## Verification

node node_modules/vitest/vitest.mjs run --config vitest.config.ts packages/compaction/compaction-basic/tests/incremental.spec.ts: 7 tests passed. node node_modules/vitest/vitest.mjs run --config vitest.config.ts scripts/stable-compaction-hybrid.spec.ts: 21 tests passed.

<a id="dev-note"></a>
## Dev Note

None.
