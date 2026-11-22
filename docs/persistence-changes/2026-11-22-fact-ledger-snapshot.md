---
description: "Records a persistence type transition and its compatibility acknowledgement."
kind: persistence-change
---

# 2026-11-22-fact-ledger-snapshot

English | [中文](2026-11-22-fact-ledger-snapshot.zh.md)

## Summary

Adds an optional merged Fact Ledger snapshot to compaction/summary so repeated compaction can commit sourced facts with the checkpoint.

## Table of Contents

- [Declaration](#declaration)
- [Compatibility](#compatibility)
- [Verification](#verification)
- [Dev Note](#dev-note)

<a id="declaration"></a>
## Declaration

```yaml persistence-change
schemaVersion: 1
id: 2026-11-22-fact-ledger-snapshot
baseline: false
changes:
  - root: "event:compaction/summary"
    previous: "2026-11-22-incremental-raw-coverage"
    after: "be8be7717d085cb7380fb0df12239f2996ef85fdf55e26cc1f8240e2ff2b6bd2"
    decision: same-version
```

<a id="compatibility"></a>
## Compatibility

Existing records remain valid and keep their full-surface behavior. Absence of the field means no committed ledger. Older V4 readers retain the payload as data and restore the same Surface. Current readers validate source event ids, content digests, seen bounds and supersede lineage before using a snapshot, and a summary without its replacement message or a clean compaction/end never commits derived state.

<a id="verification"></a>
## Verification

node node_modules/vitest/vitest.mjs run --config vitest.config.ts packages/compaction/compaction-basic/tests/fact-ledger.spec.ts packages/compaction/compaction-basic/tests/incremental.spec.ts scripts/stable-compaction-hybrid.spec.ts: 40 tests passed (9 ledger, 8 incremental, 23 hybrid).

<a id="dev-note"></a>
## Dev Note

None.
