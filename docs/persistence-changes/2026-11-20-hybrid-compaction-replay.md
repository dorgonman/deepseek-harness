---
description: "Records a persistence type transition and its compatibility acknowledgement."
kind: persistence-change
---

# 2026-11-20-hybrid-compaction-replay

English | [中文](2026-11-20-hybrid-compaction-replay.zh.md)

## Summary

Records the ordered summarizer calls of a successful Hybrid compaction in the existing V4 compaction summary.

## Table of Contents

- [Declaration](#declaration)
- [Compatibility](#compatibility)
- [Verification](#verification)
- [Dev Note](#dev-note)

<a id="declaration"></a>
## Declaration

```yaml persistence-change
schemaVersion: 1
id: 2026-11-20-hybrid-compaction-replay
baseline: false
changes:
  - root: "event:compaction/summary"
    previous: "2026-11-17-compaction-request-projection"
    after: "06fa604dd079f2c6d5d00b078c22748029abdbb2849daa56e1c18e89e66a8dbf"
    decision: same-version
```

<a id="compatibility"></a>
## Compatibility

The V4 event, replacement, and header meanings remain the same. Older V4 readers retain the extra field as payload data and restore the same Surface. An older replay adapter cannot reconstruct Hybrid calls from these receipts. Current readers accept the unmarked and single-call summaries and replay each ordered receipt when present.

<a id="verification"></a>
## Verification

The focused suites passed 2,009 tests with one skip. Synthetic Hybrid integration recovered three checkpoints and their ordered chunk calls after cold reopen. V4 relationship tests accepted valid receipts and refused malformed or conflicting records.

<a id="dev-note"></a>
## Dev Note

None.
