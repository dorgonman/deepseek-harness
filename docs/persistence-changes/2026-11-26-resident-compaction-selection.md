---
description: "Records a persistence type transition and its compatibility acknowledgement."
kind: persistence-change
---

# 2026-11-26-resident-compaction-selection

English | [中文](2026-11-26-resident-compaction-selection.zh.md)

## Summary

Records the selected resident deployment and each dispatched projection digest on a committed Hybrid summary.

## Table of Contents

- [Declaration](#declaration)
- [Compatibility](#compatibility)
- [Verification](#verification)
- [Dev Note](#dev-note)

<a id="declaration"></a>
## Declaration

```yaml persistence-change
schemaVersion: 1
id: 2026-11-26-resident-compaction-selection
baseline: false
changes:
  - root: "event:compaction/summary"
    previous: "2026-11-22-fact-ledger-snapshot"
    after: "c150bb9857d507236b493b9f6234ab1383648f98f4789bc889309a88932fe15c"
    decision: same-version
```

<a id="compatibility"></a>
## Compatibility

The optional receipt adds audit information to the V4 summary. Older V4 readers preserve the unknown field and derive the same checkpoint from the existing summary and chunk calls. Current readers accept summaries without a receipt and validate the deployment identity and projection list when present.

<a id="verification"></a>
## Verification

The V4 relationship suite passed 63 tests. It reopened the new receipt and an older summary without it, and rejected missing chunk calls, mismatched aliases or node names, missing identity, an unknown selection reason, and invalid projection digests.

<a id="dev-note"></a>
## Dev Note

None.
