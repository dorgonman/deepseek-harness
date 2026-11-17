---
description: "Records a persistence type transition and its compatibility acknowledgement."
kind: persistence-change
---

# 2026-11-17-compaction-request-projection

English | [中文](2026-11-17-compaction-request-projection.zh.md)

## Summary

Adds optional requestProjection to compaction/summary, naming the auxiliary summarizer input as conversation-prefix-v1 or text-transcript-v1.

## Table of Contents

- [Declaration](#declaration)
- [Compatibility](#compatibility)
- [Verification](#verification)
- [Dev Note](#dev-note)

<a id="declaration"></a>
## Declaration

```yaml persistence-change
schemaVersion: 1
id: 2026-11-17-compaction-request-projection
baseline: false
changes:
  - root: "event:compaction/summary"
    previous: "2026-09-16-session-format-v4"
    after: "c217c6f6a129040bcb03a388f932d1290611b54b71d40f259b96fd5d671eb80f"
    decision: same-version
```

<a id="compatibility"></a>
## Compatibility

Old records omit the field. New readers leave absence as historical or unspecified custom-summarizer behavior and continue to use the adjacent checkpoint user/message. The released V4 decoder retains extra event.data properties, while compaction admission and relationship checks inspect existing fields and ignore this tag. Older readers therefore accept the new row, preserve the tag when round-tripping it, and do not change replay, execution, recovery, or authority. SessionHeader.version remains 4.

<a id="verification"></a>
## Verification

In an isolated history-clean source checkout, regenerated the stale catalog and verified 66 roots against 24 history records. Applying only packages/compaction/compaction/src/types.ts produced one optional-property-added transition for event:compaction/summary, e2f9a41e0989f54ed8cee80f8db2bcf9d60a5c810dc9d45b83fa050b9dce7602 to c217c6f6a129040bcb03a388f932d1290611b54b71d40f259b96fd5d671eb80f. Inspected the released V4 decoder, summary admission, relationship checks, and checkpoint source. Controlled Linux keyless replay through the shipped headless profile passed the compaction-text-only-summary scenario: 1 selected test passed, 201 skipped, 4.25 s scenario time (6.37 s Vitest). The run generated 41 Typert artifacts first. Candidate manifest SHA-256 ab868ec7062ed0d7d969957440354cdfa145841e462ff03686ad4d5758fb2af6 and run receipt SHA-256 14b60c272598c999b6832dc81d7d735885081bb52f2155c70300d2bf34296a13 bind 83 unchanged staged candidate files. This is recorded replay, not a live provider or TP2 inference qualification.

<a id="dev-note"></a>
## Dev Note

None.
