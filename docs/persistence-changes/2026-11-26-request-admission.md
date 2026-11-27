---
description: "Records a persistence type transition and its compatibility acknowledgement."
kind: persistence-change
---

# 2026-11-26-request-admission

English | [中文](2026-11-26-request-admission.zh.md)

## Summary

Adds request/admission, a log-only receipt for the local estimate of the final model request's input and reserved output.

## Table of Contents

- [Declaration](#declaration)
- [Compatibility](#compatibility)
- [Verification](#verification)
- [Dev Note](#dev-note)

<a id="declaration"></a>
## Declaration

```yaml persistence-change
schemaVersion: 1
id: 2026-11-26-request-admission
baseline: false
changes:
  - root: "event:request/admission"
    previous: null
    after: "5c383e739520794452f88610f9cf2e046f080c310a16b22118cbcd667fbf5e07"
    decision: same-version
```

<a id="compatibility"></a>
## Compatibility

Current readers accept existing logs without a receipt. The receipt leaves reconstructed model history and request controls unchanged. The writer marks it ignorable. Older V4 readers whose vocabulary lacks the receipt retain it as opaque metadata and reconstruct the same model history.

<a id="verification"></a>
## Verification

The composed Hybrid admission tests reopened persisted V4 receipts before adapter dispatch and checked that a flush failure prevented dispatch. The prior daily and sealed Basic V4 readers retained a synthetic marked receipt with unchanged model history and rejected the same receipt without the marker.

<a id="dev-note"></a>
## Dev Note

None.
