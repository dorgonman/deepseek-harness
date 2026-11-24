---
description: "Records a persistence type transition and its compatibility acknowledgement."
kind: persistence-change
---

# 2026-11-21-decision-relevance

English | [中文](2026-11-21-decision-relevance.zh.md)

## Summary

Records the Decision Relevance audit event and the added source kind in four declared V4 event roots.

## Table of Contents

- [Declaration](#declaration)
- [Compatibility](#compatibility)
- [Verification](#verification)
- [Dev Note](#dev-note)

<a id="declaration"></a>
## Declaration

```yaml persistence-change
schemaVersion: 1
id: 2026-11-21-decision-relevance
baseline: false
changes:
  - root: "event:agent/inbox/spliced"
    previous: "2026-10-27-retry-control-input"
    after: "04ecd4e613e53758fd3685fc6c4de447d94810fee27870849cae52eac8c63fd9"
    decision: same-version
  - root: "event:context/decision-relevance"
    previous: null
    after: "139b033b629515085d96a415d6b66224e2daffd9cd3aead52bd16332dedab104"
    decision: same-version
  - root: "event:developer/message"
    previous: "2026-10-27-retry-control-input"
    after: "10cb7902aecbf17953cbb6feed3ffa07a0ec0da8a8fab34bcf927b8b2fc66bb4"
    decision: same-version
  - root: "event:session/title-llm-request"
    previous: "2026-10-27-retry-control-input"
    after: "e760910a671c184f2a847900ed94b21e9ac0f30236f972d5b26445b3cbfc6df9"
    decision: same-version
  - root: "event:user/message"
    previous: "2026-10-27-retry-control-input"
    after: "8935c0cca8a025706c44317e7137032217155375252af456bcba478315297594"
    decision: same-version
```

<a id="compatibility"></a>
## Compatibility

Current readers preserve existing V4 messages and accept the Decision Relevance audit event. The open producer-source discriminator preserves the new kind in user messages, developer messages, and inbox inserts, including on older readers. The title request's declared message union also widens. Its reader still requires the `dsh-session-title-llm` source. A reader whose vocabulary lacks `context/decision-relevance` refuses the required audit event as unsupported before returning a restored Session. Sessions without the audit event retain their existing read path.

<a id="verification"></a>
## Verification

The Decision Relevance and V4 reader suites passed 99 tests. The sealed Basic rollback and Hybrid candidate catalogs round-tripped earlier V4 records and mixed Decision Relevance records in both directions, preserving all events and source values. Both refused the audit event under a vocabulary without that type and rejected a title request with the Decision Relevance source.

<a id="dev-note"></a>
## Dev Note

None.
