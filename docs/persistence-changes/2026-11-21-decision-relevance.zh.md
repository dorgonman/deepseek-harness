---
description: "记录持久化类型更改及其兼容性确认。"
kind: persistence-change
---

# 2026-11-21-decision-relevance

[English](2026-11-21-decision-relevance.md) | 中文

## 概述

记录 Decision Relevance 审计事件及四个 V4 事件根类型中新增的来源类别。

## 目录

- [声明](#declaration)
- [兼容性](#compatibility)
- [验证](#verification)
- [开发备注](#dev-note)

<a id="declaration"></a>
## 声明

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
## 兼容性

当前读取器保留已有的 V4 消息，并接受 Decision Relevance 审计事件。包括旧版读取器在内，开放的消息来源判别字段会保留用户消息、开发者消息及收件箱插入项中的新类别。标题请求声明的消息联合类型也随之扩展，其读取器仍要求来源为 `dsh-session-title-llm`。若读取器不认识 `context/decision-relevance`，会在返回恢复后的会话前拒绝此必需事件并报告不支持。不含该审计事件的会话沿用现有读取流程。

<a id="verification"></a>
## 验证

Decision Relevance 与 V4 读取器测试套件共通过 99 项。封存的 Basic 回滚版本与 Hybrid 候选版本的会话目录对早期 V4 记录及混合 Decision Relevance 记录执行双向往返验证，保留所有事件和来源值。在移除该事件类型的测试词汇表中，两者均拒绝审计事件，也均拒绝把 Decision Relevance 作为标题请求来源。

<a id="dev-note"></a>
## 开发备注

无。
