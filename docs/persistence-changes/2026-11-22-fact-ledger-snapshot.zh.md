---
description: "记录持久化类型更改及其兼容性确认。"
kind: persistence-change
---

# 2026-11-22-fact-ledger-snapshot

[English](2026-11-22-fact-ledger-snapshot.md) | 中文

## 概述

为 compaction/summary 增加可选的合并 Fact Ledger 快照，使重复压缩可以把有来源的事实与检查点一起提交。

## 目录

- [声明](#declaration)
- [兼容性](#compatibility)
- [验证](#verification)
- [开发备注](#dev-note)

<a id="declaration"></a>
## 声明

```yaml persistence-change
schemaVersion: 1
id: 2026-11-22-fact-ledger-snapshot
baseline: false
changes:
  - root: "event:compaction/summary"
    previous: "2026-11-22-incremental-raw-coverage"
    after: "d40046c5f7d040ae58a4550a8a606ee68fa5b48cf367b742b8000435af2ff4e8"
    decision: same-version
```

<a id="compatibility"></a>
## 兼容性

已有记录仍然有效并保持全表面行为。字段缺失表示没有已提交的账本。旧版 V4 读取器把新增负载作为数据保留并恢复同一 Surface。当前读取器在使用快照前校验来源事件 ID、内容摘要、首末序号与取代关系；没有替换消息或干净 compaction/end 的摘要绝不提交派生状态。账本的来源标记在记录定稿前原地改名为 `basis`（取值仍为 `asserted | inferred`）：没有任何已发布格式使用过旧标记，而使用旧标记的快照会被拒绝校验，不会被静默重新解释。

<a id="verification"></a>
## 验证

node node_modules/vitest/vitest.mjs run --config vitest.config.ts packages/compaction/compaction-basic/tests/fact-extraction.spec.ts packages/compaction/compaction-basic/tests/fact-ledger.spec.ts packages/compaction/compaction-basic/tests/incremental.spec.ts scripts/stable-compaction-hybrid.spec.ts：53 个测试通过（抽取 6、账本 9、增量 8、Hybrid 30）。

<a id="dev-note"></a>
## 开发备注

无。
