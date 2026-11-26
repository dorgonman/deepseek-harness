---
description: "记录持久化类型更改及其兼容性确认。"
kind: persistence-change
---

# 2026-11-26-resident-compaction-selection

[English](2026-11-26-resident-compaction-selection.md) | 中文

## 概述

Hybrid 提交摘要时保存已选常驻部署及每个已发送投影的摘要值。

## 目录

- [声明](#declaration)
- [兼容性](#compatibility)
- [验证](#verification)
- [开发备注](#dev-note)

<a id="declaration"></a>
## 声明

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
## 兼容性

这一可选回执为 V4 摘要补充审计信息。旧版 V4 读取器保留未知字段，并根据原有摘要和分块调用还原同一检查点。当前读取器接受没有回执的摘要，并在回执存在时校验部署身份和投影列表。

<a id="verification"></a>
## 验证

V4 关系测试通过 63 项。测试重新打开了带回执的新摘要和无回执的旧摘要，并拒绝缺少分块调用、别名或节点名不匹配、身份缺失、选择理由未知和投影摘要值无效的记录。

<a id="dev-note"></a>
## 开发备注

无。
