---
description: "记录持久化类型更改及其兼容性确认。"
kind: persistence-change
---

# 2026-11-20-hybrid-compaction-replay

[English](2026-11-20-hybrid-compaction-replay.md) | 中文

## 概述

在现有 V4 压缩摘要中，按顺序记录 Hybrid 压缩成功时的摘要调用。

## 目录

- [声明](#declaration)
- [兼容性](#compatibility)
- [验证](#verification)
- [开发备注](#dev-note)

<a id="declaration"></a>
## 声明

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
## 兼容性

V4 事件、替换和 header 的含义保持不变。旧版 V4 读取器把新增字段保留为 payload 数据，并恢复同一 Surface。旧版重放适配器无法从这些记录还原 Hybrid 的调用。当前读取器接受未标记摘要和单调用摘要，并按顺序重放新增记录。

<a id="verification"></a>
## 验证

受影响测试通过 2,009 项，跳过 1 项。Hybrid 合成集成测试在冷启动后恢复了三个检查点及其有序分块调用。V4 关系测试接受有效记录，并拒绝格式错误或冲突的记录。

<a id="dev-note"></a>
## 开发备注

无。
