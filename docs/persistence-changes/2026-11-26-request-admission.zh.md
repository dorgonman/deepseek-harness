---
description: "记录持久化类型更改及其兼容性确认。"
kind: persistence-change
---

# 2026-11-26-request-admission

[English](2026-11-26-request-admission.md) | 中文

## 概述

新增仅写入日志的 request/admission 回执，记录最终模型请求在本地估算的输入量和预留输出量。

## 目录

- [声明](#declaration)
- [兼容性](#compatibility)
- [验证](#verification)
- [开发备注](#dev-note)

<a id="declaration"></a>
## 声明

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
## 兼容性

当前读取器接受没有回执的现有日志。回执不改变重建后的模型历史或请求控制参数。写入器会将它标记为可忽略。不识别这种回执的旧版 V4 读取器将其保留为不解析的元数据，并重建相同的模型历史。

<a id="verification"></a>
## 验证

Hybrid 准入组合测试在适配器派发前重新读取了已持久化的 V4 回执，并确认持久化写入失败会阻止派发。此前日用版本和已封存的 Basic V4 读取器均保留了带标记的合成回执，且模型历史没有变化。去除标记后，同一回执被拒绝。

<a id="dev-note"></a>
## 开发备注

无。
