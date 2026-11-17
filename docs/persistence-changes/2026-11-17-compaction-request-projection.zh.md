---
description: "记录持久化类型更改及其兼容性确认。"
kind: persistence-change
---

# 2026-11-17-compaction-request-projection

[English](2026-11-17-compaction-request-projection.md) | 中文

## 概述

在 compaction/summary 中加入可选的 requestProjection，标记辅助摘要请求使用 conversation-prefix-v1 或 text-transcript-v1 输入表示。

## 目录

- [声明](#declaration)
- [兼容性](#compatibility)
- [验证](#verification)
- [开发备注](#dev-note)

<a id="declaration"></a>
## 声明

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
## 兼容性

旧记录没有此字段；新读取器将缺失值视为历史记录或未说明的自定义摘要器行为，继续使用相邻的检查点 user/message。已发布的 V4 解码器保留 event.data 的额外字段，而压缩准入与关系检查只读取既有字段，不依赖此标记。因此旧读取器可接受新记录，往返写入时保留此标记，且不会改变重放、执行、恢复或权限。SessionHeader.version 保持为 4。

<a id="verification"></a>
## 验证

在隔离且持久化历史校验通过的源代码检出中重生成过期目录后，验证 66 个根与 24 条历史记录。仅应用 packages/compaction/compaction/src/types.ts 的改动后，预检只报告 event:compaction/summary 的一个可选属性新增，摘要从 e2f9a41e0989f54ed8cee80f8db2bcf9d60a5c810dc9d45b83fa050b9dce7602 变为 c217c6f6a129040bcb03a388f932d1290611b54b71d40f259b96fd5d671eb80f。已检查 V4 解码器、摘要准入、关系检查与检查点来源。受控 Linux 环境通过正式 headless profile 的无密钥重放 compaction-text-only-summary 场景：选中的 1 项测试通过，201 项跳过，场景耗时 4.25 秒（Vitest 总耗时 6.37 秒）。运行前生成了 41 项 Typert 工件。候选清单 SHA-256 ab868ec7062ed0d7d969957440354cdfa145841e462ff03686ad4d5758fb2af6 与运行回执 SHA-256 14b60c272598c999b6832dc81d7d735885081bb52f2155c70300d2bf34296a13 绑定 83 个未变化的候选文件。这是录制内容重放，不代表真实提供方或 TP2 推理资格验证。

<a id="dev-note"></a>
## 开发备注

无。
