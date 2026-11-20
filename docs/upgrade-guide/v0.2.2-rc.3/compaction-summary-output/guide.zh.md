---
kind: upgrade-guide
description: "压缩会校验摘要器容量和文本能力，并拒绝工具调用输出。"
---

# 摘要器必须返回不含工具调用的文本

[English](guide.md) | 中文

## 变更

`compaction-basic` 会以 `UNSUPPORTED_CONTENT` 拒绝包含 `tool-call` 块或以 `tool-calls` 结束的摘要输出，包括未包含块的 tool-calls 结束。此前的版本会丢弃工具调用块，并保留返回的文本。这会影响同时返回文本与工具调用的摘要模型和适配器。推理内容可以随文本一同返回，但仍不进入检查点。

基础后端和混合后端会以 `SUMMARY_TARGET_NOT_CHAT` 拒绝声明的输入模态排除文本的路由。使用 `summarizationFallback: error` 的基础后端和混合后端还会在发送历史记录前以 `SUMMARY_TARGET_UNVERIFIED` 拒绝缺失或无效的上下文容量。使用 `latest-route` 的基础后端仍允许未声明的容量。

摘要被拒绝后不会替换其所选历史。现有 Session 日志仍可读取，无需重写。

## 迁移

1. 检查 `summarizationProvider` 与 `summarizationModel` 选择的模型。该模型必须接受文本，以文本回答压缩请求，且不能调用工具。使用混合后端或设有 `summarizationFallback: error` 的基础策略时，如果适配器未声明容量，请在适配器模型上配置 `contextWindow`。
2. 通过 `/compact` 或自动压缩运行一次自有的合成压缩。确认有效文本输出会生成 `compaction/summary` 事件与替换检查点。目标或响应被拒绝时，所选历史会保留。

配置文件键保持不变。[基础后端参考](../../../../packages/compaction/compaction-basic/README.zh.md)说明摘要请求投影和验证。
