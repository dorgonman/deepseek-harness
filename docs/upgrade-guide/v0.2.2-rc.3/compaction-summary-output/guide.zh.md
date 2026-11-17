---
kind: upgrade-guide
description: "压缩会校验摘要目标，并拒绝工具调用输出，不再丢弃它。"
---

# 摘要器必须返回不含工具调用的文本

[English](guide.md) | 中文

## 变更

`compaction-basic` 会以 `UNSUPPORTED_CONTENT` 拒绝包含 `tool-call` 块或以 `tool-calls` 结束的摘要输出，包括未包含块的 tool-calls 结束。此前的版本会丢弃工具调用块，并保留返回的文本。这会影响同时返回文本与工具调用的摘要模型和适配器。推理内容可以随文本一同返回，但仍不进入检查点。

摘要调用还会以 `SUMMARY_TARGET_NOT_CHAT` 拒绝已声明输入模态排除文本的所选路由。使用 `summarizationFallback: error` 时，缺少上下文容量会在派发前以 `SUMMARY_TARGET_UNVERIFIED` 失败。此前的版本可能在上述任一条件下继续派发。未声明模态仍按普通 Chat 处理；默认尽力执行策略仍允许未知容量。

摘要被拒绝后不会替换其所选历史。现有 Session 日志仍可读取，无需重写。

## 迁移

1. 检查 `summarizationProvider` 与 `summarizationModel` 选择的模型及其适配器。该模型必须接受文本，以文本回答压缩请求，且不能调用工具；不要依赖工具调用过滤。当回退设为 `error` 时，提供经过验证的 `contextWindow` 元数据，或选择已声明容量的目标。部署禁止回退时，应保留 `error`。
2. 通过 `/compact` 或自动压缩运行一次自有的合成压缩。确认有效文本输出会生成 `compaction/summary` 事件与替换检查点。使用 `error` 时缺少容量，或显式不支持文本，都必须在调用提供方前失败；元数据或工具调用输出被拒绝时，必须保留所选历史。

配置文件键保持不变。[基础后端参考](../../../../packages/compaction/compaction-basic/README.zh.md)说明摘要请求投影和验证。
