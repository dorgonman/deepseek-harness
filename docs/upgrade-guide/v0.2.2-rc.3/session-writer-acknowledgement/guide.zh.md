---
kind: upgrade-guide
description: "第三方 Session 后端必须确认用于持久化模型请求准入回执的写句柄。"
---

# Session 后端必须确认写入回执的句柄

[English](guide.md) | 中文

## 变更

此前的版本将 `session/flush` 完成视为持久化检查点。启用自动压缩且路由声明了上下文窗口时，持久化 Agent 现在要求其获取的写句柄确认 `request/admission` 回执已持久化。后端监听器完成后如果没有返回该句柄，可能触发 `CONTEXT_ADMISSION_REFUSED`。随产品提供的 JSONL 后端在排空并刷新后返回活跃句柄。

## 迁移

1. 在持久化后端的 `session/flush` 监听器中排空 Session 的排队事件，并等待 `handle.flush()` 完成。返回后端在 Agent 创建 Session 或以写入方式打开 Session 时交给它的同一个 `SessionHandle` 对象。
2. 使用更新后的后端重新打开一个自行控制的合成 Session。在启用自动压缩且声明了上下文窗口的路由上执行模型请求。确认在适配器收到请求之前，已能读取持久化的 `request/admission` 回执。

参见[存储后端约定](../../../../packages/session/session-persistence/README.zh.md#understand-the-implementation)。
