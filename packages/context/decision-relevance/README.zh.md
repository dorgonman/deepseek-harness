---
description: "面向组合 Decision 路由部署场景的模型判定工具输出相关性：判断是否应让无关工具结果离开请求，或排查工具结果为何被省略。"
kind: "package-reference"
---

# @deepseek-ai/dsh-context-decision-relevance

[English](README.md) | 中文

## 概述

`dsh-context-decision-relevance` 防止不再服务于当前目标的工具结果在每次请求中反复随行。对于出站请求中的每个超大工具结果，它会向一个独立配置的 Decision 模型发起单 token 的相关性判定，并把判定为无关的结果正文替换为简短的省略标记。完整原始结果仍保留在会话日志中，可供精确回放与检查；只有该次请求的线上副本发生变化。每次评估都有边界，任何不确定、失败或被取消的评估都会保留内容，因此该功能只会缩小请求，绝不会丢失历史。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [延伸阅读](#further-exploration)
- [模型体验](#model-experience)
- [已知限制与后续工作](#known-limitations-and-deferred-work)
- [开发备注](#dev-note)

-----

<a id="use-this-package"></a>
## 使用本包

当长时间运行的会话积累了大量不再重要的大体积工具结果——旧的文件读取、已被取代的搜索输出、耗尽的命令日志——并希望模型停止重复阅读它们时，挂载本包。该功能默认关闭：没有任何配置时本包不注册任何能力，普通请求完全不受影响。

### 最小可用组合

本包需要 LLM 运行时以访问 Decision 路由，并需要会话存储以记录其证据：

```yaml
- name: '@deepseek-ai/dsh-llm'
- name: '@deepseek-ai/dsh-session'
- name: '@deepseek-ai/dsh-context-decision-relevance'
  config:
    mode: 'on'
    provider: 'my-decision-provider'
    model: 'my-decision-model'
```

`provider` 与 `model` 必须同时配置；只配置其中一个时，无论 `mode` 为何，功能都处于非激活状态。该路由独立于对话自身的模型进行解析，因此聊天模型保持原样。你可以通过检查后续请求中原先无关结果的位置是否出现省略标记来验证成功；完整原始结果仍保留在会话日志中。

### 哪些内容会被考虑

只有工具结果才是候选，而且只是其中一部分：

- 结果不能是错误——失败的工具结果保持可见，因为其错误是「曾尝试过什么」这一记录的组成部分；
- 其文本必须达到 `minCandidateChars`，因此小结果不值得发起判定调用；
- 每次请求最多评估 `maxCandidatesPerRequest` 个候选，按消息顺序进行；其余全部保留。

每个候选只被判定一次，其结论会在会话期间记住，因此长对话不会在每一轮重复询问同一结果。

### 设置阈值

所有设置都是可选的；生成的[配置目录](../../../docs/config-catalog.zh.md#deepseek-aidsh-context-decision-relevance)是完整权威来源。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `mode` | `off` | `off` 禁用该功能；`on` 将经过校验的选择应用到出站投影。 |
| `provider` | 未设置 | Decision 路由的规范 provider。 |
| `model` | 未设置 | Decision 路由的规范 model。 |
| `retainThreshold` | `0.6` | 保留工具结果所需的相关性分数。 |
| `uncertaintyMargin` | `0.1` | `retainThreshold` 周围视为不确定的区间半宽。 |
| `maxCandidatesPerRequest` | `4` | 每次请求评估的最大候选数。 |
| `minCandidateChars` | `512` | 结果成为候选所需的最小文本大小（Unicode 码点）。 |
| `maxInputChars` | `4096` | 发送给 Decision 模型的最大文本大小（Unicode 码点）。 |
| `timeoutMs` | `15000` | 单次评估的端到端截止时间。 |
| `elisionText` | 内置 | 被省略正文的替换文本。 |

分数低于 `retainThreshold - uncertaintyMargin` 时省略该结果；分数不低于 `retainThreshold + uncertaintyMargin` 时保留；介于两者之间视为不确定并保留。未知设置会在构造时拒绝该插件。

### 判定何时运行

评估在 LLM 运行时的请求投影阶段运行：出站消息列表解析完成之后、适配器分派之前。它们是使用独立路由的后台调用：绝不会改变对话的模型、采样参数或工具。

-----

<a id="understand-the-implementation"></a>
## 理解实现

<details>
<summary>实现内部细节 — 点击展开</summary>

本节解释过滤器背后的设计决策；可观察行为已在[使用本包](#use-this-package)中完整覆盖。

### 设计理念

过滤器建立在四项承诺之上：

- **出站投影，而非持久重写。** 会话日志是仅追加的，且循环构造的请求必须是它的纯函数。因此选择只重写单次请求的线上副本，保留每条原始消息及其工具调用关联关系完整不变。
- **独立的合格路由。** Decision 模型通过其自身的 `prepareCall` 解析，而非沿用对话的路由。其实际服务身份会与请求进行校验，因此静默提供了其他模型的路由会被视为失败，而非一个判定结果。
- **处处失败即保留。** 传输错误、格式错误的读数、服务模型不匹配、落在不确定区间的分数、以及取消，都会保留内容。过滤器绝不会删除它不确定的内容。
- **有界的持久证据。** 每轮处理追加一条 `context/decision-relevance` 事件，记录对话路由、是否发生了省略，以及每个候选一条有界记录：候选身份、请求与服务的规范身份、分数、处置方式与简短理由。不含凭据，也不含历史转储。

### 评估机制

对每个候选，过滤器由最近一条用户消息和候选正文的码点有界副本构造纯文本提示，将其冻结为一个单 token、零温度、`label_scores` 模式、purpose 为 `decision-relevance` 的请求，并在与调用方信号融合的截止时间下装配适配器响应。读数必须携带预期的 schema、mode、label 以及 `[0, 1]` 内的有限分数；其他任何情况都是格式错误的读数并保留内容。精确签名见 [`src/index.ts`](src/index.ts)。

### 源码地图

| 文件 | 职责 |
|---|---|
| [`src/index.ts`](src/index.ts) | 插件入口：`DecisionRelevanceFilter` 服务、`project`、读数解析 |
| [`src/config.ts`](src/config.ts) | 默认值、稀疏配置校验、码点边界处理 |
| [`src/types.ts`](src/types.ts) | `DecisionRelevanceConfig`、`ResolvedConfig`、`RelevanceJudgment`、`ProjectionResult` |

</details>

-----

<a id="further-exploration"></a>
## 延伸阅读

当包级契约不足以满足需要时阅读这些页面；它们从投影接缝延伸到 Decision 协议与持久词汇表。

- [LLM 运行时](../../llm/llm/README.zh.md) — 应用选择逻辑的请求投影接缝。
- [Pi AI provider](../../llm/llm-pi-ai/README.zh.md) — 渲染 Decision 读数的适配器。
- [压缩子系统参考](../../../docs/subsystems/compaction.zh.md) — 本包证据所加入的持久词汇表。
- [生成的配置目录](../../../docs/config-catalog.zh.md#deepseek-aidsh-context-decision-relevance) — 每个可接受配置字段及其源声明。

-----

<a id="model-experience"></a>
## 模型体验

### 被省略的工具结果

#### 模型看到什么

对于某一次请求，无关工具结果的正文被替换为 `elisionText`。该消息保留其身份、`tool` 角色和工具调用关联，因此助手早先的调用仍可解析；只有文本消失了。相关的、不确定的和失败的结果原样发送。

#### Token 影响

每个被省略的结果花费 `elisionText` 而非其原始正文，每个被评估的候选在 Decision 路由上花费一次有界的单 token Decision 调用。没有合格候选的请求完全不发起 Decision 调用。

#### KV Cache 影响

省略会改变第一条被省略消息处的提示，因此首次应用该省略的请求会失去从该点起的复用。由于结论按会话记住，后续请求保持相同的省略，并从该点起正常复用。

## 已知限制与后续工作

<a id="known-limitations-and-deferred-work"></a>


这些限制界定了何时不适合使用过滤或需要特别留意；它们是当前包的约束。

- **按结果而非按片段判定相关性** — 一半仍然相关的结果会被整体省略或整体保留；部分选择尚未提供。
- **目标是最近的用户消息** — 提示将最近的用户文本作为目标陈述，因此服务于较早长期指令的结果可能被判定为无关。
- **字符边界不是 token 边界** — `minCandidateChars` 与 `maxInputChars` 是 Unicode 码点；各 provider 的 token 密度不同，因此请求压力仍以 token meter 为权威。
- **没有回退路由** — 当配置的 Decision 路由失败时内容被保留；过滤器绝不会改用其他 Decision 模型。

<a id="dev-note"></a>
### 开发备注

<details>
<summary>面向维护者的工作上下文 — 点击展开</summary>

本开发备注是面向维护者的工作上下文，明确不具有规范性；已交付行为以上文、包代码与所链接的 Agent Notes 为准。

- **轮次与步骤归属，延后处理** — 投影接缝不会接收循环的 turn 与 step，因此证据记录对话路由而不记录位置。
- **按候选选择，尚未决定** — 结论按消息身份在服务生命周期内缓存；目标变化是否应使早先结论失效尚未决定。

</details>
