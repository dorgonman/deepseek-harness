---
kind: upgrade-guide
description: "The repeat-tool-reminder guard blocks an identical tool call at its blockAfter count instead of only reminding."
---

# repeat-tool-reminder guard 会阻止重复的工具调用

[English](guide.md) | 中文

## 变更

`@deepseek-ai/dsh-repeat-tool-reminder` 新增了 `blockAfter` 配置字段（默认 `8`）。此前 guard 只在对话中追加提醒，绝不拒绝调用，因此模型持续以相同参数重复同一工具调用时，只要超过最高提醒阈值就会无限制地继续执行。现在 guard 在第 `blockAfter` 次连续相同调用时返回 `block` 决策：调用不再执行，工具结果以错误形式携带阻止原因，供模型阅读。

设置 `blockAfter: 0` 可保持原有的仅建议行为。`dsh` 基础组合包以 `blockAfter: 8` 启用该 guard，因此挂载组合包的部署现在会阻止第八次相同调用。

## 迁移

1. 如果部署依赖同一工具调用连续执行超过八次，请在插件配置中设置 `blockAfter: 0` 以恢复仅建议行为：

   ```yaml
   - name: '@deepseek-ai/dsh-repeat-tool-reminder'
     config:
       thresholds: [3, 5, 8]
       blockAfter: 0
   ```

2. 其余情况保持默认值。若要更早或更晚阻止，请把 `blockAfter` 设为大于等于 `2` 的整数；`1` 或小数会在插件加载时失败。

3. 通过运行一个以相同参数重复超过 `blockAfter` 次的工具来确认行为：最后一次尝试的工具结果会携带错误，其文本以 `Repeated tool call blocked:` 开头。参见[包配置参考](../../../../packages/guard/repeat-tool-reminder/README.zh.md#use-this-package)。
