---
kind: upgrade-guide
description: "The repeat-tool-reminder guard blocks an identical tool call at its blockAfter count instead of only reminding."
---

# The repeat-tool-reminder guard blocks a repeated tool call

English | [中文](guide.zh.md)

## Change

`@deepseek-ai/dsh-repeat-tool-reminder` gained a `blockAfter` configuration field (default `8`). Previously the guard only appended a reminder to the conversation and never refused a call, so a model that kept repeating the same tool call with identical arguments ran it without limit once the highest reminder threshold was passed. The guard now returns a `block` decision at the `blockAfter`-th consecutive identical call: the call does not run, and the tool result carries the block reason as an error the model reads.

Set `blockAfter: 0` to keep the previous advisory-only behavior. The `dsh` base bundle enables the guard with `blockAfter: 8`, so deployments that mount the bundle now block the eighth identical call.

## Migration

1. If a deployment relies on an identical tool call running more than eight times in a row, set `blockAfter: 0` in the plugin's configuration to restore advisory-only behavior:

   ```yaml
   - name: '@deepseek-ai/dsh-repeat-tool-reminder'
     config:
       thresholds: [3, 5, 8]
       blockAfter: 0
   ```

2. Otherwise keep the default. To block earlier or later, set `blockAfter` to an integer of at least `2`; a value of `1` or a fractional value fails at plugin load.

3. Confirm the behavior by running a tool that repeats identical arguments more than `blockAfter` times: the last attempt's tool result carries an error whose text starts with `Repeated tool call blocked:`. See the [package configuration reference](../../../../packages/guard/repeat-tool-reminder/README.md#use-this-package).
