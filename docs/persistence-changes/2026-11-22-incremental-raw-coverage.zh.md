---
description: "记录持久化类型更改及其兼容性确认。"
kind: persistence-change
---

# 2026-11-22-incremental-raw-coverage

[English](2026-11-22-incremental-raw-coverage.md) | 中文

## 概述

为 compaction/summary 增加可选的增量 RAW 覆盖记录，使重复压缩能够记录检查点已经涵盖的追加原始事件范围。

## 目录

- [声明](#declaration)
- [兼容性](#compatibility)
- [验证](#verification)
- [开发备注](#dev-note)

<a id="declaration"></a>
## 声明

```yaml persistence-change
schemaVersion: 1
id: 2026-11-22-incremental-raw-coverage
baseline: false
changes:
  - root: "event:compaction/summary"
    previous: "2026-11-20-hybrid-compaction-replay"
    after: "ed393982b25498a87088b799e3cc6f41a9a3a3a431d3141cbe19582503057dfd"
    decision: same-version
```

<a id="compatibility"></a>
## 兼容性

已有记录仍然有效并保持全表面行为。字段缺失绝不表示已有覆盖。旧版 V4 读取器把新增负载作为数据保留并恢复同一 Surface；当前读取器接受缺失，并在使用前校验记录。写入方仅对增量压缩在同一摘要事件中写入该字段。

<a id="verification"></a>
## 验证

node node_modules/vitest/vitest.mjs run --config vitest.config.ts packages/compaction/compaction-basic/tests/incremental.spec.ts：7 个测试通过。node node_modules/vitest/vitest.mjs run --config vitest.config.ts scripts/stable-compaction-hybrid.spec.ts：21 个测试通过。

<a id="dev-note"></a>
## 开发备注

无。
