---
kind: upgrade-guide
description: "Third-party Session backends must acknowledge the write handle used to persist a model request's admission receipt."
---

# Session backends must acknowledge the receipt writer

English | [中文](guide.zh.md)

## Change

Earlier releases treated `session/flush` completion as the durability checkpoint. With automatic compaction enabled on a route with a declared context window, a persisted Agent now requires the write handle it acquired to acknowledge the durable `request/admission` receipt. Backends whose listener completes without returning that handle can trigger `CONTEXT_ADMISSION_REFUSED`. The shipped JSONL backend returns the active handle after draining and flushing it.

## Migration

1. In your persistence backend's `session/flush` listener, drain the Session's queued events and await `handle.flush()`. Return the same `SessionHandle` object your backend gave the Agent when it created or opened the Session for writing.
2. Reopen an owned synthetic Session with the updated backend. Run a model request with automatic compaction enabled and a declared context window. Verify that the stored `request/admission` receipt is readable before the adapter receives the request.

See the [storage backend contract](../../../../packages/session/session-persistence/README.md#understand-the-implementation).
