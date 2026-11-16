---
kind: upgrade-guide
description: "Glob patterns containing a slash are relative to the explicit search directory rather than the Session working directory."
---

# Slash-containing glob patterns use the search directory

English | [中文](guide.zh.md)

## Change

When `glob` receives an explicit `path`, a pattern containing `/` is matched relative to that directory. Previously, ripgrep matched these patterns against paths in the Session working directory's coordinates, so a call could miss existing files or require the directory prefix in both arguments.

Patterns without `/` still match basenames at any depth. Calls without `path`, `grep`, the Session working directory, and the returned path coordinates do not change. See the [filesystem search reference](../../../../packages/fs/tool-fs-search/README.md) for limits and traversal rules.

## Migration

Remove the duplicated search-directory prefix from slash-containing patterns. For example, to find files directly under `src/nested`, replace:

```json
{"pattern":"src/nested/*.ts","path":"src"}
```

with:

```json
{"pattern":"nested/*.ts","path":"src"}
```

Keep `nested/*.ts` when depth matters; do not add `**/` unless recursive matching is intended.

`path` names a directory. Pass a file's parent directory and match its name in `pattern` rather than passing the file as `path`. On Windows, use an absolute directory when selecting a different drive; a different-drive drive-relative root such as `D:src` is rejected for slash-containing patterns instead of consulting the host process's per-drive current directory.

Confirm that the call finds a known file at the intended depth and that a deeper file remains excluded. Returned paths keep their existing coordinates; the search root does not become the Session working directory. Use the returned paths, not pattern-relative paths, for follow-up filesystem calls in a co-located workspace.
