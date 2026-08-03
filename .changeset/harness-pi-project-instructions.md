---
'@ai-sdk/harness-pi': patch
---

feat(harness-pi): surface agent guidance in the system prompt via `projectInstructions`

The harness-v1 contract applies `HarnessV1PromptTurnOptions.instructions` by prepending them to the first user message of a fresh session. That placement is fragile for guidance meant to hold for the whole session: compaction can summarize the message away, and the adapter must not re-apply the instructions on resume, so a rebuilt session depends on the persisted history still carrying them.

`createPi({ projectInstructions })` offers the system-prompt route instead. The value is registered with Pi's resource loader as a synthetic workspace-root `AGENTS.md` context file, which Pi renders into the system prompt's `<project_context>` block — the same channel a project's own `AGENTS.md` uses. Nothing is written to the workspace, so the sandbox mirror stays a faithful view of the sandbox.

Because it is sourced from harness settings rather than a per-turn field, it is available before the resource loader's first `reload()` and applies to every session build — fresh, resumed, and `doContinueTurn` (which carries no instructions at all). It is appended after any discovered ancestor `AGENTS.md`, replaces rather than duplicates a file at the same path, and is independent of the per-turn `instructions` field, which is unchanged.
