---
'@ai-sdk/harness': patch
---

fix(harness): pass tool context to host tool `execute()` and honor `settings.runtimeContext`

`HarnessAgent` always passed `context: undefined` to host tool `execute()` calls and fabricated an empty runtime context, so `stopWhen` conditions and `StepResult.runtimeContext` saw `{}` regardless of configuration. Callers had to smuggle request-scoped values (acting user, tenant, turn id) around the SDK, typically through `AsyncLocalStorage`.

`generate` / `stream` / `continueGenerate` / `continueStream` now accept a per-call `toolsContext` map, validated against each tool's `contextSchema` and delivered as the `context` execution option — the same contract as `generateText` / `streamText`. `HarnessAgentSettings.runtimeContext` mirrors `ToolLoopAgentSettings.runtimeContext`.
