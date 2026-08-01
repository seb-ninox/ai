---
'@ai-sdk/sandbox-just-bash': patch
---

feat(sandbox-just-bash): durable sessions via a pluggable snapshot store

A just-bash filesystem is process memory, so a session could not outlive the process that created it: the provider had no `resumeSession`, and `HarnessAgent` rejected every resume against it with `HarnessCapabilityUnsupportedError`. That forced deployments to pin a conversation to one replica.

`createJustBashSandbox({ store })` now snapshots each session's filesystem into a caller-supplied `JustBashSandboxStore` and rehydrates it in `resumeSession()`, so a session can resume in another process. Snapshots are written on `session.stop()`, dropped on `session.destroy()`, and can be taken mid-session with `provider.persistSession({ sessionId })` for sessions held open across turns. The snapshot is plain JSON, so any backend works; the package ships no backend client and gains no dependencies.

Providers without a `store` are unchanged, and still do not advertise resume. Also exports `createJustBashSnapshot` / `restoreJustBashSandbox` for driving snapshots directly.
