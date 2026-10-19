---
'@ai-sdk/harness': patch
---

fix(harness): create the sandbox work directory without relying on `env` expansion

The session work-directory bootstrap ran `mkdir -p "$WORK_DIR"` with the path supplied through the sandbox `env` option. Providers are only required to put `env` in the process environment, not to expose it to the shell's parameter expansion; on a provider that does not (`@ai-sdk/sandbox-just-bash`), the command degraded to `mkdir -p ""`, which exits 0 without creating anything. The directory silently did not exist and the failure surfaced much later as the adapter failing to `cd` into it, aborting `createSession()`.

The path is now shell-quoted directly into the command.
