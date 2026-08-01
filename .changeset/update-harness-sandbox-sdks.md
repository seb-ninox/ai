---
'@ai-sdk/sandbox-just-bash': patch
'@ai-sdk/harness-pi': patch
---

chore: update just-bash to 3.2.0 and pi-coding-agent to 0.82.1

`just-bash` 3.x reaches `Sandbox.create`, so `createJustBashSandbox` now forwards the capability flags that were previously dropped on the floor — `python`, `javascript`, `commands`, `customCommands` and `fetch` (just-bash 3.1.0). It also brings the 3.2.0 hardening pass for untrusted execution: shared aggregate budgets, transactional filesystem and shell state, and request-bound network validation. The 3.0.0 breaking change is confined to stdin byte/utf8 handling in custom commands, which this provider does not register.

`@earendil-works/pi-coding-agent` 0.82.1 adds Claude Opus 5 on Anthropic and Amazon Bedrock with adaptive thinking, inference profiles and prompt caching, and fixes an active-response session replacement that could leave dangling tool calls. 0.83.0 is held back by the workspace's `minimumReleaseAge`; it also carries a TypeBox 1.3.7 upgrade that removes deprecated APIs, so it wants its own pass.
