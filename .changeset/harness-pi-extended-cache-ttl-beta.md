---
'@ai-sdk/harness-pi': patch
---

fix(harness-pi): send the extended-cache-ttl beta so `PI_CACHE_RETENTION=long` works

pi-ai already emits `cache_control: { type: 'ephemeral', ttl: '1h' }` under
`PI_CACHE_RETENTION=long`, but never sends the `extended-cache-ttl-2025-04-11`
beta header, so Anthropic ignores the `ttl` field and applies the default five
minutes. The anthropic provider now registers that beta when long retention is
requested.
