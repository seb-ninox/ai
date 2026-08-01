---
'@ai-sdk/harness-pi': patch
---

Deliver a tool outcome the host submits before Pi drives the matching call, instead of dropping it.

A turn parked on host input is normally continued on the parked in-process session, which still holds the tool's waiter. When that session is gone — evicted, or the process restarted — the continuation runs against a session rebuilt from the persisted journal, so the host submits the outcome before Pi re-drives the recorded call. The submission found no waiter and returned silently, stranding the tool's `execute` promise forever: an approved call re-ran and looked to the model like a failure worth retrying (often decomposed, so input-hash replay guards could not catch it), and a denied call hung the turn with no error at all. Outcomes that arrive first are now held and consumed when the call is driven, making the two arrival orders equivalent.
