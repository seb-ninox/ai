---
'@ai-sdk/harness': patch
---

fix(harness): keep text and reasoning blocks self-contained within each result stream

A runtime that keeps its native turn alive across a tool-approval pause (Pi does) resumes the text block it had open with a plain `text-delta` on the continuation stream, whose `text-start` went to the previous result. UI message assembly is per result, so `processUIMessageStream` rejected that delta with a `UIMessageStreamError` and the rest of the turn was lost. Adapters could also finish a stream leaving a block open, stranding the text part in the `streaming` state.

The harness now synthesizes the missing `-start` for a resumed block, drops an `-end` for a block the stream never opened, and closes blocks the runtime leaves open before a step or the stream settles.
