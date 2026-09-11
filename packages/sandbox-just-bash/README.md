# AI SDK - just-bash Sandbox

_This package is **experimental**._

`HarnessV1SandboxProvider` implementation for [`just-bash`](https://github.com/vercel-labs/just-bash), an in-process JavaScript bash environment with a virtual filesystem. File operations and spawned processes share the same in-memory state.

This provider does not expose ports, so it cannot be used with features that require actual network sandboxes. It is primarily useful for handing a local `Experimental_SandboxSession` to AI SDK tools that accept `experimental_sandbox`.

## Setup

```bash
npm i @ai-sdk/sandbox-just-bash
```

## Usage

The factory is synchronous. The returned provider is stable; the actual `just-bash` `Sandbox` is created on demand inside `provider.createSession()`.

```ts
import { createJustBashSandbox } from '@ai-sdk/sandbox-just-bash';

const justBashSandbox = createJustBashSandbox({ cwd: '/work' });

const networkSandboxSession = await justBashSandbox.createSession();
const sandboxSession = networkSandboxSession.restricted();

await sandboxSession.writeTextFile({ path: '/work/hello.txt', content: 'hi' });

const { stdout } = await sandboxSession.run({
  command: 'cat /work/hello.txt',
});
console.log(stdout); // "hi"
```

`networkSandboxSession.restricted()` is typed as `Experimental_SandboxSession` and is the surface to hand to user tools. The network sandbox session's `getPortEndpoint` throws (just-bash has no port story), `getPortUrl` remains as a deprecated compatibility wrapper, and `setNetworkPolicy` is omitted (no local enforcement primitive).

To wrap an already-created `just-bash` `Sandbox` (e.g. with a custom `fs`), pass it via `sandbox`. Install `just-bash` directly if your application imports `Sandbox`, `OverlayFs`, or other `just-bash` exports:

```ts
import { createJustBashSandbox } from '@ai-sdk/sandbox-just-bash';
import { OverlayFs, Sandbox } from 'just-bash';

const overlay = new OverlayFs({ root: process.cwd() });
const sandbox = createJustBashSandbox({
  sandbox: await Sandbox.create({ fs: overlay, cwd: overlay.getMountPoint() }),
});
```

## Sandbox Capabilities

Everything `just-bash`'s `Sandbox.create` accepts is forwarded, including the capability flags that are off by default:

```ts
const sandbox = createJustBashSandbox({
  cwd: '/work',
  python: true, // enable python3/python
  javascript: true, // enable js-exec (QuickJS)
  commands: ['cat', 'grep', 'ls'], // restrict the built-in command set
  network: { allowedUrls: ['https://api.example.com/*'] },
});
```

Network access is off unless you configure `network` or `fetch`, and Python and js-exec are off unless enabled — each widens the code an agent can run, so turn them on deliberately.

## Durable Sessions

A just-bash filesystem is process memory, so by default a session dies with the process that created it and `HarnessAgent` refuses to resume one. Supply a `store` and the provider snapshots each session's filesystem into it, then rehydrates it in `resumeSession()` — potentially in a different process or on a different replica.

The store is yours to implement against whatever you already run; the package ships no backend client.

```ts
import {
  createJustBashSandbox,
  type JustBashSandboxStore,
} from '@ai-sdk/sandbox-just-bash';

const store: JustBashSandboxStore = {
  async load({ sessionId }) {
    const raw = await redis.get(`agent:workdir:${sessionId}`);
    return raw == null ? undefined : JSON.parse(raw);
  },
  async save({ sessionId, snapshot }) {
    await redis.set(`agent:workdir:${sessionId}`, JSON.stringify(snapshot), {
      EX: 60 * 60 * 24,
    });
  },
  async remove({ sessionId }) {
    await redis.del(`agent:workdir:${sessionId}`);
  },
};

const sandbox = createJustBashSandbox({ cwd: '/work', store });
```

Snapshots are written when a session is created with a `sessionId` and then:

| Event                                    | Effect                                     |
| ---------------------------------------- | ------------------------------------------ |
| `session.stop()`                         | `store.save` — the session is resumable    |
| `session.destroy()`                      | `store.remove` — resumability is discarded |
| `provider.persistSession({ sessionId })` | `store.save` without ending the session    |

`stop()` covers sessions that end cleanly. Call `persistSession` at any boundary you want to be able to recover from — typically the end of a turn — for sessions held open across turns, such as one parked on a pending tool approval. Sessions created without a `sessionId` are never persisted, since nothing identifies them.

`resumeSession()` starts with an empty filesystem when the store holds no snapshot, rather than failing: an expired or evicted entry is a normal outcome for a TTL'd store. Check `store.load` yourself first if your product needs to tell "expired" from "never existed".

Snapshots also carry whatever a harness adapter keeps inside the sandbox. Pi persists its session journal into the session work directory, so a resumed session recovers its conversation history along with the files.

`createJustBashSnapshot` and `restoreJustBashSandbox` are exported if you want to drive snapshots yourself.

### Limits

Snapshots are eager and complete: the whole filesystem is serialized on every save, with file contents base64-encoded. Cost scales with workdir size, so cap what agents write and keep large artifacts in object storage. Snapshot contents are tenant data — uploads, model output, conversation history — so scope keys per tenant, expire them, and hold them to the same encryption-at-rest posture as the rest of that store.

`store` cannot be combined with `fs` or `overlayRoot`: those are forwarded to every session, so all sessions would share one filesystem and overwrite each other's snapshots. The provider throws at construction rather than letting that happen at runtime.
