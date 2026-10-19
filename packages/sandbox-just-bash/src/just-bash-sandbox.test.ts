import { InMemoryFs, Sandbox } from 'just-bash';
import { describe, expect, it, vi } from 'vitest';
import { createJustBashSandbox } from './just-bash-sandbox';
import type {
  JustBashSandboxSnapshot,
  JustBashSandboxStore,
} from './just-bash-snapshot';

/** In-process stand-in for the Redis/Postgres/S3 store a consumer supplies. */
function memoryStore(): JustBashSandboxStore & {
  entries: Map<string, JustBashSandboxSnapshot>;
  save: ReturnType<typeof vi.fn>;
  remove: ReturnType<typeof vi.fn>;
} {
  const entries = new Map<string, JustBashSandboxSnapshot>();
  const save = vi.fn(
    async ({
      sessionId,
      snapshot,
    }: {
      sessionId: string;
      snapshot: JustBashSandboxSnapshot;
    }) => {
      // Store what a real backend would: bytes, not a live reference.
      entries.set(
        sessionId,
        JSON.parse(JSON.stringify(snapshot)) as JustBashSandboxSnapshot,
      );
    },
  );
  const remove = vi.fn(async ({ sessionId }: { sessionId: string }) => {
    entries.delete(sessionId);
  });
  return {
    entries,
    save,
    remove,
    load: async ({ sessionId }) => entries.get(sessionId),
  };
}

describe('JustBashSandboxProvider persistence', () => {
  it('does not advertise resume without a store', async () => {
    const provider = createJustBashSandbox({ cwd: '/work' });

    // The harness reads the property's presence to decide whether resume is
    // supported at all, so an unbacked provider must not define it.
    expect(provider.resumeSession).toBeUndefined();
  });

  it('does not advertise resume for a caller-owned sandbox', async () => {
    const provider = createJustBashSandbox({
      sandbox: await Sandbox.create({ cwd: '/work' }),
      store: memoryStore(),
    });

    expect(provider.resumeSession).toBeUndefined();
  });

  it('rejects a store combined with a shared filesystem', () => {
    // Every session would get the same `fs`, so snapshots would overwrite each
    // other and sessions would read each other's files.
    expect(() =>
      createJustBashSandbox({
        cwd: '/work',
        fs: new InMemoryFs(),
        store: memoryStore(),
      }),
    ).toThrow(/`store` cannot be combined with `fs`/);

    expect(() =>
      createJustBashSandbox({
        overlayRoot: '/tmp',
        store: memoryStore(),
      }),
    ).toThrow(/`store` cannot be combined with `overlayRoot`/);
  });

  it('still allows a shared filesystem without a store', () => {
    expect(() =>
      createJustBashSandbox({ cwd: '/work', fs: new InMemoryFs() }),
    ).not.toThrow();
  });

  it('restores a stopped session in a later provider instance', async () => {
    const store = memoryStore();
    const settings = { cwd: '/work', store };

    const first = createJustBashSandbox(settings);
    const session = await first.createSession({ sessionId: 'conversation-1' });
    await session.run({ command: 'echo drafted > /work/plan.md' });
    await session.stop();

    // A different provider instance stands in for a different pod.
    const second = createJustBashSandbox(settings);
    const resumed = await second.resumeSession!({
      sessionId: 'conversation-1',
    });

    expect(await resumed.readTextFile({ path: '/work/plan.md' })).toBe(
      'drafted\n',
    );
  });

  it('persists a live session without ending it', async () => {
    const store = memoryStore();
    const provider = createJustBashSandbox({ cwd: '/work', store });
    const session = await provider.createSession({ sessionId: 'c1' });

    await session.run({ command: 'echo turn-one > /work/a.txt' });
    await provider.persistSession({ sessionId: 'c1' });

    // Still usable after the snapshot.
    await session.run({ command: 'echo turn-two > /work/b.txt' });
    expect(await session.readTextFile({ path: '/work/b.txt' })).toBe(
      'turn-two\n',
    );

    // The store holds the turn-one state; turn two is not in it yet.
    const resumed = await provider.resumeSession!({ sessionId: 'c1' });
    expect(await resumed.readTextFile({ path: '/work/a.txt' })).toBe(
      'turn-one\n',
    );
    expect(await resumed.readTextFile({ path: '/work/b.txt' })).toBeNull();
  });

  it('drops the snapshot when a session is destroyed', async () => {
    const store = memoryStore();
    const provider = createJustBashSandbox({ cwd: '/work', store });
    const session = await provider.createSession({ sessionId: 'c1' });
    await session.run({ command: 'echo x > /work/a.txt' });
    await provider.persistSession({ sessionId: 'c1' });
    expect(store.entries.has('c1')).toBe(true);

    // `destroy` means "discard resumability" in the harness session contract.
    await session.destroy!();

    expect(store.entries.has('c1')).toBe(false);
    expect(store.save).toHaveBeenCalledTimes(1);
  });

  it('resumes with an empty filesystem when nothing is stored', async () => {
    const provider = createJustBashSandbox({
      cwd: '/work',
      store: memoryStore(),
    });

    // An expired or evicted entry is normal for a TTL'd store; the session
    // starts clean rather than failing.
    const resumed = await provider.resumeSession!({ sessionId: 'expired' });

    expect(await resumed.readTextFile({ path: '/work/plan.md' })).toBeNull();
    expect(await resumed.run({ command: 'echo ok' })).toMatchObject({
      exitCode: 0,
      stdout: 'ok\n',
    });
  });

  it('keeps sessions isolated from each other', async () => {
    const store = memoryStore();
    const provider = createJustBashSandbox({ cwd: '/work', store });

    const a = await provider.createSession({ sessionId: 'a' });
    const b = await provider.createSession({ sessionId: 'b' });
    await a.run({ command: 'echo from-a > /work/f.txt' });
    await b.run({ command: 'echo from-b > /work/f.txt' });
    await a.stop();
    await b.stop();

    const resumedA = await provider.resumeSession!({ sessionId: 'a' });
    const resumedB = await provider.resumeSession!({ sessionId: 'b' });
    expect(await resumedA.readTextFile({ path: '/work/f.txt' })).toBe(
      'from-a\n',
    );
    expect(await resumedB.readTextFile({ path: '/work/f.txt' })).toBe(
      'from-b\n',
    );
  });

  it('does not persist a session created without an id', async () => {
    const store = memoryStore();
    const provider = createJustBashSandbox({ cwd: '/work', store });

    const session = await provider.createSession();
    await session.run({ command: 'echo x > /work/a.txt' });
    await session.stop();

    // Nothing identifies it, so there is nothing to resume.
    expect(store.save).not.toHaveBeenCalled();
    expect(store.entries.size).toBe(0);
  });

  it('settles the lifecycle once', async () => {
    const store = memoryStore();
    const provider = createJustBashSandbox({ cwd: '/work', store });
    const session = await provider.createSession({ sessionId: 'c1' });

    await session.stop();
    await session.stop();
    // `destroy` after `stop` must not erase the snapshot `stop` just wrote.
    await session.destroy!();

    expect(store.save).toHaveBeenCalledTimes(1);
    expect(store.remove).not.toHaveBeenCalled();
    expect(store.entries.has('c1')).toBe(true);
  });

  it('leaves a caller-owned sandbox untouched by the lifecycle', async () => {
    const store = memoryStore();
    const provider = createJustBashSandbox({
      sandbox: await Sandbox.create({ cwd: '/work' }),
      store,
    });

    const session = await provider.createSession({ sessionId: 'c1' });
    await session.run({ command: 'echo x > /work/a.txt' });
    await session.destroy!();

    expect(store.save).not.toHaveBeenCalled();
    expect(store.remove).not.toHaveBeenCalled();
  });
});
