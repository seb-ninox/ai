import type {
  HarnessV1NetworkSandboxSession,
  HarnessV1SandboxProvider,
} from '@ai-sdk/harness';
import type { Experimental_SandboxSession as SandboxSession } from '@ai-sdk/provider-utils';
import { Sandbox } from 'just-bash';
import { JustBashNetworkSandboxSession } from './just-bash-network-sandbox-session';
import {
  createJustBashSnapshot,
  restoreJustBashSandbox,
  type JustBashSandboxStore,
} from './just-bash-snapshot';

/**
 * Parameters forwarded to `just-bash`'s `Sandbox.create` when creating a
 * sandbox from scratch. Aliased directly from the underlying SDK so the full
 * surface is available without us re-declaring it.
 */
type JustBashSandboxCreateParams = NonNullable<
  Parameters<typeof Sandbox.create>[0]
>;

/**
 * Durable session storage. Opt in by supplying a {@link JustBashSandboxStore};
 * without one the provider keeps its original behaviour and sessions live and
 * die with the host process.
 */
type JustBashSandboxPersistence = {
  /**
   * Backend the provider snapshots session filesystems into, enabling
   * `resumeSession` across processes.
   *
   * Only sessions created with a `sessionId` are persisted — the harness
   * supplies one for every session it intends to resume.
   */
  readonly store?: JustBashSandboxStore;
};

/**
 * Settings for {@link createJustBashSandbox}. Two mutually-exclusive shapes:
 *
 * - `{ sandbox }` — wrap an already-created `just-bash` `Sandbox`. The caller
 *   owns its lifecycle.
 * - {@link JustBashSandboxCreateParams} fields — provider calls
 *   `Sandbox.create(settings)` on every `createSession()`.
 *
 * just-bash has no port exposure and no native snapshot mechanism, so image
 * management is a per-call no-op: if an adapter declares a bootstrap recipe
 * the provider runs it once on the freshly-created sandbox before returning.
 * Session persistence is opt-in through `store`.
 */
export type JustBashSandboxSettings =
  | ({ sandbox: Sandbox } & JustBashSandboxPersistence)
  | (JustBashSandboxCreateParams & {
      sandbox?: never;
    } & JustBashSandboxPersistence);

const JUST_BASH_PROVIDER_ID = 'just-bash-sandbox';

export function createJustBashSandbox(
  settings: JustBashSandboxSettings = {} as JustBashSandboxSettings,
): JustBashSandboxProvider {
  return new JustBashSandboxProvider(settings);
}

/**
 * `HarnessV1SandboxProvider` implementation backed by `just-bash`. Useful for
 * non-bridge harness flows and for handing a local `Experimental_SandboxSession`
 * to AI SDK tools — use `provider.createSession()` then
 * `sandboxSession.restricted()` to get the latter.
 *
 * Note: just-bash cannot expose ports, so bridge-backed harness adapters
 * (claude-code, codex) will reject this provider at start.
 *
 * ## Session persistence
 *
 * A just-bash filesystem is in-process memory, so by default a session cannot
 * outlive the process that created it. Supply a {@link JustBashSandboxStore}
 * and the provider snapshots each session's filesystem into it — on
 * `session.stop()`, and whenever the caller asks via {@link persistSession} —
 * then rehydrates it in `resumeSession()`, potentially in another process.
 *
 * That covers more than the workdir: harness adapters that keep their
 * conversation journal inside the sandbox (Pi does) get their history restored
 * with it, so a resumed session keeps its native history.
 */
export class JustBashSandboxProvider implements HarnessV1SandboxProvider {
  readonly specificationVersion = 'harness-sandbox-v1' as const;
  readonly providerId = JUST_BASH_PROVIDER_ID;

  /**
   * Live persisted sessions by id, so {@link persistSession} can snapshot a
   * session the caller does not hold a reference to. Entries are added on
   * create/resume and removed when the session stops or is destroyed.
   */
  private readonly liveSessions = new Map<
    string,
    JustBashNetworkSandboxSession
  >();

  constructor(private readonly settings: JustBashSandboxSettings) {
    const store = settings.store;
    /*
     * `fs` and `overlayRoot` are forwarded to every `Sandbox.create` call, so
     * all sessions would share one filesystem — snapshots would overwrite each
     * other and sessions would read each other's files. Fail loudly at
     * construction instead of corrupting sessions at runtime.
     */
    if (store != null && !('sandbox' in settings && settings.sandbox)) {
      const shared =
        settings.fs != null
          ? 'fs'
          : settings.overlayRoot != null
            ? 'overlayRoot'
            : undefined;
      if (shared != null) {
        throw new Error(
          `createJustBashSandbox: \`store\` cannot be combined with \`${shared}\`, which every session would share. ` +
            'Persistence requires each session to own its filesystem.',
        );
      }
    }
    /*
     * The harness treats the mere presence of `resumeSession` as "this provider
     * supports resume", so it is installed only when something can actually
     * satisfy it. A caller-owned sandbox is excluded for the same reason
     * `createSession` does not snapshot one: the provider does not own its
     * state.
     */
    if (store != null && !('sandbox' in settings && settings.sandbox)) {
      this.resumeSession = async ({ sessionId, abortSignal }) => {
        abortSignal?.throwIfAborted();
        const snapshot = await store.load({ sessionId });
        const sandbox =
          snapshot == null
            ? await Sandbox.create(this.createParams())
            : await restoreJustBashSandbox({
                snapshot,
                options: this.createParams(),
              });
        return this.track({ sandbox, sessionId });
      };
    }
  }

  createSession = async (options?: {
    sessionId?: string;
    abortSignal?: AbortSignal;
    identity?: string;
    onFirstCreate?: (
      session: SandboxSession,
      opts: { abortSignal?: AbortSignal },
    ) => Promise<void>;
  }): Promise<HarnessV1NetworkSandboxSession> => {
    options?.abortSignal?.throwIfAborted();

    if ('sandbox' in this.settings && this.settings.sandbox) {
      // The caller owns this sandbox and its lifetime; snapshotting it would
      // persist state the provider does not control.
      return new JustBashNetworkSandboxSession({
        sandbox: this.settings.sandbox,
        ownsLifecycle: false,
      });
    }

    const sandbox = await Sandbox.create(this.createParams());
    const sandboxSession = this.track({
      sandbox,
      sessionId: options?.sessionId,
    });

    if (options?.onFirstCreate != null) {
      await options.onFirstCreate(sandboxSession.restricted(), {
        abortSignal: options?.abortSignal,
      });
    }

    return sandboxSession;
  };

  /**
   * Rehydrate the session's filesystem from the store. Present only when a
   * store is configured — the harness reads this property to decide whether
   * resume is supported at all, so it must be absent otherwise.
   *
   * A session with no stored snapshot resumes with an empty filesystem rather
   * than failing: an expired or evicted entry is a normal outcome of a TTL'd
   * store, and adapters fall back to a fresh conversation when their journal
   * is missing. Distinguish the two by checking `store.load` yourself before
   * resuming if the difference matters to your product.
   */
  resumeSession?: (options: {
    sessionId: string;
    abortSignal?: AbortSignal;
  }) => Promise<HarnessV1NetworkSandboxSession>;

  /**
   * Snapshot a live session into the store now, without ending it.
   *
   * `session.stop()` already persists, but a long-lived session that is never
   * stopped — one held open across turns for a pending approval — would other-
   * wise keep everything in memory. Call this at whatever boundary you want to
   * be able to recover from, typically the end of a turn.
   *
   * No-op when no store is configured or `sessionId` is not a live session.
   */
  persistSession = async (options: { sessionId: string }): Promise<void> => {
    const store = this.settings.store;
    const session = this.liveSessions.get(options.sessionId);
    if (store == null || session == null) return;
    await store.save({
      sessionId: options.sessionId,
      snapshot: await createJustBashSnapshot(session.sandboxInstance),
    });
  };

  /** `Sandbox.create` options, with the persistence fields stripped. */
  private createParams(): JustBashSandboxCreateParams {
    const { store: _store, ...createParams } = this
      .settings as JustBashSandboxCreateParams & JustBashSandboxPersistence;
    return createParams;
  }

  /**
   * Wrap a sandbox in a session and, when persistence applies, register it so
   * `stop` saves its snapshot and `destroy` drops it.
   */
  private track(input: {
    sandbox: Sandbox;
    sessionId: string | undefined;
  }): JustBashNetworkSandboxSession {
    const store = this.settings.store;
    const { sessionId } = input;
    if (store == null || sessionId == null) {
      return new JustBashNetworkSandboxSession({
        sandbox: input.sandbox,
        ownsLifecycle: true,
      });
    }

    const session = new JustBashNetworkSandboxSession({
      sandbox: input.sandbox,
      ownsLifecycle: true,
      /*
       * `stop` means "persist enough to resume later" and `destroy` means
       * "discard resumability", per the harness session contract — so they map
       * onto saving and removing the snapshot.
       */
      onStop: async () => {
        this.liveSessions.delete(sessionId);
        await store.save({
          sessionId,
          snapshot: await createJustBashSnapshot(input.sandbox),
        });
      },
      onDestroy: async () => {
        this.liveSessions.delete(sessionId);
        await store.remove({ sessionId });
      },
    });
    this.liveSessions.set(sessionId, session);
    return session;
  }
}
