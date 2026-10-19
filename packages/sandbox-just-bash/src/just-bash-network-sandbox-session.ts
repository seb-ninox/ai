import {
  HarnessCapabilityUnsupportedError,
  type HarnessV1NetworkSandboxSession,
  type HarnessV1PortEndpoint,
} from '@ai-sdk/harness';
import type { Experimental_SandboxSession as SandboxSession } from '@ai-sdk/provider-utils';
import { randomUUID } from 'node:crypto';
import type { Sandbox } from 'just-bash';
import { JustBashSandboxSession } from './just-bash-sandbox-session';

const JUST_BASH_PROVIDER_ID = 'just-bash-sandbox';

/**
 * `HarnessV1NetworkSandboxSession` backed by a `just-bash` `Sandbox`. It extends
 * {@link JustBashSandboxSession} with the infra surface. Exposes no ports —
 * there is no network namespace and no way to publish an HTTP endpoint
 * reachable from outside the host process. Bridge-backed harness adapters that
 * need `getPortEndpoint` will fail with `HarnessCapabilityUnsupportedError` at
 * start.
 *
 * Network policy is not implementable locally either — `setNetworkPolicy` is
 * omitted.
 */
export class JustBashNetworkSandboxSession
  extends JustBashSandboxSession
  implements HarnessV1NetworkSandboxSession
{
  /**
   * Minted at construct time. just-bash has no native identifier and no
   * cross-process reattach surface; the value exists only to satisfy the
   * `HarnessV1NetworkSandboxSession.id` contract.
   */
  readonly id: string;
  readonly defaultWorkingDirectory: string;
  private readonly ownsLifecycle: boolean;
  private readonly onStop: (() => Promise<void>) | undefined;
  private readonly onDestroy: (() => Promise<void>) | undefined;
  private ended = false;

  constructor(input: {
    sandbox: Sandbox;
    ownsLifecycle: boolean;
    /**
     * Runs once, when the session stops and its state should survive. The
     * provider uses it to snapshot the filesystem into a configured store.
     */
    onStop?: () => Promise<void>;
    /**
     * Runs once, when the session is destroyed and its state should not
     * survive. The provider uses it to drop the stored snapshot.
     */
    onDestroy?: () => Promise<void>;
  }) {
    super(input.sandbox);
    this.ownsLifecycle = input.ownsLifecycle;
    this.onStop = input.onStop;
    this.onDestroy = input.onDestroy;
    this.id = randomUUID();
    this.defaultWorkingDirectory = input.sandbox.bashEnvInstance.getCwd();
  }

  /**
   * The underlying `just-bash` sandbox.
   *
   * @internal — used by the provider to snapshot a live session. Tools receive
   * the narrowed `restricted()` view and never reach this.
   */
  get sandboxInstance(): Sandbox {
    return this.sandbox;
  }

  readonly ports: ReadonlyArray<number> = [];

  restricted(): SandboxSession {
    return new JustBashSandboxSession(this.sandbox);
  }

  getPortEndpoint = async (_options: {
    port: number;
    protocol?: 'http' | 'https' | 'ws';
  }): Promise<HarnessV1PortEndpoint> => {
    throw new HarnessCapabilityUnsupportedError({
      harnessId: JUST_BASH_PROVIDER_ID,
      message:
        'just-bash sandboxes run in-process and cannot expose a port URL. ' +
        'Use a hosted sandbox (e.g. @ai-sdk/sandbox-vercel) for bridge-backed harness adapters.',
    });
  };

  /**
   * @deprecated Use `getPortEndpoint` instead.
   */
  getPortUrl = async (options: {
    port: number;
    protocol?: 'http' | 'https' | 'ws';
  }): Promise<string> => {
    return (await this.getPortEndpoint(options)).url;
  };

  /*
   * just-bash has no explicit shutdown; the sandbox is garbage-collected along
   * with its in-memory filesystem once references drop. The lifecycle hooks are
   * the only real work here — they give a configured store its last chance to
   * capture the filesystem before it goes.
   *
   * Both settle at most once: `destroy` falls through to `stop` in the harness
   * contract, and a snapshot must not be written after the session was
   * deliberately discarded.
   */
  stop = async (): Promise<void> => {
    if (!this.ownsLifecycle || this.ended) return;
    this.ended = true;
    await this.onStop?.();
  };

  destroy = async (): Promise<void> => {
    if (!this.ownsLifecycle || this.ended) return;
    this.ended = true;
    await this.onDestroy?.();
  };
}
