import { HarnessCapabilityUnsupportedError } from '@ai-sdk/harness';
import { Sandbox } from 'just-bash';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { JustBashNetworkSandboxSession } from './just-bash-network-sandbox-session';

describe('JustBashNetworkSandboxSession', () => {
  let sandbox: JustBashNetworkSandboxSession;

  beforeEach(async () => {
    sandbox = new JustBashNetworkSandboxSession({
      sandbox: await Sandbox.create({ cwd: '/work' }),
      ownsLifecycle: true,
    });
  });

  it('rejects getPortEndpoint because ports are unsupported', async () => {
    await expect(
      sandbox.getPortEndpoint({ port: 4319, protocol: 'ws' }),
    ).rejects.toBeInstanceOf(HarnessCapabilityUnsupportedError);
  });

  it('keeps getPortUrl as a compatibility wrapper', async () => {
    await expect(
      sandbox.getPortUrl({ port: 4319, protocol: 'ws' }),
    ).rejects.toBeInstanceOf(HarnessCapabilityUnsupportedError);
  });

  // The fork keeps `stop` and `destroy` apart on purpose: `stop` gives a configured store its
  // last chance to snapshot the filesystem, while `destroy` means the session was deliberately
  // discarded and must never be snapshotted. Upstream's version delegates one to the other.
  it('ends the session on destroy without running the stop hook', async () => {
    const stop = vi.spyOn(sandbox, 'stop');

    await sandbox.destroy();

    expect(stop).not.toHaveBeenCalled();
  });
});
