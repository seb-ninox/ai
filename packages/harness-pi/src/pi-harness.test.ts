import { describe, expect, it, vi } from 'vitest';
import type {
  HarnessV1NetworkSandboxSession,
  HarnessV1Session,
} from '@ai-sdk/harness';
import { createPi } from './pi-harness';
import { createPiSession } from './pi-session';

vi.mock('./pi-session', () => ({
  createPiSession: vi.fn(async () => ({}) as HarnessV1Session),
}));

function startOptions() {
  return {
    sessionId: 'session-1',
    sandboxSession: {} as HarnessV1NetworkSandboxSession,
    sessionWorkDir: '/sandbox/work',
  };
}

describe('createPi adapter', () => {
  it('declares the harness id and builtin tools', () => {
    const harness = createPi();
    expect(harness.harnessId).toBe('pi');
    expect(harness.specificationVersion).toBe('harness-v1');
    expect(harness.supportsBuiltinToolApprovals).toBe(true);
    expect(harness.supportsBuiltinToolFiltering).toBe(true);
    expect(Object.keys(harness.builtinTools).sort()).toEqual([
      'bash',
      'edit',
      'glob',
      'grep',
      'ls',
      'read',
      'write',
    ]);
    expect(harness.builtinTools.read.nativeName).toBe('read');
    expect(harness.builtinTools.read.commonName).toBe('read');
    expect(harness.builtinTools.read.toolUseKind).toBe('readonly');
    expect(harness.builtinTools.write.toolUseKind).toBe('edit');
    expect(harness.builtinTools.bash.toolUseKind).toBe('bash');
    // `glob` is the common-name key; the native Pi name is `find`.
    expect(harness.builtinTools.glob.nativeName).toBe('find');
    expect(harness.builtinTools.glob.commonName).toBe('glob');
    // `ls` is Pi-specific and intentionally has no common equivalent.
    expect(harness.builtinTools.ls.nativeName).toBe('ls');
    expect(harness.builtinTools.ls.commonName).toBeUndefined();
  });

  it('exposes a lifecycle-state schema', () => {
    const harness = createPi();
    expect(harness.lifecycleStateSchema).toBeDefined();
  });

  it('omits getBootstrap (no in-sandbox install needed)', () => {
    const harness = createPi();
    expect(harness.getBootstrap).toBeUndefined();
  });

  it('forwards projectInstructions to the session', async () => {
    vi.mocked(createPiSession).mockClear();
    await createPi({ projectInstructions: 'Be concise.' }).doStart(
      startOptions(),
    );

    expect(vi.mocked(createPiSession).mock.calls[0]?.[0]).toMatchObject({
      projectInstructions: 'Be concise.',
    });
  });

  it('omits projectInstructions when unset', async () => {
    vi.mocked(createPiSession).mockClear();
    await createPi().doStart(startOptions());

    expect(vi.mocked(createPiSession).mock.calls[0]?.[0]).not.toHaveProperty(
      'projectInstructions',
    );
  });

  it('forwards sequentialToolCalls to the session', async () => {
    vi.mocked(createPiSession).mockClear();
    await createPi({ sequentialToolCalls: true }).doStart(startOptions());

    expect(vi.mocked(createPiSession).mock.calls[0]?.[0]).toMatchObject({
      settings: { sequentialToolCalls: true },
    });
  });

  it('omits sequentialToolCalls when unset', async () => {
    vi.mocked(createPiSession).mockClear();
    await createPi().doStart(startOptions());

    expect(
      vi.mocked(createPiSession).mock.calls[0]?.[0].settings,
    ).not.toHaveProperty('sequentialToolCalls');
  });
});
