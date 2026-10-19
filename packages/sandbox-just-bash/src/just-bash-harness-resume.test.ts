import { HarnessAgent } from '@ai-sdk/harness/agent';
import type {
  HarnessV1,
  HarnessV1NetworkSandboxSession,
  HarnessV1PromptControl,
  HarnessV1SandboxProvider,
  HarnessV1Session,
} from '@ai-sdk/harness';
import { describe, expect, it } from 'vitest';
import { createJustBashSandbox } from './just-bash-sandbox';
import type {
  JustBashSandboxSnapshot,
  JustBashSandboxStore,
} from './just-bash-snapshot';

const WORKDIR_FILE = '/work/plan.md';

/**
 * Minimal adapter. It does nothing with the model — the point is the sandbox
 * lifecycle around it, and whether `HarnessAgent` accepts a resume at all.
 */
function mockHarness(): HarnessV1 {
  const lifecycleState = {
    type: 'resume-session' as const,
    harnessId: 'mock',
    specificationVersion: 'harness-v1' as const,
    data: {},
  };
  const control: HarnessV1PromptControl = {
    submitToolResult: async () => {},
    done: Promise.resolve(),
  };
  const session: Omit<HarnessV1Session, 'isResume'> = {
    sessionId: 'mock-session',
    doPromptTurn: async () => control,
    doContinueTurn: async () => control,
    doCompact: async () => {},
    doDetach: async () => lifecycleState,
    doStop: async () => lifecycleState,
    doDestroy: async () => {},
    doSuspendTurn: async () => ({
      ...lifecycleState,
      type: 'continue-turn' as const,
    }),
  };
  return {
    specificationVersion: 'harness-v1',
    harnessId: 'mock',
    builtinTools: {},
    // Report the resume the framework asked for, so a test can tell a resumed
    // session from a fresh one.
    doStart: async options => ({
      ...session,
      isResume: options.resumeFrom != null || options.continueFrom != null,
    }),
  };
}

function memoryStore(): JustBashSandboxStore {
  const entries = new Map<string, JustBashSandboxSnapshot>();
  return {
    load: async ({ sessionId }) => entries.get(sessionId),
    save: async ({ sessionId, snapshot }) => {
      entries.set(sessionId, snapshot);
    },
    remove: async ({ sessionId }) => {
      entries.delete(sessionId);
    },
  };
}

/**
 * Wraps a provider to capture the sandbox sessions it hands the agent. The
 * agent keeps its sandbox session internal, so this is how a test reaches the
 * filesystem the agent is actually running against.
 */
function recording(provider: HarnessV1SandboxProvider): {
  provider: HarnessV1SandboxProvider;
  sessions: HarnessV1NetworkSandboxSession[];
} {
  const sessions: HarnessV1NetworkSandboxSession[] = [];
  const capture = async (
    session: PromiseLike<HarnessV1NetworkSandboxSession>,
  ) => {
    const resolved = await session;
    sessions.push(resolved);
    return resolved;
  };
  return {
    sessions,
    provider: {
      ...provider,
      createSession: options => capture(provider.createSession(options)),
      ...(provider.resumeSession != null
        ? {
            resumeSession: options => capture(provider.resumeSession!(options)),
          }
        : {}),
    },
  };
}

describe('HarnessAgent resume over a persisted just-bash sandbox', () => {
  it('rejects resume when the provider has no store', async () => {
    const agent = new HarnessAgent({
      harness: mockHarness(),
      sandbox: createJustBashSandbox({ cwd: '/work' }),
    });
    const session = await agent.createSession({ sessionId: 's1' });
    const resumeFrom = await session.stop();

    await expect(
      agent.createSession({ sessionId: 's1', resumeFrom }),
    ).rejects.toThrow(/does not support resume/);
  });

  it('carries the sandbox filesystem across a stop and resume', async () => {
    const { provider, sessions } = recording(
      createJustBashSandbox({ cwd: '/work', store: memoryStore() }),
    );
    const agent = new HarnessAgent({
      harness: mockHarness(),
      sandbox: provider,
    });

    const session = await agent.createSession({ sessionId: 's1' });
    await sessions[0]!.writeTextFile({
      path: WORKDIR_FILE,
      content: 'drafted',
    });
    const resumeFrom = await session.stop();

    const resumed = await agent.createSession({ sessionId: 's1', resumeFrom });

    /*
     * This is what lets an adapter that persists its journal into the sandbox
     * (Pi writes its session file there) recover conversation history on a
     * different process.
     */
    expect(await sessions[1]!.readTextFile({ path: WORKDIR_FILE })).toBe(
      'drafted',
    );
    expect(resumed.isResume).toBe(true);
    await resumed.destroy();
  });

  it('starts clean when the stored snapshot has expired', async () => {
    const store = memoryStore();
    const { provider, sessions } = recording(
      createJustBashSandbox({ cwd: '/work', store }),
    );
    const agent = new HarnessAgent({
      harness: mockHarness(),
      sandbox: provider,
    });

    const session = await agent.createSession({ sessionId: 's1' });
    await sessions[0]!.writeTextFile({
      path: WORKDIR_FILE,
      content: 'drafted',
    });
    const resumeFrom = await session.stop();
    await store.remove({ sessionId: 's1' });

    const resumed = await agent.createSession({ sessionId: 's1', resumeFrom });

    expect(await sessions[1]!.readTextFile({ path: WORKDIR_FILE })).toBeNull();
    await resumed.destroy();
  });
});
