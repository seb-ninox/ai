import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  type AgentSession,
  type ToolDefinition,
  ModelRuntime,
  SettingsManager,
} from '@earendil-works/pi-coding-agent';
import type {
  HarnessV1NetworkSandboxSession,
  HarnessV1ToolSpec,
} from '@ai-sdk/harness';
import { createPiSession } from './pi-session';

type FakePiTool = Pick<ToolDefinition, 'name' | 'execute'>;

const piMock = vi.hoisted(() => {
  return {
    createAgentSession: vi.fn(),
    customTools: [] as FakePiTool[],
    session: undefined as AgentSession | undefined,
  };
});

vi.mock('@earendil-works/pi-coding-agent', () => {
  return {
    createAgentSession: piMock.createAgentSession,
    DefaultResourceLoader: class {
      async reload() {}
    },
    defineTool: vi.fn(tool => tool),
    ModelRegistry: class {
      getAll = vi.fn(() => []);
      registerProvider = vi.fn();
    },
    ModelRuntime: {
      create: vi.fn(async () => ({
        setRuntimeApiKey: vi.fn(async () => {}),
      })),
    },
    SessionManager: {
      create: vi.fn(() => ({
        getSessionFile: () => undefined,
      })),
      open: vi.fn(() => ({
        getSessionFile: () => undefined,
      })),
    },
    SettingsManager: {
      inMemory: vi.fn(() => ({})),
      create: vi.fn(() => ({})),
    },
  };
});

describe('createPiSession', () => {
  beforeEach(() => {
    piMock.customTools = [];
    piMock.session = undefined;
    piMock.createAgentSession.mockReset();
    piMock.createAgentSession.mockImplementation(async options => {
      piMock.customTools = options.customTools;
      return { session: piMock.session };
    });
  });

  it('rejects unsafe resume session filenames before sandbox restore', async () => {
    const sandboxSession = createSandboxSession();

    await expect(
      createPiSession({
        sessionId: 'session-unsafe',
        sandboxSession,
        sessionWorkDir: '/sandbox/work',
        skills: [],
        settings: {},
        clientApp: 'ai-sdk/harness-pi/0.0.0-test',
        isResume: true,
        resumeSessionFileName: '../session.jsonl',
      }),
    ).rejects.toThrow('Invalid Pi session file name');

    expect(sandboxSession.readBinaryFile).not.toHaveBeenCalled();
  });

  it('parks a pending tool turn on suspend and resumes it in-process', async () => {
    const toolStarted = createDeferred<void>();
    let resolvedToolResult: unknown;
    const prompt = vi.fn(async () => {
      const tool = piMock.customTools.find(tool => tool.name === 'weather');
      if (!tool) throw new Error('Expected weather tool.');
      const toolResultPromise = tool.execute(
        'tool-1',
        {},
        undefined,
        undefined,
        undefined as never,
      );
      toolStarted.resolve();
      resolvedToolResult = await toolResultPromise;
    });
    const abort = vi.fn(async () => {});
    piMock.session = {
      abort,
      compact: vi.fn(async () => {}),
      dispose: vi.fn(),
      getSessionStats: () => ({
        tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
      prompt,
      steer: vi.fn(async () => {}),
      subscribe: vi.fn(() => () => {}),
    } as unknown as AgentSession;

    const sandboxSession = createSandboxSession();
    const session = await createPiSession({
      sessionId: 'session-1',
      sandboxSession,
      sessionWorkDir: '/sandbox/work',
      skills: [],
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: false,
    });
    const toolSpecs: HarnessV1ToolSpec[] = [{ name: 'weather' }];
    const control = await session.doPromptTurn({
      prompt: 'go',
      tools: toolSpecs,
      emit: vi.fn(),
    });

    await toolStarted.promise;
    await expect(session.doSuspendTurn()).resolves.toEqual({
      type: 'continue-turn',
      harnessId: 'pi',
      specificationVersion: 'harness-v1',
      data: {},
    });
    expect(abort).not.toHaveBeenCalled();

    const resumedSession = await createPiSession({
      sessionId: 'session-1',
      sandboxSession,
      sessionWorkDir: '/sandbox/work',
      skills: [],
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
    });
    const resumedControl = await resumedSession.doContinueTurn({
      tools: toolSpecs,
      emit: vi.fn(),
    });

    await resumedControl.submitToolResult({
      toolCallId: 'tool-1',
      output: { weather: 'sunny' },
    });
    await resumedControl.done;
    await control.done;

    expect(prompt).toHaveBeenCalledTimes(1);
    expect(resolvedToolResult).toMatchInlineSnapshot(
      {
        content: [{ type: 'text', text: '{"weather":"sunny"}' }],
        details: undefined,
      },
      `
      {
        "content": [
          {
            "text": "{"weather":"sunny"}",
            "type": "text",
          },
        ],
        "details": undefined,
      }
    `,
    );
  });

  /*
   * A turn parked on host input is normally continued on the parked in-process
   * session, which still holds the tool's waiter. When that session is gone the
   * continuation runs against one rebuilt from the journal, so the host submits
   * the outcome BEFORE Pi re-drives the recorded call. These two cover that
   * order; dropping the outcome there stranded the tool promise forever, which
   * surfaced as an approved write the model re-proposed (it never saw a result)
   * and a denied one that hung the turn with no error at all.
   */
  it('delivers a tool result submitted before Pi drove the call', async () => {
    const outcomeSubmitted = createDeferred<void>();
    let resolvedToolResult: unknown;
    const prompt = vi.fn(async () => {
      await outcomeSubmitted.promise;
      const tool = piMock.customTools.find(tool => tool.name === 'weather');
      if (!tool) throw new Error('Expected weather tool.');
      resolvedToolResult = await tool.execute(
        'tool-1',
        {},
        undefined,
        undefined,
        undefined as never,
      );
    });
    piMock.session = {
      abort: vi.fn(async () => {}),
      compact: vi.fn(async () => {}),
      dispose: vi.fn(),
      getSessionStats: () => ({
        tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
      prompt,
      steer: vi.fn(async () => {}),
      subscribe: vi.fn(() => () => {}),
    } as unknown as AgentSession;

    const session = await createPiSession({
      sessionId: 'session-early-result',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
      skills: [],
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
    });
    const control = await session.doPromptTurn({
      prompt: 'go',
      tools: [{ name: 'weather' }] satisfies HarnessV1ToolSpec[],
      emit: vi.fn(),
    });

    await control.submitToolResult({
      toolCallId: 'tool-1',
      output: { weather: 'sunny' },
    });
    outcomeSubmitted.resolve();
    await control.done;

    expect(resolvedToolResult).toEqual({
      content: [{ type: 'text', text: '{"weather":"sunny"}' }],
      details: undefined,
    });
  });

  it('settles a denied tool call submitted before Pi drove the call', async () => {
    const outcomeSubmitted = createDeferred<void>();
    let resolvedToolResult: unknown;
    const prompt = vi.fn(async () => {
      await outcomeSubmitted.promise;
      const tool = piMock.customTools.find(tool => tool.name === 'weather');
      if (!tool) throw new Error('Expected weather tool.');
      resolvedToolResult = await tool.execute(
        'tool-1',
        {},
        undefined,
        undefined,
        undefined as never,
      );
    });
    piMock.session = {
      abort: vi.fn(async () => {}),
      compact: vi.fn(async () => {}),
      dispose: vi.fn(),
      getSessionStats: () => ({
        tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
      prompt,
      steer: vi.fn(async () => {}),
      subscribe: vi.fn(() => () => {}),
    } as unknown as AgentSession;

    const session = await createPiSession({
      sessionId: 'session-early-denial',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
      skills: [],
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
    });
    const control = await session.doPromptTurn({
      prompt: 'go',
      tools: [{ name: 'weather' }] satisfies HarnessV1ToolSpec[],
      emit: vi.fn(),
    });

    // The exact payload the harness submits for a denied approval.
    await control.submitToolResult({
      toolCallId: 'tool-1',
      output: { type: 'execution-denied', reason: 'Denied by user' },
    });
    outcomeSubmitted.resolve();
    await control.done;

    expect(resolvedToolResult).toEqual({
      content: [
        {
          type: 'text',
          text: '{"type":"execution-denied","reason":"Denied by user"}',
        },
      ],
      details: undefined,
    });
  });

  /*
   * Cross-process continuation of a turn parked on host input: the parked live
   * session is gone and Pi never re-drives a recorded call after a journal
   * rebuild, so `doContinueTurn` must hold the rerun until the host settles the
   * recorded calls, write each outcome into the rebuilt context, and only then
   * re-drive. Without that, the rerun ships a context whose dangling call gets
   * a synthetic "No result provided" error and the model retries an
   * already-applied call.
   */
  it('injects host outcomes into the rebuilt context before re-driving a parked continuation', async () => {
    const stateMessages: Array<Record<string, unknown>> = [
      {
        role: 'user',
        content: [{ type: 'text', text: 'insert the rows' }],
        timestamp: 1,
      },
      {
        role: 'assistant',
        stopReason: 'toolUse',
        content: [
          { type: 'toolCall', id: 'tool-1', name: 'applyChanges', arguments: {} },
        ],
        timestamp: 2,
      },
    ];
    const appendMessage = vi.fn();
    const prompt = vi.fn(async () => {});
    piMock.session = {
      abort: vi.fn(async () => {}),
      agent: { state: { messages: stateMessages } },
      compact: vi.fn(async () => {}),
      dispose: vi.fn(),
      getSessionStats: () => ({
        tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
      messages: stateMessages,
      prompt,
      sessionManager: { appendMessage },
      steer: vi.fn(async () => {}),
      subscribe: vi.fn(() => () => {}),
    } as unknown as AgentSession;

    const session = await createPiSession({
      sessionId: 'session-inject-outcome',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
      skills: [],
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
      pendingHostToolCalls: [
        {
          toolCallId: 'tool-1',
          toolName: 'applyChanges',
          input: '{"op":"insert"}',
        },
      ],
    });

    const emit = vi.fn();
    const control = await session.doContinueTurn({
      tools: [{ name: 'applyChanges' }] satisfies HarnessV1ToolSpec[],
      emit,
    });

    // The recorded call is replayed so the framework can settle it against
    // its continuations (or re-emit its approval request).
    expect(emit).toHaveBeenCalledWith({
      type: 'tool-call',
      toolCallId: 'tool-1',
      toolName: 'applyChanges',
      input: '{"op":"insert"}',
      providerExecuted: false,
    });
    // The rerun is held until the host settles the recorded call.
    expect(prompt).not.toHaveBeenCalled();

    await control.submitToolResult({
      toolCallId: 'tool-1',
      output: { applied: true, revision: 42 },
    });
    await control.done;

    const injected = stateMessages.at(-1);
    expect(injected).toMatchObject({
      role: 'toolResult',
      toolCallId: 'tool-1',
      toolName: 'applyChanges',
      content: [{ type: 'text', text: '{"applied":true,"revision":42}' }],
      isError: false,
    });
    expect(appendMessage).toHaveBeenCalledWith(injected);
    // The outcome landed in the context (state + journal) before the rerun.
    expect(appendMessage.mock.invocationCallOrder[0]).toBeLessThan(
      prompt.mock.invocationCallOrder[0]!,
    );
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(prompt).toHaveBeenCalledWith('');
  });

  it('keeps the continuation parked until every recorded call is settled', async () => {
    const stateMessages: Array<Record<string, unknown>> = [
      {
        role: 'assistant',
        stopReason: 'toolUse',
        content: [
          { type: 'toolCall', id: 'tool-1', name: 'applyChanges', arguments: {} },
          { type: 'toolCall', id: 'tool-2', name: 'applyChanges', arguments: {} },
        ],
        timestamp: 1,
      },
    ];
    const prompt = vi.fn(async () => {});
    piMock.session = {
      abort: vi.fn(async () => {}),
      agent: { state: { messages: stateMessages } },
      compact: vi.fn(async () => {}),
      dispose: vi.fn(),
      getSessionStats: () => ({
        tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
      messages: stateMessages,
      prompt,
      sessionManager: { appendMessage: vi.fn() },
      steer: vi.fn(async () => {}),
      subscribe: vi.fn(() => () => {}),
    } as unknown as AgentSession;

    const session = await createPiSession({
      sessionId: 'session-partial-settle',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
      skills: [],
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
      pendingHostToolCalls: [
        { toolCallId: 'tool-1', toolName: 'applyChanges', input: '{}' },
        { toolCallId: 'tool-2', toolName: 'applyChanges', input: '{}' },
      ],
    });

    const control = await session.doContinueTurn({
      tools: [{ name: 'applyChanges' }] satisfies HarnessV1ToolSpec[],
      emit: vi.fn(),
    });

    await control.submitToolResult({ toolCallId: 'tool-1', output: 'ok' });
    expect(prompt).not.toHaveBeenCalled();

    await control.submitToolResult({ toolCallId: 'tool-2', output: 'ok' });
    await control.done;
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(
      stateMessages.filter(message => message['role'] === 'toolResult'),
    ).toHaveLength(2);
  });

  it('does not hold the continuation for calls the journal already answers', async () => {
    const stateMessages: Array<Record<string, unknown>> = [
      {
        role: 'assistant',
        stopReason: 'toolUse',
        content: [
          { type: 'toolCall', id: 'tool-1', name: 'applyChanges', arguments: {} },
        ],
        timestamp: 1,
      },
      {
        role: 'toolResult',
        toolCallId: 'tool-1',
        toolName: 'applyChanges',
        content: [{ type: 'text', text: '{"applied":true}' }],
        isError: false,
        timestamp: 2,
      },
    ];
    const appendMessage = vi.fn();
    const prompt = vi.fn(async () => {});
    piMock.session = {
      abort: vi.fn(async () => {}),
      agent: { state: { messages: stateMessages } },
      compact: vi.fn(async () => {}),
      dispose: vi.fn(),
      getSessionStats: () => ({
        tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
      messages: stateMessages,
      prompt,
      sessionManager: { appendMessage },
      steer: vi.fn(async () => {}),
      subscribe: vi.fn(() => () => {}),
    } as unknown as AgentSession;

    const session = await createPiSession({
      sessionId: 'session-already-answered',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
      skills: [],
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
      pendingHostToolCalls: [
        { toolCallId: 'tool-1', toolName: 'applyChanges', input: '{}' },
      ],
    });

    const control = await session.doContinueTurn({
      tools: [{ name: 'applyChanges' }] satisfies HarnessV1ToolSpec[],
      emit: vi.fn(),
    });
    await control.done;

    // Injecting again would put a duplicate result into the provider payload.
    expect(appendMessage).not.toHaveBeenCalled();
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(prompt).toHaveBeenCalledWith('');
  });

  it('closes the open step before terminal finish when prompt resolves without turn_end', async () => {
    const listeners: Array<(raw: unknown) => void> = [];
    const prompt = vi.fn(async () => {
      // The model stream is cut mid-message: text streams, then Pi resolves
      // `prompt()` with no message_end/turn_end and no terminal error event.
      for (const event of [
        { type: 'turn_start' },
        { type: 'message_start', message: { role: 'assistant', content: [] } },
        {
          type: 'message_update',
          assistantMessageEvent: { type: 'text_delta', delta: 'Hi there' },
        },
      ]) {
        for (const listener of [...listeners]) listener(event);
      }
    });
    piMock.session = {
      abort: vi.fn(async () => {}),
      compact: vi.fn(async () => {}),
      dispose: vi.fn(),
      getSessionStats: () => ({
        tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
      prompt,
      steer: vi.fn(async () => {}),
      subscribe: vi.fn((listener: (raw: unknown) => void) => {
        listeners.push(listener);
        return () => {
          const index = listeners.indexOf(listener);
          if (index >= 0) listeners.splice(index, 1);
        };
      }),
    } as unknown as AgentSession;

    const session = await createPiSession({
      sessionId: 'session-cut-mid-message',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
      skills: [],
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: false,
    });
    const emit = vi.fn();
    const control = await session.doPromptTurn({
      prompt: 'go',
      tools: [],
      emit,
    });
    await control.done;

    const types = emit.mock.calls.map(
      call => (call[0] as { type: string }).type,
    );
    // The harness contract: `finish-step` must precede the terminal `finish`,
    // and the open text block must be closed first.
    expect(types).not.toContain('error');
    const textEndIndex = types.indexOf('text-end');
    const finishStepIndex = types.indexOf('finish-step');
    const finishIndex = types.indexOf('finish');
    expect(textEndIndex).toBeGreaterThan(-1);
    expect(finishStepIndex).toBeGreaterThan(textEndIndex);
    expect(finishIndex).toBeGreaterThan(finishStepIndex);
  });

  it('uses agentDir for auth, models, and settings when provided', async () => {
    vi.mocked(ModelRuntime.create).mockClear();
    vi.mocked(SettingsManager.inMemory).mockClear();
    vi.mocked(SettingsManager.create).mockClear();

    const sandboxSession = createSandboxSession();
    await createPiSession({
      sessionId: 'session-agentdir',
      sandboxSession,
      sessionWorkDir: '/sandbox/work',
      skills: [],
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: false,
      agentDir: '/custom/.pi/agent',
    });

    expect(ModelRuntime.create).toHaveBeenCalledWith({
      authPath: '/custom/.pi/agent/auth.json',
      modelsPath: '/custom/.pi/agent/models.json',
      allowModelNetwork: false,
    });
    expect(SettingsManager.create).toHaveBeenCalledTimes(1);
    expect(SettingsManager.inMemory).not.toHaveBeenCalled();
  });

  it('falls back to temp dir and inMemory settings when agentDir is omitted', async () => {
    vi.mocked(ModelRuntime.create).mockClear();
    vi.mocked(SettingsManager.inMemory).mockClear();
    vi.mocked(SettingsManager.create).mockClear();

    const sandboxSession = createSandboxSession();
    await createPiSession({
      sessionId: 'session-no-agentdir',
      sandboxSession,
      sessionWorkDir: '/sandbox/work',
      skills: [],
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: false,
    });

    expect(SettingsManager.inMemory).toHaveBeenCalledTimes(1);
    expect(SettingsManager.create).not.toHaveBeenCalled();
  });
});

function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function createSandboxSession(): HarnessV1NetworkSandboxSession {
  const sandbox = {
    defaultWorkingDirectory: '/sandbox',
    destroy: vi.fn(async () => {}),
    getPortUrl: vi.fn(),
    readBinaryFile: vi.fn(async () => undefined),
    restricted: vi.fn(() => sandbox),
    run: vi.fn(async () => ({ stdout: '', stderr: '', exitCode: 0 })),
    stop: vi.fn(async () => {}),
    writeBinaryFile: vi.fn(async () => {}),
    writeTextFile: vi.fn(async () => {}),
  };
  return sandbox as unknown as HarnessV1NetworkSandboxSession;
}
