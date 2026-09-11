import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  type AgentSession,
  type ExtensionAPI,
  type ExtensionFactory,
  type ToolDefinition,
  ModelRuntime,
  SessionManager,
  SettingsManager,
} from '@earendil-works/pi-coding-agent';
import type {
  HarnessV1NetworkSandboxSession,
  HarnessV1SandboxProvider,
  HarnessV1ToolSpec,
} from '@ai-sdk/harness';
import {
  HarnessAgent,
  type HarnessAgentContinueTurnState,
} from '@ai-sdk/harness/agent';
import { tool } from '@ai-sdk/provider-utils';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { z } from 'zod/v4';
import { createPi } from './pi-harness';
import { sequentialToolCallsExtension } from './pi-sequential-tools';
import { createPiSession } from './pi-session';

type FakePiTool = Pick<ToolDefinition, 'name' | 'execute'>;
type FakeExtensionsResult = {
  readonly errors: unknown[];
  readonly extensions: unknown[];
  readonly runtime: object;
};
type FakeAgentsFile = { path: string; content: string };
type FakeAgentsFilesOverride = (base: { agentsFiles: FakeAgentsFile[] }) => {
  agentsFiles: FakeAgentsFile[];
};
type FakeSystemPromptOverride = (
  base: string | undefined,
) => string | undefined;

type ResourceLoaderOptions = {
  readonly agentsFilesOverride?: FakeAgentsFilesOverride;
  readonly appendSystemPromptOverride?: (base: string[]) => string[];
  readonly extensionFactories?: Array<ExtensionFactory>;
  readonly systemPromptOverride?: FakeSystemPromptOverride;
  readonly extensionsOverride?: (
    base: FakeExtensionsResult,
  ) => FakeExtensionsResult;
  readonly noExtensions?: boolean;
  readonly noPromptTemplates?: boolean;
  readonly noThemes?: boolean;
};

const piMock = vi.hoisted(() => {
  const extensionHandlers = new Map<string, Array<() => unknown>>();
  return {
    agentSessionExtensionResults: [] as FakeExtensionsResult[],
    createAgentSession: vi.fn(),
    customTools: [] as FakePiTool[],
    appendSystemPrompts: [] as string[][],
    extensionApi: {
      on: vi.fn((eventType: string, handler: () => unknown) => {
        const handlers = extensionHandlers.get(eventType) ?? [];
        handlers.push(handler);
        extensionHandlers.set(eventType, handlers);
      }),
    } as unknown as ExtensionAPI,
    extensionFactoryInputs: [] as Array<{
      readonly reference: Array<ExtensionFactory>;
      readonly snapshot: Array<ExtensionFactory>;
    }>,
    extensionHandlers,
    resourceLoaderReloadCount: 0,
    resourceLoaderOptions: [] as ResourceLoaderOptions[],
    session: undefined as AgentSession | undefined,
    sessionManagerOpen: vi.fn(),
  };
});

const mcpAdapterMock = vi.hoisted(() => {
  const mcpExtensionFactory = vi.fn();
  return {
    createMcpAdapter: vi.fn(() => mcpExtensionFactory),
    mcpExtensionFactory,
  };
});

vi.mock('pi-mcp-adapter', () => ({
  createMcpAdapter: mcpAdapterMock.createMcpAdapter,
}));

vi.mock('@earendil-works/pi-coding-agent', () => {
  return {
    createAgentSession: piMock.createAgentSession,
    DefaultResourceLoader: class {
      private extensionsResult: FakeExtensionsResult = {
        errors: [],
        extensions: [],
        runtime: {},
      };

      constructor(private readonly options: ResourceLoaderOptions) {
        piMock.resourceLoaderOptions.push(options);
        const extensionFactories = options.extensionFactories ?? [];
        piMock.extensionFactoryInputs.push({
          reference: extensionFactories,
          snapshot: [...extensionFactories],
        });
      }

      async reload() {
        piMock.resourceLoaderReloadCount += 1;
        piMock.appendSystemPrompts.push(
          this.options.appendSystemPromptOverride?.([]) ?? [],
        );
        for (const factory of this.options.extensionFactories ?? []) {
          await factory(piMock.extensionApi);
        }
        const base = { errors: [], extensions: [], runtime: {} };
        this.extensionsResult = this.options.extensionsOverride?.(base) ?? base;
      }

      getExtensions() {
        return this.extensionsResult;
      }
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
      open: piMock.sessionManagerOpen,
    },
    SettingsManager: {
      inMemory: vi.fn(() => ({})),
      create: vi.fn(() => ({})),
    },
  };
});

describe('createPiSession', () => {
  beforeEach(() => {
    piMock.agentSessionExtensionResults = [];
    piMock.customTools = [];
    piMock.appendSystemPrompts = [];
    piMock.extensionFactoryInputs = [];
    piMock.extensionHandlers.clear();
    piMock.resourceLoaderReloadCount = 0;
    piMock.resourceLoaderOptions = [];
    piMock.session = undefined;
    mcpAdapterMock.createMcpAdapter.mockClear();
    mcpAdapterMock.mcpExtensionFactory.mockClear();
    piMock.createAgentSession.mockReset();
    piMock.createAgentSession.mockImplementation(async options => {
      piMock.agentSessionExtensionResults.push(
        options.resourceLoader.getExtensions(),
      );
      piMock.customTools = options.customTools;
      return { session: piMock.session };
    });
    piMock.sessionManagerOpen.mockReset();
    piMock.sessionManagerOpen.mockImplementation(() => ({
      getSessionFile: () => undefined,
    }));
    // Restated per test: one case below points it at a real host journal so the
    // persist path runs, and that must not leak into the others.
    vi.mocked(SessionManager.create).mockReturnValue({
      getSessionFile: () => undefined,
    } as unknown as ReturnType<typeof SessionManager.create>);
  });

  it('rejects structured output turns', async () => {
    const session = await createPiSession({
      sessionId: 'session-structured-output',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: false,
    });

    try {
      await expect(
        session.doPromptTurn({
          skills: [],
          tools: [],
          prompt: 'Generate an object.',
          responseFormat: {
            type: 'json',
            schema: { type: 'object' },
          },
          emit: () => {},
        }),
      ).rejects.toMatchObject({
        name: 'AI_HarnessCapabilityUnsupportedError',
        harnessId: 'pi',
      });
    } finally {
      await session.doDestroy();
    }
  });

  it('loads a caller-supplied inline extension factory through createPi', async () => {
    const factory = vi.fn((piApi: ExtensionAPI) => {
      expect(piApi).toBe(piMock.extensionApi);
    });

    const session = await createPi({
      extensionFactories: [factory],
    }).doStart({
      sessionId: 'session-inline-extension',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
    });

    try {
      expect(factory).toHaveBeenCalledOnce();
    } finally {
      await session.doDestroy();
    }
  });

  it('lets inline extensions read a host-backed session workspace', async () => {
    const sessionWorkDir = mkdtempSync(
      path.join(tmpdir(), 'pi-host-workspace-'),
    );
    const projectFile = path.join(sessionWorkDir, 'graphify-out', 'graph.json');
    mkdirSync(path.dirname(projectFile), { recursive: true });
    writeFileSync(projectFile, '{}');
    const factory = vi.fn(() => {
      expect(existsSync(projectFile)).toBe(true);
      expect(readFileSync(projectFile, 'utf8')).toBe('{}');
    });

    try {
      const session = await createPi({
        extensionFactories: [factory],
      }).doStart({
        sessionId: 'session-host-workspace-extension',
        sandboxSession: createSandboxSession({ hostFilesystem: true }),
        sessionWorkDir,
      });

      try {
        expect(factory).toHaveBeenCalledOnce();
      } finally {
        await session.doDestroy();
      }
    } finally {
      rmSync(sessionWorkDir, { recursive: true, force: true });
    }
  });

  it('keeps tools registered by inline extensions enabled', async () => {
    const factory = vi.fn();
    piMock.session = createFakePiSession().session;

    const session = await createPi({
      extensionFactories: [factory],
    }).doStart({
      sessionId: 'session-extension-tools',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
    });

    try {
      const control = await session.doPromptTurn({
        skills: [],
        prompt: 'Use an extension tool.',
        tools: [],
        emit: vi.fn(),
      });
      await control.done;

      const createOptions = piMock.createAgentSession.mock.lastCall?.[0];
      expect(createOptions).toMatchObject({ noTools: 'builtin' });
      expect(createOptions).not.toHaveProperty('tools');
    } finally {
      await session.doDestroy();
    }
  });

  it('preserves caller order and passes a fresh mutable factory array', async () => {
    const callOrder: string[] = [];
    const firstFactory: ExtensionFactory = () => {
      callOrder.push('first');
    };
    const secondFactory: ExtensionFactory = () => {
      callOrder.push('second');
    };
    const extensionFactories = [firstFactory, secondFactory] as const;

    const session = await createPi({ extensionFactories }).doStart({
      sessionId: 'session-ordered-extensions',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
    });

    try {
      const extensionFactoryInput = piMock.extensionFactoryInputs.at(-1);
      expect(callOrder).toEqual(['first', 'second']);
      expect(extensionFactoryInput?.snapshot).toEqual(extensionFactories);
      expect(extensionFactoryInput?.reference).not.toBe(extensionFactories);
    } finally {
      await session.doDestroy();
    }
  });

  it('loads sequential tool handling before caller extensions', async () => {
    const factory: ExtensionFactory = vi.fn();
    const session = await createPi({
      extensionFactories: [factory],
      sequentialToolCalls: true,
    }).doStart({
      sessionId: 'session-sequential-tools',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
    });

    try {
      expect(piMock.extensionFactoryInputs.at(-1)?.snapshot).toEqual([
        sequentialToolCallsExtension,
        factory,
      ]);
    } finally {
      await session.doDestroy();
    }
  });

  it('does not reload inline extensions between turns', async () => {
    const observedEvents: string[] = [];
    const factory = vi.fn((piApi: ExtensionAPI) => {
      piApi.on('agent_start', () => {
        observedEvents.push('agent_start');
      });
    });
    piMock.session = {
      abort: vi.fn(async () => {}),
      compact: vi.fn(async () => {}),
      dispose: vi.fn(),
      getSessionStats: () => ({
        tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
      prompt: vi.fn(async () => {
        for (const handler of piMock.extensionHandlers.get('agent_start') ??
          []) {
          await handler();
        }
      }),
      steer: vi.fn(async () => {}),
      subscribe: vi.fn(() => () => {}),
    } as unknown as AgentSession;

    const session = await createPi({
      extensionFactories: [factory],
    }).doStart({
      sessionId: 'session-multiple-extension-turns',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
    });

    try {
      for (const prompt of ['first turn', 'second turn']) {
        const control = await session.doPromptTurn({
          skills: [],
          prompt,
          tools: [],
          emit: vi.fn(),
        });
        await control.done;
      }

      expect(factory).toHaveBeenCalledOnce();
      expect(observedEvents).toEqual(['agent_start', 'agent_start']);
      expect(piMock.resourceLoaderReloadCount).toBe(3);
      expect(piMock.agentSessionExtensionResults).toHaveLength(1);
    } finally {
      await session.doDestroy();
    }
  });

  it('materializes skills under sandbox HOME on prompt turn', async () => {
    piMock.session = {
      abort: vi.fn(async () => {}),
      compact: vi.fn(async () => {}),
      dispose: vi.fn(),
      getSessionStats: () => ({
        tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
      prompt: vi.fn(async () => {}),
      steer: vi.fn(async () => {}),
      subscribe: vi.fn(() => () => {}),
    } as unknown as AgentSession;
    const sandboxSession = createSandboxSession();
    const session = await createPiSession({
      sessionId: 'session-skills',
      sandboxSession,
      sessionWorkDir: '/sandbox/work',
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: false,
    });

    try {
      const control = await session.doPromptTurn({
        skills: [
          {
            name: 'demo-skill',
            description: 'Demo skill description',
            content: 'Skill instructions content',
          },
        ],
        prompt: 'test prompt',
        tools: [],
        emit: vi.fn(),
      });
      await control.done;

      expect(sandboxSession.run).toHaveBeenCalledWith(
        expect.objectContaining({
          command: "mkdir -p '/sandbox/home/.agents/skills'",
        }),
      );
      expect(sandboxSession.writeTextFile).toHaveBeenCalledWith(
        expect.objectContaining({
          path: '/sandbox/home/.agents/skills/demo-skill/SKILL.md',
          content:
            '---\nname: demo-skill\ndescription: Demo skill description\n---\n\nSkill instructions content',
        }),
      );
    } finally {
      await session.doDestroy();
    }
  });

  it('steers the active Pi session', async () => {
    let finishPrompt!: () => void;
    const promptDone = new Promise<void>(resolve => {
      finishPrompt = resolve;
    });
    const steer = vi.fn(async () => {});
    piMock.session = {
      abort: vi.fn(async () => {}),
      compact: vi.fn(async () => {}),
      dispose: vi.fn(),
      getSessionStats: () => ({
        tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
      prompt: vi.fn(async () => promptDone),
      steer,
      subscribe: vi.fn(() => () => {}),
    } as unknown as AgentSession;
    const session = await createPiSession({
      sessionId: 'session-steering',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: false,
    });

    try {
      const control = await session.doPromptTurn({
        skills: [],
        prompt: 'Weather in Paris?',
        tools: [],
        emit: vi.fn(),
      });
      await control.submitUserMessage?.('Actually, Paris, Texas.');

      expect(steer).toHaveBeenCalledExactlyOnceWith('Actually, Paris, Texas.');
      finishPrompt();
      await control.done;
    } finally {
      await session.doDestroy();
    }
  });

  it('reloads inline extensions when the Pi session is rebuilt', async () => {
    const factory = vi.fn();
    piMock.session = {
      abort: vi.fn(async () => {}),
      compact: vi.fn(async () => {}),
      dispose: vi.fn(),
      getSessionStats: () => ({
        tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
      prompt: vi.fn(async () => {}),
      steer: vi.fn(async () => {}),
      subscribe: vi.fn(() => () => {}),
    } as unknown as AgentSession;

    const session = await createPi({
      extensionFactories: [factory],
    }).doStart({
      sessionId: 'session-rebuilt-extension-runtime',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
    });

    try {
      const firstControl = await session.doPromptTurn({
        skills: [],
        prompt: 'first turn',
        tools: [],
        emit: vi.fn(),
      });
      await firstControl.done;
      const secondControl = await session.doPromptTurn({
        skills: [],
        prompt: 'second turn',
        tools: [{ name: 'new-tool' }],
        emit: vi.fn(),
      });
      await secondControl.done;

      expect(factory).toHaveBeenCalledTimes(2);
      expect(piMock.resourceLoaderReloadCount).toBe(3);
      expect(piMock.agentSessionExtensionResults).toHaveLength(2);
      expect(piMock.agentSessionExtensionResults[0]).not.toBe(
        piMock.agentSessionExtensionResults[1],
      );
    } finally {
      await session.doDestroy();
    }
  });

  it('keeps filesystem extensions and other resources disabled by default', async () => {
    const session = await createPi().doStart({
      sessionId: 'session-no-extensions',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
    });

    try {
      expect(piMock.resourceLoaderOptions.at(-1)).toMatchObject({
        extensionFactories: [],
        noExtensions: true,
        noPromptTemplates: true,
        noThemes: true,
      });
      expect(
        piMock.resourceLoaderOptions.at(-1)?.extensionsOverride,
      ).toBeUndefined();
    } finally {
      await session.doDestroy();
    }
  });

  it('registers configured MCP servers as direct Pi extension tools', async () => {
    const bindExtensions = vi.fn(async () => {});
    const dispose = vi.fn();
    const reload = vi.fn(async () => {});
    piMock.session = {
      bindExtensions,
      dispose,
      getSessionStats: () => ({
        tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
      prompt: vi.fn(async () => {}),
      reload,
      subscribe: vi.fn(() => () => {}),
    } as unknown as AgentSession;

    const sandboxSession = createSandboxSession();
    const session = await createPiSession({
      sessionId: 'session-mcp',
      sandboxSession,
      sessionWorkDir: '/sandbox/work',
      settings: {
        mcpServers: {
          memory: { command: 'memory-mcp', args: [] },
        },
      },
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: false,
    });
    const control = await session.doPromptTurn({
      skills: [],
      prompt: 'Use an MCP tool.',
      tools: [],
      emit: vi.fn(),
    });
    await control.done;
    await session.doDestroy();

    expect(mcpAdapterMock.createMcpAdapter).toHaveBeenCalledWith({
      config: {
        mcpServers: {
          memory: { command: 'memory-mcp', args: [] },
        },
        settings: {
          directTools: true,
          toolPrefix: 'mcp',
          disableProxyTool: true,
        },
      },
    });
    expect(piMock.resourceLoaderOptions.at(-1)?.extensionFactories).toEqual([
      mcpAdapterMock.mcpExtensionFactory,
    ]);
    expect(piMock.createAgentSession).toHaveBeenCalledWith(
      expect.objectContaining({ noTools: 'builtin' }),
    );
    expect(bindExtensions).toHaveBeenCalledWith({ mode: 'print' });
    expect(reload).toHaveBeenCalledTimes(1);
    expect(dispose).toHaveBeenCalledTimes(1);
  });

  it('loads configured MCP servers alongside caller-supplied extension factories', async () => {
    const factory = vi.fn();

    const session = await createPi({
      extensionFactories: [factory],
      mcpServers: { memory: { command: 'memory-mcp', args: [] } },
    }).doStart({
      sessionId: 'session-mcp-and-extensions',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
    });

    try {
      expect(piMock.extensionFactoryInputs.at(-1)?.snapshot).toEqual([
        factory,
        mcpAdapterMock.mcpExtensionFactory,
      ]);
      expect(factory).toHaveBeenCalledOnce();
      expect(mcpAdapterMock.mcpExtensionFactory).toHaveBeenCalledOnce();
    } finally {
      await session.doDestroy();
    }
  });

  it('rejects unsafe resume session filenames before sandbox restore', async () => {
    const sandboxSession = createSandboxSession();

    await expect(
      createPiSession({
        sessionId: 'session-unsafe',
        sandboxSession,
        sessionWorkDir: '/sandbox/work',
        settings: {},
        clientApp: 'ai-sdk/harness-pi/0.0.0-test',
        isResume: true,
        resumeSessionFileName: '../session.jsonl',
      }),
    ).rejects.toThrow('Invalid Pi session file name');

    expect(sandboxSession.readBinaryFile).not.toHaveBeenCalled();
  });

  it('appends instructions without changing the user prompt or reloading MCP extensions', async () => {
    const prompt = vi.fn(async () => {});
    piMock.session = {
      abort: vi.fn(async () => {}),
      bindExtensions: vi.fn(async () => {}),
      compact: vi.fn(async () => {}),
      dispose: vi.fn(),
      getActiveToolNames: vi.fn(() => []),
      getSessionStats: () => ({
        tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
      prompt,
      reload: vi.fn(async () => {}),
      setActiveToolsByName: vi.fn(),
      steer: vi.fn(async () => {}),
      subscribe: vi.fn(() => () => {}),
    } as unknown as AgentSession;

    const session = await createPiSession({
      sessionId: 'session-instructions',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
      settings: {
        mcpServers: { memory: { command: 'memory-mcp', args: [] } },
      },
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: false,
    });
    const control = await session.doPromptTurn({
      skills: [],
      tools: [],
      prompt: 'do the thing',
      instructions: 'Use turbo build.',
      emit: vi.fn(),
    });
    await control.done;

    expect(piMock.appendSystemPrompts.at(-1)).toEqual(['Use turbo build.']);
    expect(prompt).toHaveBeenCalledWith('do the thing');
    expect(mcpAdapterMock.mcpExtensionFactory).toHaveBeenCalledOnce();
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
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: false,
    });
    const toolSpecs: HarnessV1ToolSpec[] = [{ name: 'weather' }];
    const control = await session.doPromptTurn({
      skills: [],
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
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
    });
    const resumedControl = await resumedSession.doContinueTurn({
      skills: [],
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

  it('holds a cross-process rerun until dangling host tool results arrive, then injects them into the journal', async () => {
    const { session: fakePiSession, prompt } = createFakePiSession();
    piMock.session = fakePiSession;
    const { journal, appendedMessages } = createJournal([
      userMessage('ask the user something'),
      assistantMessageWithToolCalls([{ id: 'tool-1', name: 'askUser' }]),
    ]);
    piMock.sessionManagerOpen.mockImplementation(() => journal);

    const sandboxSession = createSandboxSession({
      sessionFileContent: 'pi-journal',
    });
    const session = await createPiSession({
      sessionId: 'session-cross-process',
      sandboxSession,
      sessionWorkDir: '/sandbox/work',
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
      resumeSessionFileName: 'pi-session.jsonl',
    });

    const emit = vi.fn();
    const control = await session.doContinueTurn({
      skills: [],
      tools: [{ name: 'askUser' }],
      instructions: 'Return the tool result exactly.',
      emit,
    });

    // The rerun must wait for the framework to re-deliver the result of the
    // journal-pending tool call; starting it eagerly would resolve the call
    // as a synthetic empty result and drop the submission below.
    expect(prompt).not.toHaveBeenCalled();

    await control.submitToolResult({
      toolCallId: 'tool-1',
      output: { selection: 'Option A' },
    });
    await control.done;

    expect(appendedMessages).toEqual([
      {
        role: 'toolResult',
        toolCallId: 'tool-1',
        toolName: 'askUser',
        content: [{ type: 'text', text: '{"selection":"Option A"}' }],
        isError: false,
        timestamp: expect.any(Number),
      },
    ]);
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(prompt).toHaveBeenCalledWith('');
    expect(emit.mock.calls.slice(0, 2)).toEqual([
      [{ type: 'stream-start' }],
      [
        {
          type: 'finish-step',
          finishReason: { unified: 'tool-calls', raw: undefined },
          usage: {
            inputTokens: {
              total: 0,
              noCache: 0,
              cacheRead: 0,
              cacheWrite: 0,
            },
            outputTokens: {
              total: 0,
              text: 0,
              reasoning: 0,
            },
          },
          harnessMetadata: { pi: { inferredStep: true } },
        },
      ],
    ]);
    expect(piMock.appendSystemPrompts.at(-1)).toEqual([
      'Return the tool result exactly.',
    ]);
    expect(piMock.sessionManagerOpen).toHaveBeenCalledTimes(1);
  });

  it('preserves the error flag when injecting a cross-process tool result', async () => {
    const { session: fakePiSession } = createFakePiSession();
    piMock.session = fakePiSession;
    const { journal, appendedMessages } = createJournal([
      userMessage('call the tool'),
      assistantMessageWithToolCalls([{ id: 'tool-1', name: 'askUser' }]),
    ]);
    piMock.sessionManagerOpen.mockImplementation(() => journal);

    const session = await createPiSession({
      sessionId: 'session-cross-process-error',
      sandboxSession: createSandboxSession({
        sessionFileContent: 'pi-journal',
      }),
      sessionWorkDir: '/sandbox/work',
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
      resumeSessionFileName: 'pi-session.jsonl',
    });
    const control = await session.doContinueTurn({
      skills: [],
      tools: [{ name: 'askUser' }],
      emit: vi.fn(),
    });

    await control.submitToolResult({
      toolCallId: 'tool-1',
      output: { error: 'answer unavailable' },
      isError: true,
    });
    await control.done;

    expect(appendedMessages).toMatchInlineSnapshot(
      [
        {
          timestamp: expect.any(Number),
        },
      ],
      `
      [
        {
          "content": [
            {
              "text": "{\"error\":\"answer unavailable\"}",
              "type": "text",
            },
          ],
          "isError": true,
          "role": "toolResult",
          "timestamp": Any<Number>,
          "toolCallId": "tool-1",
          "toolName": "askUser",
        },
      ]
    `,
    );
  });

  it('reruns immediately on cross-process resume when the journal has no dangling host tool calls', async () => {
    const { session: fakePiSession, prompt } = createFakePiSession();
    piMock.session = fakePiSession;
    const { journal } = createJournal([
      userMessage('ask the user something'),
      assistantMessageWithToolCalls([{ id: 'tool-1', name: 'askUser' }]),
      {
        role: 'toolResult',
        toolCallId: 'tool-1',
        toolName: 'askUser',
        content: [{ type: 'text', text: 'already answered' }],
        isError: false,
        timestamp: 0,
      },
    ]);
    piMock.sessionManagerOpen.mockImplementation(() => journal);

    const sandboxSession = createSandboxSession({
      sessionFileContent: 'pi-journal',
    });
    const session = await createPiSession({
      sessionId: 'session-cross-process-resolved',
      sandboxSession,
      sessionWorkDir: '/sandbox/work',
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
      resumeSessionFileName: 'pi-session.jsonl',
    });

    const control = await session.doContinueTurn({
      skills: [],
      tools: [{ name: 'askUser' }],
      emit: vi.fn(),
    });
    await control.done;

    expect(prompt).toHaveBeenCalledTimes(1);
    expect(prompt).toHaveBeenCalledWith('');
    expect(journal.appendMessage).not.toHaveBeenCalled();
  });

  it('does not re-await results already delivered by a previous continuation of the same session', async () => {
    const { session: fakePiSession, prompt } = createFakePiSession();
    piMock.session = fakePiSession;
    const { journal, appendedMessages } = createJournal([
      userMessage('ask the user two things'),
      assistantMessageWithToolCalls([
        { id: 'tool-1', name: 'askUser' },
        { id: 'tool-2', name: 'askUser' },
      ]),
    ]);
    piMock.sessionManagerOpen.mockImplementation(() => journal);

    const sandboxSession = createSandboxSession({
      sessionFileContent: 'pi-journal',
    });
    const session = await createPiSession({
      sessionId: 'session-cross-process-reentrant',
      sandboxSession,
      sessionWorkDir: '/sandbox/work',
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
      resumeSessionFileName: 'pi-session.jsonl',
    });

    // First continuation delivers only tool-1's result, then ends (e.g. it
    // paused again awaiting a tool-result continuation for tool-2).
    const firstControl = await session.doContinueTurn({
      skills: [],
      tools: [{ name: 'askUser' }],
      emit: vi.fn(),
    });
    await firstControl.submitToolResult({
      toolCallId: 'tool-1',
      output: 'first answer',
    });
    expect(prompt).not.toHaveBeenCalled();

    // Second continuation: the framework will only re-deliver tool-2 — it
    // marked tool-1 settled. The new barrier must not wait on tool-1 again.
    const secondControl = await session.doContinueTurn({
      skills: [],
      tools: [{ name: 'askUser' }],
      emit: vi.fn(),
    });
    // Installing the new barrier settles the abandoned first turn cleanly.
    await expect(firstControl.done).resolves.toBeUndefined();
    expect(prompt).not.toHaveBeenCalled();

    await secondControl.submitToolResult({
      toolCallId: 'tool-2',
      output: 'second answer',
    });
    await secondControl.done;

    expect(appendedMessages).toEqual([
      {
        role: 'toolResult',
        toolCallId: 'tool-1',
        toolName: 'askUser',
        content: [{ type: 'text', text: 'first answer' }],
        isError: false,
        timestamp: expect.any(Number),
      },
      {
        role: 'toolResult',
        toolCallId: 'tool-2',
        toolName: 'askUser',
        content: [{ type: 'text', text: 'second answer' }],
        isError: false,
        timestamp: expect.any(Number),
      },
    ]);
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(prompt).toHaveBeenCalledWith('');
  });

  it('flushes results delivered before a suspend into the journal so a later resume sees them', async () => {
    const { session: fakePiSession, prompt } = createFakePiSession();
    piMock.session = fakePiSession;
    const { journal, appendedMessages } = createJournal([
      userMessage('ask the user two things'),
      assistantMessageWithToolCalls([
        { id: 'tool-1', name: 'askUser' },
        { id: 'tool-2', name: 'askUser' },
      ]),
    ]);
    piMock.sessionManagerOpen.mockImplementation(() => journal);

    const sandboxSession = createSandboxSession({
      sessionFileContent: 'pi-journal',
    });
    const session = await createPiSession({
      sessionId: 'session-cross-process-partial',
      sandboxSession,
      sessionWorkDir: '/sandbox/work',
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
      resumeSessionFileName: 'pi-session.jsonl',
    });

    const control = await session.doContinueTurn({
      skills: [],
      tools: [{ name: 'askUser' }],
      emit: vi.fn(),
    });
    await control.submitToolResult({
      toolCallId: 'tool-1',
      output: 'first answer',
    });

    // Still awaiting tool-2; the rerun has not started.
    expect(prompt).not.toHaveBeenCalled();

    await expect(session.doSuspendTurn()).resolves.toEqual({
      type: 'continue-turn',
      harnessId: 'pi',
      specificationVersion: 'harness-v1',
      data: { sessionFileName: 'pi-session.jsonl' },
    });
    await control.done;

    expect(appendedMessages).toEqual([
      {
        role: 'toolResult',
        toolCallId: 'tool-1',
        toolName: 'askUser',
        content: [{ type: 'text', text: 'first answer' }],
        isError: false,
        timestamp: expect.any(Number),
      },
    ]);
    expect(prompt).not.toHaveBeenCalled();
    // The updated journal is pushed back into the sandbox for the next resume.
    expect(sandboxSession.writeBinaryFile).toHaveBeenCalled();
  });

  it('cancels a deferred rerun initialization before suspending', async () => {
    const setup = await startDeferredCrossProcessRerun({
      sessionId: 'session-cross-process-suspend-during-startup',
    });

    const suspension = setup.session.doSuspendTurn();
    setup.agentSessionCreation.resolve({ session: setup.fakePiSession });

    await expect(suspension).resolves.toEqual({
      type: 'continue-turn',
      harnessId: 'pi',
      specificationVersion: 'harness-v1',
      data: { sessionFileName: 'pi-session.jsonl' },
    });
    await expect(setup.control.done).resolves.toBeUndefined();
    expect(setup.prompt).not.toHaveBeenCalled();
    expect(setup.dispose).toHaveBeenCalledOnce();
  });

  it('waits for deferred rerun initialization to cancel before stopping', async () => {
    const setup = await startDeferredCrossProcessRerun({
      sessionId: 'session-cross-process-stop-during-startup',
    });
    const done = expect(setup.control.done).rejects.toMatchObject({
      name: 'AbortError',
    });

    const stopping = setup.session.doStop();
    setup.agentSessionCreation.resolve({ session: setup.fakePiSession });

    await expect(stopping).resolves.toEqual({
      type: 'resume-session',
      harnessId: 'pi',
      specificationVersion: 'harness-v1',
      data: { sessionFileName: 'pi-session.jsonl' },
    });
    await done;
    expect(setup.prompt).not.toHaveBeenCalled();
    expect(setup.dispose).toHaveBeenCalledOnce();
  });

  it('rejects an abort received during deferred rerun initialization', async () => {
    const abortController = new AbortController();
    const setup = await startDeferredCrossProcessRerun({
      sessionId: 'session-cross-process-abort-during-startup',
      abortSignal: abortController.signal,
    });
    const done = expect(setup.control.done).rejects.toThrow(
      'cancel deferred rerun',
    );

    abortController.abort(new Error('cancel deferred rerun'));
    setup.agentSessionCreation.resolve({ session: setup.fakePiSession });

    await done;
    expect(setup.prompt).not.toHaveBeenCalled();
    await setup.session.doDestroy();
    expect(setup.dispose).toHaveBeenCalledOnce();
  });

  it('attaches another continuation to deferred rerun initialization', async () => {
    const setup = await startDeferredCrossProcessRerun({
      sessionId: 'session-cross-process-attach-during-startup',
    });
    const attachedEmit = vi.fn();

    const attachedControl = await setup.session.doContinueTurn({
      skills: [],
      tools: [{ name: 'askUser' }],
      emit: attachedEmit,
    });
    setup.agentSessionCreation.resolve({ session: setup.fakePiSession });

    await Promise.all([setup.control.done, attachedControl.done]);
    expect(piMock.createAgentSession).toHaveBeenCalledOnce();
    expect(setup.prompt).toHaveBeenCalledOnce();
    expect(setup.emit).not.toHaveBeenCalledWith({ type: 'stream-start' });
    expect(attachedEmit).toHaveBeenCalledWith({ type: 'stream-start' });
    await setup.session.doDestroy();
  });

  it('replays an approved host tool through HarnessAgent before rerunning Pi', async () => {
    const execute = vi.fn(async () => ({ selection: 'Option A' }));
    const askUser = tool({
      description: 'Ask the user to select an option.',
      inputSchema: z.object({ question: z.string() }),
      execute,
    });
    const { session: fakePiSession, prompt } = createFakePiSession({
      promptEvents: [
        { type: 'turn_start' },
        {
          type: 'message_start',
          message: { role: 'assistant', content: [] },
        },
        {
          type: 'message_update',
          assistantMessageEvent: {
            type: 'text_delta',
            delta: 'Option A',
          },
        },
        {
          type: 'message_end',
          message: {
            role: 'assistant',
            content: [{ type: 'text', text: 'Option A' }],
            stopReason: 'stop',
          },
        },
        {
          type: 'turn_end',
          message: {
            role: 'assistant',
            content: [{ type: 'text', text: 'Option A' }],
            stopReason: 'stop',
          },
        },
      ],
    });
    piMock.session = fakePiSession;
    const { journal, appendedMessages } = createJournal([
      userMessage('ask the user something'),
      assistantMessageWithToolCalls([{ id: 'tool-1', name: 'askUser' }]),
    ]);
    piMock.sessionManagerOpen.mockImplementation(() => journal);
    const sandboxSession = createSandboxSession({
      sessionFileContent: 'pi-journal',
    });
    const continueFrom: HarnessAgentContinueTurnState = {
      type: 'continue-turn',
      harnessId: 'pi',
      specificationVersion: 'harness-v1',
      data: { sessionFileName: 'pi-session.jsonl' },
      pendingToolApprovals: [
        {
          approvalId: 'approval-1',
          toolCallId: 'tool-1',
          toolName: 'askUser',
          input: '{"question":"Choose an option"}',
          kind: 'custom',
          providerExecuted: false,
        },
      ],
    };
    const agent = new HarnessAgent({
      harness: createPi(),
      sandbox: createSandboxProvider({ sandboxSession }),
      tools: { askUser },
      toolApproval: { askUser: 'user-approval' },
    });
    const session = await agent.createSession({
      sessionId: 'session-harness-agent-cross-process',
      continueFrom,
    });

    try {
      const result = await agent.continueGenerate({
        session,
        toolApprovalContinuations: [
          {
            type: 'tool-approval-response',
            approvalId: 'approval-1',
            approved: true,
          },
        ],
      });

      expect(execute).toHaveBeenCalledOnce();
      expect(result.text).toBe('Option A');
      expect(appendedMessages).toEqual([
        {
          role: 'toolResult',
          toolCallId: 'tool-1',
          toolName: 'askUser',
          content: [{ type: 'text', text: '{"selection":"Option A"}' }],
          isError: false,
          timestamp: expect.any(Number),
        },
      ]);
      expect(prompt).toHaveBeenCalledWith('');
    } finally {
      await session.destroy();
    }
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
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
    });
    const control = await session.doPromptTurn({
      skills: [],
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
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
    });
    const control = await session.doPromptTurn({
      skills: [],
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

  it('dispatches a validated host call only from execute and receives its outcome', async () => {
    const listeners = new Set<(event: unknown) => void>();
    const publish = (event: unknown) => {
      for (const listener of listeners) listener(event);
    };
    const ready = createDeferred<void>();
    const emit = vi.fn();
    let result: unknown;
    piMock.session = {
      abort: vi.fn(async () => {}),
      dispose: vi.fn(),
      getSessionStats: () => ({
        tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
      subscribe: (listener: (event: unknown) => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
      prompt: async () => {
        await ready.promise;
        publish({
          type: 'tool_execution_start',
          toolCallId: 'valid',
          toolName: 'applyChanges',
          args: { revision: 1 },
        });
        expect(
          emit.mock.calls.filter(([part]) => part.type === 'tool-call'),
        ).toHaveLength(0);
        const tool = piMock.customTools.find(
          tool => tool.name === 'applyChanges',
        )!;
        const pending = tool.execute(
          'valid',
          { revision: 1 },
          undefined,
          undefined,
          {} as Parameters<typeof tool.execute>[4],
        );
        expect(
          emit.mock.calls.filter(([part]) => part.type === 'tool-call'),
        ).toHaveLength(1);
        await control.submitToolResult({
          toolCallId: 'valid',
          output: { revision: 2 },
        });
        result = await pending;
        publish({
          type: 'tool_execution_end',
          toolCallId: 'valid',
          toolName: 'applyChanges',
          result,
        });
      },
    } as unknown as AgentSession;
    const session = await createPiSession({
      sessionId: 'validated-write',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
      settings: {},
      clientApp: 'test',
      isResume: false,
    });
    const control = await session.doPromptTurn({
      skills: [],
      prompt: 'go',
      tools: [{ name: 'applyChanges' }],
      emit,
    });
    ready.resolve();
    await control.done;
    expect(result).toMatchObject({
      content: [{ type: 'text', text: '{"revision":2}' }],
    });
    await session.doDestroy();
  });

  it('does not dispatch a host write rejected by native argument validation', async () => {
    const listeners = new Set<(event: unknown) => void>();
    const publish = (event: unknown) => {
      for (const listener of listeners) listener(event);
    };
    piMock.session = {
      abort: vi.fn(async () => {}),
      dispose: vi.fn(),
      getSessionStats: () => ({
        tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
      subscribe: (listener: (event: unknown) => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
      prompt: async () => {
        publish({
          type: 'tool_execution_start',
          toolCallId: 'invalid-write',
          toolName: 'applyChanges',
          args: {},
        });
        publish({
          type: 'tool_execution_end',
          toolCallId: 'invalid-write',
          toolName: 'applyChanges',
          isError: true,
          result: {
            content: [{ type: 'text', text: 'Missing required revision' }],
          },
        });
      },
    } as unknown as AgentSession;
    const session = await createPiSession({
      sessionId: 'rejected-write',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
      settings: {},
      clientApp: 'test',
      isResume: false,
    });
    const emit = vi.fn();
    const control = await session.doPromptTurn({
      skills: [],
      prompt: 'go',
      tools: [{ name: 'applyChanges' }],
      emit,
    });
    await control.done;
    const calls = emit.mock.calls
      .map(([part]) => part)
      .filter(part => part.type === 'tool-call');
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({
      toolCallId: 'invalid-write',
      providerExecuted: true,
    });
    expect(emit).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'tool-result', isError: true }),
    );
    await session.doDestroy();
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
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: false,
    });
    const emit = vi.fn();
    const control = await session.doPromptTurn({
      skills: [],
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

  it('reports per-turn token deltas instead of cumulative session totals', async () => {
    const tokens = { input: 100, output: 40, cacheRead: 20, cacheWrite: 10 };
    const prompt = vi.fn(async () => {
      if (prompt.mock.calls.length === 1) {
        Object.assign(tokens, {
          input: 130,
          output: 65,
          cacheRead: 28,
          cacheWrite: 14,
        });
      } else {
        Object.assign(tokens, {
          input: 139,
          output: 72,
          cacheRead: 31,
          cacheWrite: 16,
        });
      }
    });
    piMock.session = {
      abort: vi.fn(async () => {}),
      compact: vi.fn(async () => {}),
      dispose: vi.fn(),
      getSessionStats: () => ({ tokens: { ...tokens } }),
      prompt,
      steer: vi.fn(async () => {}),
      subscribe: vi.fn(() => () => {}),
    } as unknown as AgentSession;

    const session = await createPiSession({
      sessionId: 'session-usage-deltas',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: false,
    });
    const firstEmit = vi.fn();
    const first = await session.doPromptTurn({
      skills: [],
      prompt: 'first',
      tools: [],
      emit: firstEmit,
    });
    await first.done;
    const secondEmit = vi.fn();
    const second = await session.doPromptTurn({
      skills: [],
      prompt: 'second',
      tools: [],
      emit: secondEmit,
    });
    await second.done;

    const finishUsage = (emit: ReturnType<typeof vi.fn>) =>
      emit.mock.calls.map(call => call[0]).find(part => part.type === 'finish')
        .totalUsage;
    expect(finishUsage(firstEmit)).toMatchObject({
      inputTokens: { total: 42, noCache: 30, cacheRead: 8, cacheWrite: 4 },
      outputTokens: { total: 25 },
    });
    expect(finishUsage(secondEmit)).toMatchObject({
      inputTokens: { total: 14, noCache: 9, cacheRead: 3, cacheWrite: 2 },
      outputTokens: { total: 7 },
    });
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
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: false,
    });

    expect(SettingsManager.inMemory).toHaveBeenCalledTimes(1);
    expect(SettingsManager.create).not.toHaveBeenCalled();
  });

  describe('projectInstructions', () => {
    async function loadAgentsFiles(
      projectInstructions: string | undefined,
      discovered: FakeAgentsFile[] = [],
    ): Promise<FakeAgentsFile[]> {
      await createPiSession({
        sessionId: 'session-project-instructions',
        sandboxSession: createSandboxSession(),
        sessionWorkDir: '/sandbox/work',
        settings: {},
        clientApp: 'ai-sdk/harness-pi/0.0.0-test',
        isResume: false,
        ...(projectInstructions === undefined ? {} : { projectInstructions }),
      });

      const override = piMock.resourceLoaderOptions.at(-1)?.agentsFilesOverride;
      if (!override) throw new Error('Expected an agentsFiles override.');
      return override({ agentsFiles: discovered }).agentsFiles;
    }

    it('surfaces instructions as a workspace-root AGENTS.md context file', async () => {
      expect(await loadAgentsFiles('Be concise.')).toEqual([
        { path: '/sandbox/work/AGENTS.md', content: 'Be concise.' },
      ]);
    });

    it('leaves discovered context files untouched when unset', async () => {
      const discovered = [{ path: '/sandbox/AGENTS.md', content: 'ancestor' }];
      expect(await loadAgentsFiles(undefined, discovered)).toEqual(discovered);
    });

    it('ranks after ancestor context files', async () => {
      const discovered = [{ path: '/sandbox/AGENTS.md', content: 'ancestor' }];
      expect(await loadAgentsFiles('Be concise.', discovered)).toEqual([
        ...discovered,
        { path: '/sandbox/work/AGENTS.md', content: 'Be concise.' },
      ]);
    });

    it('replaces rather than duplicates a workspace AGENTS.md', async () => {
      expect(
        await loadAgentsFiles('Be concise.', [
          { path: '/sandbox/work/AGENTS.md', content: 'from the workspace' },
        ]),
      ).toEqual([{ path: '/sandbox/work/AGENTS.md', content: 'Be concise.' }]);
    });

    it('ignores whitespace-only instructions', async () => {
      expect(await loadAgentsFiles('   \n  ')).toEqual([]);
    });
  });

  describe('systemPrompt', () => {
    async function loadSystemPrompt(
      systemPrompt: string | undefined,
      piBase:
        | string
        | undefined = 'You are an expert coding assistant operating inside pi.',
    ): Promise<string | undefined> {
      await createPiSession({
        sessionId: 'session-system-prompt',
        sandboxSession: createSandboxSession(),
        sessionWorkDir: '/sandbox/work',
        settings: {},
        clientApp: 'ai-sdk/harness-pi/0.0.0-test',
        isResume: false,
        ...(systemPrompt === undefined ? {} : { systemPrompt }),
      });

      const override =
        piMock.resourceLoaderOptions.at(-1)?.systemPromptOverride;
      return override ? override(piBase) : piBase;
    }

    it('replaces Pi\u2019s base system prompt', async () => {
      expect(await loadSystemPrompt('You are the Ninox agent.')).toBe(
        'You are the Ninox agent.',
      );
    });

    it('leaves Pi\u2019s own prompt in place when unset', async () => {
      expect(await loadSystemPrompt(undefined)).toBe(
        'You are an expert coding assistant operating inside pi.',
      );
    });

    // A blank setting is a configuration mistake, not a request for an empty
    // system prompt: falling back to Pi's prompt keeps the agent usable.
    it('ignores a whitespace-only prompt', async () => {
      expect(await loadSystemPrompt('   \n ')).toBe(
        'You are an expert coding assistant operating inside pi.',
      );
    });
  });

  /*
   * Skills are materialized into sandbox HOME on EVERY turn, resumed sessions
   * included, and hosts depend on that rather than on the sandbox snapshot carrying them:
   * a snapshot holding every skill file costs tens of kilobytes on every turn's write and
   * read for bytes that are overwritten on restore anyway, so a host that persists its own
   * sandbox is right to strip them before storing.
   *
   * The failure mode if this regresses is silent — a resumed agent comes back with a
   * `<available_skills>` section pointing at files that no longer exist, and simply stops
   * being able to read any of them — so the guarantee is pinned here rather than left to
   * the reader of an ungated call site.
   */
  describe('skills on a resumed session', () => {
    async function skillWritesFor(isResume: boolean): Promise<string[]> {
      piMock.session = createFakePiSession().session;
      const sandbox = createSandboxSession(
        isResume ? { sessionFileContent: 'pi-journal' } : undefined,
      );
      const session = await createPiSession({
        sessionId: `session-skills-${String(isResume)}`,
        sandboxSession: sandbox,
        sessionWorkDir: '/sandbox/work',
        settings: {},
        clientApp: 'ai-sdk/harness-pi/0.0.0-test',
        isResume,
        ...(isResume ? { resumeSessionFileName: 'session.jsonl' } : {}),
      });

      try {
        const control = await session.doPromptTurn({
          skills: [
            { name: 'demo', description: 'Demo skill.', content: 'Body.' },
          ],
          prompt: 'go',
          tools: [],
          emit: vi.fn(),
        });
        await control.done;
      } finally {
        await session.doDestroy();
      }

      return vi
        .mocked(sandbox.writeTextFile)
        .mock.calls.map(([input]) => (input as { path: string }).path);
    }

    it('materializes skills on a fresh session', async () => {
      expect(await skillWritesFor(false)).toContain(
        '/sandbox/home/.agents/skills/demo/SKILL.md',
      );
    });

    it('materializes skills again on a resumed session', async () => {
      expect(await skillWritesFor(true)).toContain(
        '/sandbox/home/.agents/skills/demo/SKILL.md',
      );
    });
  });

  /*
   * A parked session outlives the sandbox it was built with: the host stops the
   * harness session at the pause (stopping that sandbox) and the resume hands it
   * a new one restored from the snapshot. Reattaching used to keep the original
   * handle, so every later `persistSessionFile` wrote the journal into a stopped
   * sandbox while the running one — the one whose snapshot the host stores — kept
   * whatever it was restored with. The durable journal then never advanced, and
   * the first cold resume replayed the model from a context missing every turn
   * since, which the user saw as the assistant re-asking questions it had already
   * had answered.
   */
  it('persists the journal into the sandbox a reattached session was resumed on', async () => {
    const sessionId = 'session-rebind';
    const sessionFileName = '2026-01-01T00-00-00-000Z_rebind.jsonl';
    const hostSessionDir = path.join(
      tmpdir(),
      'ai-sdk-harness',
      'pi',
      sessionId,
      'sessions',
    );
    await mkdir(hostSessionDir, { recursive: true });
    const hostJournalPath = path.join(hostSessionDir, sessionFileName);
    await writeFile(hostJournalPath, 'parked\n');
    vi.mocked(SessionManager.create).mockReturnValue({
      getSessionFile: () => hostJournalPath,
    } as unknown as ReturnType<typeof SessionManager.create>);

    const toolStarted = createDeferred<void>();
    const prompt = vi.fn(async () => {
      const tool = piMock.customTools.find(tool => tool.name === 'weather');
      if (!tool) throw new Error('Expected weather tool.');
      const pending = tool.execute(
        'tool-1',
        {},
        undefined,
        undefined,
        undefined as never,
      );
      toolStarted.resolve();
      await pending;
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

    const parkedSandbox = createSandboxSession();
    const session = await createPiSession({
      sessionId,
      sandboxSession: parkedSandbox,
      sessionWorkDir: '/sandbox/work',
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: false,
    });
    const control = await session.doPromptTurn({
      skills: [],
      prompt: 'go',
      tools: [{ name: 'weather' }] satisfies HarnessV1ToolSpec[],
      emit: vi.fn(),
    });
    await toolStarted.promise;
    await session.doSuspendTurn();

    // The host stops the parked slice's sandbox; the resume gets a new one.
    const resumedSandbox = createSandboxSession();
    const resumed = await createPiSession({
      sessionId,
      sandboxSession: resumedSandbox,
      sessionWorkDir: '/sandbox/work',
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
    });

    // Pi appended this slice's entries to its host journal; only a persist into
    // the RUNNING sandbox puts them where the next cold resume will find them.
    await writeFile(hostJournalPath, 'parked\ncontinued\n');
    const resumedControl = await resumed.doContinueTurn({
      skills: [],
      tools: [{ name: 'weather' }] satisfies HarnessV1ToolSpec[],
      emit: vi.fn(),
    });
    await resumedControl.submitToolResult({
      toolCallId: 'tool-1',
      output: { weather: 'sunny' },
    });
    await resumedControl.done;
    await control.done;
    await resumed.doStop();

    expect(journalWritesTo(resumedSandbox)).toEqual(['parked\ncontinued\n']);
    expect(journalWritesTo(parkedSandbox)).not.toContain('parked\ncontinued\n');
  });

  it('rebuilds instead of reattaching when the slice resolved a different work dir', async () => {
    const toolStarted = createDeferred<void>();
    const prompt = vi.fn(async () => {
      const tool = piMock.customTools.find(tool => tool.name === 'weather');
      if (!tool) throw new Error('Expected weather tool.');
      const pending = tool.execute(
        'tool-1',
        {},
        undefined,
        undefined,
        undefined as never,
      );
      toolStarted.resolve();
      await pending;
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
      sessionId: 'session-workdir-moved',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/work',
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: false,
    });
    const control = await session.doPromptTurn({
      skills: [],
      prompt: 'go',
      tools: [{ name: 'weather' }] satisfies HarnessV1ToolSpec[],
      emit: vi.fn(),
    });
    await toolStarted.promise;
    await session.doDetach();

    const resumed = await createPiSession({
      sessionId: 'session-workdir-moved',
      sandboxSession: createSandboxSession(),
      sessionWorkDir: '/sandbox/other',
      settings: {},
      clientApp: 'ai-sdk/harness-pi/0.0.0-test',
      isResume: true,
    });

    // A reattach would have handed back the parked session verbatim, still
    // addressing `/sandbox/work` — paths the new sandbox does not have.
    expect(resumed).not.toBe(session);
    expect(
      (piMock.session as unknown as { dispose: ReturnType<typeof vi.fn> })
        .dispose,
    ).toHaveBeenCalled();
    await control.done;
    await resumed.doDestroy();
  });
});

/** The journal bodies a session wrote into `sandbox`, in call order. */
function journalWritesTo(sandbox: HarnessV1NetworkSandboxSession): string[] {
  return vi
    .mocked(sandbox.writeBinaryFile)
    .mock.calls.map(([input]) => input as { content: Uint8Array; path: string })
    .filter(({ path: written }) => written.endsWith('.jsonl'))
    .map(({ content }) => Buffer.from(content).toString('utf8'));
}

function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function createFakePiSession({
  promptEvents = [],
}: {
  promptEvents?: unknown[];
} = {}) {
  const subscribers = new Set<(event: unknown) => void>();
  const prompt = vi.fn(async (_text: string) => {
    for (const event of promptEvents) {
      for (const subscriber of subscribers) {
        subscriber(event);
      }
    }
  });
  const abort = vi.fn(async () => {});
  const dispose = vi.fn();
  const session = {
    abort,
    compact: vi.fn(async () => {}),
    dispose,
    getSessionStats: () => ({
      tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    }),
    prompt,
    steer: vi.fn(async () => {}),
    subscribe: vi.fn((subscriber: (event: unknown) => void) => {
      subscribers.add(subscriber);
      return () => subscribers.delete(subscriber);
    }),
  } as unknown as AgentSession;
  return { session, prompt, abort, dispose };
}

async function startDeferredCrossProcessRerun({
  sessionId,
  abortSignal,
}: {
  sessionId: string;
  abortSignal?: AbortSignal;
}) {
  const { session: fakePiSession, prompt, dispose } = createFakePiSession();
  const { journal } = createJournal([
    userMessage('ask the user something'),
    assistantMessageWithToolCalls([{ id: 'tool-1', name: 'askUser' }]),
  ]);
  piMock.sessionManagerOpen.mockImplementation(() => journal);
  const agentSessionCreation = createDeferred<{ session: AgentSession }>();
  piMock.createAgentSession.mockImplementation(
    async () => agentSessionCreation.promise,
  );
  const session = await createPiSession({
    sessionId,
    sandboxSession: createSandboxSession({
      sessionFileContent: 'pi-journal',
    }),
    sessionWorkDir: '/sandbox/work',
    settings: {},
    clientApp: 'ai-sdk/harness-pi/0.0.0-test',
    isResume: true,
    resumeSessionFileName: 'pi-session.jsonl',
  });
  const emit = vi.fn();
  const control = await session.doContinueTurn({
    skills: [],
    tools: [{ name: 'askUser' }],
    emit,
    ...(abortSignal ? { abortSignal } : {}),
  });
  await control.submitToolResult({
    toolCallId: 'tool-1',
    output: 'answer',
  });
  await vi.waitFor(() => {
    expect(piMock.createAgentSession).toHaveBeenCalledOnce();
  });

  return {
    session,
    control,
    emit,
    agentSessionCreation,
    fakePiSession,
    prompt,
    dispose,
  };
}

/**
 * Fake `SessionManager` handle over a fixed restored journal. Records
 * messages appended by the adapter (the injected host tool results).
 */
function createJournal(messages: unknown[]) {
  const appendedMessages: unknown[] = [];
  const journal = {
    getSessionFile: () => 'pi-session.jsonl',
    buildSessionContext: () => ({ messages }),
    appendMessage: vi.fn((message: unknown) => {
      appendedMessages.push(message);
      return 'appended-entry';
    }),
  };
  return { journal, appendedMessages };
}

function userMessage(text: string) {
  return {
    role: 'user',
    content: [{ type: 'text', text }],
    timestamp: 0,
  };
}

function assistantMessageWithToolCalls(
  toolCalls: Array<{ id: string; name: string }>,
) {
  return {
    role: 'assistant',
    content: toolCalls.map(toolCall => ({
      type: 'toolCall',
      id: toolCall.id,
      name: toolCall.name,
      arguments: {},
    })),
    stopReason: 'toolUse',
    timestamp: 0,
  };
}

function createSandboxSession(options?: {
  /** When set, resume-path `readBinaryFile` finds a persisted session file. */
  sessionFileContent?: string;
  /** When set, file reads use the host filesystem at the same paths. */
  hostFilesystem?: boolean;
}): HarnessV1NetworkSandboxSession {
  const textFiles = new Map<string, string>();
  const sandbox = {
    id: 'sandbox',
    defaultWorkingDirectory: '/sandbox',
    ports: [],
    destroy: vi.fn(async () => {}),
    getPortEndpoint: vi.fn(),
    getPortUrl: vi.fn(),
    readBinaryFile: vi.fn(async ({ path }: { path: string }) => {
      if (options?.hostFilesystem) {
        try {
          return new Uint8Array(readFileSync(path));
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
            return undefined;
          }
          throw error;
        }
      }
      return options?.sessionFileContent != null
        ? new TextEncoder().encode(options.sessionFileContent)
        : undefined;
    }),
    readTextFile: vi.fn(async ({ path }: { path: string }) =>
      textFiles.get(path),
    ),
    restricted: vi.fn(() => sandbox),
    run: vi.fn(async ({ command }: { command: string }) => {
      const manifestMove = command.match(/^mv -f '([^']+)' '([^']+)'$/);
      if (manifestMove != null) {
        const content = textFiles.get(manifestMove[1]!);
        if (content != null) textFiles.set(manifestMove[2]!, content);
      }
      return {
        stdout: command === 'printf "%s" "$HOME"' ? '/sandbox/home' : '',
        stderr: '',
        exitCode: 0,
      };
    }),
    stop: vi.fn(async () => {}),
    writeBinaryFile: vi.fn(async () => {}),
    writeTextFile: vi.fn(
      async ({ path, content }: { path: string; content: string }) => {
        textFiles.set(path, content);
      },
    ),
  };
  return sandbox as unknown as HarnessV1NetworkSandboxSession;
}

function createSandboxProvider({
  sandboxSession,
}: {
  sandboxSession: HarnessV1NetworkSandboxSession;
}): HarnessV1SandboxProvider {
  return {
    specificationVersion: 'harness-sandbox-v1',
    providerId: 'test-sandbox',
    createSession: async () => sandboxSession,
    resumeSession: async () => sandboxSession,
  };
}
