import {
  createAgentSession,
  DefaultResourceLoader,
  defineTool,
  ModelRegistry,
  ModelRuntime,
  SessionManager,
  SettingsManager,
  type AgentSession,
  type AgentToolResult,
  type ExtensionFactory,
  type Skill,
  type ToolDefinition,
} from '@earendil-works/pi-coding-agent';
import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { Type } from 'typebox';
import {
  type HarnessV1BuiltinToolFiltering,
  type HarnessV1ContinueTurnOptions,
  type HarnessV1ContinueTurnState,
  type HarnessV1PromptControl,
  type HarnessV1PromptTurnOptions,
  type HarnessV1NetworkSandboxSession,
  type HarnessV1PermissionMode,
  type HarnessV1ResumeSessionState,
  type HarnessV1Session,
  type HarnessV1Skill,
  type HarnessV1StreamPart,
  type HarnessV1ToolSpec,
} from '@ai-sdk/harness';
import { resolveSandboxHomeDir } from '@ai-sdk/harness/utils';
import {
  registerPiProviders,
  resolvePiEnv,
  type PiAuthOptions,
} from './pi-auth';
import {
  createPiTerminalError,
  getPiTerminalError,
  parseNativeEvent,
} from './pi-events';
import { createPiModelResolver } from './pi-model-resolver';
import { createPiPathMapper } from './pi-paths';
import { createPiRemoteOps, type PiRemoteOps } from './pi-remote-ops';
import { writePiSkills } from './pi-skills';
import {
  persistSessionFileToSandbox,
  pullSessionFileFromSandbox,
  safePiSessionFileName,
} from './pi-resume-state';
import {
  consumePendingPiStepUsage,
  createPiTranslatorState,
  finishPiApprovalStep,
  flushOpenStep,
  piTurnFinishReason,
  translatePiEvent,
  type PiTranslatorState,
} from './pi-translate';
import { toolSpecToTypeBoxParameters } from './pi-typebox-adapter';
import {
  extractUserText,
  safePiMetadataSegment,
  serializeToolOutput,
} from './pi-utils';
import { PiWorkspaceVfs } from './pi-workspace-vfs';
import { syncHostWorkspaceFromSandbox } from './pi-workspace-mirror';

const HARNESS_ID = 'pi';

/*
 * pi-mcp-adapter publishes TypeScript source as its package entry point. A
 * non-literal specifier keeps the repository type-check focused on this
 * package's compatibility boundary instead of compiling dependency internals.
 */
const PI_MCP_ADAPTER_PACKAGE: string = 'pi-mcp-adapter';

type PiMcpAdapterModule = {
  createMcpAdapter(options: {
    config: {
      mcpServers: Record<string, unknown>;
      settings: {
        directTools: boolean;
        toolPrefix: string;
        disableProxyTool: boolean;
      };
    };
  }): ExtensionFactory;
};

/*
 * Pi runs in this Node process, not behind an attachable in-sandbox bridge.
 * During a tool approval pause the Pi turn is still alive and blocked on the
 * custom tool promise, so detach must park that live session for the next
 * same-process resume instead of stopping it and resolving the promise as an
 * error. Cross-process resume still falls back to the persisted session file.
 */
const parkedPiSessions = new Map<string, ParkedPiSession>();

/**
 * A parked session together with the rebind the NEXT slice must perform before
 * using it.
 *
 * Parking outlives the sandbox it was created with: the host stops the harness
 * session at the pause, which stops that sandbox, and the resume builds a new
 * one from the persisted snapshot. A reattach that kept the original sandbox
 * handle therefore kept writing into a stopped sandbox — most damagingly in
 * `persistSessionFile`, so the journal the next cold resume reads never
 * advanced past the last slice that went through the rebuild path. The live
 * conversation looked healthy (Pi's context is in memory) right up to the first
 * event that dropped the parked session, at which point the model resumed from
 * a journal missing every turn since, and redid work it had already done.
 */
interface ParkedPiSession {
  /**
   * Points the session's host-side handles at `input`'s sandbox. Returns false
   * when this session cannot serve that slice, in which case the caller must
   * discard it and rebuild from the journal rather than reattach.
   */
  readonly rebindToSlice: (input: CreatePiSessionInput) => Promise<boolean>;
  readonly session: HarnessV1Session;
}

/**
 * Whether a discovered resource path belongs to a specific directory.
 */
function isWithinDirectory(parent: string, child: string): boolean {
  const rel = path.relative(parent, child);
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel));
}

/**
 * Whether a discovered resource path belongs to the session workspace — either
 * the sandbox-side working directory the model sees (`sessionWorkDir`) or its
 * host-side mirror (`hostWorkDir`).
 */
function isWithinWorkspace(
  candidate: string,
  sessionWorkDir: string,
  hostWorkDir: string,
): boolean {
  return (
    isWithinDirectory(sessionWorkDir, candidate) ||
    isWithinDirectory(hostWorkDir, candidate)
  );
}

function createHarnessPiSkills({
  skills,
  sandboxSkillRootDir,
}: {
  skills: ReadonlyArray<HarnessV1Skill>;
  sandboxSkillRootDir: string;
}): Skill[] {
  return skills.map(skill => {
    const name = safePiMetadataSegment(skill.name, 'skill');
    const baseDir = path.posix.join(sandboxSkillRootDir, name);
    const filePath = path.posix.join(baseDir, 'SKILL.md');
    return {
      name: skill.name,
      description: skill.description,
      filePath,
      baseDir,
      sourceInfo: {
        path: filePath,
        source: 'harness',
        scope: 'temporary',
        origin: 'top-level',
        baseDir,
      },
      disableModelInvocation: false,
    };
  });
}

const PI_NATIVE_BUILTIN_NAMES = [
  'read',
  'write',
  'edit',
  'bash',
  'grep',
  'find',
  'ls',
] as const;

const NATIVE_TO_COMMON: Readonly<Record<string, string>> = {
  find: 'glob',
};

const PUBLIC_TO_NATIVE: Readonly<
  Record<string, (typeof PI_NATIVE_BUILTIN_NAMES)[number]>
> = {
  read: 'read',
  write: 'write',
  edit: 'edit',
  bash: 'bash',
  grep: 'grep',
  glob: 'find',
  ls: 'ls',
};

const PI_NATIVE_TOOL_KINDS: Readonly<
  Record<(typeof PI_NATIVE_BUILTIN_NAMES)[number], 'readonly' | 'edit' | 'bash'>
> = {
  read: 'readonly',
  write: 'edit',
  edit: 'edit',
  bash: 'bash',
  grep: 'readonly',
  find: 'readonly',
  ls: 'readonly',
};

function resolveActivePiBuiltinNames(
  toolFiltering: HarnessV1BuiltinToolFiltering | undefined,
): ReadonlyArray<(typeof PI_NATIVE_BUILTIN_NAMES)[number]> {
  if (toolFiltering == null) return PI_NATIVE_BUILTIN_NAMES;
  if (toolFiltering.mode === 'allow') {
    return toolFiltering.toolNames
      .map(name => PUBLIC_TO_NATIVE[name])
      .filter(
        (name): name is (typeof PI_NATIVE_BUILTIN_NAMES)[number] =>
          name != null,
      );
  }
  return PI_NATIVE_BUILTIN_NAMES.filter(
    native =>
      !toolFiltering.toolNames.includes(NATIVE_TO_COMMON[native] ?? native),
  );
}

export type PiThinkingLevel =
  | 'off'
  | 'minimal'
  | 'low'
  | 'medium'
  | 'high'
  | 'xhigh';

export interface PiSessionSettings {
  readonly auth?: PiAuthOptions;
  readonly model?: string;
  readonly thinkingLevel?: PiThinkingLevel;
  readonly mcpServers?: Record<string, unknown>;
  readonly extensionFactories?: ReadonlyArray<ExtensionFactory>;
}

/**
 * A custom (host-executed) tool call that was still waiting on host input —
 * an approval decision or the executed result — when its turn was parked.
 *
 * On a cross-process continuation the parked live session (and with it the
 * call's promise waiter) is gone. Pi never re-drives a recorded call after a
 * journal rebuild, so without these records `doContinueTurn` would rerun the
 * model on a context whose dangling call gets patched with a synthetic
 * `"No result provided"` error (see pi-ai's `transformMessages`) — the model
 * then retries a call whose effect may already be applied. `doContinueTurn`
 * instead holds the turn until the host submits each recorded call's outcome,
 * writes those outcomes into the session context, and only then re-drives.
 */
export interface PiPendingHostToolCall {
  readonly toolCallId: string;
  readonly toolName: string;
  /** JSON-encoded tool input, as recorded by the framework's pending records. */
  readonly input: string;
}

export interface CreatePiSessionInput {
  readonly sessionId: string;
  readonly sandboxSession: HarnessV1NetworkSandboxSession;
  readonly sessionWorkDir: string;
  readonly skills: ReadonlyArray<HarnessV1Skill>;
  readonly settings: PiSessionSettings;
  readonly clientApp: string;
  readonly isResume: boolean;
  readonly permissionMode?: HarnessV1PermissionMode;
  readonly builtinToolFiltering?: HarnessV1BuiltinToolFiltering;
  readonly resumeSessionFileName?: string;
  /**
   * Host tool calls the parked turn was waiting on when its lifecycle state
   * was captured. Only meaningful together with `isResume`; consumed by
   * `doContinueTurn`.
   */
  readonly pendingHostToolCalls?: ReadonlyArray<PiPendingHostToolCall>;
  readonly abortSignal?: AbortSignal;
  /**
   * Directory holding Pi's global agent config (auth.json, models.json,
   * settings.json). When omitted, a per-session temp dir is used (the
   * harness cannot reuse existing CLI logins). Pass the user's agent dir
   * (e.g. `~/.pi/agent/`) to reuse their CLI auth and model settings.
   */
  readonly agentDir?: string;
  /**
   * Operating guidance surfaced as a synthetic workspace-root `AGENTS.md`
   * context file, so it lands in the system prompt rather than in the first
   * user message. See `PiHarnessSettings.projectInstructions`.
   */
  readonly projectInstructions?: string;
  /**
   * Replaces Pi's own base system prompt. See
   * `PiHarnessSettings.systemPrompt`.
   */
  readonly systemPrompt?: string;
}

interface PendingToolResult {
  resolve: (value: unknown) => void;
}

/** The verdict a host returns for one approval request. */
interface ToolApprovalDecision {
  approved: boolean;
  reason?: string;
}

interface PendingToolApproval {
  resolve: (value: ToolApprovalDecision) => void;
}

interface ActivePiTurn {
  readonly token: object;
  readonly done: Promise<void>;
}

interface PiTokenSnapshot {
  readonly input: number;
  readonly output: number;
  readonly cacheRead: number;
  readonly cacheWrite: number;
}

function tokenSnapshot(session: AgentSession): PiTokenSnapshot {
  return { ...session.getSessionStats().tokens };
}

function usageSince(after: PiTokenSnapshot, before: PiTokenSnapshot) {
  const delta = (current: number, baseline: number) =>
    Math.max(0, current - baseline);
  const noCache = delta(after.input, before.input);
  const cacheRead = delta(after.cacheRead, before.cacheRead);
  const cacheWrite = delta(after.cacheWrite, before.cacheWrite);
  return {
    inputTokens: {
      total: noCache + cacheRead + cacheWrite,
      noCache,
      cacheRead,
      cacheWrite,
    },
    outputTokens: {
      total: delta(after.output, before.output),
      text: undefined,
      reasoning: undefined,
    },
  };
}

/*
 * Pi's message types, derived from the `AgentSession` surface instead of
 * imported: `@earendil-works/pi-agent-core` / `pi-ai` are transitive
 * dependencies this package deliberately does not declare.
 */
type PiAgentMessage = AgentSession['messages'][number];
type PiToolResultMessage = Extract<PiAgentMessage, { role: 'toolResult' }>;

export async function createPiSession(
  input: CreatePiSessionInput,
): Promise<HarnessV1Session> {
  if (input.isResume) {
    const parked = parkedPiSessions.get(input.sessionId);
    if (parked) {
      parkedPiSessions.delete(input.sessionId);
      // A rebind that throws leaves this slice's sandbox unusable, which the
      // fresh build below runs into as well and reports properly — so it is
      // the same outcome as a refusal, and handling it as one keeps the parked
      // session from being stranded live but unreachable.
      const reattached = await parked.rebindToSlice(input).catch(() => false);
      if (reattached) {
        return {
          ...parked.session,
          isResume: true,
        };
      }
      // Reattaching would leave the session addressing paths that do not exist
      // in this slice's sandbox. Drop it and fall through to a fresh build off
      // the persisted journal — lossy, but coherent. Destroying before the
      // `mkdir`s below matters: both sessions derive `hostRoot` from the same
      // session id, and `doDestroy` removes that tree.
      await parked.session.doDestroy();
    }
  }

  // Host-side mirror layout under tmpdir. Replace path-separator characters
  // that would otherwise turn a session id like `2026-05-29T17:54:27` into a
  // sub-directory tree on disk.
  const safeSessionId = input.sessionId.replace(/[\\/: ]/g, '-');
  const hostRoot = path.join(tmpdir(), 'ai-sdk-harness', 'pi', safeSessionId);
  const hostWorkDir = path.join(hostRoot, 'workspace');
  const hostAgentDir = path.join(hostRoot, 'agent');
  const hostSessionDir = path.join(hostRoot, 'sessions');

  // Pi runs in this host process but must behave as though it lives in the
  // sandbox workspace: its working directory is the real `sessionWorkDir`
  // (where `setup()` clones and where the sandbox-backed tools operate), so the
  // paths Pi advertises to the model — most notably the "Current working
  // directory" line in its system prompt — resolve inside the sandbox. The
  // workspace VFS maps that sandbox path to the host-side mirror so Pi's own
  // `fs`-based resource loading (`.pi/`, `AGENTS.md`) still works on the host.
  // `sessionWorkDir` is a sandbox path (e.g. `/vercel/sandbox/...`) that does
  // not exist on the host, so it is a safe, collision-free VFS mount point.
  const sessionWorkDir = input.sessionWorkDir;

  await mkdir(hostWorkDir, { recursive: true });
  await mkdir(hostAgentDir, { recursive: true });
  await mkdir(hostSessionDir, { recursive: true });

  /*
   * Rebound by `rebindToSlice` when this session is reattached after a pause:
   * the sandbox it was created with is stopped by then. Everything that reaches
   * the sandbox at turn time must read THIS binding rather than capture it, so
   * a reattached session writes into the sandbox that is actually running — and,
   * critically, the one whose snapshot becomes the next persisted boundary.
   */
  let sandbox = input.sandboxSession.restricted();
  const permissionMode = input.permissionMode ?? 'allow-all';
  let sandboxSkillRootDir: string | undefined;
  let harnessSkills: Skill[] = [];

  /*
   * Materialise harness-provided skills into sandbox HOME, not the workspace.
   *
   * Runs on every start AND on every reattach: a reattach binds this session to
   * a sandbox restored from a snapshot, and hosts deliberately exclude the skill
   * tree from what they persist precisely because it is rewritten here. Skipping
   * it would leave the model with skill paths its own resource list advertises
   * but the running sandbox does not have.
   */
  async function writeSkillsToSandbox(
    abortSignal: AbortSignal | undefined,
  ): Promise<void> {
    if (input.skills.length === 0) {
      return;
    }
    const sandboxHomeDir = await resolveSandboxHomeDir({
      sandbox,
      ...(abortSignal ? { abortSignal } : {}),
    });
    sandboxSkillRootDir = path.posix.join(sandboxHomeDir, '.agents', 'skills');
    harnessSkills = createHarnessPiSkills({
      skills: input.skills,
      sandboxSkillRootDir,
    });
    await writePiSkills({
      sandbox,
      sandboxHomeDir,
      skills: input.skills,
      ...(abortSignal ? { abortSignal } : {}),
    });
  }

  await writeSkillsToSandbox(input.abortSignal);

  // On resume: pull the Pi session file out of the sandbox into the fresh
  // host mirror so SessionManager.open can read it.
  let resumeSessionFilePath: string | undefined;
  if (input.isResume && input.resumeSessionFileName) {
    const resumeSessionFileName = safePiSessionFileName(
      input.resumeSessionFileName,
    );
    resumeSessionFilePath = await pullSessionFileFromSandbox({
      sandbox,
      sessionWorkDir: input.sessionWorkDir,
      hostSessionDir,
      sessionFileName: resumeSessionFileName,
      ...(input.abortSignal ? { abortSignal: input.abortSignal } : {}),
    });
  }

  // Snapshot sandbox state into the host mirror BEFORE the VFS goes live so
  // Pi sees the workspace as soon as it boots.
  await syncHostWorkspaceFromSandbox({
    sandbox,
    sandboxWorkDir: input.sessionWorkDir,
    hostWorkDir,
  });

  // Mount only the workspace: the model's view of the workspace lives at
  // `sessionWorkDir` and is backed by `hostWorkDir`. The agent and session
  // directories stay on the real host filesystem (below) — they are host-only
  // Pi state (auth, model registry, session journal) that must never surface
  // in the sandbox or the workspace mirror.
  const workspaceVfs = new PiWorkspaceVfs();
  workspaceVfs.mount(hostWorkDir, sessionWorkDir);

  const paths = createPiPathMapper({
    hostWorkDir,
    sandboxWorkDir: sessionWorkDir,
    readableRoots: sandboxSkillRootDir
      ? [{ sandboxDir: sandboxSkillRootDir }]
      : [],
  });

  // Pi auth + model registry are global to this Pi session. These live on the
  // real host filesystem, never in the sandbox/workspace.
  // When `agentDir` is provided, use it instead so the harness can reuse
  // existing CLI logins and model/settings config.
  const agentDir = input.agentDir ?? hostAgentDir;
  const modelRuntime = await ModelRuntime.create({
    authPath: path.join(agentDir, 'auth.json'),
    modelsPath: path.join(agentDir, 'models.json'),
    allowModelNetwork: false,
  });
  const modelRegistry = new ModelRegistry(modelRuntime);
  const settingsManager =
    input.agentDir != null
      ? SettingsManager.create(hostWorkDir, agentDir)
      : SettingsManager.inMemory();

  // Run-scoped env (for the model resolver's gateway fallback heuristic).
  const resolverEnv = resolvePiEnv({
    options: input.settings.auth,
    env: process.env,
  });
  await registerPiProviders({
    options: input.settings.auth,
    resolvedEnv: resolverEnv,
    registries: {
      modelRegistry,
      modelRuntime,
    },
    clientApp: input.clientApp,
  });
  const resolveModel = createPiModelResolver({
    modelRegistry,
    env: resolverEnv,
  });
  // Resolve once: deterministic given the configured model. This is the Pi
  // `Model` object handed to `createAgentSession`.
  const resolvedModel = resolveModel(input.settings.model);
  const mcpServers = resolvePiMcpServers({
    mcpServers: input.settings.mcpServers,
  });
  const hasMcpServers = Object.keys(mcpServers).length > 0;

  let sessionInstructions: string | undefined;

  /*
   * Configured MCP servers are served by an inline Pi extension, so they share
   * the extension runtime with the caller-supplied factories: both are loaded
   * by the resource loader below and both are subject to the reload handling
   * that keeps the active runtime alive across resource-only reloads.
   */
  const extensionFactories: ExtensionFactory[] = [
    ...(input.settings.extensionFactories ?? []),
  ];
  if (hasMcpServers) {
    const { createMcpAdapter } = (await import(
      PI_MCP_ADAPTER_PACKAGE
    )) as PiMcpAdapterModule;
    extensionFactories.push(
      createMcpAdapter({
        config: {
          mcpServers,
          settings: {
            directTools: true,
            toolPrefix: 'mcp',
            disableProxyTool: true,
          },
        },
      }),
    );
  }
  const hasExtensionFactories = extensionFactories.length > 0;
  let preserveExtensionsResult = false;
  let currentExtensionsResult:
    | ReturnType<DefaultResourceLoader['getExtensions']>
    | undefined;

  /*
   * Surface the configured agent definition the way Pi surfaces project
   * guidance: as an `AGENTS.md` context file, which `buildSystemPrompt` renders
   * into the system prompt's `<project_context>` block. Nothing is written to
   * the workspace — Pi reads context files through the resource loader, so a
   * synthetic entry is indistinguishable from a discovered one, and the
   * workspace mirror stays a faithful view of the sandbox.
   *
   * Sourced from session settings rather than the per-turn `instructions`
   * field so it is known before the loader's first `reload()`, and so it also
   * covers resumed sessions and `doContinueTurn` (which carries no
   * instructions at all).
   */
  const projectInstructions = input.projectInstructions?.trim();
  const projectInstructionsPath = path.posix.join(sessionWorkDir, 'AGENTS.md');

  /*
   * Replace Pi's base system prompt when the host supplies one.
   *
   * Pi's default opens with "You are an expert coding assistant operating inside
   * pi" and carries a block of absolute paths to Pi's own README, docs and
   * examples. For a harness embedding Pi as an agent runtime that is both a wrong
   * persona and a standing invitation to read Pi's documentation, and it is paid
   * for on every turn. Routed through the resource loader — the same seam
   * `projectInstructions` uses — because that is what `_rebuildSystemPrompt`
   * reads as `customPrompt`, so the replacement survives every session rebuild
   * and every resume without touching Pi itself. Context files, the skills
   * section and the cwd line are still appended by Pi, so a custom prompt loses
   * neither the agent definition nor skill discovery.
   */
  const systemPrompt = input.systemPrompt?.trim();

  const resourceLoader = new DefaultResourceLoader({
    cwd: sessionWorkDir,
    agentDir: hostAgentDir,
    settingsManager,
    appendSystemPromptOverride: () =>
      sessionInstructions ? [sessionInstructions] : [],
    ...(systemPrompt ? { systemPromptOverride: () => systemPrompt } : {}),
    extensionFactories,
    ...(hasExtensionFactories
      ? {
          // DefaultResourceLoader invokes inline factories on every reload.
          // Resource-only reloads retain the active extension runtime, while a
          // genuine Pi session rebuild is allowed to replace that runtime.
          extensionsOverride: extensions => {
            if (preserveExtensionsResult && currentExtensionsResult != null) {
              return currentExtensionsResult;
            }
            currentExtensionsResult = extensions;
            return extensions;
          },
        }
      : {}),
    // Appended last so it outranks any ancestor `AGENTS.md`. A file discovered
    // at the same path is replaced rather than duplicated, keeping one entry
    // per path in the rendered prompt.
    agentsFilesOverride: base =>
      projectInstructions
        ? {
            agentsFiles: [
              ...base.agentsFiles.filter(
                file => file.path !== projectInstructionsPath,
              ),
              { path: projectInstructionsPath, content: projectInstructions },
            ],
          }
        : base,
    // Pi runs in the host process, so its default resource discovery reaches
    // the host developer's personal config (`~/.pi/agent/*`, `~/.agents/*`).
    // The harness exposes only explicitly supplied inline extension factories;
    // disable filesystem extension discovery entirely to avoid loading and
    // executing a host developer's personal or project Pi extensions inside
    // the server process. Themes and prompt templates stay disabled. Skills
    // are kept but filtered to workspace project skills plus harness-provided
    // skills whose files live in sandbox HOME.
    noExtensions: true,
    noThemes: true,
    noPromptTemplates: true,
    skillsOverride: base => ({
      ...base,
      skills: [
        ...base.skills.filter(skill =>
          isWithinWorkspace(skill.filePath, sessionWorkDir, hostWorkDir),
        ),
        ...harnessSkills,
      ],
    }),
  });
  await resourceLoader.reload();

  async function reloadResourcesOnly(): Promise<void> {
    if (!hasExtensionFactories) {
      await resourceLoader.reload();
      return;
    }

    const factories = extensionFactories.splice(0);
    preserveExtensionsResult = true;
    try {
      await resourceLoader.reload();
    } finally {
      preserveExtensionsResult = false;
      extensionFactories.push(...factories);
    }
  }

  // Per-session mutable state we hold across prompts.
  let piSession: AgentSession | undefined;
  let unsubscribe: (() => void) | undefined;
  let lastToolsSignature: string | undefined;
  let sessionFileName: string | undefined;
  let stopped = false;
  /*
   * Set by `doSuspendTurn` before it aborts the in-flight host turn at a slice
   * boundary. The turn's catch settles silently when this is set, so the stream
   * closes cleanly (no spurious `error` chunk) — the next slice rerun-continues
   * from the persisted journal.
   */
  let suspending = false;
  const pendingToolResults = new Map<string, PendingToolResult>();
  const pendingToolApprovals = new Map<string, PendingToolApproval>();
  /*
   * Outcomes the host submitted for a call that has no waiter yet, keyed by the
   * id the host addressed.
   *
   * The host and Pi do not have to arrive in that order. A turn parked on host
   * input is normally continued on the parked in-process session, whose pending
   * maps still hold the waiter — but when that session is gone (evicted, or the
   * process restarted) the continuation runs against a session rebuilt from the
   * persisted journal, and Pi re-drives the recorded tool call only after the
   * host has already submitted the outcome for it. Dropping the outcome in that
   * window is silent and unrecoverable: the tool's `execute` promise never
   * settles, so an approved call re-runs and looks to the model like a failure
   * it should retry, and a denied one strands the turn forever with no error.
   *
   * Holding the outcome here instead makes the two orders equivalent — whichever
   * side arrives second consumes what the first left. Entries are consumed on
   * take and cleared when the session settles, so nothing outlives its turn.
   */
  const earlyToolResults = new Map<string, unknown>();
  const earlyToolApprovals = new Map<string, ToolApprovalDecision>();
  /*
   * Recorded host tool calls a cross-process continuation must settle before
   * the model turn may run (see `PiPendingHostToolCall`). Entries are removed
   * as outcomes are injected into the session context; while any remain, the
   * turn stays parked on host input.
   */
  const awaitingHostToolCalls = new Map<string, PiPendingHostToolCall>(
    (input.isResume ? (input.pendingHostToolCalls ?? []) : []).map(call => [
      call.toolCallId,
      call,
    ]),
  );
  /*
   * The one live deferred continuation (a `doContinueTurn` parked on
   * `awaitingHostToolCalls`). Lifecycle methods settle it so its stream
   * closes; a newer deferred continuation supersedes it.
   */
  let deferredContinuation:
    | { resolve: () => void; reject: (error: unknown) => void }
    | undefined;

  // Emit channel set at the start of every doPromptTurn and cleared on end.
  const validatingHostCalls = new Map<
    string,
    ReturnType<typeof parseNativeEvent>
  >();
  let currentEmit: ((part: HarnessV1StreamPart) => void) | undefined;
  let translatorState: PiTranslatorState | undefined;
  let activeTurn: ActivePiTurn | undefined;
  let turnUsageBaseline: PiTokenSnapshot | undefined;

  function accountForPendingHostStepUsage(): void {
    if (!translatorState || !turnUsageBaseline) return;
    const usage = consumePendingPiStepUsage(translatorState);
    if (!usage) return;
    const after = piSession ? tokenSnapshot(piSession) : turnUsageBaseline;
    turnUsageBaseline = {
      input: Math.max(
        after.input,
        turnUsageBaseline.input + (usage.inputTokens.noCache ?? 0),
      ),
      output: Math.max(
        after.output,
        turnUsageBaseline.output + (usage.outputTokens.total ?? 0),
      ),
      cacheRead: Math.max(
        after.cacheRead,
        turnUsageBaseline.cacheRead + (usage.inputTokens.cacheRead ?? 0),
      ),
      cacheWrite: Math.max(
        after.cacheWrite,
        turnUsageBaseline.cacheWrite + (usage.inputTokens.cacheWrite ?? 0),
      ),
    };
  }
  /*
   * Compaction parts produced while no turn is active. Pi's `compact()` aborts
   * the current turn before it summarizes, so a manually triggered compaction
   * (and any compaction that lands between turns) emits its `compaction_end`
   * after `currentEmit` has been cleared. Buffer those parts and flush them on
   * the next turn's stream so the observation is not lost. Auto-compaction that
   * runs mid-turn still emits inline via `currentEmit`.
   */
  const pendingCompactionParts: HarnessV1StreamPart[] = [];

  async function applySessionInstructions(
    instructions: string | undefined,
  ): Promise<void> {
    if (instructions === sessionInstructions) return;
    sessionInstructions = instructions;
    await reloadResourcesOnly();
    piSession?.setActiveToolsByName(piSession.getActiveToolNames());
  }

  const remoteOps = createPiRemoteOps({
    // A getter, not the value: `remoteOps` is built once and captured by every
    // builtin tool, so this is what carries a reattach's rebound sandbox
    // through to the tools without rebuilding them.
    get sandbox() {
      return sandbox;
    },
    paths,
    onFileChange: (event, relPath) => {
      currentEmit?.({ type: 'file-change', event, path: relPath });
    },
  });

  /**
   * Points this session at `next`'s sandbox so a reattach stops addressing the
   * stopped one, and re-materialises what that sandbox is missing. See
   * {@link ParkedPiSession}.
   *
   * Only the sandbox handle is rebindable. `sessionWorkDir` is baked into the
   * workspace VFS mount, the path mapper, and Pi's own advertised cwd (already
   * written into the journal and the system prompt), so a slice that resolved a
   * different one cannot be served by this session at all — it is refused here
   * rather than reattached onto paths the new sandbox does not have. The host
   * derives that directory from the session id, so this is a guard, not a case
   * the current callers reach.
   */
  async function rebindToSlice(next: CreatePiSessionInput): Promise<boolean> {
    if (next.sessionWorkDir !== sessionWorkDir) {
      return false;
    }
    sandbox = next.sandboxSession.restricted();
    await writeSkillsToSandbox(next.abortSignal);
    return true;
  }

  function settlePendingToolResults(reason: string): void {
    for (const pending of pendingToolResults.values()) {
      pending.resolve({ error: reason });
    }
    pendingToolResults.clear();
    // The session is going away, so nothing can still come along to claim an
    // unconsumed outcome — keeping it would only leak into the next session
    // that happens to reuse the id.
    earlyToolResults.clear();
  }

  function settlePendingToolApprovals(reason: string): void {
    for (const pending of pendingToolApprovals.values()) {
      pending.resolve({ approved: false, reason });
    }
    pendingToolApprovals.clear();
    earlyToolApprovals.clear();
  }

  async function persistSessionFile(): Promise<void> {
    if (!sessionFileName) return;
    await persistSessionFileToSandbox({
      sandbox,
      sessionWorkDir: input.sessionWorkDir,
      hostSessionDir,
      sessionFileName,
    });
  }

  function deliverHostToolResult(args: {
    toolCallId: string;
    output: unknown;
  }): void {
    /*
     * Preserve the original output so the result projection can surface it
     * unchanged. The tool handler stringifies the output for the runtime
     * (so the model reads it), and Pi echoes that text back — without this
     * the consumer-facing result would be the serialized string instead of
     * the original object. Recorded before the waiter lookup because it is
     * needed either way: an outcome that lands ahead of Pi is consumed by
     * the tool handler later, and the projection still has to un-stringify
     * it when the matching result event arrives.
     */
    translatorState?.hostToolResults.set(args.toolCallId, args.output);
    const pending = pendingToolResults.get(args.toolCallId);
    if (!pending) {
      earlyToolResults.set(args.toolCallId, args.output);
      return;
    }
    pendingToolResults.delete(args.toolCallId);
    pending.resolve(args.output);
  }

  function deliverHostToolApproval(args: {
    approvalId: string;
    approved: boolean;
    reason?: string;
  }): void {
    const pending = pendingToolApprovals.get(args.approvalId);
    if (!pending) {
      earlyToolApprovals.set(args.approvalId, {
        approved: args.approved,
        reason: args.reason,
      });
      return;
    }
    pendingToolApprovals.delete(args.approvalId);
    pending.resolve({
      approved: args.approved,
      reason: args.reason,
    });
  }

  /**
   * Where `toolCallId` stands in the rebuilt session context: recorded on a
   * live (non-errored) assistant message and still unanswered (`dangling`),
   * already answered (`resolved`), or not in the context at all (`absent`).
   *
   * Errored/aborted assistant messages don't count as recording the call:
   * `transformMessages` drops them from the replayed LLM context entirely, so
   * a result injected for such a call would reach the provider without its
   * matching tool call.
   */
  function classifyHostToolCall(
    toolCallId: string,
  ): 'dangling' | 'resolved' | 'absent' {
    if (piSession == null) return 'absent';
    let recorded = false;
    for (const message of piSession.messages) {
      if (message.role === 'toolResult' && message.toolCallId === toolCallId) {
        return 'resolved';
      }
      if (message.role !== 'assistant') continue;
      if (message.stopReason === 'error' || message.stopReason === 'aborted') {
        continue;
      }
      for (const block of message.content) {
        if (block.type === 'toolCall' && block.id === toolCallId) {
          recorded = true;
        }
      }
    }
    return recorded ? 'dangling' : 'absent';
  }

  /**
   * Write a host-submitted outcome into the session context, exactly where the
   * in-process turn would have put it had the call's waiter survived: the same
   * dual write Pi performs when a tool settles (the live context array feeds
   * the next LLM call; the session manager persists the journal entry).
   */
  function injectHostToolResult(args: {
    toolCallId: string;
    toolName: string;
    output: unknown;
    isError?: boolean;
  }): void {
    const session = piSession;
    if (session == null) return;
    const message: PiToolResultMessage = {
      role: 'toolResult',
      toolCallId: args.toolCallId,
      toolName: args.toolName,
      content: [{ type: 'text', text: serializeToolOutput(args.output) }],
      details: undefined,
      isError: args.isError === true,
      timestamp: Date.now(),
    };
    session.agent.state.messages.push(message);
    session.sessionManager.appendMessage(message);
  }

  function createPromptControl(input: {
    done: Promise<void>;
    abortSignal?: AbortSignal;
  }): HarnessV1PromptControl {
    const abortHandler = () => {
      piSession?.abort().catch(() => {});
    };
    if (input.abortSignal) {
      input.abortSignal.addEventListener('abort', abortHandler, {
        once: true,
      });
      void input.done.then(
        () => {
          input.abortSignal?.removeEventListener('abort', abortHandler);
        },
        () => {
          input.abortSignal?.removeEventListener('abort', abortHandler);
        },
      );
    }

    return {
      async submitToolResult(args) {
        deliverHostToolResult(args);
      },
      async submitToolApproval(args) {
        deliverHostToolApproval(args);
      },
      async submitUserMessage(text) {
        await piSession?.steer(text);
      },
      done: input.done,
    };
  }

  async function requestBuiltinToolApproval(args: {
    toolCallId: string;
    nativeName: (typeof PI_NATIVE_BUILTIN_NAMES)[number];
  }): Promise<ToolApprovalDecision> {
    // A decision the host already submitted for this call — it continued a
    // parked turn whose session no longer holds the waiter, so Pi is only now
    // re-driving the recorded call. Answer from it instead of asking the user
    // to decide the same call twice.
    const earlyDecision = earlyToolApprovals.get(args.toolCallId);
    if (earlyDecision !== undefined) {
      earlyToolApprovals.delete(args.toolCallId);
      return earlyDecision;
    }
    if (
      !piBuiltinToolRequiresApproval({
        permissionMode,
        kind: PI_NATIVE_TOOL_KINDS[args.nativeName],
      })
    ) {
      return { approved: true };
    }
    if (translatorState) {
      const session = piSession;
      const before = turnUsageBaseline;
      const after = session == null ? undefined : tokenSnapshot(session);
      for (const part of finishPiApprovalStep(
        translatorState,
        args.toolCallId,
        before === undefined || after === undefined
          ? undefined
          : usageSince(after, before),
      )) {
        currentEmit?.(part);
      }
      if (after !== undefined) turnUsageBaseline = after;
    }
    currentEmit?.({
      type: 'tool-approval-request',
      approvalId: args.toolCallId,
      toolCallId: args.toolCallId,
    });
    return new Promise(resolve => {
      pendingToolApprovals.set(args.toolCallId, { resolve });
    });
  }

  function buildToolDefinitions(userTools: ReadonlyArray<HarnessV1ToolSpec>): {
    customTools: ToolDefinition[];
    builtinNames: string[];
  } {
    const builtinNames = resolveActivePiBuiltinNames(
      input.builtinToolFiltering,
    );
    const customTools: ToolDefinition[] = [
      ...builtinNames.map(native =>
        buildBuiltinToolDefinition({
          native,
          remoteOps,
          requestApproval: requestBuiltinToolApproval,
        }),
      ),
      ...userTools.map(spec =>
        buildUserToolDefinition(
          spec,
          pendingToolResults,
          earlyToolResults,
          (toolCallId, args) => {
            validatingHostCalls.delete(toolCallId);
            if (!translatorState) return;
            for (const part of translatePiEvent(
              {
                type: 'tool_execution_start',
                toolCallId,
                toolName: spec.name,
                args,
              },
              translatorState,
            )) {
              currentEmit?.(part);
            }
          },
        ),
      ),
    ];
    return {
      customTools,
      builtinNames: [...builtinNames],
    };
  }

  async function disposePiSession(): Promise<void> {
    unsubscribe?.();
    unsubscribe = undefined;

    const session = piSession;
    piSession = undefined;
    if (!session) return;

    if (hasMcpServers) {
      await session.reload().catch(() => {});
    }
    session.dispose();
  }

  async function rebuildPiSession(
    userTools: ReadonlyArray<HarnessV1ToolSpec>,
    isFirstBuild: boolean,
  ): Promise<boolean> {
    let resourcesReloaded = false;
    if (piSession) {
      await disposePiSession();
      // Original adapter waits 25 ms here to let Pi's teardown microtasks
      // settle before the next createAgentSession. Port verbatim.
      // TODO(pi-0.77): verify the race still exists; original SDK had a
      // teardown microtask the host needed to wait on.
      await new Promise(resolve => setTimeout(resolve, 25));
      if (hasExtensionFactories) {
        // dispose() invalidates Pi's current extension runtime, so a replacement
        // AgentSession needs factories to create a fresh runtime before build.
        await resourceLoader.reload();
        resourcesReloaded = true;
      }
    }

    const { customTools, builtinNames } = buildToolDefinitions(userTools);
    const toolNames = customTools.map(t => t.name);

    // SessionManager: open the resumed file on the first build of a resumed
    // session; create fresh otherwise.
    const sessionManager =
      isFirstBuild && resumeSessionFilePath
        ? SessionManager.open(
            resumeSessionFilePath,
            hostSessionDir,
            sessionWorkDir,
          )
        : SessionManager.create(sessionWorkDir, hostSessionDir);

    const { session } = await createAgentSession({
      cwd: sessionWorkDir,
      agentDir: hostAgentDir,
      modelRuntime,
      sessionManager,
      settingsManager,
      resourceLoader,
      customTools,
      ...(hasMcpServers
        ? { noTools: 'builtin' as const }
        : { tools: toolNames }),
      ...(input.settings.thinkingLevel
        ? { thinkingLevel: input.settings.thinkingLevel }
        : {}),
      ...(resolvedModel ? { model: resolvedModel } : {}),
    });
    piSession = session;
    if (hasMcpServers) {
      await piSession.bindExtensions({ mode: 'print' });
    }

    // Pick up the actual session file path so doStop can persist it. Pi
    // 0.77 emits `.jsonl` files; older builds used `.json`. Persist the
    // basename verbatim — including the extension — so the resume path can
    // round-trip it without guessing the extension.
    const candidatePath = sessionManager.getSessionFile();
    if (candidatePath) {
      sessionFileName = safePiSessionFileName(path.basename(candidatePath));
    }

    translatorState = createPiTranslatorState({
      builtinToolNames: builtinNames,
      hostToolNames: userTools.map(tool => tool.name),
      nativeToCommon: NATIVE_TO_COMMON,
    });
    unsubscribe = piSession.subscribe(rawEvent => {
      if (!translatorState) return;
      const event = parseNativeEvent(rawEvent);
      if (!event) return;
      // Pi emits start before argument validation and before its execution policy.
      // Only the custom tool's execute callback may dispatch a host write.
      if (
        event.type === 'tool_execution_start' &&
        event.toolCallId &&
        event.toolName &&
        translatorState.hostToolNames.has(event.toolName)
      ) {
        validatingHostCalls.set(event.toolCallId, event);
        return;
      }
      if (event.type === 'tool_execution_end' && event.toolCallId) {
        const rejectedCall = validatingHostCalls.get(event.toolCallId);
        if (rejectedCall) {
          validatingHostCalls.delete(event.toolCallId);
          for (const part of translatePiEvent(rejectedCall, translatorState)) {
            currentEmit?.(
              part.type === 'tool-call'
                ? { ...part, providerExecuted: true }
                : part,
            );
          }
        }
      }
      for (const part of translatePiEvent(event, translatorState)) {
        if (currentEmit) {
          currentEmit(part);
        } else if (part.type === 'compaction') {
          // No active turn: defer compaction observations to the next turn.
          pendingCompactionParts.push(part);
        }
        // Other event types outside a turn have no consumer and are dropped.
      }
    });
    return resourcesReloaded;
  }

  /*
   * Drive one turn against the Pi session and return the control surface.
   * Shared by `doPromptTurn` (a fresh user prompt) and `doContinueTurn` (an empty
   * prompt that asks Pi to continue its own thread after a rerun resume).
   */
  async function runTurn(turnOpts: {
    text: string;
    tools: ReadonlyArray<HarnessV1ToolSpec>;
    emit: (part: HarnessV1StreamPart) => void;
    abortSignal?: AbortSignal;
  }): Promise<HarnessV1PromptControl> {
    if (stopped) {
      throw new Error('Pi session has been stopped.');
    }

    const userTools = turnOpts.tools;
    const signature = JSON.stringify(userTools.map(t => t.name).sort());
    const needsRebuild = piSession == null || signature !== lastToolsSignature;
    let resourcesReloaded = false;
    if (needsRebuild) {
      resourcesReloaded = await rebuildPiSession(userTools, piSession == null);
      lastToolsSignature = signature;
    }

    if (!resourcesReloaded) {
      await reloadResourcesOnly();
    }
    await syncHostWorkspaceFromSandbox({
      sandbox,
      sandboxWorkDir: input.sessionWorkDir,
      hostWorkDir,
    });

    validatingHostCalls.clear();
    currentEmit = turnOpts.emit;
    // Fresh translator state for the new turn — keep the tool sets the
    // session was built with.
    translatorState = createPiTranslatorState({
      builtinToolNames: [...PI_NATIVE_BUILTIN_NAMES],
      hostToolNames: userTools.map(tool => tool.name),
      nativeToCommon: NATIVE_TO_COMMON,
    });
    turnUsageBaseline = tokenSnapshot(piSession!);

    turnOpts.emit({ type: 'stream-start' });

    const turnPromise = (async () => {
      let terminalError: string | undefined;
      let recoverableModelError = false;
      const session = piSession!;

      // We subscribed in rebuild, but the translator may need to detect
      // terminal errors too — wrap a second listener that records them.
      const unsubErr = session.subscribe(raw => {
        const ev = parseNativeEvent(raw);
        if (!ev) return;
        if (
          ev.type === 'auto_retry_end' &&
          ev.success === true &&
          recoverableModelError
        ) {
          terminalError = undefined;
          recoverableModelError = false;
          return;
        }
        const err = getPiTerminalError(ev);
        // Cancellation wins over an earlier retryable error. Pi can report a
        // successful retry reset after an aborted assistant message.
        if (err && ev.message?.stopReason === 'aborted') {
          terminalError = err;
          recoverableModelError = false;
        } else if (err && !terminalError) {
          terminalError = err;
          recoverableModelError =
            ev.message?.role === 'assistant' &&
            ev.message.stopReason === 'error';
        }
      });

      try {
        await session.prompt(turnOpts.text);

        if (terminalError) {
          /*
           * A `doSuspendTurn` aborts the in-flight turn on purpose. Pi surfaces
           * that abort as a *resolved* prompt with a recorded terminal error
           * ("This operation was aborted") rather than a thrown exception, so the
           * `catch` guard below never sees it. Swallow it here too — but only if
           * it's actually the abort: the stream then closes cleanly (no spurious
           * `error` chunk) and the next slice rerun-continues from the journal.
           * Any other terminal error mid-suspend is unanticipated and must
           * surface.
           */
          if (suspending && isAbortError(terminalError)) return;
          currentEmit?.({
            type: 'error',
            error: createPiTerminalError(terminalError),
          });
          return;
        }

        /*
         * Pi can resolve `prompt()` without a `turn_end` (and without a
         * recognized terminal error) when the model stream ends abnormally —
         * e.g. cut mid-message. The translator then still holds an open text
         * block and an unfinished step, and emitting the terminal `finish`
         * below would violate the harness contract (`finish-step` before
         * `finish`) and fail the turn, discarding already-streamed content.
         * Flush whatever is still open; a no-op after a normal `turn_end`.
         */
        if (translatorState) {
          for (const part of flushOpenStep(translatorState)) {
            currentEmit?.(part);
          }
        }

        const usage = usageSince(
          tokenSnapshot(session),
          turnUsageBaseline ?? tokenSnapshot(session),
        );
        // The provider's real stop reason (captured from the last assistant
        // message) — `length` and friends must reach consumers instead of a
        // hard-coded `stop`. Only when Pi reported none is `stop` inferred,
        // and then it is marked as inferred so evidence stays distinguishable
        // from a guess.
        const { finishReason, inferred } = piTurnFinishReason(translatorState);
        currentEmit?.({
          type: 'finish',
          finishReason,
          totalUsage: usage,
          ...(inferred
            ? { harnessMetadata: { pi: { inferredFinishReason: true } } }
            : {}),
        });
      } catch (err) {
        // A `doSuspendTurn` aborts the in-flight turn on purpose — settle silently
        // so the stream closes cleanly without a spurious `error` chunk; the
        // next slice rerun-continues from the persisted journal.
        // Same rule as the resolved-with-terminalError path: only swallow the
        // abort our own suspend caused; surface anything unanticipated.
        if (suspending && isAbortError(err)) return;
        currentEmit?.({ type: 'error', error: err });
      } finally {
        unsubErr();
      }
    })();

    const activeTurnToken = {};
    const done = turnPromise.finally(() => {
      if (activeTurn?.token === activeTurnToken) {
        activeTurn = undefined;
      }
      currentEmit = undefined;
    });
    activeTurn = {
      token: activeTurnToken,
      done,
    };

    return createPromptControl({
      done,
      abortSignal: turnOpts.abortSignal,
    });
  }

  /*
   * Cross-process continuation of a turn parked on host input. The parked live
   * session — and with it the recorded call's promise waiter — is gone, and Pi
   * does not re-drive recorded calls after a journal rebuild: a plain rerun
   * (`prompt('')`) would hand the model a context whose dangling call
   * `transformMessages` patches with a synthetic `"No result provided"` error,
   * so the model retries a call whose effect is already applied.
   *
   * Hold the model turn instead: the host's continuations arrive through
   * `submitToolResult`, each outcome is written into the session context where
   * the in-process turn would have put it, and the turn re-drives only once no
   * recorded call is left unanswered. The recorded calls are also replayed as
   * `tool-call` parts so the framework can settle the decided ones and re-emit
   * an approval request for any the host has not decided — parking the turn
   * again instead of hanging it when a continuation batch is partial.
   */
  async function deferContinueTurnForHostOutcomes(
    continueOpts: HarnessV1ContinueTurnOptions,
  ): Promise<HarnessV1PromptControl> {
    if (stopped) {
      throw new Error('Pi session has been stopped.');
    }

    const userTools = continueOpts.tools ?? [];
    const signature = JSON.stringify(userTools.map(t => t.name).sort());
    if (piSession == null || signature !== lastToolsSignature) {
      await rebuildPiSession(userTools, piSession == null);
      lastToolsSignature = signature;
    }

    /*
     * An outcome can only be attached to a call the journal recorded as
     * unanswered. Anything else — a call missing from a stale journal copy, or
     * one already answered by a previous partial continuation — must neither
     * gate the turn nor be injected: a result without its call (or a duplicate
     * result) makes the provider reject the whole request.
     */
    for (const toolCallId of [...awaitingHostToolCalls.keys()]) {
      if (classifyHostToolCall(toolCallId) !== 'dangling') {
        awaitingHostToolCalls.delete(toolCallId);
      }
    }
    if (awaitingHostToolCalls.size === 0) {
      return runTurn({
        text: '',
        tools: userTools,
        emit: continueOpts.emit,
        abortSignal: continueOpts.abortSignal,
      });
    }

    // Only one deferred continuation can own the next turn; a superseded one
    // (its consumer re-continued after a partial settle) closes cleanly.
    deferredContinuation?.resolve();

    let state: 'waiting' | 'running' | 'settled' = 'waiting';
    let resolveDone!: () => void;
    let rejectDone!: (error: unknown) => void;
    const done = new Promise<void>((resolve, reject) => {
      resolveDone = resolve;
      rejectDone = reject;
    });
    function onAbort(): void {
      finish({
        error:
          continueOpts.abortSignal?.reason ??
          new DOMException('Aborted', 'AbortError'),
      });
    }
    function finish(outcome: { error?: unknown } = {}): void {
      if (state === 'settled') return;
      state = 'settled';
      continueOpts.abortSignal?.removeEventListener('abort', onAbort);
      if (deferredContinuation === deferred) {
        deferredContinuation = undefined;
      }
      if ('error' in outcome) {
        rejectDone(outcome.error);
      } else {
        resolveDone();
      }
    }
    const deferred = {
      resolve: () => finish(),
      reject: (error: unknown) => finish({ error }),
    };
    deferredContinuation = deferred;
    continueOpts.abortSignal?.addEventListener('abort', onAbort, {
      once: true,
    });

    const startContinuationTurn = async (): Promise<void> => {
      if (state !== 'waiting') return;
      state = 'running';
      try {
        const control = await runTurn({
          text: '',
          tools: userTools,
          emit: continueOpts.emit,
          abortSignal: continueOpts.abortSignal,
        });
        control.done.then(
          () => finish(),
          (error: unknown) => finish({ error }),
        );
      } catch (error) {
        finish({ error });
      }
    };

    for (const call of awaitingHostToolCalls.values()) {
      continueOpts.emit({
        type: 'tool-call',
        toolCallId: call.toolCallId,
        toolName: call.toolName,
        input: call.input,
        providerExecuted: false,
      });
    }

    return {
      async submitToolResult(args) {
        const awaited =
          state === 'waiting'
            ? awaitingHostToolCalls.get(args.toolCallId)
            : undefined;
        if (awaited == null) {
          deliverHostToolResult(args);
          return;
        }
        awaitingHostToolCalls.delete(args.toolCallId);
        injectHostToolResult({
          toolCallId: args.toolCallId,
          toolName: awaited.toolName,
          output: args.output,
          ...(args.isError !== undefined ? { isError: args.isError } : {}),
        });
        if (awaitingHostToolCalls.size > 0) {
          // Partial settle: the turn stays parked on the remaining calls.
          // Make the injected outcome durable now — the next continuation may
          // run in another process and must find it in the journal.
          await persistSessionFile().catch(() => {});
          return;
        }
        await startContinuationTurn();
      },
      async submitToolApproval(args) {
        deliverHostToolApproval(args);
      },
      async submitUserMessage(text) {
        await piSession?.steer(text);
      },
      done,
    };
  }

  const doStop = async (): Promise<HarnessV1ResumeSessionState> => {
    if (stopped) {
      throw new Error('Pi session has been stopped.');
    }
    stopped = true;
    parkedPiSessions.delete(input.sessionId);
    deferredContinuation?.reject(new Error('Pi session stopped'));
    settlePendingToolResults('Pi session stopped');
    settlePendingToolApprovals('Pi session stopped');

    // Persist the Pi session file into the sandbox so a future process
    // can pick it up after `provider.resumeSession({ sessionId })` reattaches.
    if (sessionFileName) {
      try {
        await persistSessionFile();
      } catch {
        // Best-effort: a missing session file means resume returns to a
        // fresh conversation rather than failing stop.
      }
    }

    await disposePiSession();
    workspaceVfs.unmount();
    await rm(hostRoot, { recursive: true, force: true });

    return {
      type: 'resume-session',
      harnessId: HARNESS_ID,
      specificationVersion: 'harness-v1',
      data: sessionFileName ? { sessionFileName } : {},
    };
  };

  const sessionImpl: HarnessV1Session = {
    sessionId: input.sessionId,
    isResume: input.isResume,
    // The model Pi actually resolves to (the configured id, or its default when
    // unset) — `gen_ai.request.model`.
    ...(resolvedModel?.id ? { modelId: resolvedModel.id } : {}),

    // Pi has no bridge to attach to and no on-disk event log to replay; its
    // only resume path is restoring the session file on a fresh/snapshotted
    // sandbox, i.e. `rerun`.

    doPromptTurn: async (
      promptOpts: HarnessV1PromptTurnOptions,
    ): Promise<HarnessV1PromptControl> => {
      await applySessionInstructions(promptOpts.instructions);

      return runTurn({
        text: extractUserText(promptOpts.prompt),
        tools: promptOpts.tools ?? [],
        emit: promptOpts.emit,
        abortSignal: promptOpts.abortSignal,
      });
    },

    doContinueTurn: async (
      continueOpts: HarnessV1ContinueTurnOptions,
    ): Promise<HarnessV1PromptControl> => {
      if (activeTurn != null) {
        currentEmit = continueOpts.emit;
        return createPromptControl({
          done: activeTurn.done,
          abortSignal: continueOpts.abortSignal,
        });
      }

      /*
       * The parked turn was waiting on host input for recorded tool calls and
       * its live session did not survive to here: hold the rerun until the
       * host settles those calls, injecting each outcome into the rebuilt
       * context first.
       */
      if (awaitingHostToolCalls.size > 0) {
        return deferContinueTurnForHostOutcomes(continueOpts);
      }

      /*
       * Pi runs the model on the host, so there is no live turn in the sandbox
       * to attach to — the previous slice's turn died with its process.
       * Rerun-continue: re-drive the agent from the journal restored on resume.
       * An empty prompt asks Pi to continue its own thread. Lossy — any work in
       * flight at the slice boundary is recomputed because a host-resident
       * runtime cannot do a lossless attach.
       */
      await applySessionInstructions(continueOpts.instructions);
      return runTurn({
        text: '',
        tools: continueOpts.tools ?? [],
        emit: continueOpts.emit,
        abortSignal: continueOpts.abortSignal,
      });
    },

    doCompact: async (customInstructions?: string) => {
      if (stopped) {
        throw new Error('Pi session has been stopped.');
      }
      /*
       * Pi owns the compaction. We just request it; the resulting
       * `compaction_end` event is observed by the session subscription and
       * translated into a `compaction` stream part. The returned
       * `CompactionResult` is intentionally discarded here.
       */
      await piSession?.compact(customInstructions);
    },

    doDestroy: async () => {
      if (stopped) return;
      stopped = true;
      parkedPiSessions.delete(input.sessionId);
      deferredContinuation?.reject(new Error('Pi session stopped'));
      settlePendingToolResults('Pi session stopped');
      settlePendingToolApprovals('Pi session stopped');
      await disposePiSession();
      workspaceVfs.unmount();
      await rm(hostRoot, { recursive: true, force: true });
    },

    doStop,

    doDetach: async (): Promise<HarnessV1ResumeSessionState> => {
      if (
        activeTurn != null ||
        pendingToolResults.size > 0 ||
        awaitingHostToolCalls.size > 0
      ) {
        accountForPendingHostStepUsage();
        parkedPiSessions.set(input.sessionId, {
          rebindToSlice,
          session: sessionImpl,
        });
        // A deferred continuation still parked on host input hands over to
        // the next resume; close its stream cleanly.
        deferredContinuation?.resolve();
        if (sessionFileName) {
          try {
            await persistSessionFile();
          } catch {
            /*
             * The parked in-process session is the authoritative continuation
             * path while the live turn is waiting on host input. Persistence is
             * only a fallback for later non-live resumes.
             */
          }
        }
        return {
          type: 'resume-session',
          harnessId: HARNESS_ID,
          specificationVersion: 'harness-v1',
          data: sessionFileName ? { sessionFileName } : {},
        };
      }
      return doStop();
    },

    doSuspendTurn: async (): Promise<HarnessV1ContinueTurnState> => {
      if (stopped) {
        throw new Error('Pi session has been stopped.');
      }
      if (
        (activeTurn != null &&
          (pendingToolResults.size > 0 || pendingToolApprovals.size > 0)) ||
        (activeTurn == null && awaitingHostToolCalls.size > 0)
      ) {
        accountForPendingHostStepUsage();
        parkedPiSessions.set(input.sessionId, {
          rebindToSlice,
          session: sessionImpl,
        });
        // A deferred continuation still parked on host input hands over to
        // the next resume; close its stream cleanly.
        deferredContinuation?.resolve();
        if (sessionFileName) {
          try {
            await persistSessionFile();
          } catch {
            /*
             * While waiting on host input, the live parked session is the
             * authoritative same-process continuation path. The sandbox copy
             * remains a best-effort fallback for a later cold resume.
             */
          }
        }
        return {
          type: 'continue-turn',
          harnessId: HARNESS_ID,
          specificationVersion: 'harness-v1',
          data: sessionFileName ? { sessionFileName } : {},
        };
      }
      /*
       * Pi's model runs in this host process, which is about to be suspended at
       * the slice boundary — the in-flight turn cannot survive it. Abort it (the
       * turn settles silently via the `suspending` guard so the stream closes
       * cleanly), persist the journal into the sandbox, and tear down host-side
       * resources. The sandbox itself is left running; the next slice pulls the
       * journal after `provider.resumeSession({ sessionId })` and rerun-continues. The
       * tail in flight at the boundary is recomputed — Pi cannot freeze a live
       * turn the way a bridge adapter can.
       */
      suspending = true;
      await Promise.resolve(piSession?.abort()).catch(() => {});

      if (sessionFileName) {
        try {
          await persistSessionFile();
        } catch {
          // Best-effort: a missing/failed copy leaves the previously persisted
          // journal in place, so the next slice resumes from a slightly older
          // (still valid) state.
        }
      }

      stopped = true;
      parkedPiSessions.delete(input.sessionId);
      settlePendingToolResults('Pi session suspended');
      settlePendingToolApprovals('Pi session suspended');
      await disposePiSession();
      workspaceVfs.unmount();
      await rm(hostRoot, { recursive: true, force: true });

      return {
        type: 'continue-turn',
        harnessId: HARNESS_ID,
        specificationVersion: 'harness-v1',
        data: sessionFileName ? { sessionFileName } : {},
      };
    },
  };

  return sessionImpl;
}

function resolvePiMcpServers({
  mcpServers,
}: {
  mcpServers: Record<string, unknown> | undefined;
}): Record<string, unknown> {
  if (mcpServers == null) return {};
  for (const [name, value] of Object.entries(mcpServers)) {
    if (value == null || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error(
        `Pi MCP server ${JSON.stringify(name)} must be configured with an object value.`,
      );
    }
  }
  return mcpServers;
}

/**
 * Whether a terminal error (string from Pi's event stream, or a thrown error)
 * is an abort — the expected result of `doSuspendTurn` aborting the in-flight
 * turn. Only these are safe to swallow while `suspending`; any other error is
 * unanticipated and must surface as an `error` chunk.
 */
function isAbortError(value: unknown): boolean {
  if (value == null) return false;
  if (
    typeof value === 'object' &&
    (value as { name?: unknown }).name === 'AbortError'
  ) {
    return true;
  }
  const text =
    typeof value === 'string'
      ? value
      : value instanceof Error
        ? value.message
        : String(value);
  return /\baborted\b|AbortError|operation was aborted/i.test(text);
}

function asPiToolResult(text: string): AgentToolResult<unknown> {
  return {
    content: [{ type: 'text', text }],
    details: undefined,
  };
}

async function maybeDenyPiBuiltinTool(input: {
  toolCallId: string;
  nativeName: (typeof PI_NATIVE_BUILTIN_NAMES)[number];
  requestApproval: (args: {
    toolCallId: string;
    nativeName: (typeof PI_NATIVE_BUILTIN_NAMES)[number];
  }) => Promise<{ approved: boolean; reason?: string }>;
}): Promise<AgentToolResult<unknown> | undefined> {
  const decision = await input.requestApproval({
    toolCallId: input.toolCallId,
    nativeName: input.nativeName,
  });
  if (decision.approved) return undefined;
  return asPiToolResult(
    serializeToolOutput({
      type: 'execution-denied',
      reason: decision.reason,
    }),
  );
}

function piBuiltinToolRequiresApproval(input: {
  permissionMode: HarnessV1PermissionMode;
  kind: 'readonly' | 'edit' | 'bash';
}): boolean {
  if (input.permissionMode === 'allow-all') return false;
  if (input.permissionMode === 'allow-edits') return input.kind === 'bash';
  return input.kind === 'edit' || input.kind === 'bash';
}

function buildBuiltinToolDefinition(input: {
  native: (typeof PI_NATIVE_BUILTIN_NAMES)[number];
  remoteOps: PiRemoteOps;
  requestApproval: (args: {
    toolCallId: string;
    nativeName: (typeof PI_NATIVE_BUILTIN_NAMES)[number];
  }) => Promise<{ approved: boolean; reason?: string }>;
}): ToolDefinition {
  switch (input.native) {
    case 'read':
      return defineTool({
        name: 'read',
        label: 'read',
        description: 'Read file contents.',
        parameters: Type.Object({ file_path: Type.String() }),
        async execute(toolCallId, params) {
          const denied = await maybeDenyPiBuiltinTool({
            toolCallId,
            nativeName: 'read',
            requestApproval: input.requestApproval,
          });
          if (denied) return denied;
          const buf = await input.remoteOps.readBuffer(params.file_path);
          return asPiToolResult(buf.toString('utf8'));
        },
      });
    case 'write':
      return defineTool({
        name: 'write',
        label: 'write',
        description: 'Write content to a file.',
        parameters: Type.Object({
          file_path: Type.String(),
          content: Type.String(),
        }),
        async execute(toolCallId, params) {
          const denied = await maybeDenyPiBuiltinTool({
            toolCallId,
            nativeName: 'write',
            requestApproval: input.requestApproval,
          });
          if (denied) return denied;
          await input.remoteOps.writeFile(params.file_path, params.content);
          return asPiToolResult(`Wrote ${params.file_path}`);
        },
      });
    case 'edit':
      return defineTool({
        name: 'edit',
        label: 'edit',
        description: 'Edit a file by exact-string replacement.',
        parameters: Type.Object({
          file_path: Type.String(),
          old_string: Type.String(),
          new_string: Type.String(),
        }),
        async execute(toolCallId, params) {
          const denied = await maybeDenyPiBuiltinTool({
            toolCallId,
            nativeName: 'edit',
            requestApproval: input.requestApproval,
          });
          if (denied) return denied;
          await input.remoteOps.editFile(
            params.file_path,
            params.old_string,
            params.new_string,
          );
          return asPiToolResult(`Edited ${params.file_path}`);
        },
      });
    case 'bash':
      return defineTool({
        name: 'bash',
        label: 'bash',
        description: 'Execute a shell command.',
        parameters: Type.Object({
          command: Type.String(),
          timeout: Type.Optional(
            Type.Number({ description: 'Timeout in seconds.' }),
          ),
        }),
        async execute(toolCallId, params, signal) {
          const denied = await maybeDenyPiBuiltinTool({
            toolCallId,
            nativeName: 'bash',
            requestApproval: input.requestApproval,
          });
          if (denied) return denied;
          const chunks: Buffer[] = [];
          const result = await input.remoteOps.exec(params.command, '.', {
            onData(data) {
              chunks.push(data);
            },
            ...(signal ? { signal } : {}),
            ...(typeof params.timeout === 'number'
              ? { timeout: params.timeout }
              : {}),
          });
          const out = Buffer.concat(chunks).toString('utf8');
          const text = `${out}${
            result.exitCode != null ? `\n\n(exit ${result.exitCode})` : ''
          }`.trim();
          return asPiToolResult(text);
        },
      });
    case 'grep':
      return defineTool({
        name: 'grep',
        label: 'grep',
        description: 'Search file contents with regex.',
        parameters: Type.Object({
          pattern: Type.String(),
          path: Type.Optional(Type.String()),
          glob: Type.Optional(Type.String()),
          ignoreCase: Type.Optional(Type.Boolean()),
          literal: Type.Optional(Type.Boolean()),
          context: Type.Optional(Type.Number()),
          limit: Type.Optional(Type.Number()),
        }),
        async execute(toolCallId, params) {
          const denied = await maybeDenyPiBuiltinTool({
            toolCallId,
            nativeName: 'grep',
            requestApproval: input.requestApproval,
          });
          if (denied) return denied;
          const out = await input.remoteOps.grepFiles(params.pattern, params);
          return asPiToolResult(out);
        },
      });
    case 'find':
      return defineTool({
        name: 'find',
        label: 'find',
        description: 'Find files matching a glob pattern.',
        parameters: Type.Object({
          pattern: Type.String(),
          path: Type.Optional(Type.String()),
          limit: Type.Optional(Type.Number()),
        }),
        async execute(toolCallId, params) {
          const denied = await maybeDenyPiBuiltinTool({
            toolCallId,
            nativeName: 'find',
            requestApproval: input.requestApproval,
          });
          if (denied) return denied;
          const matches = await input.remoteOps.findFiles(
            params.pattern,
            params.path ?? '.',
            params.limit ?? 1_000,
          );
          return asPiToolResult(matches.join('\n'));
        },
      });
    case 'ls':
      return defineTool({
        name: 'ls',
        label: 'ls',
        description: 'List directory entries.',
        parameters: Type.Object({
          path: Type.Optional(Type.String()),
          limit: Type.Optional(Type.Number()),
        }),
        async execute(toolCallId, params) {
          const denied = await maybeDenyPiBuiltinTool({
            toolCallId,
            nativeName: 'ls',
            requestApproval: input.requestApproval,
          });
          if (denied) return denied;
          const entries = await input.remoteOps.listDirectory(
            params.path ?? '.',
            params.limit ?? 500,
          );
          return asPiToolResult(entries.join('\n'));
        },
      });
  }
}

function buildUserToolDefinition(
  spec: HarnessV1ToolSpec,
  pending: Map<string, PendingToolResult>,
  early: Map<string, unknown>,
  dispatch: (toolCallId: string, args: unknown) => void,
): ToolDefinition {
  const schema = spec.inputSchema ?? {
    type: 'object',
    properties: {},
    additionalProperties: true,
  };
  return defineTool({
    name: spec.name,
    label: spec.name,
    description: spec.description ?? `User-registered tool ${spec.name}`,
    parameters: toolSpecToTypeBoxParameters(schema),
    async execute(toolCallId, args) {
      dispatch(toolCallId, args);
      /*
       * An outcome the host submitted before Pi drove this call. Returning it
       * is what makes the host tool run exactly once: without it Pi would sit
       * on a promise nobody can resolve any more, and the host's already-applied
       * effect would be invisible to the model. Checked with `has` rather than a
       * truthiness test because `undefined` is a legitimate tool output.
       */
      if (early.has(toolCallId)) {
        const output = early.get(toolCallId);
        early.delete(toolCallId);
        return asPiToolResult(serializeToolOutput(output));
      }
      return new Promise<unknown>(resolve => {
        pending.set(toolCallId, { resolve });
      }).then(output => asPiToolResult(serializeToolOutput(output)));
    },
  });
}
