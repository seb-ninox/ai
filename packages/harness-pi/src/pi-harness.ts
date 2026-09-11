import {
  commonTool,
  type HarnessV1,
  type HarnessV1BuiltinTool,
} from '@ai-sdk/harness';
import { tool } from '@ai-sdk/provider-utils';
import type { ExtensionFactory } from '@earendil-works/pi-coding-agent';
import { z } from 'zod/v4';
import type { PiAuthenticationMode } from './pi-auth';
import { piResumeStateSchema } from './pi-resume-state';
import { createPiSession, type PiThinkingLevel } from './pi-session';
import { VERSION } from './version';

/**
 * Value to use in User-Agent and `x-client-app` headers.
 */
const PI_CLIENT_APP = `ai-sdk/harness-pi/${VERSION}`;

/**
 * Configuration knobs for `createPi`. Pi runs as an in-process Node library
 * (no bridge), so there's no `port` or `startupTimeoutMs` to set.
 */
export type PiHarnessSettings = {
  /** Where Pi sources API keys / gateway credentials from. */
  readonly auth?: PiAuthenticationMode;
  /**
   * Pi's extended-thinking budget level. Maps directly to the SDK's
   * `thinkingLevel` option on `createAgentSession`.
   */
  readonly thinkingLevel?: PiThinkingLevel;
  /**
   * Directory holding Pi's global agent config (auth.json, models.json,
   * settings.json). When omitted, a per-session temp dir is used. Pass the
   * user's agent dir (e.g. `~/.pi/agent/`) to reuse their CLI auth and
   * model settings.
   */
  readonly agentDir?: string;
  /**
   * Operating guidance for the agent, surfaced to Pi the way a project's own
   * `AGENTS.md` is: as a context file rendered into the system prompt's
   * `<project_context>` block. Nothing is written to disk.
   *
   * This is the system-prompt counterpart of the per-turn
   * `HarnessV1PromptTurnOptions.instructions`, which the contract requires the
   * adapter to prepend to the first user message instead. Prefer this setting
   * for guidance that must survive compaction and resumption: the system
   * prompt is rebuilt from scratch on every session build, whereas a first
   * user message can be summarized away and is only reapplied on fresh
   * sessions. The two are independent — setting this does not suppress the
   * per-turn field.
   */
  readonly projectInstructions?: string;
  /**
   * Replaces Pi's base system prompt — the "expert coding assistant operating
   * inside pi" preamble, its tool list, its guidelines and the paths to Pi's own
   * documentation — with the host's own. Pi still appends what it derives from
   * the session: the `<project_context>` block (see `projectInstructions`), the
   * available-skills section, and the working directory.
   *
   * Set this when the harness runs an agent that is not a coding assistant: the
   * default preamble is both a persona the host did not ask for and a per-turn
   * cost (~500 tokens, most of it Pi's documentation index). Leave it unset to
   * keep Pi's own prompt.
   */
  readonly systemPrompt?: string;
  /**
   * MCP server definitions keyed by server name. Each definition uses the
   * underlying runtime's native MCP server configuration format.
   */
  readonly mcpServers?: Record<string, unknown>;
  /**
   * Trusted inline Pi extensions loaded for each harness session.
   *
   * Filesystem-discovered user and project extensions remain disabled.
   */
  readonly extensionFactories?: ReadonlyArray<ExtensionFactory>;
  /**
   * Makes Anthropic tool calls sequential so each model step observes the
   * preceding tool result before choosing another tool.
   */
  readonly sequentialToolCalls?: boolean;
};

const PI_BUILTIN_TOOLS = {
  read: commonTool('read', {
    nativeName: 'read',
    toolUseKind: 'readonly',
    description: 'Read file contents.',
    inputSchema: z.object({
      file_path: z.string(),
    }),
  }),
  write: commonTool('write', {
    nativeName: 'write',
    toolUseKind: 'edit',
    description: 'Overwrite or create a file.',
    inputSchema: z.object({
      file_path: z.string(),
      content: z.string(),
    }),
  }),
  edit: commonTool('edit', {
    nativeName: 'edit',
    toolUseKind: 'edit',
    description: 'Edit a file by exact string replacement.',
    inputSchema: z.object({
      file_path: z.string(),
      old_string: z.string(),
      new_string: z.string(),
    }),
  }),
  bash: commonTool('bash', {
    nativeName: 'bash',
    toolUseKind: 'bash',
    description: 'Execute a shell command in the sandbox.',
    inputSchema: z.object({
      command: z.string(),
      timeout: z.number().optional(),
    }),
  }),
  grep: commonTool('grep', {
    nativeName: 'grep',
    toolUseKind: 'readonly',
    description: 'Search file contents with regex.',
    inputSchema: z.object({
      pattern: z.string(),
      path: z.string().optional(),
      glob: z.string().optional(),
      ignoreCase: z.boolean().optional(),
      literal: z.boolean().optional(),
      context: z.number().optional(),
      limit: z.number().optional(),
    }),
  }),
  glob: commonTool('glob', {
    nativeName: 'find',
    toolUseKind: 'readonly',
    description: 'Find files matching a glob pattern.',
    inputSchema: z.object({
      pattern: z.string(),
      path: z.string().optional(),
      limit: z.number().optional(),
    }),
  }),
  ls: {
    ...tool({
      description: 'List directory entries.',
      inputSchema: z.object({
        path: z.string().optional(),
        limit: z.number().optional(),
      }),
      outputSchema: z.unknown(),
    }),
    nativeName: 'ls',
    toolUseKind: 'readonly',
  } as HarnessV1BuiltinTool,
} as const satisfies Record<string, HarnessV1BuiltinTool<any, any>>;

export function createPi(
  settings: PiHarnessSettings = {},
): HarnessV1<typeof PI_BUILTIN_TOOLS> {
  return {
    specificationVersion: 'harness-v1',
    harnessId: 'pi',
    builtinTools: PI_BUILTIN_TOOLS,
    supportsBuiltinToolApprovals: true,
    supportsBuiltinToolFiltering: true,
    lifecycleStateSchema: piResumeStateSchema,
    doStart: async startOpts => {
      const lifecycleState = startOpts.continueFrom ?? startOpts.resumeFrom;
      const resumeData = lifecycleState?.data as
        | { sessionFileName?: string }
        | undefined;
      return createPiSession({
        sessionId: startOpts.sessionId,
        sandboxSession: startOpts.sandboxSession,
        sessionWorkDir: startOpts.sessionWorkDir,
        settings: {
          ...(settings.auth ? { auth: settings.auth } : {}),
          ...(settings.thinkingLevel
            ? { thinkingLevel: settings.thinkingLevel }
            : {}),
          ...(settings.mcpServers ? { mcpServers: settings.mcpServers } : {}),
          ...(settings.extensionFactories
            ? { extensionFactories: settings.extensionFactories }
            : {}),
          ...(settings.sequentialToolCalls
            ? { sequentialToolCalls: true }
            : {}),
          ...(startOpts.headers ? { headers: startOpts.headers } : {}),
        },
        clientApp: PI_CLIENT_APP,
        isResume: lifecycleState != null,
        permissionMode: startOpts.permissionMode,
        builtinToolFiltering: startOpts.builtinToolFiltering,
        ...(resumeData?.sessionFileName
          ? { resumeSessionFileName: resumeData.sessionFileName }
          : {}),
        ...(startOpts.abortSignal
          ? { abortSignal: startOpts.abortSignal }
          : {}),
        ...(settings.agentDir ? { agentDir: settings.agentDir } : {}),
        ...(settings.projectInstructions
          ? { projectInstructions: settings.projectInstructions }
          : {}),
        ...(settings.systemPrompt
          ? { systemPrompt: settings.systemPrompt }
          : {}),
      });
    },
  };
}
