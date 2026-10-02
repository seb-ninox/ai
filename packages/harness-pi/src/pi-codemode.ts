import {
  CodemodeSandbox,
  parseCodemodeSource,
  renderToolSample,
  toCodemodeIdentifier,
  type CodemodeResult,
  type CodemodeTool as SandboxTool,
} from '@earendil-works/pi-codemode';
import {
  createCodemodeExtension,
  type AgentToolResult,
  type ExtensionAPI,
  type ExtensionFactory,
  type ExtensionToolContext,
  type ToolDefinition,
} from '@earendil-works/pi-coding-agent';
import type { TSchema } from 'typebox';

/**
 * Pi's `codemode` tool, which lets the model write one JavaScript script that
 * calls other tools instead of issuing the calls one step at a time. Pi
 * registers the tool inactive; the session activates it on every build.
 */
export interface PiCodemodeSettings {
  /**
   * `on` (Pi's default) keeps every script-callable tool declared to the model
   * and appends its codemode declaration to the tool description. `only` hides
   * those declarations and lists the tools in the codemode description instead.
   */
  readonly mode?: 'on' | 'only';
  /**
   * Estimated tokens the codemode description may spend on tool declarations.
   * Pi's default is 3000.
   */
  readonly inlineBudget?: number;
  /**
   * Heap the QuickJS VM of one script may allocate, in bytes. A script that
   * allocates past it fails with `InternalError: out of memory`. Absent means
   * Pi's own 256 MiB.
   */
  readonly memoryLimitBytes?: number;
  /**
   * Deadline for one script, in milliseconds, counting time spent in nested
   * tool calls. A script's own `// @options: {"timeout_ms": N}` may shorten
   * it, never extend it. Absent means no deadline, as in Pi.
   */
  readonly timeoutMs?: number;
}

/** The name Pi registers its codemode tool under; the package does not export it. */
export const PI_CODEMODE_TOOL_NAME = 'codemode';

/**
 * Pi's own heap cap, a constant in
 * `@earendil-works/pi-coding-agent/dist/extensions/codemode/execute.js`.
 */
const PI_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024;
const ARGS_PREVIEW_CHARS = 200;
const ERROR_PREVIEW_CHARS = 500;
const DEFAULT_MAX_OUTPUT_TOKENS = 10_000;
const CHARS_PER_TOKEN = 4;
const DEFAULT_SEARCH_LIMIT = 8;
/** Pi's custom session entry type that carries one script's `store()` writes. */
const STORE_ENTRY_TYPE = 'codemode-store';
/** What a script receives from a tool without an output schema. */
const TEXT_OUTPUT_SCHEMA = { type: 'string' };

/**
 * Pi's codemode extension, with the script limits in `settings` applied.
 *
 * Pi builds every script's sandbox with a fixed 256 MiB heap and no deadline,
 * and `createCodemodeExtension` takes neither. With a limit set, the extension
 * Pi returns is loaded against an API whose `registerTool` hands Pi's
 * definition to {@link withScriptLimits}. Without a limit, Pi's extension is
 * returned as is. Scripts get no `models` globals either way.
 */
export function createPiCodemodeExtension(
  settings: PiCodemodeSettings,
): ExtensionFactory {
  const extension = createCodemodeExtension({
    ...(settings.mode ? { mode: settings.mode } : {}),
    ...(settings.inlineBudget !== undefined
      ? { inlineBudget: settings.inlineBudget }
      : {}),
    models: false,
  });
  const limits: ScriptLimits = {
    memoryLimitBytes: settings.memoryLimitBytes,
    timeoutMs: settings.timeoutMs,
  };
  if (limits.memoryLimitBytes === undefined && limits.timeoutMs === undefined) {
    return extension;
  }
  return pi =>
    extension({
      ...pi,
      registerTool: tool => {
        if (tool.name === PI_CODEMODE_TOOL_NAME) {
          pi.registerTool(withScriptLimits(tool, pi, limits));
        } else {
          pi.registerTool(tool);
        }
      },
    });
}

/**
 * Pi's codemode definition with its `execute` replaced by
 * {@link executeWithLimits}. The name, schema, description, `prepareLoadout`
 * and call renderer stay Pi's. `renderResult` is dropped: it is typed to Pi's
 * result details, and the harness has no TUI to render into.
 */
function withScriptLimits<TParams extends TSchema, TDetails, TState>(
  tool: ToolDefinition<TParams, TDetails, TState>,
  pi: ExtensionAPI,
  limits: ScriptLimits,
): ToolDefinition<TParams, CodemodeDetails, TState> {
  return {
    ...tool,
    execute: (toolCallId, params, signal, onUpdate, ctx) =>
      executeWithLimits(
        toolCallId,
        readCode(params),
        signal,
        onUpdate,
        ctx,
        pi,
        limits,
      ),
    renderResult: undefined,
  };
}

/**
 * The deadline one script runs under: the shorter of its own `timeout_ms` and
 * the configured default, and none when neither is set.
 */
export function resolveScriptTimeoutMs(
  scriptTimeoutMs: number | undefined,
  defaultTimeoutMs: number | undefined,
): number {
  return Math.min(
    scriptTimeoutMs ?? Number.POSITIVE_INFINITY,
    defaultTimeoutMs ?? Number.POSITIVE_INFINITY,
  );
}

type ScriptLimits = Pick<PiCodemodeSettings, 'memoryLimitBytes' | 'timeoutMs'>;

type CallStatus = 'running' | 'ok' | 'error' | 'cancelled';

/** One nested call as Pi's result details and renderer record it. */
interface CallRecord {
  id: string;
  name: string;
  args: string;
  status: CallStatus;
  durationMs?: number;
  error?: string;
}

interface CodemodeDetails {
  calls: CallRecord[];
}

type CodemodeToolResult = AgentToolResult<CodemodeDetails>;
type ContentItem = CodemodeToolResult['content'][number];
type NestedTool = ExtensionToolContext['tools'][number];
type SessionBranch = ReturnType<
  ExtensionToolContext['sessionManager']['getBranch']
>;

interface StoreWrites {
  set: Record<string, unknown>;
  delete: string[];
}

function readCode(params: unknown): string {
  if (
    typeof params === 'object' &&
    params !== null &&
    'code' in params &&
    typeof params.code === 'string'
  ) {
    return params.code;
  }
  throw new Error('codemode expects { code: string }');
}

/**
 * A typebox schema is a JSON Schema object at runtime; the type is opaque, so
 * the declaration renderer gets it through this check.
 */
function isJsonSchemaObject(
  schema: unknown,
): schema is { [key: string]: unknown } {
  return typeof schema === 'object' && schema !== null;
}

/**
 * Runs one script the way Pi's `executeCodemode` does, in a sandbox built with
 * the configured limits. Mirrors Pi's result format: a `Script completed` or
 * `Script failed` header, the script's output, and nested calls in `details`.
 * Three deliberate differences: `searchTools()` ranks by term overlap instead
 * of BM25, truncated output is not spilled to a host temp file, and the result
 * waits for nested host calls the script abandoned (see below).
 */
async function executeWithLimits(
  toolCallId: string,
  code: string,
  signal: AbortSignal | undefined,
  onUpdate: ((result: CodemodeToolResult) => void) | undefined,
  ctx: ExtensionToolContext,
  pi: ExtensionAPI,
  limits: ScriptLimits,
): Promise<CodemodeToolResult> {
  const startedAt = performance.now();
  const source = parseCodemodeSource(code);
  const calls: CallRecord[] = [];
  const snapshot = (): CodemodeDetails => ({
    calls: calls.map(call => ({ ...call })),
  });
  const publish = () => onUpdate?.({ content: [], details: snapshot() });
  const callable = ctx.tools.filter(
    tool => tool.name !== PI_CODEMODE_TOOL_NAME,
  );
  const samples = new Map(
    callable.map(tool => [
      tool.name,
      renderToolSample({
        name: tool.name,
        description: tool.description,
        inputSchema: isJsonSchemaObject(tool.parameters)
          ? tool.parameters
          : undefined,
        outputSchema: isJsonSchemaObject(tool.outputSchema)
          ? tool.outputSchema
          : TEXT_OUTPUT_SCHEMA,
      }),
    ]),
  );
  const runNestedCall = async (
    tool: NestedTool,
    args: unknown,
    callSignal: AbortSignal,
  ): Promise<unknown> => {
    const record: CallRecord = {
      id: `${toolCallId}/?`,
      name: tool.name,
      args: previewArgs(args),
      status: 'running',
    };
    calls.push(record);
    publish();
    const callStartedAt = performance.now();
    const outcome = await ctx.executeTool(tool.name, args, {
      signal: callSignal,
    });
    record.id = outcome.toolCall.id;
    record.durationMs = performance.now() - callStartedAt;
    // A call the script ended without consuming stays `cancelled`, as in Pi.
    if (record.status === 'running') {
      if (outcome.isError) {
        record.status = callSignal.aborted ? 'cancelled' : 'error';
        record.error = truncateText(
          textOf(outcome.result) || `Tool "${tool.name}" failed`,
          ERROR_PREVIEW_CHARS,
        );
      } else {
        record.status = 'ok';
      }
    }
    publish();
    return toScriptValue(tool, outcome);
  };
  const inFlight = new Set<Promise<unknown>>();
  const sandboxTools: SandboxTool[] = callable.map(tool => ({
    name: tool.name,
    description: samples.get(tool.name),
    execute: (args, { signal: callSignal }) => {
      const call = runNestedCall(tool, args, callSignal);
      inFlight.add(call);
      const settled = () => inFlight.delete(call);
      void call.then(settled, settled);
      return call;
    },
  }));
  const sandbox = new CodemodeSandbox({
    tools: sandboxTools,
    globals: createDiscoveryGlobals(
      callable,
      samples,
      name =>
        pi.getAllTools().find(tool => tool.name === name)?.namespace?.name,
    ),
    timeoutMs: resolveScriptTimeoutMs(
      source.options.timeoutMs,
      limits.timeoutMs,
    ),
    memoryLimitBytes: limits.memoryLimitBytes ?? PI_MEMORY_LIMIT_BYTES,
  });
  let result: CodemodeResult;
  try {
    result = await sandbox.execute(source.code, {
      signal,
      store: readStore(ctx.sessionManager.getBranch()),
    });
  } finally {
    await sandbox.close();
  }
  const wallTime = ((performance.now() - startedAt) / 1000).toFixed(1);
  // Calls still running were cut off by the script ending, a timeout, or an abort.
  for (const call of calls) {
    if (call.status === 'running') call.status = 'cancelled';
  }
  /*
   * Pi's executor returns here, leaving nested host calls the script abandoned
   * running in Pi's pipeline. The harness agent holds the step open until they
   * settle, while Pi starts the next step at once; host calls of that step are
   * then not dispatched until the agent catches up, and a later script's reads
   * died at their own deadline for it. Waiting keeps Pi in step with the agent.
   * The worker, and the memory the script held, were released when the sandbox
   * settled above.
   */
  await Promise.allSettled([...inFlight]);
  const items: ContentItem[] = result.output.map(item =>
    item.type === 'text'
      ? { type: 'text', text: item.text }
      : { type: 'image', data: item.data, mimeType: item.mimeType },
  );
  if (result.ok) {
    const { set, delete: deleted } = result.storeWrites;
    if (Object.keys(set).length > 0 || deleted.length > 0) {
      pi.appendEntry(STORE_ENTRY_TYPE, { set, delete: deleted });
    }
    if (result.value !== undefined) {
      items.push({ type: 'text', text: valueText(result.value) });
    }
  } else {
    items.push({
      type: 'text',
      text: `Script error:\n${formatError(result, calls)}`,
    });
  }
  const output = truncateOutput(
    items,
    source.options.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS,
  );
  const header = `${result.ok ? 'Script completed' : 'Script failed'}\nWall time ${wallTime} seconds\nOutput:\n`;
  return {
    content: [{ type: 'text', text: header }, ...output],
    details: snapshot(),
    ...(result.ok ? {} : { isError: true }),
  };
}

/**
 * The value a script receives for a nested call: a tool that declares
 * `outputSchema` resolves to its `structuredContent`, also for error results
 * that carry one; any other tool resolves to its text content. Other failures
 * reject with the tool's error text.
 */
function toScriptValue(
  tool: NestedTool,
  outcome: Awaited<ReturnType<ExtensionToolContext['executeTool']>>,
): unknown {
  const { result } = outcome;
  if (tool.outputSchema && result.structuredContent !== undefined) {
    return result.structuredContent;
  }
  const text = textOf(result);
  if (outcome.isError) throw new Error(text || `Tool "${tool.name}" failed`);
  return text;
}

/** `searchTools()` and `describeTool()`: ranked search and lookup over the script's nested tools. */
function createDiscoveryGlobals(
  tools: readonly NestedTool[],
  samples: ReadonlyMap<string, string>,
  namespaceOf: (name: string) => string | undefined,
): SandboxTool[] {
  const entry = (tool: NestedTool) => ({
    name: toCodemodeIdentifier(tool.name),
    description: samples.get(tool.name) ?? '',
  });
  return [
    {
      name: 'searchTools',
      spread: true,
      execute: args => {
        const [query, options] = argumentsOf(args);
        if (typeof query !== 'string') {
          throw new Error('searchTools() expects a query string');
        }
        const limit =
          readSearchOption(options, 'limit') ?? DEFAULT_SEARCH_LIMIT;
        if (
          typeof limit !== 'number' ||
          !Number.isInteger(limit) ||
          limit <= 0
        ) {
          throw new Error('searchTools() limit must be a positive integer');
        }
        const namespace = readSearchOption(options, 'namespace');
        if (
          namespace !== undefined &&
          namespace !== null &&
          typeof namespace !== 'string'
        ) {
          throw new Error('searchTools() namespace must be a string');
        }
        const candidates = namespace
          ? tools.filter(tool => namespaceOf(tool.name) === namespace)
          : tools;
        return rankTools(query, candidates, limit).map(entry);
      },
    },
    {
      name: 'describeTool',
      spread: true,
      execute: args => {
        const [name] = argumentsOf(args);
        if (typeof name !== 'string') {
          throw new Error('describeTool() expects a tool name');
        }
        const tool = tools.find(
          candidate =>
            candidate.name === name ||
            toCodemodeIdentifier(candidate.name) === name,
        );
        return tool ? samples.get(tool.name) : undefined;
      },
    },
  ];
}

function argumentsOf(args: unknown): readonly unknown[] {
  return Array.isArray(args) ? args : [];
}

function readSearchOption(
  options: unknown,
  key: 'limit' | 'namespace',
): unknown {
  return typeof options === 'object' && options !== null && key in options
    ? Reflect.get(options, key)
    : undefined;
}

/** Tools whose name or description share the most terms with the query, best first; ties keep tool order. */
function rankTools(
  query: string,
  tools: readonly NestedTool[],
  limit: number,
): NestedTool[] {
  const terms = new Set(termsOf(query));
  return tools
    .map(tool => ({
      tool,
      score: termsOf(`${tool.name} ${tool.description}`).filter(term =>
        terms.has(term),
      ).length,
    }))
    .filter(ranked => ranked.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(ranked => ranked.tool);
}

function termsOf(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
}

/** Values of `load()`: the `codemode-store` entries on the branch, applied from the root. */
function readStore(branch: SessionBranch): Record<string, unknown> {
  const store = new Map<string, unknown>();
  for (const entry of branch) {
    if (
      entry.type !== 'custom' ||
      entry.customType !== STORE_ENTRY_TYPE ||
      !isStoreWrites(entry.data)
    ) {
      continue;
    }
    for (const key of entry.data.delete) store.delete(key);
    for (const [key, value] of Object.entries(entry.data.set)) {
      store.set(key, value);
    }
  }
  return Object.fromEntries(store);
}

function isStoreWrites(data: unknown): data is StoreWrites {
  if (typeof data !== 'object' || data === null) return false;
  const { set, delete: deleted } = data as Partial<StoreWrites>;
  return (
    typeof set === 'object' &&
    set !== null &&
    Array.isArray(deleted) &&
    deleted.every(key => typeof key === 'string')
  );
}

function formatError(
  result: Extract<CodemodeResult, { ok: false }>,
  calls: readonly CallRecord[],
): string {
  const { error } = result;
  const head =
    error.kind === 'script'
      ? (error.stack ?? `${error.name ?? 'Error'}: ${error.message}`)
      : error.kind === 'timeout'
        ? `Script timed out: ${error.message}`
        : error.kind === 'aborted'
          ? `Script aborted: ${error.message}`
          : `Script sandbox failed: ${error.message}`;
  const summary =
    calls.length === 0
      ? 'No tool calls were made.'
      : `Tool calls made before the failure (they are not undone): ${calls
          .map(call => `${call.name} (${call.status})`)
          .join(', ')}`;
  return `${head}\n\n${summary}`;
}

/**
 * Pi's token budget for script output: past it, the text items become one
 * item keeping the start and end of the text, and images follow it.
 */
function truncateOutput(
  items: readonly ContentItem[],
  maxTokens: number,
): ContentItem[] {
  const texts = items.flatMap(item =>
    item.type === 'text' ? [item.text] : [],
  );
  const combined = texts.join('\n');
  const budget = maxTokens * CHARS_PER_TOKEN;
  if (texts.length === 0 || combined.length <= budget) return [...items];
  const headChars = Math.floor(budget / 2);
  const tailChars = budget - headChars;
  const removed = combined.length - headChars - tailChars;
  const text = `Warning: truncated output (original token count: ${Math.ceil(
    combined.length / CHARS_PER_TOKEN,
  )})\nTotal output lines: ${combined.split('\n').length}\n\n${combined.slice(
    0,
    headChars,
  )}…${Math.ceil(removed / CHARS_PER_TOKEN)} tokens truncated…${
    tailChars > 0 ? combined.slice(-tailChars) : ''
  }`;
  return [
    { type: 'text', text },
    ...items.filter(item => item.type === 'image'),
  ];
}

/** Like the script's `text()`: strings as is, other values as compact JSON. */
function valueText(value: unknown): string {
  if (typeof value === 'string') return value;
  return JSON.stringify(value) ?? String(value);
}

function textOf(result: AgentToolResult<unknown>): string {
  return result.content
    .flatMap(block => (block.type === 'text' ? [block.text] : []))
    .join('\n');
}

function previewArgs(args: unknown): string {
  if (args === undefined) return '';
  try {
    return truncateText(JSON.stringify(args) ?? '', ARGS_PREVIEW_CHARS);
  } catch {
    return '';
  }
}

function truncateText(text: string, maxChars: number): string {
  return text.length > maxChars ? `${text.slice(0, maxChars - 3)}...` : text;
}
