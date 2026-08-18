import { randomBytes } from 'node:crypto';
import type { HarnessV1StreamPart } from '@ai-sdk/harness';
import { secureJsonParse } from '@ai-sdk/provider-utils';
import { extractAssistantText, type PiSessionEvent } from './pi-events';
import { serializeToolOutput } from './pi-utils';

/**
 * Translator state shared across all events of a single turn. Reset at the
 * start of every `doPromptTurn`. Callers update the same instance and read it to
 * decide when a turn has settled into a steady state (e.g. for gap-filling).
 */
export interface PiTranslatorState {
  /**
   * True once a `turn_start` or assistant `message_start` event has been
   * observed. Suppresses spurious deltas that arrive before the turn opens.
   */
  promptStarted: boolean;
  /** Accumulated assistant text from `text_delta` events. */
  streamedAssistantText: string;
  /**
   * The most recent `stopReason` Pi reported on an assistant message
   * (`message_end` / `turn_end`). Carried onto the harness finish reason so
   * consumers see the provider's real evidence (`length`, `toolUse`, …)
   * instead of a fabricated `stop`. `undefined` until Pi reports one; the
   * finish reason is then an inferred `stop` marked as such in metadata.
   */
  lastStopReason: string | undefined;
  /** Stream-part id for the active text block; synthesized on first delta. */
  currentTextId: string | undefined;
  /**
   * Stream-part id for the active reasoning block; synthesized lazily on
   * first `thinking_delta`.
   */
  currentReasoningId: string | undefined;
  /** Whether a `reasoning-start` event has already been emitted. */
  reasoningStarted: boolean;
  /** Tool-call id → tool name (used to fill in `toolName` on results). */
  observedToolNames: Map<string, string>;
  /** Tool ids requested by the current assistant message but not yet completed. */
  pendingStepToolCallIds: Set<string>;
  /** Whether the current assistant message has opened a visible step. */
  stepOpen: boolean;
  /** Exact provider usage from the assistant message that opened the step. */
  pendingStepUsage: PiStepUsage | undefined;
  /**
   * Tool-call id → the exact output value the host submitted for a
   * user-registered (host-executed) tool. Pi only echoes the tool result back
   * as serialized text (the tool handler stringifies the output before handing
   * it to the runtime so the model can read it), which would otherwise reach
   * consumers as a string and lose the original object structure. Keeping the
   * submitted value here lets the result projection surface the original object
   * — matching the other adapters — while the model still receives the text.
   * Populated by the session's `submitToolResult`; consumed (and cleared) when
   * the matching `tool_result`/`tool_execution_end` event is translated.
   */
  hostToolResults: Map<string, unknown>;
  dynamicToolCallIds: Set<string>;
  /**
   * Names of tools that Pi executes natively (read/write/edit/bash/grep/
   * find/ls). `tool-call` events for these get `providerExecuted: true`
   * so the harness host doesn't try to dispatch them. User-registered
   * tools are not in this set.
   */
  readonly builtinToolNames: ReadonlySet<string>;
  readonly hostToolNames: ReadonlySet<string>;
  /**
   * Map of native tool name → common name. `find` → `glob`, etc. Pi emits
   * native names on its events; the wire `toolName` is the common name when
   * one exists.
   */
  readonly nativeToCommonNameMap: ReadonlyMap<string, string>;
}

export interface PiTranslatorStateOptions {
  readonly builtinToolNames?: ReadonlyArray<string>;
  readonly hostToolNames?: ReadonlyArray<string>;
  readonly nativeToCommon?:
    | ReadonlyMap<string, string>
    | Record<string, string>;
}

export function createPiTranslatorState(
  options: PiTranslatorStateOptions = {},
): PiTranslatorState {
  const map =
    options.nativeToCommon instanceof Map
      ? options.nativeToCommon
      : new Map(Object.entries(options.nativeToCommon ?? {}));
  return {
    promptStarted: false,
    streamedAssistantText: '',
    lastStopReason: undefined,
    currentTextId: undefined,
    currentReasoningId: undefined,
    reasoningStarted: false,
    observedToolNames: new Map(),
    pendingStepToolCallIds: new Set(),
    stepOpen: false,
    pendingStepUsage: undefined,
    hostToolResults: new Map(),
    dynamicToolCallIds: new Set(),
    builtinToolNames: new Set(options.builtinToolNames ?? []),
    hostToolNames: new Set(options.hostToolNames ?? []),
    nativeToCommonNameMap: map,
  };
}

function newId(): string {
  return randomBytes(8).toString('hex');
}

/**
 * Pi's `tool_execution_end` event payload (`result`) is a Pi `AgentToolResult`
 * envelope `{ content: (TextContent | ImageContent)[], details, terminate? }`.
 * The `tool_result` event uses a flat shape with `content` and `details` at
 * the top level. In both cases we extract just the text payload (joined when
 * multiple text parts are present) so the AI SDK consumer sees the raw
 * string the tool produced.
 */
function unwrapPiToolResult(event: PiSessionEvent): never {
  const candidates: unknown[] = [];
  const result = event.result as unknown;
  if (result && typeof result === 'object') {
    const inner = (result as { content?: unknown }).content;
    if (Array.isArray(inner)) candidates.push(inner);
  }
  if (Array.isArray(event.content)) candidates.push(event.content);

  for (const content of candidates) {
    if (!Array.isArray(content)) continue;
    const text = content
      .filter(
        (p): p is { type: 'text'; text: string } =>
          !!p &&
          typeof p === 'object' &&
          (p as { type?: unknown }).type === 'text' &&
          typeof (p as { text?: unknown }).text === 'string',
      )
      .map(p => p.text)
      .join('');
    if (text) return text as never;
  }

  if (typeof event.result === 'string') return event.result as never;
  if (typeof event.content === 'string') return event.content as never;
  return (event.result ?? event.content ?? null) as never;
}

function parseMcpToolResult(content: unknown): unknown {
  if (typeof content !== 'string') return content;
  try {
    return secureJsonParse(content);
  } catch {
    return content;
  }
}

function resolveToolName(
  state: PiTranslatorState,
  nativeName: string,
): { wire: string; native: string } {
  const common = state.nativeToCommonNameMap.get(nativeName);
  return { wire: common ?? nativeName, native: nativeName };
}

type PiStepUsage = Extract<
  HarnessV1StreamPart,
  { type: 'finish-step' }
>['usage'];

function emptyStepUsage(): PiStepUsage {
  return {
    inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 0, text: 0, reasoning: 0 },
  };
}

type PiFinishReason = Extract<
  HarnessV1StreamPart,
  { type: 'finish' }
>['finishReason'];

/**
 * Pi's `stopReason` vocabulary mapped onto the unified harness finish
 * reasons. Values Pi may add later fall through to `other` with the raw
 * string preserved.
 */
const PI_STOP_REASON_TO_UNIFIED: Record<string, PiFinishReason['unified']> = {
  aborted: 'other',
  error: 'error',
  length: 'length',
  stop: 'stop',
  toolUse: 'tool-calls',
};

/**
 * The harness finish reason for the turn, mapped from the last `stopReason`
 * Pi reported on an assistant message. `raw` always carries Pi's actual
 * string — never a fabricated value. When Pi reported no reason at all the
 * result is `stop` with `raw: undefined` and `inferred: true`, which callers
 * record as metadata so consumers can tell evidence from a guess.
 */
export function piTurnFinishReason(
  state: Pick<PiTranslatorState, 'lastStopReason'> | undefined,
): { finishReason: PiFinishReason; inferred: boolean } {
  const stopReason = state?.lastStopReason;
  if (!stopReason) {
    return {
      finishReason: { unified: 'stop', raw: undefined },
      inferred: true,
    };
  }
  return {
    finishReason: {
      unified: PI_STOP_REASON_TO_UNIFIED[stopReason] ?? 'other',
      raw: stopReason,
    },
    inferred: false,
  };
}

function finishStep(
  state: PiTranslatorState,
  usage?: PiStepUsage,
): HarnessV1StreamPart[] {
  if (!state.stepOpen || state.pendingStepToolCallIds.size > 0) return [];
  state.stepOpen = false;
  state.pendingStepToolCallIds.clear();
  const resolvedUsage = state.pendingStepUsage ?? usage ?? emptyStepUsage();
  state.pendingStepUsage = undefined;
  const { finishReason, inferred } = piTurnFinishReason(state);
  return [
    {
      type: 'finish-step',
      finishReason,
      usage: resolvedUsage,
      harnessMetadata: {
        pi: {
          inferredStep: true,
          ...(inferred ? { inferredFinishReason: true } : {}),
        },
      },
    },
  ];
}

export function finishPiApprovalStep(
  state: PiTranslatorState,
  toolCallId: string,
  usage: PiStepUsage = emptyStepUsage(),
): HarnessV1StreamPart[] {
  state.stepOpen = true;
  state.pendingStepToolCallIds.delete(toolCallId);
  return finishStep(state, usage);
}

export function consumePendingPiStepUsage(
  state: PiTranslatorState,
): PiStepUsage | undefined {
  const usage = state.pendingStepUsage;
  state.pendingStepUsage = undefined;
  return usage;
}

function extractPiStepUsage(
  message: PiSessionEvent['message'],
): PiStepUsage | undefined {
  const usage = message?.usage;
  if (!usage) return undefined;
  return {
    inputTokens: {
      total: usage.input + usage.cacheRead + usage.cacheWrite,
      noCache: usage.input,
      cacheRead: usage.cacheRead,
      cacheWrite: usage.cacheWrite,
    },
    outputTokens: {
      total: usage.output,
      text:
        usage.reasoning === undefined
          ? undefined
          : Math.max(0, usage.output - usage.reasoning),
      reasoning: usage.reasoning,
    },
  };
}

/**
 * Close whatever the turn left open — active text/reasoning blocks and the
 * step itself — so the session layer can emit a terminal `finish` without
 * violating the harness contract (`finish-step` must precede `finish`).
 *
 * Pi resolves `session.prompt()` without a `turn_end` event when a turn ends
 * abnormally but not with a recognized terminal error (e.g. the provider
 * stream is cut mid-message). Without this flush the streamed content stays
 * attributed to an unfinished step and `HarnessAgent` fails the whole turn,
 * discarding text the user already saw. Returns an empty array when the turn
 * closed normally, so callers can emit it unconditionally.
 */
export function flushOpenStep(state: PiTranslatorState): HarnessV1StreamPart[] {
  const parts: HarnessV1StreamPart[] = [];
  if (state.currentTextId) {
    parts.push({ type: 'text-end', id: state.currentTextId });
    state.currentTextId = undefined;
  }
  if (state.reasoningStarted && state.currentReasoningId) {
    parts.push({ type: 'reasoning-end', id: state.currentReasoningId });
    state.reasoningStarted = false;
    state.currentReasoningId = undefined;
  }
  // An open block proves step content exists even if no assistant
  // `message_start` ever set `stepOpen` (deltas alone open a step downstream).
  if (parts.length > 0) {
    state.stepOpen = true;
  }
  // Tool calls whose results never arrived cannot complete anymore — the turn
  // is over. Clear them so the step can close.
  state.pendingStepToolCallIds.clear();
  return [...parts, ...finishStep(state)];
}

function extractPiToolCallIds(message: PiSessionEvent['message']): string[] {
  if (!message || message.role !== 'assistant') return [];
  if (!Array.isArray(message.content)) return [];
  return message.content.flatMap(part => {
    if (!part || typeof part !== 'object') return [];
    const block = part as Record<string, unknown>;
    if (block.type !== 'toolCall') return [];
    const id = block.id ?? block.toolCallId;
    return typeof id === 'string' && id.length > 0 ? [id] : [];
  });
}

/**
 * Translate a single Pi `session.subscribe` event into zero or more
 * `HarnessV1StreamPart`s, updating the translator state in place. Returns
 * an empty array for events that produce no output (e.g. events emitted
 * before `turn_start`).
 *
 * The translator does NOT emit `stream-start`/`finish` — those are
 * lifecycle signals owned by the session layer.
 */
export function translatePiEvent(
  event: PiSessionEvent,
  state: PiTranslatorState,
): HarnessV1StreamPart[] {
  switch (event.type) {
    case 'turn_start':
    case 'message_start': {
      if (
        event.type === 'message_start' &&
        event.message?.role !== 'assistant'
      ) {
        return [];
      }
      state.promptStarted = true;
      if (event.type === 'message_start') {
        state.stepOpen = true;
        state.pendingStepToolCallIds.clear();
      }
      state.streamedAssistantText = '';
      state.lastStopReason = undefined;
      state.currentTextId = undefined;
      state.currentReasoningId = undefined;
      state.reasoningStarted = false;
      return [];
    }

    case 'message_update': {
      if (!state.promptStarted) return [];
      const update = event.assistantMessageEvent;
      if (!update) return [];
      if (update.type === 'text_delta' && typeof update.delta === 'string') {
        const parts: HarnessV1StreamPart[] = [];
        // If reasoning was active, close it before opening the text block so
        // consumers can reset block-scoped formatting (ANSI colors, etc.)
        // between sections.
        if (state.reasoningStarted && state.currentReasoningId) {
          parts.push({
            type: 'reasoning-end',
            id: state.currentReasoningId,
          });
          state.reasoningStarted = false;
          state.currentReasoningId = undefined;
        }
        if (!state.currentTextId) {
          state.currentTextId = newId();
          parts.push({ type: 'text-start', id: state.currentTextId });
        }
        state.streamedAssistantText += update.delta;
        parts.push({
          type: 'text-delta',
          id: state.currentTextId,
          delta: update.delta,
        });
        return parts;
      }
      if (
        update.type === 'thinking_delta' &&
        typeof update.delta === 'string'
      ) {
        const parts: HarnessV1StreamPart[] = [];
        // Symmetric to the text branch: close any open text block before
        // starting a fresh reasoning block.
        if (state.currentTextId) {
          parts.push({ type: 'text-end', id: state.currentTextId });
          state.currentTextId = undefined;
        }
        if (!state.currentReasoningId) {
          state.currentReasoningId = newId();
        }
        if (!state.reasoningStarted) {
          state.reasoningStarted = true;
          parts.push({ type: 'reasoning-start', id: state.currentReasoningId });
        }
        parts.push({
          type: 'reasoning-delta',
          id: state.currentReasoningId,
          delta: update.delta,
        });
        return parts;
      }
      return [];
    }

    case 'message_end':
    case 'turn_end': {
      if (!state.promptStarted) return [];
      const parts: HarnessV1StreamPart[] = [];
      state.pendingStepUsage =
        extractPiStepUsage(event.message) ?? state.pendingStepUsage;
      const stopReason = event.message?.stopReason?.trim();
      if (stopReason) {
        state.lastStopReason = stopReason;
      }
      const fullText = extractAssistantText(event.message);
      // Text the final message carries beyond what was streamed as deltas.
      // Covers both the gap-fill case (deltas were a prefix) and the
      // final-only case (no deltas at all — `streamedAssistantText` is ''),
      // which previously lost the entire response because emission was gated
      // on an already-open text block. Emitting only a proven suffix is what
      // keeps already-streamed text from ever being duplicated — including
      // on the `turn_end` that follows a `message_end` for the same message.
      const missingText =
        fullText.startsWith(state.streamedAssistantText) &&
        fullText.length > state.streamedAssistantText.length
          ? fullText.slice(state.streamedAssistantText.length)
          : '';
      let textId = state.currentTextId;
      if (missingText) {
        if (!textId) {
          // Mirror the delta handler: close an open reasoning block before
          // opening a text block, so the two never interleave.
          if (state.reasoningStarted && state.currentReasoningId) {
            parts.push({
              type: 'reasoning-end',
              id: state.currentReasoningId,
            });
            state.reasoningStarted = false;
            state.currentReasoningId = undefined;
          }
          textId = newId();
          state.currentTextId = textId;
          parts.push({ type: 'text-start', id: textId });
        }
        state.streamedAssistantText = fullText;
        parts.push({ type: 'text-delta', id: textId, delta: missingText });
      }
      if (textId) {
        parts.push({ type: 'text-end', id: textId });
        state.currentTextId = undefined;
      }
      if (state.reasoningStarted && state.currentReasoningId) {
        parts.push({ type: 'reasoning-end', id: state.currentReasoningId });
        state.reasoningStarted = false;
        state.currentReasoningId = undefined;
      }
      if (event.type === 'message_end') {
        for (const toolCallId of extractPiToolCallIds(event.message)) {
          state.pendingStepToolCallIds.add(toolCallId);
        }
      } else {
        state.pendingStepToolCallIds.clear();
        parts.push(...finishStep(state));
      }
      return parts;
    }

    case 'tool_execution_start': {
      if (!event.toolCallId || !event.toolName) return [];
      const { wire, native } = resolveToolName(state, event.toolName);
      state.observedToolNames.set(event.toolCallId, wire);
      const isMcpTool =
        !state.hostToolNames.has(native) &&
        (native === 'mcp' || native.startsWith('mcp__'));
      const providerExecuted = state.builtinToolNames.has(native) || isMcpTool;
      if (isMcpTool) state.dynamicToolCallIds.add(event.toolCallId);
      const input = serializeToolOutput(event.args ?? event.input ?? {});
      return [
        {
          type: 'tool-call',
          toolCallId: event.toolCallId,
          toolName: wire,
          input,
          ...(wire !== native ? { nativeName: native } : {}),
          ...(providerExecuted ? { providerExecuted: true } : {}),
          ...(!providerExecuted && state.pendingStepUsage
            ? { usage: state.pendingStepUsage }
            : {}),
          ...(isMcpTool ? { dynamic: true } : {}),
        } as HarnessV1StreamPart,
      ];
    }

    case 'tool_execution_end':
    case 'tool_result': {
      if (!event.toolCallId) return [];
      const recordedName = state.observedToolNames.get(event.toolCallId);
      const nativeName = event.toolName;
      const wire =
        recordedName ??
        (nativeName ? resolveToolName(state, nativeName).wire : undefined);
      if (!wire) return [];
      const dynamic = state.dynamicToolCallIds.delete(event.toolCallId);
      /*
       * Prefer the exact value the host submitted for user-registered tools
       * (see `hostToolResults`). Built-in tools, whose results Pi produces and
       * reports as text, are not in the map and fall back to unwrapping the
       * event's text payload.
       */
      const result = state.hostToolResults.has(event.toolCallId)
        ? ((state.hostToolResults.get(event.toolCallId) ?? null) as Extract<
            HarnessV1StreamPart,
            { type: 'tool-result' }
          >['result'])
        : dynamic
          ? parseMcpToolResult(unwrapPiToolResult(event))
          : unwrapPiToolResult(event);
      state.hostToolResults.delete(event.toolCallId);
      state.pendingStepToolCallIds.delete(event.toolCallId);
      return [
        {
          type: 'tool-result',
          toolCallId: event.toolCallId,
          toolName: wire,
          result,
          ...(event.isError ? { isError: true } : {}),
          ...(dynamic ? { dynamic: true } : {}),
        } as HarnessV1StreamPart,
        ...finishStep(state),
      ];
    }

    case 'compaction_end': {
      /*
       * Pi performs the compaction itself; we observe its result. Skip aborted
       * or result-less compactions (nothing happened). A result with no summary
       * still represents a real compaction, so emit it with a placeholder
       * rather than dropping the event. `reason` is `'manual'` for an explicit
       * `session.compact()` call, `'threshold'`/`'overflow'` for Pi's automatic
       * compaction — both map to `'auto'` on the wire. Pi reports `tokensBefore`
       * but not `tokensAfter`.
       */
      if (event.aborted) return [];
      const result = event.result;
      if (!result || typeof result !== 'object') return [];
      const rawSummary = (result as { summary?: unknown }).summary;
      const summary =
        typeof rawSummary === 'string' ? rawSummary : '(no summary provided)';
      const tokensBefore = (result as { tokensBefore?: unknown }).tokensBefore;
      return [
        {
          type: 'compaction',
          trigger: event.reason === 'manual' ? 'manual' : 'auto',
          summary,
          ...(typeof tokensBefore === 'number' ? { tokensBefore } : {}),
        },
      ];
    }

    default:
      return [];
  }
}
