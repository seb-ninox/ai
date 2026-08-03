import { tool } from '@ai-sdk/provider-utils';
import type { TextStreamPart, ToolSet } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { z } from 'zod/v4';

/*
 * `parseToolCall` reports an unparseable tool call in one of two ways depending on the `ai`
 * version the harness RUNS against: the workspace version returns an `invalid: true` part
 * (covered in `validate-tool-call.test.ts`), while published `ai@7.x` throws
 * `AI_InvalidToolInputError`. Only the returning shape was ever handled, so a host on the
 * published package got the throw out of a stream transform with no handler — which errors the
 * consumer stream and fails the whole turn. A model that omitted one required field then had no
 * way to learn that and retry; the user saw a generic "could not be completed".
 *
 * This file pins the throwing version's behaviour by mocking the throw, since the installed
 * package here cannot produce it.
 */
vi.mock('ai/internal', async importOriginal => {
  const original = await importOriginal<typeof import('ai/internal')>();
  return {
    ...original,
    parseToolCall: vi.fn(() => {
      throw new Error('AI_InvalidToolInputError: expectedRevision is required');
    }),
  };
});

const { validateToolCall } = await import('./run-prompt');

const tools = {
  modify_schema: tool({
    description: 'Apply a batch of data model changes',
    inputSchema: z.object({ expectedRevision: z.number() }),
  }),
} as const satisfies ToolSet;

type ToolCall = Extract<TextStreamPart<typeof tools>, { type: 'tool-call' }>;

describe('validateToolCall against an `ai` version that throws', () => {
    it('returns the documented invalid part instead of throwing into the stream', async () => {
        const result = await validateToolCall<typeof tools>({
            event: {
                type: 'tool-call',
                toolCallId: 'c1',
                toolName: 'modify_schema',
                input: '{"create":[{"kind":"App","name":"gym_routine"}]}',
            },
            tools,
        });

        expect(result).toMatchObject({
            type: 'tool-call',
            toolCallId: 'c1',
            toolName: 'modify_schema',
            invalid: true,
        });
        // The raw input survives, so the transcript still shows what the model actually sent and
        // the tool boundary can reject it with a message the model can act on.
        expect((result as ToolCall & { input: unknown }).input).toEqual({
            create: [{ kind: 'App', name: 'gym_routine' }],
        });
        expect((result as ToolCall & { error?: unknown }).error).toBeInstanceOf(Error);
        // A declared tool stays non-dynamic; only an unknown tool is dynamic.
        expect((result as ToolCall & { dynamic?: boolean }).dynamic).toBeUndefined();
    });

    it('marks an unknown tool dynamic so the consumer can still render it', async () => {
        const result = await validateToolCall<typeof tools>({
            event: {
                type: 'tool-call',
                toolCallId: 'c2',
                toolName: 'not_a_tool',
                input: 'not json at all',
            },
            tools,
        });

        expect(result).toMatchObject({
            type: 'tool-call',
            toolCallId: 'c2',
            toolName: 'not_a_tool',
            invalid: true,
            dynamic: true,
        });
        // Unparseable JSON is passed through verbatim rather than dropped.
        expect((result as ToolCall & { input: unknown }).input).toBe('not json at all');
    });
});
