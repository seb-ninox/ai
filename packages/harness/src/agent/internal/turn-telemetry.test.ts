import type { Telemetry } from 'ai';
import { createNullLanguageModelUsage, DefaultStepResult } from 'ai/internal';
import { describe, expect, test, vi } from 'vitest';
import { createTurnLifecycle } from './turn-telemetry';

describe('createTurnLifecycle', () => {
  test('includes the current stepNumber on onStepEnd events', async () => {
    const stepStartNumbers: number[] = [];
    const stepEndNumbers: number[] = [];
    const integration = {
      onStepStart: event => {
        stepStartNumbers.push(event.stepNumber);
      },
      onStepEnd: event => {
        stepEndNumbers.push(event.stepNumber);
      },
    } satisfies Telemetry;
    const lifecycle = createTurnLifecycle({
      callId: 'call-1',
      telemetry: { integrations: [integration] },
      callbacks: {},
      harnessId: 'mock',
      modelId: 'mock-model',
      instructions: undefined,
      tools: {},
      toolsContext: {},
      activeToolNames: [],
      toolSpecs: [],
      messages: [{ role: 'user', content: 'go' }],
      runtimeContext: {},
      output: undefined,
    });

    for (const stepNumber of [0, 1]) {
      await lifecycle.ensureStepOpen();
      await lifecycle.stepEnd(
        new DefaultStepResult({
          callId: 'call-1',
          stepNumber,
          provider: 'harness:mock',
          modelId: 'mock-model',
          runtimeContext: {},
          toolsContext: {},
          content: [],
          finishReason: 'stop',
          rawFinishReason: 'stop',
          usage: createNullLanguageModelUsage(),
          performance: {
            effectiveOutputTokensPerSecond: 0,
            outputTokensPerSecond: undefined,
            inputTokensPerSecond: undefined,
            effectiveTotalTokensPerSecond: 0,
            stepTimeMs: 0,
            responseTimeMs: 0,
            toolExecutionMs: {},
            timeToFirstOutputMs: undefined,
          },
          warnings: undefined,
          request: {},
          response: {
            id: `response-${stepNumber}`,
            modelId: 'mock-model',
            timestamp: new Date(0),
            messages: [],
          },
          providerMetadata: undefined,
        }),
      );
    }

    expect(stepStartNumbers).toEqual([0, 1]);
    expect(stepEndNumbers).toEqual([0, 1]);
  });

  test('reports time to the first generated output', async () => {
    const performanceEvents: unknown[] = [];
    const integration = {
      onLanguageModelCallEnd: event => {
        performanceEvents.push(event.performance);
      },
    } satisfies Telemetry;

    vi.useFakeTimers();
    try {
      const lifecycle = createTurnLifecycle({
        callId: 'call-1',
        telemetry: { integrations: [integration] },
        callbacks: {},
        harnessId: 'mock',
        modelId: 'mock-model',
        instructions: undefined,
        tools: {},
        toolsContext: {},
        activeToolNames: [],
        toolSpecs: [],
        messages: [{ role: 'user', content: 'go' }],
        runtimeContext: {},
        output: undefined,
      });

      await lifecycle.start();
      await lifecycle.ensureStepOpen();
      vi.advanceTimersByTime(42);
      lifecycle.markOutput();
      await lifecycle.languageModelCallEnd({
        finishReason: 'stop',
        usage: createNullLanguageModelUsage(),
        content: [{ type: 'text', text: 'done' }],
        providerMetadata: undefined,
      });
      await lifecycle.stepEnd(
        new DefaultStepResult({
          callId: 'call-1',
          stepNumber: 0,
          provider: 'harness:mock',
          modelId: 'mock-model',
          runtimeContext: {},
          toolsContext: {},
          content: [],
          finishReason: 'stop',
          rawFinishReason: 'stop',
          usage: createNullLanguageModelUsage(),
          performance: {
            effectiveOutputTokensPerSecond: 0,
            outputTokensPerSecond: undefined,
            inputTokensPerSecond: undefined,
            effectiveTotalTokensPerSecond: 0,
            stepTimeMs: 0,
            responseTimeMs: 0,
            toolExecutionMs: {},
            timeToFirstOutputMs: undefined,
          },
          warnings: undefined,
          request: {},
          response: {
            id: 'response-0',
            modelId: 'mock-model',
            timestamp: new Date(0),
            messages: [],
          },
          providerMetadata: undefined,
        }),
      );

      await lifecycle.ensureStepOpen();
      lifecycle.markOutput();
      await lifecycle.languageModelCallEnd({
        finishReason: 'stop',
        usage: createNullLanguageModelUsage(),
        content: [{ type: 'text', text: 'still done' }],
        providerMetadata: undefined,
      });

      expect(performanceEvents).toEqual([
        expect.objectContaining({ timeToFirstOutputMs: 42 }),
        expect.objectContaining({ timeToFirstOutputMs: undefined }),
      ]);
    } finally {
      vi.useRealTimers();
    }
  });

  test('leaves time to first output absent when the step has no output', async () => {
    const performanceEvents: unknown[] = [];
    const integration = {
      onLanguageModelCallEnd: event => {
        performanceEvents.push(event.performance);
      },
    } satisfies Telemetry;

    const lifecycle = createTurnLifecycle({
      callId: 'call-1',
      telemetry: { integrations: [integration] },
      callbacks: {},
      harnessId: 'mock',
      modelId: 'mock-model',
      instructions: undefined,
      tools: {},
      toolsContext: {},
      activeToolNames: [],
      toolSpecs: [],
      messages: [{ role: 'user', content: 'go' }],
      runtimeContext: {},
      output: undefined,
    });

    await lifecycle.start();
    await lifecycle.ensureStepOpen();
    await lifecycle.languageModelCallEnd({
      finishReason: 'stop',
      usage: createNullLanguageModelUsage(),
      content: [],
      providerMetadata: undefined,
    });

    expect(performanceEvents).toEqual([
      expect.objectContaining({ timeToFirstOutputMs: undefined }),
    ]);
  });

  test('hands telemetry a tool error it can name, and host callbacks the raw value', async () => {
    const telemetryErrors: unknown[] = [];
    const callbackErrors: unknown[] = [];
    const errorOf = (event: { toolOutput: unknown }) =>
      (event.toolOutput as { error?: unknown }).error;
    const lifecycle = createTurnLifecycle({
      callId: 'call-1',
      telemetry: {
        integrations: [
          {
            onToolExecutionEnd: event => {
              telemetryErrors.push(errorOf(event));
            },
          } satisfies Telemetry,
        ],
      },
      callbacks: {
        onToolExecutionEnd: event => {
          callbackErrors.push(errorOf(event));
        },
      },
      harnessId: 'mock',
      modelId: 'mock-model',
      instructions: undefined,
      tools: {},
      toolsContext: {},
      activeToolNames: [],
      toolSpecs: [],
      messages: [{ role: 'user', content: 'go' }],
      runtimeContext: {},
      output: undefined,
    });
    const toolCall = {
      type: 'tool-call',
      toolCallId: 'c1',
      toolName: 'echo',
      input: { value: 1 },
    } as never;
    const thrown = new Error('host tool threw');

    await lifecycle.start();
    for (const error of ['Validation failed', { error: 'wrapped' }, thrown]) {
      await lifecycle.toolExecutionEnd({
        toolCall,
        toolOutput: {
          ...(toolCall as object),
          type: 'tool-error',
          error,
        } as never,
        toolExecutionMs: 0,
      });
    }

    expect(callbackErrors).toEqual([
      'Validation failed',
      { error: 'wrapped' },
      thrown,
    ]);
    expect(telemetryErrors.map(error => (error as Error).message)).toEqual([
      'Validation failed',
      'wrapped',
      'host tool threw',
    ]);
    expect(telemetryErrors[0]).toMatchObject({ result: 'Validation failed' });
    expect(telemetryErrors[2]).toBe(thrown);
  });

  test('settles a paused tool call as pending rather than as an error', async () => {
    const outputs: unknown[] = [];
    const startedToolCallIds: string[] = [];
    const integration = {
      onToolExecutionStart: event => {
        startedToolCallIds.push(event.toolCall.toolCallId);
      },
      onToolExecutionEnd: event => {
        outputs.push(event.toolOutput);
      },
    } satisfies Telemetry;

    const lifecycle = createTurnLifecycle({
      callId: 'call-1',
      telemetry: { integrations: [integration] },
      callbacks: {},
      harnessId: 'mock',
      modelId: 'mock-model',
      instructions: undefined,
      tools: {},
      toolsContext: {},
      activeToolNames: [],
      toolSpecs: [],
      messages: [{ role: 'user', content: 'go' }],
      runtimeContext: {},
      output: undefined,
    });

    await lifecycle.start();
    await lifecycle.pauseOpenTools({
      toolCalls: [
        {
          type: 'tool-call',
          toolCallId: 'call-1',
          toolName: 'ask_user',
          input: { question: 'which app?' },
          dynamic: true,
        },
      ] as never,
    });

    expect(startedToolCallIds).toEqual(['call-1']);
    expect(outputs).toEqual([
      {
        type: 'tool-result',
        toolCallId: 'call-1',
        toolName: 'ask_user',
        input: { question: 'which app?' },
        dynamic: true,
        output: { status: 'awaiting-host-input' },
      },
    ]);
  });
});
