import type { Telemetry } from 'ai';
import { describe, expect, test, vi } from 'vitest';
import { createTurnTelemetry } from './turn-telemetry';

const usage = {
  inputTokens: {
    total: 1,
    noCache: 1,
    cacheRead: undefined,
    cacheWrite: undefined,
  },
  outputTokens: { total: 1, text: 1, reasoning: undefined },
};

describe('createTurnTelemetry', () => {
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

    const telemetry = createTurnTelemetry({
      telemetry: { integrations: [integration] },
      harnessId: 'mock',
      modelId: 'mock-model',
      instructions: undefined,
      promptText: 'go',
      runtimeContext: undefined,
    });

    await telemetry.start();
    await telemetry.ensureStepOpen();
    await telemetry.stepFinish({
      finishReason: { unified: 'stop', raw: 'stop' },
      usage,
      content: [{ type: 'text', text: 'done' }],
    });

    await telemetry.ensureStepOpen();
    await telemetry.end({
      finishReason: { unified: 'stop', raw: 'stop' },
      usage,
    });

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
      const telemetry = createTurnTelemetry({
        telemetry: { integrations: [integration] },
        harnessId: 'mock',
        modelId: 'mock-model',
        instructions: undefined,
        promptText: 'go',
        runtimeContext: undefined,
      });

      await telemetry.start();
      await telemetry.ensureStepOpen();
      vi.advanceTimersByTime(42);
      telemetry.markOutput();
      await telemetry.stepFinish({
        finishReason: { unified: 'stop', raw: 'stop' },
        usage,
        content: [{ type: 'text', text: 'done' }],
      });

      await telemetry.ensureStepOpen();
      telemetry.markOutput();
      await telemetry.stepFinish({
        finishReason: { unified: 'stop', raw: 'stop' },
        usage,
        content: [{ type: 'text', text: 'still done' }],
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

    const telemetry = createTurnTelemetry({
      telemetry: { integrations: [integration] },
      harnessId: 'mock',
      modelId: 'mock-model',
      instructions: undefined,
      promptText: 'go',
      runtimeContext: undefined,
    });

    await telemetry.start();
    await telemetry.ensureStepOpen();
    await telemetry.stepFinish({
      finishReason: { unified: 'stop', raw: 'stop' },
      usage,
      content: [],
    });

    expect(performanceEvents).toEqual([
      expect.objectContaining({ timeToFirstOutputMs: undefined }),
    ]);
  });

  test('settles a paused tool call as pending rather than as an error', async () => {
    const outputs: unknown[] = [];
    const integration = {
      onToolExecutionEnd: event => {
        outputs.push(event.toolOutput);
      },
    } satisfies Telemetry;

    const telemetry = createTurnTelemetry({
      telemetry: { integrations: [integration] },
      harnessId: 'mock',
      modelId: 'mock-model',
      instructions: undefined,
      promptText: 'go',
      runtimeContext: undefined,
    });

    await telemetry.start();
    await telemetry.toolStart({
      toolCallId: 'call-1',
      toolName: 'ask_user',
      input: { question: 'which app?' },
    });
    await telemetry.pauseOpenTools();
    await telemetry.end({
      finishReason: { unified: 'tool-calls', raw: 'tool_use' },
      usage,
    });

    expect(outputs).toEqual([
      { type: 'tool-result', output: { status: 'awaiting-host-input' } },
    ]);
  });

  test('settles a tool call the turn never closed as an error', async () => {
    const outputs: Array<{ type: string }> = [];
    const integration = {
      onToolExecutionEnd: event => {
        outputs.push(event.toolOutput);
      },
    } satisfies Telemetry;

    const telemetry = createTurnTelemetry({
      telemetry: { integrations: [integration] },
      harnessId: 'mock',
      modelId: 'mock-model',
      instructions: undefined,
      promptText: 'go',
      runtimeContext: undefined,
    });

    await telemetry.start();
    await telemetry.toolStart({
      toolCallId: 'call-1',
      toolName: 'read',
      input: {},
    });
    await telemetry.end({
      finishReason: { unified: 'stop', raw: 'stop' },
      usage,
    });

    expect(outputs).toEqual([
      { type: 'error', error: new Error('tool span unclosed') },
    ]);
  });
});
