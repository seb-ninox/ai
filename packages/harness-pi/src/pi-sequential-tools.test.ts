import { describe, expect, it } from 'vitest';
import { makeAnthropicToolCallsSequential } from './pi-sequential-tools';

describe('makeAnthropicToolCallsSequential', () => {
  it('disables parallel Anthropic tool calls', () => {
    const payload = {
      model: 'claude-sonnet-4-6',
      tools: [{ input_schema: { type: 'object' }, name: 'write' }],
    };

    expect(makeAnthropicToolCallsSequential(payload)).toEqual({
      ...payload,
      tool_choice: { disable_parallel_tool_use: true, type: 'auto' },
    });
  });

  it('preserves an existing Anthropic tool choice', () => {
    const payload = {
      tool_choice: { name: 'write', type: 'tool' },
      tools: [{ input_schema: { type: 'object' }, name: 'write' }],
    };

    expect(makeAnthropicToolCallsSequential(payload)).toEqual({
      ...payload,
      tool_choice: {
        disable_parallel_tool_use: true,
        name: 'write',
        type: 'tool',
      },
    });
  });

  it.each([
    null,
    {},
    { tools: [] },
    { tools: [{ function: { name: 'write' }, type: 'function' }] },
    {
      tool_choice: { type: 'none' },
      tools: [{ input_schema: { type: 'object' }, name: 'write' }],
    },
  ])('leaves unsupported payloads unchanged', payload => {
    expect(makeAnthropicToolCallsSequential(payload)).toBeUndefined();
  });
});
