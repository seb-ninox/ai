import type { ExtensionFactory } from '@earendil-works/pi-coding-agent';

export function makeAnthropicToolCallsSequential(
  payload: unknown,
): unknown | undefined {
  if (
    typeof payload !== 'object' ||
    payload === null ||
    !('tools' in payload) ||
    !Array.isArray(payload.tools) ||
    !payload.tools.some(
      tool =>
        typeof tool === 'object' && tool !== null && 'input_schema' in tool,
    )
  ) {
    return undefined;
  }

  const existing = 'tool_choice' in payload ? payload.tool_choice : undefined;
  if (
    typeof existing === 'object' &&
    existing !== null &&
    'type' in existing &&
    existing.type === 'none'
  ) {
    return undefined;
  }

  const choice =
    typeof existing === 'object' && existing !== null
      ? existing
      : { type: 'auto' };
  return {
    ...payload,
    tool_choice: { ...choice, disable_parallel_tool_use: true },
  };
}

export const sequentialToolCallsExtension: ExtensionFactory = pi => {
  pi.on('before_provider_request', ({ payload }) =>
    makeAnthropicToolCallsSequential(payload),
  );
};
