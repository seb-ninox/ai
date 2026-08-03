export const DEFAULT_PUBLIC_TOOL_ERROR_MESSAGE =
  'The tool could not complete this action.';

/**
 * Carries separate error messages for the model and the public stream.
 *
 * The model message should contain enough already-sanitized detail for the
 * model to repair its next tool call. The public message is safe to persist
 * and render to end users.
 */
export class HarnessToolError extends Error {
  readonly modelMessage: string;
  readonly publicMessage: string;

  constructor(options: { modelMessage: string; publicMessage: string }) {
    super(options.modelMessage);
    this.name = 'HarnessToolError';
    this.modelMessage = options.modelMessage;
    this.publicMessage = options.publicMessage;
  }
}

export function getHarnessToolModelMessage(error: unknown): string {
  return error instanceof HarnessToolError
    ? error.modelMessage
    : String(error);
}

export function getHarnessToolPublicMessage(error: unknown): string {
  return error instanceof HarnessToolError
    ? error.publicMessage
    : DEFAULT_PUBLIC_TOOL_ERROR_MESSAGE;
}
