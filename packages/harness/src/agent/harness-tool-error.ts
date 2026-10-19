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

/**
 * The `HarnessToolError` an error carries, directly or as a cause.
 *
 * A tool's schema rejection reaches the loop wrapped in the AI SDK's
 * `InvalidToolInputError`, which keeps the underlying error as its `cause`. Only
 * looking at the outer error there loses the tool's own two messages and reports
 * the generic default to the user instead — for the one failure the tool was
 * most specific about.
 */
function findHarnessToolError(error: unknown): HarnessToolError | undefined {
  const seen = new Set<unknown>();
  let current = error;
  while (current != null && !seen.has(current)) {
    if (current instanceof HarnessToolError) {
      return current;
    }
    seen.add(current);
    current = (current as { cause?: unknown }).cause;
  }
  return undefined;
}

export function getHarnessToolModelMessage(error: unknown): string {
  return findHarnessToolError(error)?.modelMessage ?? String(error);
}

export function getHarnessToolPublicMessage(error: unknown): string {
  return (
    findHarnessToolError(error)?.publicMessage ??
    DEFAULT_PUBLIC_TOOL_ERROR_MESSAGE
  );
}
