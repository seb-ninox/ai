import type { ToolApprovalStatus } from 'ai';
import type { HarnessV1PermissionMode } from '../../v1';
import type { HarnessAgentToolApprovalConfiguration } from '../harness-agent-settings';

export const DEFAULT_PERMISSION_MODE: HarnessV1PermissionMode =
  'allow-all' as const;

export function resolvePermissionMode(input: {
  permissionMode: HarnessV1PermissionMode | undefined;
}): HarnessV1PermissionMode {
  return input.permissionMode ?? DEFAULT_PERMISSION_MODE;
}

export function permissionModeNeedsBuiltinSupport(input: {
  permissionMode: HarnessV1PermissionMode;
}): boolean {
  return input.permissionMode !== 'allow-all';
}

export type CustomToolApprovalDecision =
  | {
      readonly type: 'allow';
      readonly reason?: string;
      /**
       * `true` when the allow came from an explicit `approved` status rather
       * than `not-applicable`: an explicit approval is a policy DECISION, so
       * the caller records it on the stream as an automatic approval
       * request/response pair, exactly as automatic denials already are. A
       * `not-applicable` tool never needed approval and stays silent.
       */
      readonly recorded: boolean;
    }
  | { readonly type: 'deny'; readonly reason?: string }
  | { readonly type: 'request' };

export async function resolveCustomToolApproval(input: {
  toolName: string;
  toolInput: unknown;
  toolContext: unknown;
  toolApproval: HarnessAgentToolApprovalConfiguration | undefined;
}): Promise<CustomToolApprovalDecision> {
  const configured = input.toolApproval?.[input.toolName];
  const status = normalizeToolApprovalStatus({
    status:
      typeof configured === 'function'
        ? await configured({
            toolName: input.toolName,
            input: input.toolInput,
            context: input.toolContext,
          })
        : configured,
  });

  switch (status.type) {
    case 'not-applicable':
      return { type: 'allow', reason: status.reason, recorded: false };
    case 'approved':
      return { type: 'allow', reason: status.reason, recorded: true };
    case 'denied':
      return { type: 'deny', reason: status.reason };
    case 'user-approval':
      return { type: 'request' };
  }
}

function normalizeToolApprovalStatus(input: {
  status: ToolApprovalStatus | undefined;
}): Exclude<ToolApprovalStatus, string | undefined> {
  if (input.status === undefined) return { type: 'not-applicable' };
  if (typeof input.status === 'string') return { type: input.status };
  return input.status;
}
