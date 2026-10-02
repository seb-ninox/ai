import { describe, expect, it, vi } from 'vitest';
import type {
  AgentToolResult,
  ExtensionAPI,
  ExtensionToolContext,
  ToolDefinition,
} from '@earendil-works/pi-coding-agent';
import {
  createPiCodemodeExtension,
  resolveScriptTimeoutMs,
  type PiCodemodeSettings,
} from './pi-codemode';

const MIB = 1024 * 1024;

type NestedTool = ExtensionToolContext['tools'][number];
type Outcome = Awaited<ReturnType<ExtensionToolContext['executeTool']>>;

const queryTable: NestedTool = {
  name: 'query_table',
  label: 'query_table',
  description: 'Reads rows from a table.',
  parameters: {
    type: 'object',
    properties: { table: { type: 'string' } },
    required: ['table'],
  },
  outputSchema: { type: 'object', additionalProperties: true },
  execute: () => Promise.reject(new Error('called directly')),
};

const hang: NestedTool = {
  name: 'hang',
  label: 'hang',
  description: 'Never answers.',
  parameters: { type: 'object', properties: {} },
  execute: () => Promise.reject(new Error('called directly')),
};

function okOutcome(
  id: string,
  name: string,
  value: AgentToolResult<unknown>['structuredContent'],
): Outcome {
  return {
    toolCall: { type: 'toolCall', id, name, arguments: {} },
    result: {
      content: [{ type: 'text', text: JSON.stringify(value) }],
      structuredContent: value,
      details: undefined,
    },
    isError: false,
  };
}

/** Pi's definition registered through the fork's extension, with a stub API around it. */
async function registerCodemode(settings: PiCodemodeSettings) {
  const registered: ToolDefinition[] = [];
  const appendEntry = vi.fn();
  const pi = {
    registerTool: (tool: ToolDefinition) => {
      registered.push(tool);
    },
    getAllTools: () => [],
    appendEntry,
  } as unknown as ExtensionAPI;
  await createPiCodemodeExtension(settings)(pi);
  const definition = registered.find(tool => tool.name === 'codemode');
  if (!definition) throw new Error('Pi registered no codemode tool');
  return { definition, appendEntry };
}

function contextWith(input: {
  tools: readonly NestedTool[];
  executeTool: ExtensionToolContext['executeTool'];
  branch?: ReturnType<ExtensionToolContext['sessionManager']['getBranch']>;
}): ExtensionToolContext {
  return {
    tools: input.tools,
    executeTool: input.executeTool,
    sessionManager: { getBranch: () => input.branch ?? [] },
  } as unknown as ExtensionToolContext;
}

function textsOf(result: Awaited<ReturnType<ToolDefinition['execute']>>) {
  return result.content.flatMap(block =>
    block.type === 'text' ? [block.text] : [],
  );
}

describe('resolveScriptTimeoutMs', () => {
  it.each([
    [undefined, undefined, Number.POSITIVE_INFINITY],
    [undefined, 5000, 5000],
    [3000, undefined, 3000],
    [2000, 5000, 2000],
    [9000, 5000, 5000],
  ])(
    'script %s with default %s runs under %s',
    (scriptTimeoutMs, defaultTimeoutMs, expected) => {
      expect(resolveScriptTimeoutMs(scriptTimeoutMs, defaultTimeoutMs)).toBe(
        expected,
      );
    },
  );
});

describe('createPiCodemodeExtension with script limits', () => {
  it('hands a script the structured result of a nested call and formats the return value like Pi', async () => {
    const { definition } = await registerCodemode({ timeoutMs: 10_000 });
    const executeTool = vi.fn(async (name: string) =>
      okOutcome('call-1/1', name, { rows: [{ id: 1 }], revision: 42 }),
    );

    const result = await definition.execute(
      'call-1',
      {
        code: "const r = await tools.query_table({ table: 'orders' }); return { rows: r.rows.length, revision: r.revision };",
      },
      undefined,
      undefined,
      contextWith({ tools: [queryTable], executeTool }),
    );

    expect(textsOf(result)[0]).toMatch(
      /^Script completed\nWall time [0-9.]+ seconds\nOutput:\n$/,
    );
    expect(textsOf(result)[1]).toBe('{"rows":1,"revision":42}');
    expect(result.isError).toBeUndefined();
    expect(executeTool).toHaveBeenCalledExactlyOnceWith(
      'query_table',
      { table: 'orders' },
      { signal: expect.any(AbortSignal) },
    );
    expect(result.details).toEqual({
      calls: [
        {
          id: 'call-1/1',
          name: 'query_table',
          args: '{"table":"orders"}',
          status: 'ok',
          durationMs: expect.any(Number),
        },
      ],
    });
  });

  it('ends a script at the default deadline, aborts its pending nested call, and reports the timeout', async () => {
    const { definition } = await registerCodemode({ timeoutMs: 300 });
    let nestedSignal: AbortSignal | undefined;
    const executeTool: ExtensionToolContext['executeTool'] = (
      name,
      _args,
      options,
    ) =>
      new Promise(resolve => {
        nestedSignal = options?.signal;
        options?.signal?.addEventListener('abort', () =>
          resolve({
            ...okOutcome('call-2/1', name, null),
            isError: true,
          }),
        );
      });

    const result = await definition.execute(
      'call-2',
      { code: 'await tools.hang({}); return "unreachable";' },
      undefined,
      undefined,
      contextWith({ tools: [hang], executeTool }),
    );

    expect(result.isError).toBe(true);
    expect(textsOf(result)[0]).toMatch(/^Script failed\n/);
    expect(textsOf(result)[1]).toBe(
      'Script error:\nScript timed out: Execution timed out after 300 ms\n\nTool calls made before the failure (they are not undone): hang (cancelled)',
    );
    expect(nestedSignal?.aborted).toBe(true);
  });

  it('waits for a nested call the script abandoned before returning, and keeps it cancelled', async () => {
    const { definition } = await registerCodemode({ timeoutMs: 10_000 });
    let hostAnswered = false;
    const executeTool = vi.fn(
      (name: string) =>
        new Promise<Outcome>(resolve =>
          setTimeout(() => {
            hostAnswered = true;
            resolve(okOutcome('call-8/1', name, { rows: [] }));
          }, 200),
        ),
    );

    const result = await definition.execute(
      'call-8',
      { code: "tools.query_table({ table: 'x' }); return 'left early';" },
      undefined,
      undefined,
      contextWith({ tools: [queryTable], executeTool }),
    );

    expect(hostAnswered).toBe(true);
    expect(textsOf(result)[1]).toBe('left early');
    expect(result.details).toEqual({
      calls: [
        {
          id: 'call-8/1',
          name: 'query_table',
          args: '{"table":"x"}',
          status: 'cancelled',
          durationMs: expect.any(Number),
        },
      ],
    });
  });

  it('lets a script shorten the deadline but not extend it', async () => {
    const settlesOnAbort: ExtensionToolContext['executeTool'] = (
      name,
      _args,
      options,
    ) =>
      new Promise(resolve =>
        options?.signal?.addEventListener('abort', () =>
          resolve({ ...okOutcome('call-3/1', name, null), isError: true }),
        ),
      );
    const run = async (settings: PiCodemodeSettings, timeoutMs: number) => {
      const { definition } = await registerCodemode(settings);
      const result = await definition.execute(
        'call-3',
        {
          code: `// @options: {"timeout_ms": ${timeoutMs}}\nawait tools.hang({});`,
        },
        undefined,
        undefined,
        contextWith({ tools: [hang], executeTool: settlesOnAbort }),
      );
      return textsOf(result)[1];
    };

    await expect(run({ timeoutMs: 5000 }, 100)).resolves.toContain(
      'Script timed out: Execution timed out after 100 ms',
    );
    await expect(run({ timeoutMs: 300 }, 60_000)).resolves.toContain(
      'Script timed out: Execution timed out after 300 ms',
    );
  });

  it('fails a script inside the VM once it allocates past memoryLimitBytes', async () => {
    const { definition } = await registerCodemode({
      memoryLimitBytes: 16 * MIB,
    });

    const result = await definition.execute(
      'call-4',
      { code: "const a = []; for (;;) a.push('x'.repeat(1e6));" },
      undefined,
      undefined,
      contextWith({ tools: [], executeTool: vi.fn() }),
    );

    expect(result.isError).toBe(true);
    expect(textsOf(result)[1]).toMatch(
      /^Script error:\nInternalError: out of memory/,
    );
    expect(textsOf(result)[1]).toContain('No tool calls were made.');
  });

  it('reads the store from the branch and appends a successful script’s writes', async () => {
    const { definition, appendEntry } = await registerCodemode({
      timeoutMs: 10_000,
    });
    const branch = [
      {
        type: 'custom',
        customType: 'codemode-store',
        data: { set: { previous: 'kept', gone: 1 }, delete: [] },
      },
      {
        type: 'custom',
        customType: 'codemode-store',
        data: { set: {}, delete: ['gone'] },
      },
    ] as unknown as ReturnType<
      ExtensionToolContext['sessionManager']['getBranch']
    >;

    const result = await definition.execute(
      'call-5',
      {
        code: "const before = load('previous'); store('next', 2); store('previous', undefined); return [before, load('gone')];",
      },
      undefined,
      undefined,
      contextWith({ tools: [], executeTool: vi.fn(), branch }),
    );

    expect(textsOf(result)[1]).toBe('["kept",null]');
    expect(appendEntry).toHaveBeenCalledExactlyOnceWith('codemode-store', {
      set: { next: 2 },
      delete: ['previous'],
    });
  });

  it('rejects an unsupported @options field with Pi’s own source error', async () => {
    const { definition } = await registerCodemode({ timeoutMs: 10_000 });

    await expect(
      definition.execute(
        'call-6',
        { code: '// @options: {"bogus": 1}\nreturn 1;' },
        undefined,
        undefined,
        contextWith({ tools: [], executeTool: vi.fn() }),
      ),
    ).rejects.toThrow(
      '@options only supports `max_output_tokens` and `timeout_ms`; got `bogus`',
    );
  });

  it('answers searchTools() and describeTool() from the nested tools', async () => {
    const { definition } = await registerCodemode({ timeoutMs: 10_000 });

    const result = await definition.execute(
      'call-7',
      {
        code: "return { found: (await searchTools('rows table')).map(t => t.name), none: await searchTools('nothing matches'), described: (await describeTool('query_table')).includes('query_table') };",
      },
      undefined,
      undefined,
      contextWith({ tools: [queryTable, hang], executeTool: vi.fn() }),
    );

    expect(textsOf(result)[1]).toBe(
      '{"found":["query_table"],"none":[],"described":true}',
    );
  });
});
