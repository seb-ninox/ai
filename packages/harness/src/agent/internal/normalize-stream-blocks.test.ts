import type { ToolSet } from '@ai-sdk/provider-utils';
import type { TextStreamPart } from 'ai';
import { describe, expect, test } from 'vitest';
import { createStreamBlockNormalizer } from './normalize-stream-blocks';

type Part = TextStreamPart<ToolSet>;

/** Feed parts through one normalizer and collect everything it emits. */
function run(parts: Part[]): {
  emitted: Part[];
  closed: readonly Part[];
} {
  const normalizer = createStreamBlockNormalizer<ToolSet>();
  const emitted = parts.flatMap(part => [...normalizer.normalize(part)]);
  return { emitted, closed: normalizer.closeOpenBlocks() };
}

describe('createStreamBlockNormalizer', () => {
  test('passes a well-formed block through untouched', () => {
    const parts = [
      { type: 'text-start', id: 't1' },
      { type: 'text-delta', id: 't1', text: 'hi' },
      { type: 'text-end', id: 't1' },
    ] as Part[];

    const { emitted, closed } = run(parts);

    expect(emitted).toEqual(parts);
    expect(closed).toEqual([]);
  });

  test('synthesizes the start of a block that this stream never opened', () => {
    // A runtime that keeps its turn alive across an approval pause resumes the
    // block with a delta whose `text-start` went to the previous result.
    const { emitted } = run([
      { type: 'text-delta', id: 't1', text: 'resumed' },
      { type: 'text-end', id: 't1' },
    ] as Part[]);

    expect(emitted).toEqual([
      { type: 'text-start', id: 't1' },
      { type: 'text-delta', id: 't1', text: 'resumed' },
      { type: 'text-end', id: 't1' },
    ]);
  });

  test('synthesizes the start only once per block', () => {
    const { emitted } = run([
      { type: 'text-delta', id: 't1', text: 'a' },
      { type: 'text-delta', id: 't1', text: 'b' },
    ] as Part[]);

    expect(emitted.filter(part => part.type === 'text-start')).toHaveLength(1);
  });

  test('carries the delta provider metadata onto the synthesized start', () => {
    const providerMetadata = { pi: { signature: 'abc' } };
    const { emitted } = run([
      { type: 'reasoning-delta', id: 'r1', text: 'think', providerMetadata },
    ] as Part[]);

    expect(emitted[0]).toEqual({
      type: 'reasoning-start',
      id: 'r1',
      providerMetadata,
    });
  });

  test('drops an end for a block this stream never opened', () => {
    // Synthesizing a start to match it would add an empty block instead.
    const { emitted } = run([{ type: 'text-end', id: 't1' }] as Part[]);

    expect(emitted).toEqual([]);
  });

  test('reopens a block that receives a delta after its end', () => {
    const { emitted } = run([
      { type: 'text-start', id: 't1' },
      { type: 'text-end', id: 't1' },
      { type: 'text-delta', id: 't1', text: 'more' },
    ] as Part[]);

    expect(emitted.map(part => part.type)).toEqual([
      'text-start',
      'text-end',
      'text-start',
      'text-delta',
    ]);
  });

  test('tracks text and reasoning ids separately', () => {
    const { emitted, closed } = run([
      { type: 'text-start', id: 'shared' },
      { type: 'reasoning-delta', id: 'shared', text: 'think' },
    ] as Part[]);

    expect(emitted.map(part => part.type)).toEqual([
      'text-start',
      'reasoning-start',
      'reasoning-delta',
    ]);
    expect(closed).toEqual([
      { type: 'text-end', id: 'shared' },
      { type: 'reasoning-end', id: 'shared' },
    ]);
  });

  test('closes still-open blocks oldest first and only once', () => {
    const normalizer = createStreamBlockNormalizer<ToolSet>();
    for (const part of [
      { type: 'text-start', id: 't1' },
      { type: 'reasoning-start', id: 'r1' },
      { type: 'text-start', id: 't2' },
      { type: 'text-end', id: 't1' },
    ] as Part[]) {
      normalizer.normalize(part);
    }

    expect(normalizer.closeOpenBlocks()).toEqual([
      { type: 'reasoning-end', id: 'r1' },
      { type: 'text-end', id: 't2' },
    ]);
    expect(normalizer.closeOpenBlocks()).toEqual([]);
  });

  test('leaves ids containing the kind separator intact', () => {
    const { closed } = run([{ type: 'text-start', id: 'text:1' }] as Part[]);

    expect(closed).toEqual([{ type: 'text-end', id: 'text:1' }]);
  });

  test('passes non-block parts through', () => {
    const parts = [
      { type: 'error', error: 'boom' },
      { type: 'raw', rawValue: { a: 1 } },
    ] as Part[];

    expect(run(parts).emitted).toEqual(parts);
  });
});
