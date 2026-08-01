import type { TextStreamPart } from 'ai';
import type { ToolSet } from '@ai-sdk/provider-utils';

type BlockKind = 'text' | 'reasoning';

/**
 * Keeps the text and reasoning blocks of one result stream self-contained.
 *
 * AI SDK stream consumers assemble content per result: `toUIMessageStream`
 * throws a `UIMessageStreamError` on a `text-delta` / `reasoning-delta` whose
 * matching `-start` it has not seen, and leaves a block that never receives
 * its `-end` stuck in the `streaming` state.
 *
 * A harness turn can pause mid-block. When a runtime keeps its native turn
 * alive across a tool-approval pause (Pi does), the block that was open when
 * the turn parked continues with a delta on the *continuation* stream — whose
 * `-start` went to the previous result. Adapters can also finish a stream with
 * a block they never closed.
 *
 * The normalizer restores the invariant without the adapters having to:
 *  - a delta for an unopened block gets a synthesized `-start` first,
 *  - an `-end` for a block this stream never opened is dropped, and
 *  - {@link StreamBlockNormalizer.closeOpenBlocks} emits the missing `-end`
 *    parts when the harness closes the stream cleanly.
 *
 * State is per result stream, so ids are only ever compared within one stream.
 */
export type StreamBlockNormalizer<TOOLS extends ToolSet> = {
  /** Parts to enqueue in place of `part`. Usually just `[part]`. */
  normalize(part: TextStreamPart<TOOLS>): ReadonlyArray<TextStreamPart<TOOLS>>;

  /**
   * `-end` parts for every block still open, in the order they were opened.
   * Clears the tracked state, so calling it twice is safe.
   */
  closeOpenBlocks(): ReadonlyArray<TextStreamPart<TOOLS>>;
};

export function createStreamBlockNormalizer<
  TOOLS extends ToolSet,
>(): StreamBlockNormalizer<TOOLS> {
  // Insertion-ordered: `closeOpenBlocks` closes blocks oldest first.
  const openBlocks = new Map<string, BlockKind>();
  const keyOf = (kind: BlockKind, id: string) => `${kind}:${id}`;

  const startPart = (
    kind: BlockKind,
    id: string,
    providerMetadata: unknown,
  ): TextStreamPart<TOOLS> =>
    ({
      type: `${kind}-start`,
      id,
      ...(providerMetadata !== undefined ? { providerMetadata } : {}),
    }) as TextStreamPart<TOOLS>;

  const endPart = (kind: BlockKind, id: string): TextStreamPart<TOOLS> =>
    ({ type: `${kind}-end`, id }) as TextStreamPart<TOOLS>;

  return {
    normalize(part) {
      const kind: BlockKind | undefined = part.type.startsWith('text-')
        ? 'text'
        : part.type.startsWith('reasoning-')
          ? 'reasoning'
          : undefined;
      if (kind === undefined) return [part];

      const { id, providerMetadata } = part as {
        id?: string;
        providerMetadata?: unknown;
      };
      if (id === undefined) return [part];
      const key = keyOf(kind, id);

      switch (part.type) {
        case 'text-start':
        case 'reasoning-start':
          openBlocks.set(key, kind);
          return [part];

        case 'text-delta':
        case 'reasoning-delta': {
          if (openBlocks.has(key)) return [part];
          openBlocks.set(key, kind);
          /*
           * The synthesized start carries the delta's own provider metadata:
           * it is the only metadata this stream has for the block, and the
           * real start already delivered it to the previous stream.
           */
          return [startPart(kind, id, providerMetadata), part];
        }

        case 'text-end':
        case 'reasoning-end': {
          // Nothing in this stream opened it — an `-end` alone would throw in
          // `toUIMessageStream`, and synthesizing a start to match it would
          // add an empty block.
          if (!openBlocks.delete(key)) return [];
          return [part];
        }

        default:
          return [part];
      }
    },

    closeOpenBlocks() {
      const parts = Array.from(openBlocks, ([key, kind]) =>
        endPart(kind, key.slice(kind.length + 1)),
      );
      openBlocks.clear();
      return parts;
    },
  };
}
