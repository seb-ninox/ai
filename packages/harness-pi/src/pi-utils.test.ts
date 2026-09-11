import { describe, expect, it } from 'vitest';
import { HarnessCapabilityUnsupportedError } from '@ai-sdk/harness';
import {
  extractUserPrompt,
  frameInstructions,
  renderPiSkillFile,
  safePiMetadataSegment,
  serializeToolOutput,
} from './pi-utils';

describe('frameInstructions', () => {
  it('wraps instructions as system guidance and fences the user text', () => {
    const framed = frameInstructions('Use turbo build.', 'do the thing');
    expect(framed).toContain('<session-instructions>');
    expect(framed).toContain('Use turbo build.');
    expect(framed).toContain('</session-instructions>');
    expect(framed).toContain('<user-message>\ndo the thing\n</user-message>');
    expect(framed).toMatch(/not a message from the user/i);
    // Instructions must come before the user message.
    expect(framed.indexOf('<session-instructions>')).toBeLessThan(
      framed.indexOf('<user-message>'),
    );
  });
});

describe('extractUserPrompt', () => {
  it('returns the string prompt verbatim', () => {
    expect(extractUserPrompt('hello there')).toEqual({
      text: 'hello there',
      images: [],
    });
  });

  it('joins multiple text parts with double newlines', () => {
    const { text } = extractUserPrompt({
      role: 'user',
      content: [
        { type: 'text', text: 'first' },
        { type: 'text', text: 'second' },
      ],
    });
    expect(text).toBe('first\n\nsecond');
  });

  it('handles a single string content', () => {
    expect(extractUserPrompt({ role: 'user', content: 'just text' })).toEqual({
      text: 'just text',
      images: [],
    });
  });

  it('converts an image part with raw bytes to a Pi attachment', () => {
    const { text, images } = extractUserPrompt({
      role: 'user',
      content: [
        { type: 'text', text: 'what is this' },
        {
          type: 'image',
          image: new Uint8Array([1, 2, 3]),
          mediaType: 'image/png',
        },
      ],
    });
    expect(text).toBe('what is this');
    expect(images).toEqual([
      { type: 'image', data: 'AQID', mimeType: 'image/png' },
    ]);
  });

  it('strips a data URL prefix and prefers its media type', () => {
    const { images } = extractUserPrompt({
      role: 'user',
      content: [
        {
          type: 'image',
          image: 'data:image/webp;base64,AQID',
        },
      ],
    });
    expect(images).toEqual([
      { type: 'image', data: 'AQID', mimeType: 'image/webp' },
    ]);
  });

  it('accepts a file part carrying an image', () => {
    const { images } = extractUserPrompt({
      role: 'user',
      content: [
        {
          type: 'file',
          data: { type: 'data', data: new Uint8Array([1, 2, 3]) },
          mediaType: 'image/jpeg',
          filename: 'photo.jpg',
        },
      ],
    });
    expect(images).toEqual([
      { type: 'image', data: 'AQID', mimeType: 'image/jpeg' },
    ]);
  });

  it('sniffs the subtype when the media type is only the top-level segment', () => {
    // 8-byte PNG signature.
    const png = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
    const { images } = extractUserPrompt({
      role: 'user',
      content: [{ type: 'image', image: png, mediaType: 'image' }],
    });
    expect(images[0]?.mimeType).toBe('image/png');
  });

  it('refuses a PDF file part', () => {
    expect(() =>
      extractUserPrompt({
        role: 'user',
        content: [
          {
            type: 'file',
            data: new Uint8Array([1, 2]),
            mediaType: 'application/pdf',
          },
        ],
      }),
    ).toThrowError(HarnessCapabilityUnsupportedError);
  });

  it('refuses an image referenced by URL', () => {
    expect(() =>
      extractUserPrompt({
        role: 'user',
        content: [
          {
            type: 'image',
            image: new URL('https://example.com/a.png'),
            mediaType: 'image/png',
          },
        ],
      }),
    ).toThrowError(HarnessCapabilityUnsupportedError);
  });

  it('refuses an image whose media type cannot be resolved', () => {
    expect(() =>
      extractUserPrompt({
        role: 'user',
        content: [{ type: 'image', image: 'AQID' }],
      }),
    ).toThrowError(HarnessCapabilityUnsupportedError);
  });
});

describe('serializeToolOutput', () => {
  it('returns strings unchanged', () => {
    expect(serializeToolOutput('hello')).toBe('hello');
  });

  it('JSON-stringifies objects', () => {
    expect(serializeToolOutput({ a: 1 })).toBe('{"a":1}');
  });

  it('returns "null" for undefined', () => {
    expect(serializeToolOutput(undefined)).toBe('null');
  });
});

describe('safePiMetadataSegment', () => {
  it('accepts alphanumeric, dots, dashes, underscores', () => {
    expect(safePiMetadataSegment('my-skill_1.0', 'skill')).toBe('my-skill_1.0');
  });

  it('rejects path traversal attempts', () => {
    expect(() => safePiMetadataSegment('..', 'skill')).toThrow();
    expect(() => safePiMetadataSegment('.', 'skill')).toThrow();
    expect(() => safePiMetadataSegment('a/b', 'skill')).toThrow();
    expect(() => safePiMetadataSegment('a b', 'skill')).toThrow();
  });
});

describe('renderPiSkillFile', () => {
  it('emits YAML frontmatter followed by content', () => {
    const rendered = renderPiSkillFile({
      name: 'refactor',
      description: 'Refactor carefully.',
      content: 'Make small changes.',
    });
    expect(rendered).toBe(
      `---\nname: refactor\ndescription: Refactor carefully.\n---\n\nMake small changes.`,
    );
  });
});
