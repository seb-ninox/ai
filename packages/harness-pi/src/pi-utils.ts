import {
  HarnessCapabilityUnsupportedError,
  type HarnessV1Prompt,
  type HarnessV1Skill,
} from '@ai-sdk/harness';
import {
  convertUint8ArrayToBase64,
  detectMediaType,
  isFullMediaType,
} from '@ai-sdk/provider-utils';
import type { PromptOptions } from '@earendil-works/pi-coding-agent';

const HARNESS_ID = 'pi';

/*
 * Pi's own attachment shape, read off `AgentSession.prompt`'s options rather
 * than imported from `@earendil-works/pi-ai`: the runtime bundles its own copy
 * of that package, so a direct import can resolve to a different one and the
 * two structurally identical types stop being assignable.
 */
type PiImage = NonNullable<PromptOptions['images']>[number];

/** The user's turn, split into the two channels Pi's `prompt()` accepts. */
export interface PiUserPrompt {
  images: PiImage[];
  text: string;
}

/*
 * Matches a `data:` URL prefix and captures its media type. Browsers hand
 * attachments over in this form, and the media type in the prefix is more
 * trustworthy than an absent or wildcard `mediaType` on the part.
 */
const DATA_URL_PREFIX = /^data:([\w.+-]{1,40}\/[\w.+-]{1,40})?;base64,/i;

function unsupported(message: string): HarnessCapabilityUnsupportedError {
  return new HarnessCapabilityUnsupportedError({
    message: `pi: ${message}`,
    harnessId: HARNESS_ID,
  });
}

/**
 * A `file` part may carry its bytes bare or tagged as `{ type: 'data', data }`.
 * Every other tag (`url`, `reference`, `stream`, `text`) is left alone so the
 * caller rejects it with a message naming the part.
 */
function unwrapTaggedFileData(data: unknown): unknown {
  if (
    typeof data === 'object' &&
    data !== null &&
    'type' in data &&
    data.type === 'data' &&
    'data' in data
  ) {
    return data.data;
  }
  return data;
}

/*
 * Narrow one image/file part down to base64 bytes plus a full `image/<subtype>`
 * media type, which is all Pi (and the Anthropic API under it) will take.
 *
 * Only inline bytes work: the harness has no fetch, so a URL or an uploaded
 * provider reference has nothing to dereference it with. PDFs and other
 * non-image files are refused here too — Pi's message content union has no
 * document block, so there is no way to pass one through.
 */
function toPiImage(
  data: unknown,
  declaredMediaType: string | undefined,
  partType: string,
): PiImage {
  if (
    declaredMediaType !== undefined &&
    isFullMediaType(declaredMediaType) &&
    !declaredMediaType.startsWith('image/')
  ) {
    throw unsupported(
      `only image attachments are supported; got '${declaredMediaType}'.`,
    );
  }

  // A `file` part's data may arrive tagged; unwrap the one tag that holds bytes.
  const payload = unwrapTaggedFileData(data);

  let mediaType = declaredMediaType;
  let base64: string;

  if (typeof payload === 'string') {
    const dataUrl = DATA_URL_PREFIX.exec(payload);
    if (dataUrl) {
      base64 = payload.slice(dataUrl[0].length);
      mediaType = dataUrl[1] ?? mediaType;
    } else {
      base64 = payload;
    }
  } else if (payload instanceof Uint8Array) {
    // Node `Buffer` is a `Uint8Array`, so this covers it too.
    base64 = convertUint8ArrayToBase64(payload);
  } else if (payload instanceof ArrayBuffer) {
    base64 = convertUint8ArrayToBase64(new Uint8Array(payload));
  } else {
    throw unsupported(
      `'${partType}' parts must carry inline bytes; URLs and provider references are not supported.`,
    );
  }

  if (mediaType === undefined || !isFullMediaType(mediaType)) {
    // A bare `image` or a missing media type: sniff the subtype off the bytes.
    mediaType = detectMediaType({ data: base64, topLevelType: 'image' });
  }

  if (mediaType === undefined || !mediaType.startsWith('image/')) {
    throw unsupported(
      `could not resolve an image media type for a '${partType}' part; set 'mediaType' to a full type such as 'image/png'.`,
    );
  }

  return { type: 'image', data: base64, mimeType: mediaType };
}

/**
 * Split a `HarnessV1Prompt` into the text and the image attachments Pi's
 * `session.prompt(text, { images })` takes. Text parts are joined with blank
 * lines; `image` parts and `file` parts holding an image become attachments.
 * Every other part type is refused — Pi's message content is text or images,
 * nothing else.
 */
export function extractUserPrompt(prompt: HarnessV1Prompt): PiUserPrompt {
  if (typeof prompt === 'string') {
    return { text: prompt, images: [] };
  }

  const { content } = prompt;
  if (typeof content === 'string') {
    return { text: content, images: [] };
  }

  const parts: string[] = [];
  const images: PiImage[] = [];
  for (const part of content) {
    switch (part.type) {
      case 'text': {
        parts.push(part.text);
        break;
      }
      case 'image': {
        images.push(toPiImage(part.image, part.mediaType, 'image'));
        break;
      }
      case 'file': {
        images.push(toPiImage(part.data, part.mediaType, 'file'));
        break;
      }
      default: {
        throw unsupported(
          `only text and image user-message parts are supported; got '${(part as { type: string }).type}'.`,
        );
      }
    }
  }
  return { text: parts.join('\n\n'), images };
}

/*
 * Frame session instructions and the user's text so the runtime treats the
 * instructions as system-provided operating guidance, not something the user
 * wrote. Without the wrapper the agent can echo the prepended text back as if
 * the user had asked for it, which is confusing since the user never typed it.
 * Applied only to the first user message of a fresh session.
 */
export function frameInstructions(
  instructions: string,
  userText: string,
): string {
  return (
    '<session-instructions>\n' +
    'The block below is operating guidance from the system, not a message from the user — follow it, but do not mention it or attribute it to the user.\n\n' +
    `${instructions}\n` +
    '</session-instructions>\n\n' +
    `<user-message>\n${userText}\n</user-message>`
  );
}

/** Serialize a tool output to the string Pi feeds back to the model. */
export function serializeToolOutput(output: unknown): string {
  if (typeof output === 'string') {
    return output;
  }
  const serialized = JSON.stringify(output);
  return serialized ?? 'null';
}

export function getErrorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Validate that a name is safe to use as a filesystem path segment under
 * `.pi/skills/<name>/` or `.pi/agents/<name>.md`. Refuses anything that
 * could be interpreted as a path traversal or contains shell-sensitive
 * characters.
 */
export function safePiMetadataSegment(name: string, label: string): string {
  if (!/^[A-Za-z0-9._-]+$/.test(name) || name === '.' || name === '..') {
    throw new Error(`Invalid Pi ${label} name: ${name}`);
  }
  return name;
}

/** Frontmatter renderer for `.pi/skills/<name>/SKILL.md`. */
export function renderPiSkillFile(skill: HarnessV1Skill): string {
  return `---\nname: ${skill.name}\ndescription: ${skill.description}\n---\n\n${skill.content}`;
}
