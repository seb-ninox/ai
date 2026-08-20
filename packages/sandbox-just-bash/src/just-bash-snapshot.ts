import { InMemoryFs, Sandbox, type SandboxOptions } from 'just-bash';

/** Snapshot format version. Bumped when the entry shape changes. */
const SNAPSHOT_VERSION = 1;

type SnapshotEntryBase = {
  /** Absolute path inside the sandbox filesystem. */
  readonly path: string;
  /** POSIX permission bits. */
  readonly mode: number;
  /** Modification time as epoch milliseconds, so the entry stays JSON-safe. */
  readonly mtime: number;
};

export type JustBashFileEntry = SnapshotEntryBase & {
  readonly type: 'file';
  /** Base64-encoded bytes. Binary-safe, unlike a UTF-8 round trip. */
  readonly content: string;
};

export type JustBashDirectoryEntry = SnapshotEntryBase & {
  readonly type: 'directory';
};

export type JustBashSymlinkEntry = SnapshotEntryBase & {
  readonly type: 'symlink';
  readonly target: string;
};

export type JustBashSnapshotEntry =
  | JustBashFileEntry
  | JustBashDirectoryEntry
  | JustBashSymlinkEntry;

/**
 * Serializable state of one just-bash sandbox filesystem.
 *
 * Plain JSON so any backend can hold it — a Redis string, a Postgres `jsonb`
 * column, an object in S3. The SDK deliberately ships no backend client; see
 * {@link JustBashSandboxStore}.
 */
export type JustBashSandboxSnapshot = {
  readonly version: typeof SNAPSHOT_VERSION;
  /** Working directory the sandbox was created with. */
  readonly cwd: string;
  /**
   * Sandbox HOME, kept separate from the rest of the environment so snapshots
   * do not persist caller-provided secrets. Optional for version 1 snapshots
   * created before HOME was captured.
   */
  readonly homeDir?: string;
  readonly entries: readonly JustBashSnapshotEntry[];
};

/**
 * Durable home for sandbox snapshots, implemented by the consumer against
 * whatever it already runs — Redis, Postgres, S3.
 *
 * Snapshot contents are whatever the agent wrote to its workdir: uploads, model
 * output, and (for adapters that persist their journal into the sandbox, as Pi
 * does) the conversation history. Treat them as tenant data — scope keys per
 * tenant, expire them, and keep them inside the same encryption-at-rest posture
 * as the rest of that store.
 */
export interface JustBashSandboxStore {
  /**
   * Snapshot for `sessionId`, or `undefined` when none is stored.
   *
   * `undefined` is an expected outcome, not an error: an expired entry means
   * the session resumes with an empty workdir. Throw only when the backend
   * itself failed, so callers can tell "expired" from "unavailable".
   */
  load(options: {
    sessionId: string;
  }): PromiseLike<JustBashSandboxSnapshot | undefined>;

  /** Store the snapshot for `sessionId`, replacing any previous one. */
  save(options: {
    sessionId: string;
    snapshot: JustBashSandboxSnapshot;
  }): PromiseLike<void>;

  /** Drop the snapshot for `sessionId`. Must succeed when nothing is stored. */
  remove(options: { sessionId: string }): PromiseLike<void>;
}

/**
 * Capture the sandbox filesystem as a serializable snapshot.
 *
 * Walks `getAllPaths()` rather than recursing directories, so the cost is
 * proportional to the number of entries. Contents are read eagerly — keep
 * workdirs small, or the snapshot becomes the slow part of a turn.
 */
export async function createJustBashSnapshot(
  sandbox: Sandbox,
): Promise<JustBashSandboxSnapshot> {
  const fs = sandbox.bashEnvInstance.fs;
  const entries: JustBashSnapshotEntry[] = [];

  // Sorted so a snapshot of unchanged state is byte-identical, which lets
  // callers skip a write by comparing against what they already stored.
  for (const path of [...fs.getAllPaths()].sort()) {
    /*
     * `lstat`, not `stat`: a symlink must be captured as the link itself. A
     * path can also disappear between the listing and the read (a background
     * command, a lazy entry whose provider throws) — skip it rather than
     * failing the whole snapshot over one file.
     */
    let stat: Awaited<ReturnType<typeof fs.lstat>>;
    try {
      stat = await fs.lstat(path);
    } catch {
      continue;
    }

    const base = { path, mode: stat.mode, mtime: stat.mtime.getTime() };
    try {
      if (stat.isSymbolicLink) {
        entries.push({
          ...base,
          type: 'symlink',
          target: await fs.readlink(path),
        });
      } else if (stat.isDirectory) {
        entries.push({ ...base, type: 'directory' });
      } else {
        entries.push({
          ...base,
          type: 'file',
          content: toBase64(await fs.readFileBuffer(path)),
        });
      }
    } catch {
      continue;
    }
  }

  return {
    version: SNAPSHOT_VERSION,
    cwd: sandbox.bashEnvInstance.getCwd(),
    homeDir: sandbox.bashEnvInstance.getEnv().HOME ?? '/',
    entries,
  };
}

/**
 * Build a sandbox whose filesystem is the snapshot's.
 *
 * `options` are the same `SandboxOptions` used for a fresh sandbox; `fs` is
 * supplied by the restore and `cwd` falls back to the snapshot's.
 */
export async function restoreJustBashSandbox({
  snapshot,
  options,
}: {
  readonly snapshot: JustBashSandboxSnapshot;
  readonly options?: Omit<SandboxOptions, 'fs' | 'overlayRoot'>;
}): Promise<Sandbox> {
  if (snapshot.version !== SNAPSHOT_VERSION) {
    throw new Error(
      `Unsupported just-bash sandbox snapshot version ${String(snapshot.version)}; expected ${SNAPSHOT_VERSION}.`,
    );
  }

  /*
   * Seeding the constructor rather than writing files afterwards keeps parent
   * directories, modes and mtimes in one pass. Directories and symlinks are
   * not expressible as initial files, so they are applied below.
   */
  const fs = new InMemoryFs(
    Object.fromEntries(
      snapshot.entries
        .filter((entry): entry is JustBashFileEntry => entry.type === 'file')
        .map(entry => [
          entry.path,
          {
            content: fromBase64(entry.content),
            mode: entry.mode,
            mtime: new Date(entry.mtime),
          },
        ]),
    ),
  );

  // Shallowest first, so a directory is created before anything beneath it.
  const byDepth = (a: JustBashSnapshotEntry, b: JustBashSnapshotEntry) =>
    a.path.split('/').length - b.path.split('/').length;

  for (const entry of snapshot.entries
    .filter(
      (entry): entry is JustBashDirectoryEntry => entry.type === 'directory',
    )
    .sort(byDepth)) {
    if (await fs.exists(entry.path)) continue;
    // `mkdir` takes no mode, so permissions are applied separately.
    await fs.mkdir(entry.path, { recursive: true });
    await fs.chmod(entry.path, entry.mode);
    await fs.utimes(entry.path, new Date(entry.mtime), new Date(entry.mtime));
  }

  /*
   * Symlinks last: one can point at another, and a dangling link is legal, so
   * ordering by depth is enough — no target has to resolve at creation time.
   */
  for (const entry of snapshot.entries
    .filter((entry): entry is JustBashSymlinkEntry => entry.type === 'symlink')
    .sort(byDepth)) {
    if (await fs.exists(entry.path)) continue;
    await fs.symlink(entry.target, entry.path);
  }

  const homeDir =
    snapshot.homeDir ??
    (snapshot.entries.some(
      entry => entry.type === 'directory' && entry.path === '/home/user',
    )
      ? '/home/user'
      : undefined);
  const env =
    homeDir === undefined ? options?.env : { HOME: homeDir, ...options?.env };

  return await Sandbox.create({
    cwd: snapshot.cwd,
    ...options,
    ...(env === undefined ? {} : { env }),
    fs,
  });
}

function toBase64(bytes: Uint8Array): string {
  return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString(
    'base64',
  );
}

function fromBase64(value: string): Uint8Array {
  const buffer = Buffer.from(value, 'base64');
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}
