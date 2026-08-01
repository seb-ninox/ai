import { Sandbox } from 'just-bash';
import { describe, expect, it } from 'vitest';
import {
  createJustBashSnapshot,
  restoreJustBashSandbox,
  type JustBashSandboxSnapshot,
} from './just-bash-snapshot';

async function read(sandbox: Sandbox, path: string): Promise<string> {
  return await sandbox.bashEnvInstance.fs.readFile(path);
}

/** Snapshot a sandbox and rebuild it, the way a resume across processes does. */
async function roundTrip(sandbox: Sandbox): Promise<Sandbox> {
  const snapshot = await createJustBashSnapshot(sandbox);
  // Prove the snapshot really is transport-safe rather than sharing objects
  // with the source sandbox.
  const transported = JSON.parse(
    JSON.stringify(snapshot),
  ) as JustBashSandboxSnapshot;
  return await restoreJustBashSandbox({ snapshot: transported });
}

describe('just-bash sandbox snapshots', () => {
  it('restores file contents and the working directory', async () => {
    const sandbox = await Sandbox.create({ cwd: '/work' });
    await sandbox.writeFiles({ '/work/notes.txt': 'hello' });

    const restored = await roundTrip(sandbox);

    expect(restored.bashEnvInstance.getCwd()).toBe('/work');
    expect(await read(restored, '/work/notes.txt')).toBe('hello');
  });

  it('restores a filesystem the agent built through bash', async () => {
    const sandbox = await Sandbox.create({ cwd: '/work' });
    await sandbox.runCommand({
      cmd: 'bash',
      args: [
        '-c',
        'mkdir -p /work/out && echo generated > /work/out/report.md',
      ],
    });

    const restored = await roundTrip(sandbox);

    expect(await read(restored, '/work/out/report.md')).toBe('generated\n');
    const listing = await restored.runCommand({
      cmd: 'bash',
      args: ['-c', 'ls /work/out'],
    });
    expect(await listing.stdout()).toBe('report.md\n');
  });

  it('keeps writing to a restored filesystem working', async () => {
    const sandbox = await Sandbox.create({ cwd: '/work' });
    await sandbox.writeFiles({ '/work/a.txt': 'first' });

    const restored = await roundTrip(sandbox);
    await restored.runCommand({
      cmd: 'bash',
      args: ['-c', 'echo second > /work/b.txt'],
    });

    expect(await read(restored, '/work/a.txt')).toBe('first');
    expect(await read(restored, '/work/b.txt')).toBe('second\n');
  });

  it('round-trips binary content without corruption', async () => {
    const sandbox = await Sandbox.create({ cwd: '/work' });
    const bytes = new Uint8Array([0, 1, 2, 250, 251, 252, 253, 254, 255]);
    await sandbox.bashEnvInstance.fs.writeFile('/work/blob.bin', bytes);

    const restored = await roundTrip(sandbox);

    expect(
      await restored.bashEnvInstance.fs.readFileBuffer('/work/blob.bin'),
    ).toEqual(bytes);
  });

  it('preserves empty directories', async () => {
    const sandbox = await Sandbox.create({ cwd: '/work' });
    await sandbox.mkDir('/work/empty', { recursive: true });

    const restored = await roundTrip(sandbox);

    const stat = await restored.bashEnvInstance.fs.stat('/work/empty');
    expect(stat.isDirectory).toBe(true);
  });

  it('preserves symlinks as links rather than copies', async () => {
    const sandbox = await Sandbox.create({ cwd: '/work' });
    await sandbox.writeFiles({ '/work/target.txt': 'linked' });
    await sandbox.bashEnvInstance.fs.symlink(
      '/work/target.txt',
      '/work/link.txt',
    );

    const restored = await roundTrip(sandbox);

    expect(
      (await restored.bashEnvInstance.fs.lstat('/work/link.txt'))
        .isSymbolicLink,
    ).toBe(true);
    expect(await restored.bashEnvInstance.fs.readlink('/work/link.txt')).toBe(
      '/work/target.txt',
    );
    expect(await read(restored, '/work/link.txt')).toBe('linked');
  });

  it('preserves file permissions', async () => {
    const sandbox = await Sandbox.create({ cwd: '/work' });
    await sandbox.writeFiles({ '/work/run.sh': '#!/bin/bash\necho hi' });
    await sandbox.bashEnvInstance.fs.chmod('/work/run.sh', 0o755);

    const restored = await roundTrip(sandbox);

    expect((await restored.bashEnvInstance.fs.stat('/work/run.sh')).mode).toBe(
      0o755,
    );
  });

  it('produces a stable snapshot for unchanged state', async () => {
    const sandbox = await Sandbox.create({ cwd: '/work' });
    await sandbox.writeFiles({ '/work/b.txt': 'b', '/work/a.txt': 'a' });

    // Lets a caller skip a write by comparing against what it already stored.
    expect(JSON.stringify(await createJustBashSnapshot(sandbox))).toBe(
      JSON.stringify(await createJustBashSnapshot(sandbox)),
    );
  });

  it('rejects a snapshot from an unknown version', async () => {
    await expect(
      restoreJustBashSandbox({
        snapshot: {
          version: 99,
          cwd: '/work',
          entries: [],
        } as unknown as JustBashSandboxSnapshot,
      }),
    ).rejects.toThrow(/Unsupported just-bash sandbox snapshot version 99/);
  });

  it('applies caller options over the snapshot cwd', async () => {
    const sandbox = await Sandbox.create({ cwd: '/work' });
    await sandbox.writeFiles({ '/work/a.txt': 'a' });
    const snapshot = await createJustBashSnapshot(sandbox);

    const restored = await restoreJustBashSandbox({
      snapshot,
      options: { cwd: '/other' },
    });

    expect(restored.bashEnvInstance.getCwd()).toBe('/other');
    expect(await read(restored, '/work/a.txt')).toBe('a');
  });
});
