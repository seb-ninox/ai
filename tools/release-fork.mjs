#!/usr/bin/env node
/**
 * Publish this fork's patched packages to a private registry, under a private
 * scope, without touching the ~60 packages the fork does not change.
 *
 * The repo keeps its upstream package names (`@ai-sdk/harness`, …) so merges
 * from upstream stay clean. Renaming happens only in the published artifact:
 * each package is packed with `pnpm pack` (which resolves `workspace:*` to real
 * versions), its manifest is rewritten, and the result is published.
 *
 * Dependencies between the published packages become npm aliases, e.g.
 *
 *   "@ai-sdk/harness": "npm:@ninox/ai-sdk-harness@1.0.53-ninox.0"
 *
 * so the shipped code's `import '@ai-sdk/harness'` resolves to the patched
 * build while consumers keep writing upstream import paths. Dependencies the
 * fork does not change (`ai`, `@ai-sdk/provider-utils`, …) keep their upstream
 * versions and resolve from the public registry.
 *
 * Usage:
 *   node tools/release-fork.mjs --dry-run
 *   node tools/release-fork.mjs --build 1
 *   node tools/release-fork.mjs --build 1 --only harness-pi      # one package, others only resolve aliases
 *
 * A version that is already on the registry is skipped, so a run that failed
 * halfway can simply be repeated.
 *
 * Environment:
 *   FORK_SCOPE      private scope to publish under      (default: @ninox)
 *   FORK_PREFIX     name prefix inside the scope        (default: ai-sdk-)
 *   FORK_REGISTRY   registry URL                        (default: https://npm.pkg.github.com)
 *   FORK_ACCESS     public | restricted                 (default: public)
 *   FORK_TAG        dist-tag                            (default: latest)
 *   FORK_SUFFIX     prerelease identifier               (default: ninox)
 *
 * `--registry` and `--access` also work as flags.
 */
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

/**
 * The packages this fork patches. Everything else in the workspace is left
 * alone — nothing here publishes by iterating the workspace, so an unrelated
 * package can never be pushed by accident.
 */
const TARGETS = ['harness', 'harness-pi', 'sandbox-just-bash'];

// TODO: change to @ninox
const SCOPE = process.env.FORK_SCOPE ?? '@sebninox';
const PREFIX = process.env.FORK_PREFIX ?? 'ai-sdk-';
const TAG = process.env.FORK_TAG ?? 'latest';
// TODO: change to ninox
const SUFFIX = process.env.FORK_SUFFIX ?? 'sebninox';

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const buildNumber = readFlag(args, '--build') ?? '0';
/**
 * Packages to publish, when not all of them. The plan still covers every
 * target so a dependency between fork packages resolves to its alias.
 */
const only = readFlag(args, '--only')?.split(',').map(name => name.trim());
const REGISTRY =
  readFlag(args, '--registry') ??
  process.env.FORK_REGISTRY ??
  'https://npm.pkg.github.com';
const ACCESS =
  readFlag(args, '--access') ?? process.env.FORK_ACCESS ?? 'public';

const repoRoot = resolve(import.meta.dirname, '..');

function readFlag(argv, flag) {
  const index = argv.indexOf(flag);
  return index === -1 ? undefined : argv[index + 1];
}

for (const name of only ?? []) {
  if (!TARGETS.includes(name)) {
    console.error(`release-fork: --only names "${name}", which is not one of: ${TARGETS.join(', ')}`);
    process.exit(1);
  }
}

/** Exit with a readable message instead of a Node stack trace. */
function fail(message) {
  console.error(`\nrelease-fork: ${message}\n`);
  process.exit(1);
}

function run(command, commandArgs, options = {}) {
  try {
    return execFileSync(command, commandArgs, {
      encoding: 'utf8',
      stdio: options.capture === true ? 'pipe' : 'inherit',
      cwd: options.cwd ?? repoRoot,
    });
  } catch (error) {
    if (options.capture === true) throw error;
    // The command already streamed its own diagnostics to the terminal.
    fail(`\`${command} ${commandArgs.join(' ')}\` failed.`);
  }
}

/** Command output, or `undefined` when it exits non-zero. */
function tryRun(command, commandArgs) {
  try {
    return execFileSync(command, commandArgs, {
      encoding: 'utf8',
      stdio: 'pipe',
      cwd: repoRoot,
    }).trim();
  } catch {
    return undefined;
  }
}

function manifestPath(pkg) {
  return join(repoRoot, 'packages', pkg, 'package.json');
}

function readManifest(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

/** `@ai-sdk/harness` -> `@ninox/ai-sdk-harness`. */
function forkName(upstreamName) {
  return `${SCOPE}/${PREFIX}${upstreamName.replace(/^@ai-sdk\//, '')}`;
}

// ─── Plan ────────────────────────────────────────────────────────────────

const plan = TARGETS.map(pkg => {
  const manifest = readManifest(manifestPath(pkg));
  return {
    pkg,
    upstreamName: manifest.name,
    name: forkName(manifest.name),
    version: `${manifest.version}-${SUFFIX}.${buildNumber}`,
  };
});
const byUpstreamName = new Map(plan.map(entry => [entry.upstreamName, entry]));

console.log(`\nPublishing ${plan.length} package(s) to ${REGISTRY}\n`);
for (const entry of plan) {
  console.log(`  ${entry.upstreamName}  ->  ${entry.name}@${entry.version}`);
}
console.log(dryRun ? '\n(dry run — nothing is published)\n' : '');

// ─── Preflight ───────────────────────────────────────────────────────────

/*
 * Check credentials before building and packing. Publishing happens one
 * package at a time, so discovering a missing token on the second one would
 * leave a partial release behind.
 */
if (!dryRun) {
  const user = tryRun('npm', ['whoami', '--registry', REGISTRY]);
  if (user == null) {
    fail(
      `not authenticated to ${REGISTRY}.\n\n` +
        `  npmjs.com:\n` +
        `    npm login --registry https://registry.npmjs.org --scope ${SCOPE}\n` +
        `    (then re-run with --registry https://registry.npmjs.org)\n\n` +
        `  GitHub Packages — add to ~/.npmrc, with a token carrying write:packages:\n` +
        `    //npm.pkg.github.com/:_authToken=<token>\n\n` +
        `Current target registry comes from --registry, then FORK_REGISTRY, ` +
        `then the built-in default.`,
    );
  }
  console.log(`Authenticated to ${REGISTRY} as ${user}.`);

  /*
   * Both npmjs and GitHub Packages only accept a scope you own. Warn rather
   * than block: the scope may be an organisation the user belongs to, which
   * `whoami` cannot tell us.
   */
  if (SCOPE !== `@${user}`) {
    console.log(
      `Note: publishing under ${SCOPE} as "${user}" — this only works if ` +
        `${SCOPE} is an organisation you can publish to.`,
    );
  }
}

// ─── Build ───────────────────────────────────────────────────────────────

console.log('Building…');
run('pnpm', [
  ...TARGETS.flatMap(pkg => ['--filter', `@ai-sdk/${pkg}`]),
  'build',
]);

// ─── Pack, rewrite, publish ──────────────────────────────────────────────

const staging = mkdtempSync(join(tmpdir(), 'ai-fork-release-'));
try {
  for (const entry of plan) {
    if (only != null && !only.includes(entry.pkg)) {
      console.log(`\nSkipping ${entry.name}@${entry.version} (not in --only).`);
      continue;
    }
    // npm refuses to publish over an existing version, and that refusal used to
    // stop the whole run before the later packages had their turn.
    if (
      !dryRun &&
      tryRun('npm', ['view', `${entry.name}@${entry.version}`, 'version', '--registry', REGISTRY]) != null
    ) {
      console.log(`\nSkipping ${entry.name}@${entry.version}: already on ${REGISTRY}.`);
      continue;
    }
    const packageDir = join(repoRoot, 'packages', entry.pkg);
    const dest = join(staging, entry.pkg);

    // `pnpm pack` applies the manifest's `files` list and resolves
    // `workspace:*` to concrete versions, so the tarball is already correct
    // apart from the names.
    run('pnpm', ['pack', '--pack-destination', dest], { cwd: packageDir });
    const tarball = readdirSync(dest).find(file => file.endsWith('.tgz'));
    if (tarball == null) {
      throw new Error(`pnpm pack produced no tarball for ${entry.upstreamName}`);
    }
    run('tar', ['xzf', join(dest, tarball), '-C', dest]);

    const extracted = join(dest, 'package');
    const target = join(extracted, 'package.json');
    const manifest = readManifest(target);

    manifest.name = entry.name;
    manifest.version = entry.version;

    for (const section of ['dependencies', 'peerDependencies', 'optionalDependencies']) {
      const deps = manifest[section];
      if (deps == null) continue;
      for (const depName of Object.keys(deps)) {
        const dep = byUpstreamName.get(depName);
        if (dep == null) continue;
        /*
         * Alias, not a rename: the shipped code imports the upstream specifier,
         * so the key must stay `@ai-sdk/…` while resolving to the fork build.
         */
        deps[depName] = `npm:${dep.name}@${dep.version}`;
      }
    }

    manifest.publishConfig = {
      ...manifest.publishConfig,
      registry: REGISTRY,
      /*
       * `access` is inherited from the upstream manifest, where it is
       * `public` — correct for the real AI SDK, but on npmjs it also makes a
       * fork world-readable. Set FORK_ACCESS=restricted to keep it private
       * (npmjs requires a paid plan; GitHub Packages follows repo visibility).
       */
      access: ACCESS,
      // Provenance needs a CI runner with OIDC; it fails a local publish.
      provenance: false,
    };
    manifest.repository = manifest.repository ?? {
      type: 'git',
      url: 'git+https://github.com/seb-ninox/ai.git',
    };

    writeFileSync(target, `${JSON.stringify(manifest, null, 2)}\n`);

    if (dryRun) {
      console.log(`\n--- ${entry.name}@${entry.version} (not published)`);
      console.log(JSON.stringify(
        {
          name: manifest.name,
          version: manifest.version,
          publishConfig: manifest.publishConfig,
          dependencies: manifest.dependencies,
        },
        null,
        2,
      ));
      continue;
    }

    console.log(`\nPublishing ${entry.name}@${entry.version}…`);
    run('npm', ['publish', '--registry', REGISTRY, '--tag', TAG], {
      cwd: extracted,
    });
  }
} finally {
  rmSync(staging, { recursive: true, force: true });
}

// ─── Consumer snippet ────────────────────────────────────────────────────

console.log('\nAdd to the consuming project\'s package.json:\n');
console.log(JSON.stringify(
  Object.fromEntries(
    plan.map(entry => [entry.upstreamName, `npm:${entry.name}@${entry.version}`]),
  ),
  null,
  2,
));
console.log(`\nand to its .npmrc:\n\n  ${SCOPE}:registry=${REGISTRY}\n`);
