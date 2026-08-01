# Releasing this fork

This fork patches three packages:

| Package                     | Why it is forked                                                             |
| --------------------------- | ---------------------------------------------------------------------------- |
| `@ai-sdk/harness`           | tool context delivery, self-contained stream blocks, sandbox workdir quoting |
| `@ai-sdk/harness-pi`        | dependency bump to `pi-coding-agent` 0.82.1                                  |
| `@ai-sdk/sandbox-just-bash` | durable sessions, `just-bash` 3.2.0                                          |

Nothing else in the workspace is changed, and nothing else is published.

## How it works

The repo keeps upstream package names, so merging from upstream stays clean. Renaming happens only in the published artifact:

```
@ai-sdk/harness            ->  @ninox/ai-sdk-harness
@ai-sdk/harness-pi         ->  @ninox/ai-sdk-harness-pi
@ai-sdk/sandbox-just-bash  ->  @ninox/ai-sdk-sandbox-just-bash
```

Dependencies **between** published packages become npm aliases:

```json
"@ai-sdk/harness": "npm:@ninox/ai-sdk-harness@1.0.53-ninox.0"
```

The key stays `@ai-sdk/harness` because the shipped code imports that specifier; the value points at the fork build. Dependencies the fork does not change — `ai`, `@ai-sdk/provider`, `@ai-sdk/provider-utils` — keep their upstream versions and resolve from the public registry. That is why `ai` is deliberately **not** forked: the one helper the harness needed from `ai/internal` now lives in `packages/harness/src/agent/internal/validate-tool-context.ts`.

Versions are `<upstream>-ninox.<build>`, e.g. `1.0.53-ninox.0`. Bump `--build` to republish against the same upstream version.

## Publishing

```bash
pnpm release:fork -- --dry-run
```

Prints the rewritten manifests and publishes nothing. When it looks right:

```bash
pnpm release:fork -- --build 0
```

Authenticate first. For GitHub Packages, with a token carrying `write:packages`:

```
//npm.pkg.github.com/:_authToken=${GITHUB_TOKEN}
@ninox:registry=https://npm.pkg.github.com
```

GitHub Packages requires the scope to match the owner of the repository the package is published from. This repo is `seb-ninox/ai`, so either publish from a repository under a `ninox` organisation, or set `FORK_SCOPE=@seb-ninox`.

Overridable via environment:

| Variable        | Default                      |
| --------------- | ---------------------------- |
| `FORK_SCOPE`    | `@ninox`                     |
| `FORK_PREFIX`   | `ai-sdk-`                    |
| `FORK_REGISTRY` | `https://npm.pkg.github.com` |
| `FORK_TAG`      | `latest`                     |
| `FORK_SUFFIX`   | `ninox`                      |

## Consuming from ninox4

`.npmrc`:

```
@ninox:registry=https://npm.pkg.github.com
//npm.pkg.github.com/:_authToken=${GITHUB_TOKEN}
```

`package.json` — imports stay exactly as they are today:

```json
{
  "dependencies": {
    "@ai-sdk/harness": "npm:@ninox/ai-sdk-harness@1.0.53-ninox.0",
    "@ai-sdk/harness-pi": "npm:@ninox/ai-sdk-harness-pi@1.0.53-ninox.0",
    "@ai-sdk/sandbox-just-bash": "npm:@ninox/ai-sdk-sandbox-just-bash@1.0.53-ninox.0"
  }
}
```

Keep `ai` on the public release. It must satisfy the range the forked harness declares (`ai@7.0.47` at the time of writing) — pin the same version in ninox4 so npm does not install two copies.

## Adding a package to the release

Add its directory name to `TARGETS` in `tools/release-fork.mjs`. Only fork a package you have actually changed: every forked package is one you must re-merge and republish on each upstream release.

## After merging upstream

1. Resolve conflicts, run the affected packages' tests.
2. Confirm the fork surface has not grown:

   ```bash
   git diff --name-only <upstream-base> HEAD -- packages | cut -d/ -f2 | sort -u
   ```

   It should list exactly the three packages above.

3. Re-publish with a bumped `--build`.
