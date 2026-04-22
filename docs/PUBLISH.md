# Publishing `damoco` to PyPI

This repository is configured for publishing with GitHub Actions + PyPI Trusted Publisher (OIDC), so no PyPI API token is required in GitHub secrets.

## One-time Setup (PyPI)

1. Create or log in to your PyPI account: https://pypi.org
2. Create a new project entry name by publishing once manually or by first trusted publish (project name: `damoco`).
3. In PyPI project settings, configure **Trusted Publishers** with:
   - Owner: `cagdastopcu`
   - Repository: `damoco`
   - Workflow filename: `publish-pypi.yml`
   - Environment name: `pypi`

## One-time Setup (GitHub)

1. In repository settings, create an environment named `pypi`.
2. Optional but recommended: add protection rules (required reviewers) for that environment.

## Release Workflow

1. Update version in `pyproject.toml` (for example `0.1.1`).
2. Commit and push.
3. Create a Git tag and push it:

```bash
git tag v0.1.1
git push origin v0.1.1
```

4. Create a GitHub Release from that tag (or use the tag push + workflow dispatch).
5. Workflow `.github/workflows/publish-pypi.yml` builds and publishes to PyPI.

## Local Validation Before Release

```bash
python -m pip install -U build twine
python -m build
python -m twine check dist/*
```

## Install After Publish

```bash
pip install damoco
```
