---
name: Minterpy Release Checklist
about: Use this checklist for releasing a new version of Minterpy
title: "[RELEASE] Minterpy vxx.yy.zz"
labels: release
assignees: ''

---

This issue tracks the release process for Minterpy `vxx.yy.zz`. The checklist is organized into three phases:

- preparing the release on `dev`,
- merging to and tagging `main`, and
- verifying the release.

Steps are **sequential**. Specifically, several pre-merge steps must be completed before pushing the tag, as the tag push immediately triggers the release workflow and publishes to PyPI **with no opportunity** to intervene.

## Pre-merge (on `dev` branch)

> Complete all steps in this section before touching the `main` branch.

- [ ] All CI checks passing on `dev` (tests, build, docs-preview, coverage)
- [ ] Merge or defer all in-scope issues to `dev`
- [ ] Update `CHANGELOG.md` *(this will serve as the basis for the GitHub release notes in the post-release step)*
  - [ ] Rename `[Unreleased]` to `[vxx.yy.zz] - YYYY-MM-DD` and update its comparison link to diff against the previous tagged release *(the link will appear broken until the tag is pushed in the next phase: this is expected)*
  - [ ] Create a new empty `[Unreleased]` section at the top with a comparison link between `HEAD` on `main` and `dev`
- [ ] Build and verify documentation (no broken references, no build warnings)

> CI pipelines include a documentation preview served to GitHub pages but to save time the notebook-based documentation is not executed; make sure you can build the documentation of the `dev` branch locally.

- [ ] Create a new draft version for the release in RODARE without publishing to reserve DOI *(do this before merging to `main` so the DOI can be embedded into the README file as part of the tagged commit)*
- [ ] Update `README.md` with the new DOI (badge and citation section)

## Merge, tag, and push

- [ ] Merge `dev` into `main` (via Pull Request)

> ⚠️ **This is the point of no return**: the next step creates and pushes the tag, which triggers the release workflow automatically. The workflow pushes the artifacts to PyPI and creates a GitHub release with all attached artifacts. PyPI does not allow re-uploading a package with the same version. Verify all pre-merge steps are complete before proceeding.
>
> The package version is derived automatically from the tag via `setuptools_scm` and written to `src/minterpy/version.py` at build time. Ensure the tag follows the format `vxx.yy.zz` exactly, as this becomes the version string on PyPI.

- [ ] Create an annotated tag: `git tag -a vxx.yy.zz -m "Release vxx.yy.zz"`
- [ ] Push the tag: `git push origin vxx.yy.zz`

## Post-release verification

- [ ] Verify the GitHub Actions release workflow completed successfully
- [ ] Verify the GitHub release page was created correctly (tag and attached artifacts)
- [ ] Write and publish the release notes on the GitHub release page

> The GitHub Actions workflow creates the release page automatically upon tag push but leaves the release notes empty. Populate them manually from `CHANGELOG.md`.

- [ ] Upload the release artifacts from the GitHub release to the RODARE draft and finalize and publish the release
- [ ] Verify the DOI resolves correctly in RODARE
- [ ] Verify install from PyPI in a clean environment:
  - [ ] `pip install minterpy==xx.yy.zz`
  - [ ] `python -c "import minterpy as mp; assert mp.__version__ == 'xx.yy.zz'"`
