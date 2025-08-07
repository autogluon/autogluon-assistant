# Release Instructions

## Prior to Release

1. **Version Check**
   * Ensure version in `pyproject.toml` matches intended release version
   * Check all dependency version ranges in `pyproject.toml`
     * Ensure dependencies have appropriate upper and lower bounds
     * Avoid exact version pinning unless necessary
     * Document any version restrictions with inline comments

2. **Dependency Review**
   * Test compatibility with latest versions of key dependencies:
     * autogluon.tabular
     * langchain
     * openai
     * streamlit
   * Update version ranges if needed and document any compatibility issues

3. **Code Freeze**
   * Communicate code freeze to contributors
   * Only merge release-critical PRs
   * Wait 24 hours after code freeze for pre-release testing



## Triggering Test Release

1. **Trigger Test Release Workflow**
   * Go to Actions → Test PyPI Release workflow
   * Click "Run workflow" dropdown
   * Select branch (usually `main`)
   * Click "Run workflow"

2. **Monitor Workflow**
   * Wait for workflow to complete (~2 minutes)
   * Check build artifacts and logs for any issues
   * Note the test version number from logs (format: `{version}dev{YYYYMMDDHHSS}`)

3. **Verify on TestPyPI**
   * Go to https://test.pypi.org/project/autogluon.assistant/
   * Confirm new version is listed
   * Check package metadata and files are correct

4. **Test Installation**
   ```bash
   # Create fresh virtual environment
   python -m venv test_env
   source test_env/bin/activate  # or `test_env\Scripts\activate` on Windows

   # Install from TestPyPI
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple \
       autogluon.assistant=={test_version}

   # Verify version
   python -c "import autogluon.assistant; print(autogluon.assistant.__version__)"
   ```

5. **Test Core Functionality**
   ```bash
   # Test UI
   aga ui

   # Test CLI with toy dataset
   aga run toy_data --config_overrides "feature_transformers.enabled_models=None, autogluon.predictor_fit_kwargs.time_limit=3600"
   ```

6. **Common Issues to Check**
   * All dependencies resolved correctly
   * Package data files included (configs, UI assets)
   * Import paths working
   * No permission issues
   * UI loads correctly
   * CLI commands functional

7. **If Issues Found**
   * Fix issues in code
   * Trigger new test release
   * Repeat verification steps

Only proceed with official release after successful test release verification.

## Release

1. **Final Checks**
   * Ensure all CI checks pass on main branch
   * Verify documentation is up-to-date
   * Test installation and core features one final time

2. **Prepare Release Notes**
   * Prepare the release notes located in docs/whats_new/v0.x.y.md:
     * This will be copy-pasted into GitHub when you release.
     * Include all merged PRs into the notes and mention all PR authors / contributors (refer to past releases for examples).
     * Prioritize major features before minor features when ordering, otherwise order by merge date.
     * Review with at least 2 core maintainers to ensure release notes are correct.

   * Push the release notes to the main branch.

3. **Release Branch**
   * Create a branch with format `0.x.y` (no v prefix) from main
   * Push the branch

4. **Create GitHub Release**
   * Tag: `v{version}` (e.g., `v0.1.0`)
   * Title: Same as tag
   * Description: Include:
     * Major features/changes
     * Breaking changes (if any)
     * Bug fixes
     * Contributors
   * Target: main branch
   * DO NOT use the 'Save draft' option during the creation of the release. This breaks GitHub pipelines.
   * Copy-paste the content of docs/whats_new/v0.x.y.md into the release notes box.
   * Ensure release notes look correct and make any final formatting fixes.
   * Click 'Publish release' and the release will go live.

5. **PyPI Release**
   * Release will be automatically triggered by GitHub release
   * Monitor the `pypi_release.yml` workflow
   * Verify package appears on PyPI
   * Test installation from PyPI in fresh environment

## Post Release

1. **Version Bump**
   * On main branch, bump version in `pyproject.toml` to next dev version
     * e.g., `0.1.0` → `0.1.1.dev0`

2. **Verification**
   * Verify installation works:
     ```bash
     pip install autogluon.assistant=={version}
     ```
   * Test basic functionality
   * Check documentation is accessible

3. **Announcements**
   * Update release notes in documentation
   * Announce in relevant channels:
     * GitHub Discussions
     * AutoGluon Slack
     * Other relevant communities


## Nightly Releases

Nightly releases are automatically published to PyPI with version format:
`{version}.dev{YYYYMMDD}`

* Triggered daily by `nightly_release.yml` at 08:59 UTC
* Will attempt to publish daily regardless of changes
* Used for testing latest development changes
