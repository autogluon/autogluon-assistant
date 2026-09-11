---
name: mlzero-env
description: Use before invoking mlzero to build or reuse a pinned conda environment (autogluon==1.6.1 plus the assistant) and record its path for the run.
---

# mlzero-env

**Purpose.** Guarantee mlzero-generated code runs against a known-good, pinned environment,
and hand that environment to mlzero via `--conda-env`. Reuse an existing env when possible to
save minutes.

**Inputs.** Workspace path `<ws>`; optionally a preferred env name (default `mlzero`).

**Steps.**
1. **Reuse if valid.** List envs (`conda env list`); for a candidate, verify autogluon is
   pinned at 1.6.1 (note: `autogluon` is a namespace package with no top-level `__version__`):
   ```
   conda run -n <env> python -c "import importlib.metadata as m; assert m.version('autogluon').startswith('1.6.1')"
   ```
   If it passes and the editable assistant imports (`python -c "import autogluon.assistant"`), reuse it.
2. **Build if missing.** Create and pin:
   ```
   conda create -y -n <env> python=3.11
   conda run -n <env> pip install "autogluon==1.6.1"
   conda run -n <env> pip install -e .   # the assistant, from repo root
   ```
3. **Record the path.** Resolve the env prefix and write it:
   ```
   conda run -n <env> python -c "import sys;print(sys.prefix)" > <ws>/env/env_path.txt
   ```

**Outputs.** `<ws>/env/env_path.txt` (absolute path to the conda env prefix).

**Notes/guardrails.**
- Pin `autogluon==1.6.1` exactly — version drift is a common source of irreproducible runs.
- This env runs the *generated solution code*. The LLM agents themselves (coder/perception on
  Opus 4.8) are configured in mlzero's YAML, independent of this env.
- Do not reinstall a valid env; reuse is the fast path. Only rebuild on version/import failure.
