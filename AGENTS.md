# Repository Guidelines

## Project Structure & Module Organization
This workspace combines local scripts with two imported research codebases. Use `safety-gymnasium-main/` for Safety-Gymnasium source work: package code is in `safety_gymnasium/`, tests in `tests/`, examples in `examples/`, documentation in `docs/`, and media/assets in `images/` and `safety_gymnasium/assets/`. SafeDreamer code lives under `external/SafeDreamer/`; avoid changing it unless the task specifically targets SafeDreamer integration. Root-level `scripts/` contains local utilities such as `test.py` and `inspect_obs.py` for quick environment checks.

## Build, Test, and Development Commands
Run package commands from `safety-gymnasium-main/` unless noted.

- `python -m pip install -e .`: install Safety-Gymnasium in editable mode.
- `make test`: install test extras and run the full pytest suite with coverage.
- `make py-format`: check `isort` and `black` formatting.
- `make ruff` or `make flake8`: run static lint checks.
- `make docs`: serve the Sphinx docs locally.
- `python scripts/test.py`: from the repository root, run a short SafetyPointGoal smoke test.
- `python scripts/inspect_obs.py --env SafetyPointGoal1-v0`: inspect observation and action spaces.

## Coding Style & Naming Conventions
Python uses 4-space indentation, LF endings, UTF-8, and final newlines. Follow the settings in `safety-gymnasium-main/pyproject.toml`: Black and isort use a 100-character line length, Google-style docstrings are preferred, and inline strings generally use single quotes. Name tests `test_*.py`; keep modules and packages lowercase with underscores.

## Testing Guidelines
Use pytest for package tests. Add or update tests in `safety-gymnasium-main/tests/` when changing environment behavior, wrappers, configs, or public APIs. Prefer deterministic seeds in environment tests, and check both reward and cost behavior when safety semantics are relevant. For quick integration checks, run the root `scripts/test.py` smoke test before broader pytest runs.

## Commit & Pull Request Guidelines
The top-level checkout does not expose a reliable project Git history, so use clear imperative commits such as `fix safety goal reset cost` or `add observation inspection script`. Pull requests should describe the behavioral change, list commands run, mention affected environments, and link issues or experiment notes when applicable. Include screenshots or rendered docs only for visual or documentation changes.

## Security & Configuration Tips
Do not commit generated caches, coverage files, local virtual environments, or large archives such as `main.zip`. Keep dependency changes scoped and document version-sensitive behavior, especially Gymnasium-Robotics compatibility warnings.
