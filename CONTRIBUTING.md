# Contributing to hner

Thank you for your interest in contributing to `hner`! Contributions of all kinds are welcome — bug fixes, new features, documentation improvements, and more.

Please take a moment to read these guidelines before opening an issue or submitting a pull request.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Report a Bug](#how-to-report-a-bug)
- [How to Request a Feature](#how-to-request-a-feature)
- [Development Setup](#development-setup)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Coding Conventions](#coding-conventions)

---

## Code of Conduct

All contributors are expected to be respectful and constructive. Harassment or abusive behaviour of any kind will not be tolerated.

---

## How to Report a Bug

1. Search [existing issues](https://github.com/racai-ai/hner/issues) to check whether the bug has already been reported.
2. If not, open a new issue using the **Bug Report** template.
3. Include as much detail as possible: steps to reproduce, expected vs. actual behaviour, relevant log output, and your environment (OS, Python version, PyTorch version).

---

## How to Request a Feature

1. Search [existing issues](https://github.com/racai-ai/hner/issues) to see if the feature has already been requested.
2. Open a new issue using the **Feature Request** template.
3. Describe the problem you are trying to solve and why an existing solution does not cover it.

---

## Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/hner.git
cd hner

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt
pip install -e .
```

---

## Submitting a Pull Request

1. **Branch** off `main` and use a descriptive branch name, e.g. `fix/bert-adapter-loading` or `feat/new-model-type`.
2. **Keep changes focused** — one logical change per pull request.
3. **Write tests** when adding or changing functionality (add them under a `tests/` directory following the existing patterns).
4. **Update documentation** (docstrings, README) if your change affects the public interface or CLI.
5. **Run existing tests** before opening the PR:
   ```bash
   python -m pytest tests/
   ```
6. Open the PR against the `main` branch and fill in the pull-request description, explaining what the change does and why.
7. Link any related issues using `Fixes #<issue-number>` in the PR description.

A maintainer will review your PR as soon as possible. Feedback may be requested before the PR is merged.

---

## Coding Conventions

- Follow **PEP 8** style. Use type hints for all public functions and methods.
- Keep docstrings in [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html) to match the existing codebase.
- Use absolute imports inside the `src/` package.
- Avoid adding new top-level dependencies unless strictly necessary; discuss in an issue first.
