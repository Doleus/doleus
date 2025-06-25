# Contributing to Doleus 🐦‍⬛

Thank you for considering contributing to **Doleus**! We welcome all contributions — from bug reports and documentation to new features and improvements.

The best ways to contribute to Doleus:

- Submit and vote on [Ideas](https://github.com/doleus/doleus/discussions) and [Feature Requests](https://github.com/doleus/doleus/issues)
- Create and comment on [Issues](https://github.com/doleus/doleus/issues)
- Open a Pull Request
- Improve documentation and examples
- Help other users on [Discord](https://discord.gg/B8SzfbGRA9)

Please take a moment to review these guidelines before submitting a pull request.

> **And if you like the project, but don't have time to contribute code, that's fine! There are other ways to support the project:**
>
> - ⭐ Star the project
> - 🐦 Tweet about it
> - 📝 Reference Doleus in your project's documentation
> - 💬 Mention it at meetups and tell your colleagues
> - 🗳️ Submit feature requests and vote on [Ideas](https://github.com/doleus/doleus/discussions)
> - 🤝 Help answer questions on [Discord](https://discord.gg/B8SzfbGRA9)

---

## 🛠️ How to Contribute

### Making a Change

**Before making any significant changes, please [open an issue](https://github.com/doleus/doleus/issues).** Discussing your proposed changes ahead of time will make the contribution process smooth for everyone. Changes that were not discussed in an issue may be rejected.

A good first step is to search for open [issues](https://github.com/doleus/doleus/issues).

### Contribution Process

1. **Fork** the repository on GitHub.
2. **Clone your fork** and create a new branch:
   ```bash
   git checkout -b <issue-number>-your-branch-name
   # Example: git checkout -b 42-fix-metadata-bug
   ```
3. Make your changes and commit using **[Conventional Commits](https://www.conventionalcommits.org/)**.
4. Push your branch to your fork:
   ```bash
   git push origin <branch-name>
   ```
5. Open a **pull request** to the `main` branch of the original repo.

---

## 💡 Development Setup

### Requirements

- **Python 3.11+** (as specified in `pyproject.toml`)
- **Poetry** for dependency management and packaging

### Steps

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/doleus.git
   cd doleus
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install
   ```

3. Activate the virtual environment:

   ```bash
   poetry shell
   ```

4. Install pre-commit hooks (optional but recommended):

   ```bash
   pre-commit install
   ```

5. Verify the installation by running tests:
   ```bash
   poetry run pytest
   ```

---

## 📏 Coding Standards

We use several tools to maintain code quality:

- **[Pre-commit hooks](https://pre-commit.com/)** for automated formatting and linting
- **[Conventional Commits](https://www.conventionalcommits.org/)** for commit messages

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Examples:

- `feat: add new metadata function for medical imaging`
- `fix: resolve issue with slice inheritance of predictions`
- `docs: update quick start guide for object detection`

### Code Formatting

Code should be formatted using:

- **Black**: ensures consistent formatting across the codebase.
- **isort**: organizes imports into sections and sorts them alphabetically.

To format your code before committing:

```bash
poetry run black .
poetry run isort .
```
---

## 🧪 Running Tests

Run the test suite using Poetry:

```bash
# Run all tests
poetry run pytest
```

For specific test files:

```bash
poetry run pytest tests/test_specific_module.py
```

---

## 📬 Reporting Issues

Please use the [GitHub Issue Tracker](https://github.com/doleus/doleus/issues) to report bugs or suggest features.

Include as much detail as possible:

- **Clear and descriptive title**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (Python version, OS, etc.)
- **Relevant logs or screenshots**
- **Code snippets** demonstrating the issue (if applicable)

For feature requests, please describe:

- The use case and problem you're trying to solve
- How the feature would work
- Any alternative solutions you've considered

---

## ✅ Pull Request Guidelines

- Reference the issue number in your PR title (e.g., `[#42] Fix metadata inheritance bug`)
- Provide a clear description of the changes
- Include tests for new functionality
- Update documentation if needed
- Ensure all CI checks pass
- Keep changes focused and atomic

---

## 💬 Community

The maintainers and community are available on [Discord](https://discord.gg/B8SzfbGRA9) if you have questions or need help getting started.

---

## 📄 License Notice

By submitting a contribution, you agree that it will be licensed under the same license as this project (Apache 2.0).

---
