# What is this project about?
Package `shapiq-student` is a Group 2 project for the Softwareentwicklungspraktikum:
Game Theoretic Explainable Artificial Intelligence (SoSe 2025). It serves as
an extension of the existing Python library, [`shapiq`](https://github.com/mmschlk/shapiq),
a library for explaining machine learning models with Shapley interactions.

The main development under `shapiq_student/` contains three parts:
1. KNN-explainer
2. Conditonal Imputers
3. Coalition-Finding Algorithmus

Tutorials and demonstrations are available in EXAMPLES.

### Project Structure
* `.github/` Contains GitHub Actions workflows that run automated tests and other checks on every commit.
* `docs/` This folder is used for creating and maintaining the project documentation.
The documentation page is created with Sphinx with theme furo and published via GitHub Pages.
* `shapiq_student/` This is the where the main functionalities: KNN-explainer; Conditonal Imputers; Coalition-Finding Algorithmus are implemented.
* `tests/` Contains various Uni-tests for shapiq_student.
* `tests_grading/` Contains tests for final examination.

### Extra
To ensure a modern Python project codebase, this project incorporates package and project manager [`uv`](https://docs.astral.sh/uv/), linting tool [`Ruff`](https://docs.astral.sh/ruff/),
static type checking tool [`mypy`](https://mypy.readthedocs.io/en/stable/), and [`Commitizen`](https://commitizen-tools.github.io/commitizen/) for release management.
