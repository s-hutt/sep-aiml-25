# Changelog
This changelog is to be updated with each important change to the codebase.

## Development

## 0.7.0 (2025-07-20)

### Feat

- **runtime_analyze**: added runtime_analysis_shap_vs_shapiq.ipynb to the doc page (#45)

## 0.6.0 (2025-07-20)

### Feat

- removed the ruff check for TODO comments. finished the introduction page (#43)

## 0.5.1 (2025-07-20)

### Fix

- **api_rendering**: registered new modules to api references on the doc page (#41)

## 0.5.0 (2025-07-20)

### Feat

- **coalition_finding**: add beam_coalition, subset_finding, and belonging tests (#40)

## 0.4.5 (2025-07-18)

### Fix

- **imputer**: Remove the unnecessary attibutes that only meaningful for the generative conditional imputer. (#37)

## 0.4.4 (2025-07-17)

### Fix

- **ci**: make coverage test a separate workflow. (#35)

## 0.4.3 (2025-07-15)

### Fix

- Updated the imputer notebook; Translated english comments into German (#33)

## 0.4.2 (2025-07-06)

### Fix

- **ci**: reduce minimal test coverage from 95% to 92%. (#25)

## 0.4.1 (2025-07-06)

### Fix

- **docs**: link shapiq_student to sphinx's api references (#24)

## 0.4.0 (2025-07-06)

### Feat

- **imputers**: add imputers and tests with >95% coverage

## 0.3.3 (2025-06-26)

### Refactor

- New tests_grading files and default python modules are added for matching the new 1.3.0 version shapiq.

## 0.3.2 (2025-06-17)

### Fix

- **ci**: install pandoc to fix Sphinx doc build on CI

## 0.3.1 (2025-06-17)

### Fix

- **ci**: rename and update Sphinx workflow. sphinx.yml should trigger when the main branch is changed.

## 0.3.0 (2025-06-17)

### Feat

- **docs**: integrate Sphinx autodoc for API generation; Setup the doc page structure

## 0.2.0 (2025-06-02)

### Feat

- **ci**: automate changelog, docs deploy, and lint ignores
- Automatically update CHANGELOG.md after each push to the main branch using GitHub Actions
- Deploy documentation as a static site at https://s-hutt.github.io/sep-aiml-25/
- Configure Ruff and Mypy to ignore docs/source/conf.py
- Configure Mypy to ignore tests_grading/
- Miscellaneous improvements for smooth CIs.

## 0.1.1 (2025-06-02)

## 0.1.0 (2024-05-12)
- Initial skeleton code-base of `shapiq_student` without functionalities.
