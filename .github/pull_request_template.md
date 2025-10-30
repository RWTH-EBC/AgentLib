# Pull Request

## Description
<!-- Brief description of changes (2-3 sentences) -->


## Type of Change
<!-- Check relevant boxes with [x] -->
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Refactoring (code change that neither fixes a bug nor adds a feature)
- [ ] Documentation update

## Required Checklist

### Testing
- [ ] Unit tests have been created/updated for new/modified functionality
- [ ] CI/CD pipeline passes all tests (pytest, pylint, coverage)

### Examples
- [ ] Add examples for new features and functionality

### Compatibility
- [ ] Changes are backward compatible OR deprecation warnings added
- [ ] No breaking changes to public APIs (Agent, BaseModule, DataBroker, Environment, Model, AgentVariable)
- [ ] New dependencies added to `pyproject.toml` (required in `dependencies` or optional in `[project.optional-dependencies]`)

### OPTIONAL: Release Information
- [ ] Version number updated in `agentlib/__init__.py` (**Required for PyPI release**)
  - Version: `___.___.___`
- [ ] CHANGELOG.md updated with description of changes (**Required for PyPI release**)

### Documentation
- [ ] Docstrings added/updated for new/modified public methods (Google style)
- [ ] Type hints added for new functions/methods

## Breaking Changes
<!-- If "Breaking change" checked above, describe impact and migration path -->

## Optional

### GitHub Copilot Review
<!-- Request a review from GitHub Copilot to get AI-powered feedback on your changes -->
- [ ] Request Copilot review via GitHub UI (add 'copilot' as a reviewer)

---
**Note**: PRs will not be merged without completed required checklist items and passing CI/CD pipeline.
