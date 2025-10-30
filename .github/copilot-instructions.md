# GitHub Copilot Review Agent Instructions

## Repository Overview

**AgentLib** is a Python framework for developing and executing distributed agents for control and simulation of energy systems. It provides base classes and modules for implementing multi-agent systems with support for distributed simulations, real-time control, and various communication protocols.

### Key Information
- **Language**: Python (3.9-3.12)
- **License**: BSD-3-Clause
- **Testing**: unittest framework
- **Code Quality**: pylint (target score =10.0)
- **Dependencies**: Core (numpy, pandas, simpy, pydantic>=2.0, attrs), Optional (FMU, scipy, MQTT, matplotlib)
- **CI/CD**: Automated testing (pytest, coverage, pylint), sphinx documentation
- **Plugin Ecosystem**: Extensible via plugins (AgentLib-MPC, AgentLib-FIWARE)

## Project Structure

```
agentlib/
├── core/          # Core framework classes (Agent, BaseModule, Environment, DataBroker, Model)
├── models/        # Model implementations (FMU, scipy state-space)
├── modules/       # Reusable modules (communicator, controller, simulation, utils)
└── utils/         # Utilities (MAS, brokers, config loading, plugins)
tests/             # Unit tests mirroring agentlib structure
examples/          # Usage examples and tutorials
docs/              # Sphinx documentation
```

### Critical Architecture Components
- **Agent**: Main reactive agent class, orchestrates modules via DataBroker
- **BaseModule**: Abstract base for all modules; requires `Config` class (pydantic), `register_callbacks()`, and `process()` methods
- **DataBroker**: Event-driven communication hub between modules (Local/RT variants)
- **Environment**: SimPy-based simulation environment (simulated or real-time)
- **Model**: Abstract base for simulation models (FMU, scipy, custom)
- **AgentVariable**: Data container for agent communication with type safety

### Configuration Pattern
All components use Pydantic v2 models (e.g., `AgentConfig`, `BaseModuleConfig`, `ModelConfig`) for declarative configuration with validation.

## Review Guidelines

### Core Principle: Concise, High-Impact Reviews
**Focus only on critical issues that affect:**
- Backward compatibility and breaking changes
- CI/CD pipeline failures
- Missing tests for new functionality
- Undocumented changes to public APIs

**Avoid commenting on:**
- Minor refactoring preferences
- Well-tested internal implementation details

### Primary Objectives
1. **Provide concise PR summary** (2-3 sentences max) highlighting purpose and scope
2. **Classify changes** as bug fixes, new features, refactoring, or documentation
3. **Assess backward compatibility** - flag breaking changes to public APIs
4. **Highlight CI risks** - identify changes likely to cause test/build failures
5. **Respond in English only**

### Specific Review Focus Areas

#### 1. Breaking Changes & Compatibility
- **Flag breaking changes** to: `Agent`, `BaseModule`, `BaseModuleConfig`, `Environment`, `DataBroker`, `Model`, `AgentVariable` classes
- **Configuration schema changes**: Any modification to Config classes affects user JSON/dict configurations
- **Plugin interface changes**: Changes to `BaseModule` or registration mechanisms impact external plugins
- **Suggest deprecation warnings**: Breaking changes should ideally use `warnings.warn()` with deprecation notice before removal

#### 2. Code Quality & Standards
- **Pydantic usage**: Ensure v2 syntax (e.g., `Field()`, `field_validator`, not v1 `validator`)
- **Type hints**: Verify type annotations on public methods and new functions
- **Docstrings**: Check for docstrings on new public classes/methods (Google/NumPy style)
- **Pylint compliance**: Flag obvious issues (unused imports, naming violations, etc.)

#### 3. Testing Requirements
- **New features require tests**: Flag new modules, functions, or classes without corresponding test files
- **Test file naming**: Should match `test_<module>.py` pattern in `tests/`
- **Coverage concerns**: Highlight complex logic added without test coverage

#### 4. Documentation & Examples
- **Core API changes**: Should update `docs/source/` RST files
- **New modules/models**: Should have entry in `docs/source/ModulesList.md`
- **Configuration changes**: May require updates to `docs/source/Tutorials/` examples
- **CHANGELOG.md**: All notable changes should be documented

#### 5. CI/CD Considerations
- **Import errors**: New dependencies must be in `setup.cfg`/`pyproject.toml` optional-dependencies
- **Python version compatibility**: Code must work on Python 3.9-3.12
- **Test execution**: Changes to `tests/` structure or execution patterns
- **Example scripts**: `examples/` should remain functional (tested by CI)

#### 6. Performance & Threading
- **DataBroker changes**: Core communication path; flag potential bottlenecks or thread-safety issues
- **Environment timing**: Changes to SimPy scheduling or real-time execution
- **Queue sizing**: Modifications to queue handling or `max_queue_size` logic

#### 7. Version & Release
- **Version bump**: Check if `agentlib/__init__.py` `__version__` updated appropriately
- **CHANGELOG**: Verify changes documented under correct version header
- **Breaking changes**: Major version bump (0.x.0) vs minor (0.8.x)

### Review Output Format

Provide structured, concise feedback (max 10 bullet points total):
1. **Summary**: 2-3 sentence overview of PR purpose
2. **Change Classification**: Bug fix / Feature / Refactor / Documentation / Mixed
3. **Backward Compatibility**: Compatible / Breaking (explain) / Deprecation needed
4. **CI Risk**: Low / Medium / High (explain if Medium/High)
5. **Key Issues**: Numbered list of critical concerns only (max 5 items)
6. **Suggestions**: Optional - only for significant improvements

**Brevity is essential. Skip sections with nothing critical to report.**

### What NOT to Flag
- Minor style issues (handled by pylint)
- Personal preference on implementation approach
- Overly detailed nitpicking on internal/private methods
- Changes to examples or test code (unless broken)

## Common Patterns to Recognize

- **Module registration**: Modules use `@communicator("mqtt")` decorators and `MODULE_TYPES` dict
- **Callback pattern**: `agent.data_broker.register_callback()` for event-driven communication
- **SimPy generators**: `process()` methods yield SimPy events (`env.timeout()`, `env.event()`)
- **Config injection**: Modules receive config via `config: MyModuleConfig` class attribute
- **Thread safety**: Real-time mode uses threading; check for race conditions in shared state

---
*Instructions version: 1.0 | Target: GitHub Copilot Review Agent | Max length: 2 pages*
