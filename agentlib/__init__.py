"""
Top-level module of the agentlib.
Besides import of submodules, nothing happens here.
"""

from . import core, utils, models, modules
from .core import (
    Environment,
    BaseModule,
    BaseModuleConfig,
    Agent,
    Model,
    ModelConfig,
    AgentVariable,
    AgentVariables,
    ModelInput,
    ModelInputs,
    ModelState,
    ModelStates,
    ModelOutputs,
    ModelOutput,
    ModelParameter,
    ModelParameters,
    ModelVariable,
    Source,
    Causality,
)
from .utils.multi_agent_system import (
    LocalMASAgency,
    MultiProcessingMAS,
    LocalCloneMAPAgency,
)

__version__ = "0.8.2"


__all__ = [
    "core",
    "modules",
    "models",
    "utils",
    "LocalMASAgency",
    "MultiProcessingMAS",
    "LocalCloneMAPAgency",
    "Environment",
    "BaseModule",
    "BaseModuleConfig",
    "Agent",
    "Model",
    "ModelConfig",
    "AgentVariable",
    "AgentVariables",
    "ModelInput",
    "ModelInputs",
    "ModelState",
    "ModelStates",
    "ModelOutputs",
    "ModelOutput",
    "ModelParameter",
    "ModelParameters",
    "ModelVariable",
    "Source",
    "Causality",
]
