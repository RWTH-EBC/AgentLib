"""
Top-level module of the agentlib.
Besides import of submodules, nothing happens here.
"""

from . import core
from . import utils
from . import models
from . import modules
from .core import Agent, BaseModule, BaseModuleConfig, Environment, Model, ModelConfig
from .core.datamodels import *
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
    "Agent",
    "BaseModule",
    "BaseModuleConfig",
    "Environment",
    "Model",
    "ModelConfig",
    "AgentVariable",
    "AgentVariables",
    "LocalMASAgency",
    "MultiProcessingMAS",
    "LocalCloneMAPAgency",
]
