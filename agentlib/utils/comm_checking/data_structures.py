from dataclasses import dataclass, field
from typing import List, Dict

import networkx as nx

from agentlib.core import AgentVariable


@dataclass
class ModuleC:
    id: str
    type: str
    variables: List[AgentVariable] = field(default_factory=list)
    subscriptions: List[str] = field(default_factory=list)


@dataclass
class AgentC:
    id: str
    modules: List[ModuleC] = field(default_factory=list)


@dataclass
class CommunicationGraph:
    agents: List[AgentC]
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    vars_by_module: Dict[str, Dict[str, List[AgentVariable]]] = field(
        default_factory=dict
    )
