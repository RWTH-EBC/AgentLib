from typing import List, Dict, Optional, Tuple

import networkx as nx

from agentlib import AgentVariable, LocalMASAgency
from agentlib.core.agent import AgentConfig
from agentlib.modules.communicator.communicator import CommunicatorConfig

AG_ID = str
MOD_ID = str
Alias = str


def create_configs(configs) -> List[AgentConfig]:
    mas = LocalMASAgency(configs=configs, env={"rt": False})
    configs = []
    for agent_id, agent in mas._agents.items():
        configs.append(agent.config)
    return configs


def collect_vars(configs_):
    pass


def order_vars_by_alias(vars_by_module):
    pass


def create_comm_list(configs):
    configs_: List[AgentConfig] = create_configs(configs)
    vars_by_module: Dict[AG_ID, Dict[MOD_ID, List[AgentVariable]]] = collect_vars(
        configs_
    )

    # tuple is (name, shared, comm_conf)
    vars_by_alias: Dict[
        Alias, List[Tuple[str, bool, Optional[CommunicatorConfig]]]
    ] = order_vars_by_alias(vars_by_module)


def create_graph(vars_by_module):
    g = nx.DiGraph()
    for ag_id, modules in vars_by_module.items():
        g.add_node(ag_id, type="agent")
        for mod_id, ag_vars in modules.items():
            g.add_node(mod_id, type="module")
            g.add_edge(ag_id, mod_id, relation="has_module")
            for var in ag_vars:
                g.add_node(var.name, type="variable")
                g.add_edge(mod_id, var.name, relation="has_variable")
