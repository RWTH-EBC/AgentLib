import itertools
import typing
from typing import (
    List,
    Dict,
    Optional,
    Tuple,
    get_type_hints,
    get_origin,
    get_args,
)

from agentlib.core.errors import OptionalDependencyError

try:
    import dash
    import networkx as nx
    import plotly.graph_objects as go
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
except ImportError:
    raise OptionalDependencyError(
        used_object=f"Communication checker",
        dependency_install="gui",
    )


from agentlib import AgentVariable, Source
from agentlib.core.agent import get_module_class

AG_ID = str
MOD_ID = str
Alias = str


def get_config_class(type_):
    module_class = get_module_class(type_)
    config_class = get_type_hints(module_class)["config"]
    config_fields = {}
    agent_variables = {}

    for field_name, field_info in config_class.__fields__.items():
        if field_name.startswith("_"):
            continue

        field_type = field_info.annotation
        default_value = field_info.default

        if default_value is None or default_value == Ellipsis:
            continue

        # If our annotation is List[AgentVariable], we have to work around the
        # generic type hints
        origin = get_origin(field_type)
        generic_args = get_args(field_type)
        if origin in {List, list} and len(generic_args) == 1:
            field_type = generic_args[0]

        if not isinstance(field_type, type):
            continue

        if not issubclass(field_type, AgentVariable):
            continue

        config_fields[field_name] = default_value

        if origin in {List, list}:
            agent_variables[field_name] = default_value
        else:
            agent_variables[field_name] = [default_value]

    return config_class, config_fields, agent_variables


def create_configs(configs: list[dict]) -> List[Dict]:
    agent_configs = []
    for config in configs:
        agent_config = {"id": config["id"], "modules": []}
        for module in config["modules"]:
            module_config = module.copy()
            _conf_class, _fields, _variables = get_config_class(module)
            module_config["_config_class"] = _conf_class
            module_config["_config_fields"] = _fields
            module_config["_agent_variables"] = _variables
            agent_config["modules"].append(module_config)
        agent_configs.append(agent_config)
    return agent_configs


def collect_vars(configs_: List[Dict]) -> Dict[AG_ID, Dict[MOD_ID, List[Dict]]]:
    vars_by_module: Dict[AG_ID, Dict[MOD_ID, List[Dict]]] = {}
    for config in configs_:
        ag_id = config["id"]
        vars_by_module[ag_id] = {}
        for module in config["modules"]:
            mod_id = module.get("id", module["_config_class"].__name__)
            vars_by_module[ag_id][mod_id] = []
            config_fields = module["_config_fields"]
            shared_variable_fields = config.get(
                "shared_variable_fields",
                module["_config_class"].__fields__["shared_variable_fields"].default,
            )

            for field_name, agvars in module.items():
                if field_name not in config_fields:
                    continue
                if isinstance(agvars, list):
                    for var in agvars:
                        var_dict = var.dict() if hasattr(var, "dict") else var
                        if var_dict.get("shared") is None:
                            var_dict["shared"] = field_name in shared_variable_fields
                        vars_by_module[ag_id][mod_id].append(var_dict)
                else:
                    var_dict = agvars.dict() if hasattr(agvars, "dict") else agvars
                    if var_dict.get("shared") is None:
                        var_dict["shared"] = field_name in shared_variable_fields
                    vars_by_module[ag_id][mod_id].append(var_dict)

            for field_name, agvars in module["_agent_variables"].items():
                if isinstance(agvars, list):
                    for var in agvars:
                        var_dict = var.dict() if hasattr(var, "dict") else var
                        if var_dict.get("shared") is None:
                            var_dict["shared"] = field_name in shared_variable_fields
                        vars_by_module[ag_id][mod_id].append(var_dict)
                else:
                    var_dict = agvars.dict() if hasattr(agvars, "dict") else agvars
                    if var_dict.get("shared") is None:
                        var_dict["shared"] = field_name in shared_variable_fields
                    vars_by_module[ag_id][mod_id].append(var_dict)

    return vars_by_module


class VarTuple(typing.NamedTuple):
    ag_id: str
    mod_id: str
    var_name: str
    shared: bool
    unknown: Optional[Dict]


def order_vars_by_alias(
    vars_by_module: Dict[AG_ID, Dict[MOD_ID, List[Dict]]]
) -> Dict[Alias, List[VarTuple]]:
    vars_by_alias: Dict[Alias, List[VarTuple]] = {}
    for ag_id, modules in vars_by_module.items():
        for mod_id, ag_vars in modules.items():
            for var in ag_vars:
                alias = var.get("alias", var["name"])
                if alias not in vars_by_alias:
                    vars_by_alias[alias] = []
                vars_by_alias[alias].append(
                    VarTuple(
                        ag_id=ag_id,
                        mod_id=mod_id,
                        var_name=var["name"],
                        shared=var.get("shared", False),
                        unknown=None,
                    )
                )
    return vars_by_alias


def check_comm_conditions(
    sender_agent: str, receiver_agent: str, configs: List[Dict], vars_by_module: Dict
) -> List[str]:
    def get_comm_module(agent_config):
        return next(
            (
                module
                for module in agent_config["modules"]
                if module["type"] in ["mqtt", "local", "local_broadcast"]
            ),
            None,
        )

    sender_config = next(config for config in configs if config["id"] == sender_agent)
    receiver_config = next(
        config for config in configs if config["id"] == receiver_agent
    )

    sender_comm = get_comm_module(sender_config)
    receiver_comm = get_comm_module(receiver_config)

    if not sender_comm or not receiver_comm:
        return []

    comm_type_match = sender_comm["type"] == receiver_comm["type"]
    subscription_valid = sender_agent in receiver_comm.get("subscriptions", [])
    is_broadcast = sender_comm["type"] in {
        "local_broadcast",
        "multiprocessing_broadcast",
    }

    if not (comm_type_match and (subscription_valid or is_broadcast)):
        return []

    communicable_vars = []
    for module in vars_by_module[sender_agent].values():
        for var in module:
            sender_source = Source.create(var.get("source", None))
            receiver_source = Source.create(var.get("source", None))
            if receiver_source.matches(sender_source) and var.get("shared", False):
                communicable_vars.append(var["name"])

    return communicable_vars


def create_comm_graph(configs: List[Dict]) -> Tuple[nx.DiGraph, Dict]:
    configs_: List[Dict] = create_configs(configs)
    vars_by_module: Dict[AG_ID, Dict[MOD_ID, List[Dict]]] = collect_vars(configs_)
    vars_by_alias: Dict[Alias, List[VarTuple]] = order_vars_by_alias(vars_by_module)

    g = nx.DiGraph()
    g.add_nodes_from(vars_by_module.keys())

    for alias, var_list in vars_by_alias.items():
        for var1, var2 in itertools.combinations(var_list, 2):
            if var1.ag_id == var2.ag_id:
                continue

            # Check communication from var1.ag_id to var2.ag_id
            communicable_vars_1_to_2 = check_comm_conditions(
                var1.ag_id, var2.ag_id, configs, vars_by_module
            )
            if var1.var_name in communicable_vars_1_to_2:
                if g.has_edge(var1.ag_id, var2.ag_id):
                    g[var1.ag_id][var2.ag_id]["label"] = set(
                        g[var1.ag_id][var2.ag_id]["label"].split("\n")
                    )
                    g[var1.ag_id][var2.ag_id]["label"].add(alias)
                    g[var1.ag_id][var2.ag_id]["label"] = "\n".join(
                        g[var1.ag_id][var2.ag_id]["label"]
                    )
                else:
                    g.add_edge(var1.ag_id, var2.ag_id, label=alias)

            # Check communication from var2.ag_id to var1.ag_id
            communicable_vars_2_to_1 = check_comm_conditions(
                var2.ag_id, var1.ag_id, configs, vars_by_module
            )
            if var2.var_name in communicable_vars_2_to_1:
                if g.has_edge(var2.ag_id, var1.ag_id):
                    g[var2.ag_id][var1.ag_id]["label"] = set(
                        g[var2.ag_id][var1.ag_id]["label"].split("\n")
                    )
                    g[var2.ag_id][var1.ag_id]["label"].add(alias)
                    g[var2.ag_id][var1.ag_id]["label"] = "\n".join(
                        g[var2.ag_id][var1.ag_id]["label"]
                    )
                else:
                    g.add_edge(var2.ag_id, var1.ag_id, label=alias)

    return g, vars_by_module
