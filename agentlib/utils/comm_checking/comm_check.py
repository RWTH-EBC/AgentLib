from pathlib import Path
from typing import List, Dict, Optional, Tuple, get_type_hints, get_origin, get_args

import networkx as nx

from agentlib import AgentVariable, AgentVariables
from agentlib.core.agent import get_module_class

AG_ID = str
MOD_ID = str
Alias = str


def get_config_class(type_):
    module_class = get_module_class(type_)
    config_class = get_type_hints(module_class)["config"]
    config_fields = {}
    agent_variables = []

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

        if issubclass(field_type, AgentVariable):
            agent_variables.append(default_value)
        elif issubclass(field_type, AgentVariables):
            agent_variables.extend(default_value)

    return config_class, config_fields, agent_variables


def create_configs(configs: list[dict]) -> List[Dict]:
    agent_configs = []
    for config in configs:
        agent_config = {"id": config["id"], "modules": []}
        for module in config["modules"]:
            module_config = module.copy()
            module_config["_config_class"] = get_config_class(module)
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
            for key, value in module.items():
                if key in config_fields:
                    if isinstance(value, list):
                        vars_by_module[ag_id][mod_id].extend(value)
                    else:
                        vars_by_module[ag_id][mod_id].append(value)
            vars_by_module[ag_id][mod_id].extend(
                [var.dict() for var in module["_agent_variables"]]
            )
            vars_by_module[ag_id][mod_id] = [
                dict(t)
                for t in {tuple(d.items()) for d in vars_by_module[ag_id][mod_id]}
            ]
    return vars_by_module


def order_vars_by_alias(
    vars_by_module: Dict[AG_ID, Dict[MOD_ID, List[Dict]]]
) -> Dict[Alias, List[Tuple[str, bool, Optional[Dict]]]]:
    vars_by_alias: Dict[Alias, List[Tuple[str, bool, Optional[Dict]]]] = {}
    for ag_id, modules in vars_by_module.items():
        for mod_id, ag_vars in modules.items():
            for var in ag_vars:
                alias = var.get("alias", var["name"])
                if alias not in vars_by_alias:
                    vars_by_alias[alias] = []
                vars_by_alias[alias].append(
                    (f"{ag_id}.{mod_id}.{var['name']}", var.get("shared", False), None)
                )
    return vars_by_alias


def create_comm_graph(configs):
    configs_: List[Dict] = create_configs(configs)
    vars_by_module: Dict[AG_ID, Dict[MOD_ID, List[Dict]]] = collect_vars(configs_)
    vars_by_alias: Dict[Alias, List[Tuple[str, bool, Optional[Dict]]]] = (
        order_vars_by_alias(vars_by_module)
    )

    g = nx.Graph()
    for ag_id in vars_by_module:
        g.add_node(ag_id)

    for alias, var_list in vars_by_alias.items():
        if len(var_list) > 1:
            for i in range(len(var_list) - 1):
                for j in range(i + 1, len(var_list)):
                    ag1, mod1, var1 = var_list[i][0].split(".")
                    ag2, mod2, var2 = var_list[j][0].split(".")
                    if ag1 != ag2:
                        g.add_edge(ag1, ag2, label=alias)

    return g


import json
import os


def load_json_to_dict(input_data):
    """
    Loads JSON data from various input formats and returns a list of dictionaries.

    Args:
        input_data (str, dict, or list): The input data, which can be a file path, a JSON string, or a dictionary.

    Returns:
        list[dict]: A list of dictionaries containing the JSON data.
    """
    results = []

    # Ensure input_data is a list
    if not isinstance(input_data, list):
        input_data = [input_data]

    for item in input_data:
        if isinstance(item, str):
            # Check if the string is a file path
            if os.path.isfile(item):
                with open(item, "r") as file:
                    data = json.load(file)
            else:
                # Assume the string is a JSON string
                data = json.loads(item)
        elif isinstance(item, dict):
            data = item
        else:
            raise ValueError(
                "Input data must be a string (file path or JSON), or a dictionary."
            )

        if isinstance(data, list):
            results.extend(data)
        else:
            results.append(data)

    return results


if __name__ == "__main__":
    directory_path = Path(
        r"D:\repos\AgentLib\examples\multi-agent-systems\room_mas\configs"
    )
    configs = load_json_to_dict([str(file) for file in directory_path.glob("*")])
    create_comm_graph(configs)
