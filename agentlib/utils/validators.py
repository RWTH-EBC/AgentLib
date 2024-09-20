"""
Module with validator function used in multiple parts of the agentlib
"""

from __future__ import annotations

from typing import List, Dict, Any, Type
from copy import deepcopy
import logging

import pydantic

from agentlib.core.datamodels import (
    AgentVariable,
    AgentVariables,
    Source,
)

logger = logging.getLogger(__name__)


def convert_to_list(obj):
    """Function to convert an object to a list.
    Either is is already a list.
    Or it is None, then [] is returned,
    or it is a scalar object and thus converted to a list.
    """
    if isinstance(obj, list):
        return obj
    if obj is None:
        return list()
    return [obj]


def include_defaults_in_root(
    variables: List[Dict],
    field: pydantic.fields.FieldInfo,
    make_shared: bool,
    agent_id: str,
    field_name: str,
    type_: Type[AgentVariable] = AgentVariable,
) -> AgentVariables:
    """
    Validator building block to merge default variables with config variables in the root validator.
    Updates default variables when a variable with the same name is present in the config.
    Then returns the union of the default variables and the external config variables.
    """
    # First create a copy as otherwise multiple instances of e.g. a Model class
    # would share the same defaults
    default: AgentVariables = deepcopy(field.default)
    if default is None:
        default = []
    variables = variables.copy()
    user_variables_dict = {d["name"]: d for d in variables}
    for i, var in enumerate(default):
        if var.name not in user_variables_dict:
            if make_shared:
                var.shared = make_shared
                var.source = Source(agent_id=agent_id)
            continue
        var_to_update_with = user_variables_dict[var.name]
        variables.remove(var_to_update_with)
        default[i] = update_default_agent_variable(
            default_var=var,
            user_data=var_to_update_with,
            make_shared=make_shared,
            agent_id=agent_id,
            field_name=field_name,
        )

    # add new variables and check if they are shared
    for var_dict in variables:
        if "shared" not in var_dict:
            var_dict["shared"] = make_shared
        new_var: AgentVariable = type_.validate_data(var_dict)
        if new_var.shared:
            new_var.source = Source(agent_id=agent_id)
        default.append(new_var)

    return default


def update_default_agent_variable(
    default_var: AgentVariable,
    user_data: dict,
    make_shared: bool,
    agent_id: str,
    field_name: str,
):
    """Update a variable based on it's default"""

    if is_valid_agent_var_config(user_data, field_name):
        update_var_with = user_data
    else:
        update_var_with = {"value": user_data}

    # Setting the shared attribute first allows it to be overwritten by the user config
    if not isinstance(default_var, AgentVariable):
        default_var = AgentVariable(name=field_name)
        if "alias" not in update_var_with:
            # need exception here, as the copy below does not cover the default alias
            default_var.alias = update_var_with["name"]

    if default_var.shared is None:
        default_var.shared = make_shared
    agent_variable = default_var.copy(update=update_var_with)
    # validate the model again, otherwise there can be buggy sources
    # todo check how this works with attrs variables
    agent_variable = type(default_var).validate_data(agent_variable.dict())
    if agent_variable.shared:
        agent_variable.source = Source(agent_id=agent_id)

    return agent_variable


def is_list_of_agent_variables(ls: Any):
    # TODO move somewhere more appropriate
    return isinstance(ls, list) and (len(ls) > 0) and isinstance(ls[0], AgentVariable)


def is_valid_agent_var_config(
    data: dict, field_name: str, type_: AgentVariable = AgentVariable
):
    if data == {}:
        return True
    try:
        type_.validate_data(data)
        return True
    except Exception as err:
        logger.error(
            "Could not update the default config of field '%s'. "
            "You most probably used some validator on this field. "
            "Error message: %s",
            err,
            field_name,
        )
        return False
