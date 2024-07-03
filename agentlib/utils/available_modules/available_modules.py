import json
import re
from enum import Enum
from typing import get_origin, get_args, Union

from flask import Flask, render_template
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from agentlib.modules import get_all_module_types

app = Flask(__name__)


def get_config_info(config_class):
    fields = []
    undefined_fields = []
    for name, field in config_class.model_fields.items():
        field_info = {
            "name": name,
            "type": get_type_info(field.annotation),
            "description": field.description or "No description",
            "default": field.default if field.default is not None else "None",
        }
        if field.default is PydanticUndefined:
            undefined_fields.append(field_info)
        else:
            fields.append(field_info)
    return undefined_fields + fields


def clean_type_name(type_str):
    # Remove <class '...'> and extract the class name
    match = re.search(r"<class '(?:.*\.)?(\w+)'>", type_str)
    if match:
        return match.group(1)

    # For other cases, just return the last part after the dot
    return type_str.split(".")[-1]


def get_type_info(annotation):
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None and isinstance(annotation, type) and issubclass(annotation, Enum):
        return f"Enum({', '.join(annotation.__members__.keys())})"
    elif origin is list:
        if args:
            return f"List[{get_type_info(args[0])}]"
        return "List"
    elif origin is dict:
        if len(args) == 2:
            return f"Dict[{get_type_info(args[0])}, {get_type_info(args[1])}]"
        return "Dict"
    elif origin is Union:
        return (
            f"Union[{', '.join(clean_type_name(get_type_info(arg)) for arg in args)}]"
        )
    elif origin is not None:
        return f"{clean_type_name(str(origin))}[{', '.join(get_type_info(arg) for arg in args)}]"
    else:
        return clean_type_name(str(annotation))


def get_default_value(field):
    if field.default is not None:
        return field.default
    if get_origin(field.annotation) is list:
        return []
    if get_origin(field.annotation) is dict:
        return {}
    return None


def get_model_config(config_class):
    config = {}
    for name, field in config_class.model_fields.items():
        config[name] = get_default_value(field)
    return config


@app.route("/")
def index():
    modules = get_all_module_types(plugins=[])
    module_configs = {}
    python_configs = {}
    json_configs = {}
    for key, module in modules.items():
        try:
            class_ = module.import_class()
            config_type = class_.get_config_type()
            if issubclass(config_type, BaseModel):
                module_configs[key] = get_config_info(config_type)
                python_config = get_model_config(config_type)
                python_configs[key] = repr(python_config)
                json_configs[key] = json.dumps(python_config, indent=2, default=str)
            else:
                module_configs[key] = [
                    {"error": "Config class is not a subclass of pydantic.BaseModel"}
                ]
                python_configs[key] = "{}"
                json_configs[key] = "{}"
        except Exception as e:
            module_configs[key] = [{"error": str(e)}]
            python_configs[key] = "{}"
            json_configs[key] = "{}"

    sorted_modules = sorted(modules.items(), key=lambda x: x[1].import_path)
    return render_template(
        "index.html",
        modules=sorted_modules,
        configs=module_configs,
        python_configs=python_configs,
        json_configs=json_configs,
    )


if __name__ == "__main__":
    app.run(debug=True)
