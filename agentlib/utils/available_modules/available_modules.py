import json
from typing import get_origin

from flask import Flask, render_template
from pydantic import BaseModel

from agentlib.modules import get_all_module_types

app = Flask(__name__)


def get_config_info(config_class):
    fields = []
    for name, field in config_class.model_fields.items():
        field_info = {
            "name": name,
            "type": str(field.annotation),
            "description": field.description or "No description",
            "default": field.default if field.default is not None else "None",
        }
        fields.append(field_info)
    return fields


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
    return json.dumps(config, indent=2, default=str)


@app.route("/")
def index():
    modules = get_all_module_types(plugins=[])
    module_configs = {}
    model_configs = {}
    for key, module in modules.items():
        try:
            class_ = module.import_class()
            config_type = class_.get_config_type()
            if issubclass(config_type, BaseModel):
                module_configs[key] = get_config_info(config_type)
                model_configs[key] = get_model_config(config_type)
            else:
                module_configs[key] = [
                    {"error": "Config class is not a subclass of pydantic.BaseModel"}
                ]
                model_configs[key] = "{}"
        except Exception as e:
            module_configs[key] = [{"error": str(e)}]
            model_configs[key] = "{}"

    sorted_modules = sorted(modules.items(), key=lambda x: x[1].import_path)
    return render_template(
        "index.html",
        modules=sorted_modules,
        configs=module_configs,
        model_configs=model_configs,
    )


if __name__ == "__main__":
    app.run(debug=True)
