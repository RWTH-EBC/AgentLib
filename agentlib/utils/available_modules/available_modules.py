from flask import Flask, render_template, jsonify
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


def get_default_json(config_class):
    default_values = {}
    for name, field in config_class.model_fields.items():
        default_values[name] = field.default if field.default is not None else None
    return jsonify(default_values).get_data(as_text=True)


@app.route("/")
def index():
    modules = get_all_module_types(plugins=[])
    module_configs = {}
    default_jsons = {}
    for key, module in modules.items():
        try:
            class_ = module.import_class()
            config_type = class_.get_config_type()
            if issubclass(config_type, BaseModel):
                module_configs[key] = get_config_info(config_type)
                default_jsons[key] = get_default_json(config_type)
            else:
                module_configs[key] = [
                    {"error": "Config class is not a subclass of pydantic.BaseModel"}
                ]
                default_jsons[key] = "{}"
        except Exception as e:
            module_configs[key] = [{"error": str(e)}]
            default_jsons[key] = "{}"

    sorted_modules = sorted(modules.items(), key=lambda x: x[1].import_path)
    return render_template(
        "index.html",
        modules=sorted_modules,
        configs=module_configs,
        default_jsons=default_jsons,
    )


if __name__ == "__main__":
    app.run(debug=True)
