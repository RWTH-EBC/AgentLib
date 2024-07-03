from flask import Flask, render_template

from agentlib.modules import get_all_module_types

app = Flask(__name__)


@app.route("/")
def index():
    modules = get_all_module_types(plugins=[])
    sorted_modules = sorted(modules.items(), key=lambda x: x[1].import_path)
    return render_template("index.html", modules=sorted_modules)


if __name__ == "__main__":
    app.run(debug=True)
