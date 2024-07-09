import importlib
from pathlib import Path
from typing import List, Dict


def parse_toml(content: str) -> Dict:
    """A simple TOML parser for our specific needs."""
    result = {}
    current_section = result

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            section = line.strip("[]")
            if "." in section:
                parts = section.split(".")
                current_section = result
                for part in parts:
                    if part not in current_section:
                        current_section[part] = {}
                    current_section = current_section[part]
            else:
                result[section] = {}
                current_section = result[section]
        elif "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value.startswith("[") and value.endswith("]"):
                value = [
                    v.strip().strip('"').strip("'") for v in value[1:-1].split(",")
                ]
            current_section[key] = value

    return result


def load_pyproject_toml() -> Dict:
    """Load and parse the pyproject.toml file."""
    pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
    with open(pyproject_path, "r") as f:
        content = f.read()
    return parse_toml(content)


def get_optional_dependencies() -> Dict[str, List[str]]:
    """Get the optional dependencies from pyproject.toml."""
    pyproject = load_pyproject_toml()
    return pyproject.get("project", {}).get("optional-dependencies", {})


def is_dependency_installed(key: str) -> bool:
    """
    Check if the dependencies for a given key are installed.

    Args:
        key (str): The key for the optional dependency group (e.g., "gui", "fmu").

    Returns:
        bool: True if all dependencies for the key are installed, False otherwise.
    """
    optional_deps = get_optional_dependencies()
    if key not in optional_deps:
        raise ValueError(f"Invalid dependency key: {key}")

    dependencies = optional_deps[key]
    for dep in dependencies:
        # Extract the package name (remove version specifiers)
        package_name = dep.split(">=")[0].split("==")[0].split("<")[0].strip()
        if importlib.util.find_spec(package_name) is None:
            return False
    return True


def get_installed_optional_dependencies() -> List[str]:
    """
    Get a list of all installed optional dependencies.

    Returns:
        List[str]: A list of keys for which all dependencies are installed.
    """
    optional_deps = get_optional_dependencies()
    installed_deps = []
    for key in optional_deps:
        if key != "full" and is_dependency_installed(key):
            installed_deps.append(key)
    return installed_deps


if __name__ == "__main__":
    # Example usage
    print("Is 'gui' installed?", is_dependency_installed("gui"))
    print("Installed optional dependencies:", get_installed_optional_dependencies())
