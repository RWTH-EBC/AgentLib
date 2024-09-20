# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

# pylint: disable-all

sys.path.insert(0, os.path.abspath("../.."))
sys.setrecursionlimit(1500)

# -- Project information -----------------------------------------------------

project = "agentlib"
copyright = "2021, AGENT-Project Associates"
author = "AGENT-Project Associates"

# The full version, including alpha/beta/rc tags
with open(Path(__file__).parents[2].joinpath(project, "__init__.py"), "r") as file:
    for line in file.readlines():
        if line.startswith("__version__"):
            release = (
                line.replace("__version__", "")
                .split("=")[1]
                .strip()
                .replace("'", "")
                .replace('"', "")
            )

# The short X.Y version.
version = ".".join(release.split(".")[:2])
# The full version, including alpha/beta/rc tags.
release = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.coverage",
    "myst_parser",  # Enable .md files
    "sphinx.ext.napoleon",  # Enable google docstrings
    "sphinxcontrib.autodoc_pydantic",
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config = False
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#

source_suffix = [".rst", ".md"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# html_theme = 'sphinx_material'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

# Material theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the navigation.
    "nav_title": project,
    # Set you GA account ID to enable tracking
    # 'google_analytics_account': 'UA-XXXXX',
    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    # 'base_url': 'https://project.github.io/project',
    # Set the color and the accent color
    "color_primary": "red",
    "color_accent": "red",
    # Set the repo location to get a badge with stats
    "repo_url": "https://github.com/RWTH-EBC/AgentLib",
    "repo_name": "Agent Library for Python",
    # Visible levels of the global TOC; -1 means unlimited
    "globaltoc_depth": 3,
    # If False, expand all TOC entries
    "globaltoc_collapse": True,
    # If True, show hidden TOC entries
    "globaltoc_includehidden": False,
    # Little logo on top left
    "logo_icon": "&#xe869",
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# pydantic settings
# autodoc_pydantic_model_show_json = False
# autodoc_pydantic_model_show_config = False
