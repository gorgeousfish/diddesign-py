import os
import sys
import tomllib

sys.path.insert(0, os.path.abspath("../src"))

with open(os.path.abspath("../pyproject.toml"), "rb") as handle:
    _pyproject = tomllib.load(handle)

project = "diddesign"
author = "Xuanyu Cai, Wenli Xu"
copyright = "2025, Xuanyu Cai, Wenli Xu"
release = _pyproject["project"]["version"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

autodoc_default_options = {
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}


def _skip_compatibility_aliases(app, what, name, obj, skip, options):
    return name in {"as_payload", "named_plot_payloads"} or skip


def setup(app):
    app.connect("autodoc-skip-member", _skip_compatibility_aliases)
