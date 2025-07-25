"""Sphinx configuration for building project documentation."""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

from collections.abc import Mapping
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sphinx.application import Sphinx

import commonmark

sys.path.insert(0, os.path.abspath("../.."))  # noqa: PTH100
sys.path.insert(0, os.path.abspath("../../shapiq_student"))  # noqa: PTH100


import shapiq_student

# -- Read the Docs ---------------------------------------------------------------------------------
master_doc = "index"

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "shapiq_student"
copyright = "2025, Sep-AIML-25-Group-2"
author = "Sep-AIML-25-Group-2"
release = shapiq_student.__version__
version = shapiq_student.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.duration",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
    "sphinx_toolbox.more_autodoc.autoprotocol",
]

nbsphinx_allow_errors = True  # optional, avoids build breaking due to execution errors

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = (
    "unsrt"  # set to alpha to not confuse references the docs with the footcites in docstrings.
)

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

intersphinx_mapping = {
    "python3": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
html_favicon = "_static/logo/shapiq.ico"
pygments_dark_style = "monokai"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo/logo_shapiq_light.svg",
    "dark_logo": "logo/logo_shapiq_dark.svg",
}

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ],
}

# -- Autodoc ---------------------------------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "show-inheritance": True,
    "members": True,
    "member-order": "groupwise",
    "special-members": "__call__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autoclass_content = "both"
autodoc_inherit_docstrings = True
autodoc_member_order = "groupwise"

# -- Markdown in docstring -----------------------------------------------------------------------------
# https://gist.github.com/dmwyatt/0f555c8f9c129c0ac6fed6fabe49078b#file-docstrings-py
# based on https://stackoverflow.com/a/56428123/23972


def docstring(
    _app: Sphinx,
    _what: str,
    _name: str,
    _obj: object,
    _options: Mapping[str, object],
    lines: list[str],
) -> None:
    """Convert Markdown in docstrings to reStructuredText."""
    if len(lines) > 1 and lines[0] == "@&ismd":
        md = "\n".join(lines[1:])
        ast = commonmark.Parser().parse(md)
        rst = commonmark.ReStructuredTextRenderer().render(ast)
        lines.clear()
        lines += rst.splitlines()


def setup(app: Sphinx) -> None:
    """Setup function for the Sphinx extension to convert Markdown in docstrings to reStructuredText."""
    app.connect("autodoc-process-docstring", docstring)
