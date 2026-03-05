import os
import sys
import argparse as _argparse

# Allow autodoc to find the kapybara package
sys.path.insert(0, os.path.abspath(".."))

# Prevent module-level argparse.parse_args() calls from failing during import
# (kapybara.commands.* call parse_args() at module level as SLURM entry points)
_orig_parse_args = _argparse.ArgumentParser.parse_args
def _safe_parse_args(self, args=None, namespace=None):
    try:
        return _orig_parse_args(self, args, namespace)
    except SystemExit:
        from unittest.mock import MagicMock
        return MagicMock()
_argparse.ArgumentParser.parse_args = _safe_parse_args

# -- Project information -----------------------------------------------------
project = "KAPyBARA"
copyright = "2025, Jiho Son"
author = "Jiho Son"
release = "2.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = True
# Use :ivar: instead of .. attribute:: so Napoleon-generated attribute docs
# don't conflict with autodoc's own attribute directives for dataclass fields.
napoleon_use_ivar = True

# autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_mock_imports = ["lammps"]

# intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = []
html_title = "KAPyBARA"
