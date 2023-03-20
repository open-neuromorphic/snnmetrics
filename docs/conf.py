import snnmetrics

project = "snnmetrics"
copyright = "2023-present"
author = "Gregor Lenz"

master_doc = "index"

extensions = [
    "autoapi.extension",
    "myst_nb",
    "pbr.sphinxext",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_typehints = "both"
autoapi_type = "python"
autoapi_dirs = ["../snnmetrics"]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# MyST settings
# nb_execution_mode = "off"
nb_execution_timeout = 300
nb_execution_show_tb = True
nb_execution_excludepatterns = ["large_datasets.ipynb"]
suppress_warnings = ["myst.header"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".ipynb_checkpoints",
    "README.rst",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "snnmetrics " + snnmetrics.__version__
# html_logo = "_static/snnmetrics-logo-black.png"
# html_favicon = "_static/snnmetrics_favicon.png"
html_show_sourcelink = True
html_sourcelink_suffix = ""

html_theme_options = {
    "repository_url": "https://github.com/open-neuromorphic/snnmetrics",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "develop",
    "path_to_docs": "docs",
    "use_fullscreen_button": True,
}
