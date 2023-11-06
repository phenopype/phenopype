# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'phenopype'
copyright = 'Moritz Lürig'
author = 'Moritz Lürig'

from phenopype._version import __version__ as version
release = version

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosectionlabel',
    'myst_parser',
    'nbsphinx',
    'sphinx_design',
    "sphinx_copybutton"
    ]
    
autodoc_member_order = 'bysource'
suppress_warnings = ['autosectionlabel.*']

# The master toctree document.
master_doc = 'index'

# error handling
nbsphinx_allow_errors = True

html_codeblock_linenos_style = 'table'
copybutton_remove_prompts = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['.ipynb_checkpoints', "README.md", "conf.py", ".git"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'bootstrap'
html_logo = "../assets/phenopype_logo.png"
html_theme = "furo"
html_show_sourcelink = True
html_last_updated_fmt = "%Y-%m-%d %H:%M:%S"
html_title = "phenopype docs"
html_static_path = ['_assets']
html_css_files = ['css/custom.css']
html_js_files = ['js/custom.js']
templates_path = ["_templates"]


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'phenopypedoc'
