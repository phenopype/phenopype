# -*- coding: utf-8 -*-

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

master_doc = 'index'

nbsphinx_allow_errors = True

html_codeblock_linenos_style = 'table'
copybutton_remove_prompts = True

exclude_patterns = ['.ipynb_checkpoints', "README.md", "conf.py", ".git"]

pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

html_logo = "../assets/phenopype_logo.png"
html_theme = "furo"
html_show_sourcelink = True
html_last_updated_fmt = "%Y-%m-%d %H:%M:%S"
html_title = "phenopype docs"
html_static_path = ['_assets']
html_css_files = ['css/custom.css']
html_js_files = ['js/custom.js']
templates_path = ["_templates"]
