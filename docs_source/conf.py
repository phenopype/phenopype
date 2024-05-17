# -*- coding: utf-8 -*-

import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath('../phenopype'))
release = importlib.metadata.version("phenopype")

# -- Project information -----------------------------------------------------

project = 'phenopype'
copyright = 'Moritz Lürig'
author = 'Moritz Lürig'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosectionlabel',
    'myst_nb',
    'sphinx_design',
    'sphinx_copybutton',
    'sphinxemoji.sphinxemoji',
]

sphinxemoji_style = 'twemoji'
sphinxemoji_source = 'https://unpkg.com/twemoji@latest/dist/twemoji.min.js'

autodoc_member_order = 'bysource'
suppress_warnings = [
    'autosectionlabel.*',
    'mystnb.nbcell',
    'myst.header',
    'myst.strikethrough',
]
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3
nb_execution_mode = 'off'
# nb_remove_code_outputs = True

master_doc = 'index'
copybutton_remove_prompts = True

exclude_patterns = [
    '.ipynb_checkpoints',
    'README.md',
    'conf.py',
    '.git',
]

pygments_style = 'sphinx'

# -- HTML configuration ---------------------------------------------------

html_codeblock_linenos_style = 'table'
html_base_url = 'https://www.phenopype.org/'
html_logo = '../assets/phenopype_logo.png'
html_theme = 'furo'
html_show_sourcelink = True
html_last_updated_fmt = '%Y-%m-%d %H:%M:%S'
html_title = 'phenopype docs'
html_static_path = ['_assets']
html_css_files = ['css/custom.css']
html_js_files = [
    'js/custom.js',
    'https://cdn.jsdelivr.net/gh/mluerig/website-assets/assets/js/enforce_trailing_slash.min.js',
]
templates_path = ['_templates']
