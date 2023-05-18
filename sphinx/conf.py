# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OmeSliCC'
copyright = '2022, Francis Crick Institute'
author = 'Francis Crick Institute'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

extensions = ['autoapi.extension']
autoapi_type = 'python'
autoapi_dirs = ['../run.py', '../OmeSliCC']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'nature'
html_static_path = ['_static']
