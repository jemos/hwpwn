# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'hwpwn'
copyright = '2023, jemos'
author = 'jemos'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx_paramlinks', 'sphinx_toolbox.github', 'sphinx_toolbox.sidebar_links']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath('../hwpwn'))

# Set the autodoc options
autodoc_member_order = 'bysource'
#autodoc_typehints = 'description'


html_theme_options = {
    # Other theme options...
    #'sidebar_hide_name_link': True,
}

html_context = {
    # Other context options...
    'sidebar_links': [
        {
            'url': 'https://github.com/jemos/hwpwn',
            'name': 'hwpwn on GitHub',
        },
    ],
}

github_username = 'jemos'
github_repository = 'hwpwn'