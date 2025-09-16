# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import sphinx_rtd_theme
# Add the project directory to sys.path
sys.path.insert(0, os.path.abspath('../../'))

project = 'polygen'
copyright = '2025, Md Jalal Uddin Rumi'
author = 'Md Jalal Uddin Rumi'
release = '1.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'sphinx.ext.intersphinx',  # Add this
    'sphinx_rtd_theme',  # Add this line
]

# Theme configuration
html_theme = 'sphinx_rtd_theme'
html_baseurl = "https://Rumi381.github.io/PolyGen/"
html_static_path = ['_static']

# Configure intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'shapely': ('https://shapely.readthedocs.io/en/latest/', None),
}

# Configure Napoleon for Numpy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
autosummary_generate = True

# Configure autodoc
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}

# Handle types better
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
python_use_unqualified_type_names = True

# Prevent duplicate documentation
autodoc_member_order = 'bysource'
add_module_names = False

def skip_duplicate_member(app, what, name, obj, skip, options):
    if hasattr(obj, '__doc__') and obj.__doc__ and ':no-index:' in obj.__doc__:
        return True
    return skip

def setup(app):
    # Connect 'autodoc-skip-member' for skipping duplicate members
    app.connect('autodoc-skip-member', skip_duplicate_member)
    
    # Add custom JavaScript to open external links in a new tab
    app.add_js_file('custom.js')

templates_path = ['_templates']
exclude_patterns = []
