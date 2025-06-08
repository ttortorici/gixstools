import os
import sys
sys.path.insert(0, os.path.abspath('../../gixstools'))
sys.path.insert(0, os.path.abspath('../../gixstools.align'))
sys.path.insert(0, os.path.abspath('../../gixstools.wedge'))
sys.path.insert(0, os.path.abspath('../../gixstools.detector'))

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Grazing Incidence X-ray Scattering Tools'
copyright = '2025, Edward Tortorici'
author = 'Edward Tortorici'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
    'sphinx.ext.napoleon',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
autosummary_generate = True

bibtex_bibfiles = ['references.bib']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
