import sys
from pathlib import Path
from unittest.mock import MagicMock

gixspath = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(gixspath))

# Mock the config module
class MockConfig:
    def load(self, subdict=None, path=None):
        return {
            "horizontal_shutter": "umv s2hg 0.6",
            "vertical_shutter": "umv s2vg 0.1",
            "set_omega": "umv om {:.4f}\n",
            "set_vertical": "umv z {:.4f}\n",
            "move_beamstop": "umv wbs {:.4f}\n",
            "move_vertical": "umvr z {:.4f}\n",
            "expose": "eiger_run 0.1 {}.tif\n",
            "beamstop": True
        }

sys.modules['gixstools.config'] = MockConfig()

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
