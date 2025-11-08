import sys
from pathlib import Path
from unittest.mock import MagicMock

gixspath = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(gixspath))

# Minimal, safe mocks for external heavy dependencies so Sphinx can import
# the package and read docstrings. Do NOT mock internal `gixstools` modules
# because we want autodoc to import them and extract their docstrings.
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# External modules to mock because they may not be available during RTD builds
MOCK_MODULES = [
    'numpy',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.pylab',
    'matplotlib.colors',
    'matplotlib.ticker',
    'pyFAI',
    'fabio',
    'scipy',
    'scipy.optimize',
    'scipy.stats',
    'h5py',
    'silx',
    'toml',
    'natsort',
]

for mod_name in MOCK_MODULES:
    sys.modules.setdefault(mod_name, Mock())

# Special small helpers for common patterns
class _NDArrayMock(MagicMock):
    def max(self, *args, **kwargs):
        return 1.0

# Provide a minimal numpy-like API on the mocked numpy
if isinstance(sys.modules.get('numpy'), Mock):
    np_mock = sys.modules['numpy']
    np_mock.ndarray = _NDArrayMock
    np_mock.array = lambda *a, **k: _NDArrayMock()

# If other modules need simple attributes we can add them below, but avoid
# mocking gixstools.* so autodoc reads real docstrings.

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
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
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

# -- Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_keyword = True

# -- EPUB
epub_show_urls = 'footnote'

