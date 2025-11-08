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

# Ensure important local subpackages are importable by Sphinx/autosummary.
# In some RTD environments the installed package may shadow the source or
# package import machinery may fail; if import fails, load the subpackage
# directly from the repository source into sys.modules.
import importlib
import importlib.util

def _ensure_local_package(module_name: str, rel_dir: str):
    try:
        importlib.import_module(module_name)
        return
    except Exception:
        pkg_init = Path(__file__).resolve().parents[2] / rel_dir / "__init__.py"
        if pkg_init.is_file():
            spec = importlib.util.spec_from_file_location(module_name, str(pkg_init))
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

# Try to make `gixstools.align` and `gixstools.detector` available from source
_ensure_local_package('gixstools.align', 'gixstools/align')
_ensure_local_package('gixstools.detector', 'gixstools/detector')

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

