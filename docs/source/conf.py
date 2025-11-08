import sys
from pathlib import Path
from unittest.mock import MagicMock

gixspath = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(gixspath))

# Mock the required modules
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# List of packages to mock
MOCK_MODULES = [
    'numpy',
    'matplotlib',
class MockNumpyArray:
    def __init__(self, *args, **kwargs):
        pass

    def __array_function__(self, *args, **kwargs):
        return MagicMock()

    def max(self):
        return 1000.0

class MockDetector:
    """Mock Detector class with key attributes and methods."""
    def __init__(self):
        self.pixel1 = 1e-4
        self.pixel2 = 1e-4
        self.shape = (2167, 2070)
        self.MAX_INT = 65535

    def calc_mask(self):
        return MockNumpyArray()

    def calc_mask_dezinger(self, image):
        return MockNumpyArray()

class MockDirectBeam:
    """Mock DirectBeam class with essential functionality."""
    def __init__(self, filename):
        self.data = MockNumpyArray()
        self.center = (1000, 1000)
        self.width = (50, 50)
        self.detector = MockDetector()

    def find_center(self):
        return self.center

class MockSpatiallyResolvedScan:
    """Mock SRS class with core functionality."""
    def __init__(self, directory, approximate_detector_distance_meters, **kwargs):
        self.directory = directory
        self.dist_guess = approximate_detector_distance_meters
        self.direct_beam = MockDirectBeam(None)
        self.type = "om"
        
    def fit(self, **kwargs):
        self.omega0 = 0.5
        self.det_dist_fit = 150.0
        
    def show_omega_scan(self, **kwargs):
        return MagicMock(), MagicMock()

    'matplotlib.pyplot',
    'pyFAI',
    'matplotlib.pyplot',
    'pyFAI',
    'silx',
    'h5py',
    'toml',
    'natsort'
    sys.modules[mod_name] = Mock()

# Special mock for align module with the required functions
class MockAlign:
    @staticmethod
    def create_z_macro(*args, **kwargs):
        return {
# Set up numpy mock with array support
class MockNumpy(MagicMock):
    ndarray = MockNumpyArray
    def array(self, *args, **kwargs):
        return MockNumpyArray()
sys.modules['numpy'] = MockNumpy()

# Set up matplotlib mocks
sys.modules['matplotlib.pylab'] = MagicMock()
sys.modules['matplotlib.colors'] = MagicMock()
sys.modules['matplotlib.ticker'] = MagicMock()

# Set up scipy mocks
sys.modules['scipy.optimize'] = MagicMock()
sys.modules['scipy.special'] = MagicMock()

# Set up detector mock
sys.modules['gixstools.detector'] = MagicMock()
sys.modules['gixstools.detector'].Detector = MockDetector
sys.modules['gixstools.detector'].DirectBeam = MockDirectBeam
sys.modules['gixstools.detector'].SpatiallyResolvedScan = MockSpatiallyResolvedScan
            "horizontal_shutter": "umv s2hg 0.6",
            "vertical_shutter": "umv s2vg 0.1",
            "set_omega": "umv om {:.4f}\n",
            "set_vertical": "umv z {:.4f}\n",
        default_config = {
            "move_vertical": "umvr z {:.4f}\n",
            "expose": "eiger_run 0.1 {}.tif\n",
            "beamstop": True
        }

sys.modules['gixstools.config'] = MockConfig()

# Configuration file for the Sphinx documentation builder.

        align_config = {
            'info': 'title',
            'im_filetype': '.tif',
            'beam_position': {'x': 'C', 'y': 'C'},
            'unit': 'mm'
        }
        if subdict == "align":
            return align_config
        return default_config
# -- Project information

project = 'Grazing Incidence X-ray Scattering Tools'
copyright = '2025, Edward Tortorici'
author = 'Edward Tortorici'
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'

# -- General configuration

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_keyword = True

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
