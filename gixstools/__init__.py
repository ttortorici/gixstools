"""Top-level package for gixstools.

Avoid importing heavy subpackages at import time so Sphinx / ReadTheDocs
can import the package without executing detector/align initialisation
that depends on third-party binaries or configuration files.
"""

from importlib import metadata
import os

# Expose package metadata
try:
	__version__ = metadata.version(__name__)
except Exception:
	__version__ = "0.0"

# Lazy or guarded imports: don't import heavy subpackages when building docs
SPHINX_BUILD = os.environ.get("SPHINX_BUILD") or os.environ.get("READTHEDOCS")

if not SPHINX_BUILD:
	# Normal runtime: import subpackages for convenience
	try:
		from . import wedge  # noqa: F401
		from . import align  # noqa: F401
		from . import config  # noqa: F401
		from . import detector  # noqa: F401
		from . import _programs  # noqa: F401
	except Exception:
		# Fail silently during import to avoid breaking environments where
		# optional dependencies are missing. Individual submodules can be
		# imported by users when needed.
		pass
else:
	# When building docs, avoid importing subpackages here. Sphinx will
	# import submodules directly (and our conf.py provides mocks for
	# heavy external dependencies).
	__all__ = []