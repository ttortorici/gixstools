from setuptools import setup, Extension, find_packages
import numpy as np
from pathlib import Path

with open ("README.md", "r") as f:
    long_description = f.read()

c_module = Extension(
    name="_transform",
    sources=[str(Path("gixstools") / "wedge" / "transform.c")],
    include_dirs=[np.get_include()],
    language="c",
)

# See pyproject.toml for metadata
setup(
    name="gixstools",
    packages=find_packages(include=["gixstools", "gixstools.*"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=[c_module],
)
