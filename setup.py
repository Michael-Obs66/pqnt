import os
import sys
from setuptools import setup, Extension, find_packages
import pybind11

# C++ include dir
include_dirs = [
    pybind11.get_include(),
    os.path.join(os.path.dirname(__file__), "include"),
]

# Sources for pquant C++ module
sources = [
    "src/pqnt.cpp",
    "src/bindings.cpp"
]

ext_modules = [
    Extension(
        "pqnt",
        sources=sources,
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"]
    )
]

setup(
    name="pqnt",
    version="0.1.0",
    author="Michael Anggi Gilang Angkasa",
    description="Power-law Quantization Engine",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    include_package_data=True,
    install_requires=[
        "pybind11>=2.10",
        "numpy"
    ],
    python_requires=">=3.8",
)
