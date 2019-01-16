from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = Extension(
    name = "cython_functions",
    sources = ["cython_functions.pyx"],
    include_dirs = [numpy.get_include()]
)

setup(
    ext_modules = cythonize(extensions)
)
