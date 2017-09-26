from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
  name = 'gauss_seidel',
  ext_modules = cythonize("gauss_seidel.pyx"),
  include_dirs=[numpy.get_include()]
)