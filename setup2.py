from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
  name = 'test2',
  ext_modules = cythonize("integration2.pyx"),
  include_dirs=[numpy.get_include()]
)