#!/usr/bin/python3

from distutils.core import setup
from Cython.Build import cythonize
import sys

sys.path.append('/home/jacob/Studies/Oxford/Programming/Python/ForceMetric/')

setup(
    ext_modules=cythonize("ForceMetric.pyx")
)
