#!/usr/bin/python3

from distutils.core import setup, Extension
from Cython.Build import cythonize
import sys

sys.path.append('/home/jacob/Studies/Oxford/Programming/Python/ForceMetric/')

cythonize("ForceMetric.pyx")

setup(
    name='ForceMetric',
    version='0.1',
    description='AFM package for Asylum Research .ibw files',
    author='Jacob Seifert',
    author_email='jacob.seifert@gmx.net',
    url='https://github.com/jcbs/ForceMetric',
    py_modules=['Tools', 'ContactPointDetermination'],
    ext_modules=[Extension('ForceMetric', ['ForceMetric.c'])]
)
