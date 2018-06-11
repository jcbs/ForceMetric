#!/usr/bin/python3

import setuptools
# from distutils.core import setup, Extension
# from Cython.Build import cythonize
# import sys

# sys.path.append('/home/jacob/Studies/Oxford/Programming/Python/ForceMetric/')
with open("README.md", "r") as fh:
    long_description = fh.read()

# cythonize("ForceMetric.pyx")

setuptools.setup(
    name='ForceMetric',
    version='0.0.7',
    description='AFM package for Asylum Research .ibw files',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Jacob Seifert',
    author_email='jacob.seifert@gmx.net',
    url='https://github.com/jcbs/ForceMetric',
    packages=setuptools.find_packages(),
    install_requires=["numpy", "h5py", "scipy", "igor", "matplotlib"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    py_modules=['Tools', 'ContactPointDetermination'],
        # ext_modules=[Extension('ForceMetric', ['ForceMetric.c'])]
)
