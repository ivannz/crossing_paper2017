from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from pip.req import parse_requirements
from pip.download import PipSession

import numpy

install_reqs = parse_requirements("requirements.txt", session=PipSession())

# python setup.py build_ext [--inplace]
setup(
    name="crossing_tree",
    ext_modules=cythonize([Extension("crossing_tree/_crossing", 
                                     ["crossing_tree/_crossing.pyx"],
                                     include_dirs=[numpy.get_include()],),]),
    cmdclass={"build_ext": build_ext},
    packages=["crossing_tree",],
    author='Ivan Nazarov',
    version='0.5.0',
    description="A library for experimentation with crossing trees for analysis of self-similarity",
    install_requires=[str(ir.req) for ir in install_reqs],
)
