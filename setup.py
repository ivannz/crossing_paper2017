from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy

setup(
    name="crossing_tree",
    version="0.9.3",
    description="""A library for experimentation with crossing trees """
                """for analysis of self-similarity.""",
    license="MIT License",
    author="Ivan Nazarov",
    author_email="ivan.nazarov@skolkovotech.ru",
    ext_modules=cythonize([
        Extension(
            "crossing_tree._crossing",
            ["crossing_tree/_crossing.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-std=c99", "-O3", "-Ofast"]
        )
    ]),
    packages=[
        "crossing_tree",
        "crossing_tree.processes"
    ],
    install_requires=[
        "numpy>=1.11.1",
        "pyfftw>=0.10.4",
        "cython>=0.24.1"
    ]
)
