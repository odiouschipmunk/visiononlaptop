from setuptools import setup
from Cython.Build import cythonize

setup(
    name='visionapp',
    ext_modules=cythonize("main.py"),
)