import os
import unittest
from setuptools import setup, find_packages


def discover_tests():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test_*.py")
    return test_suite


def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, mode="r", encoding="utf-8") as f:
        return f.read()


setup(
    name="catalyst-dynamic-text-classification",
    description="Reimplementation of the paper by ASAPP Research using catalyst",
    author="Dmitry Kryuchkov",
    packages=find_packages(),
    install_requires=read("requirements.txt").splitlines(),
    tests_require=["pytest"],
    test_suite="setup.discover_tests",
    zip_safe=True,
)
