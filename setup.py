import os
import sys
from typing import Dict
from typing import List
from typing import Optional

import pkg_resources
from setuptools import find_packages
from setuptools import setup

def get_version() -> str:

    version_filepath = os.path.join(os.path.dirname(__file__), "optlearn", "version.py")
    with open(version_filepath) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]
    assert False

def get_long_description() -> str:

    readme_filepath = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_filepath) as f:
        return f.read()

def get_install_requires() -> List[str]:

    requirements = [
        "optuna"
    ]
    return requirements

setup(
    name="optuna-learn",
    version=get_version(),
    description="A hyperparameter optimization framework via optuna.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="zhangxjohn",
    author_email="zhangxjohn@yeah.net",
    packages=find_packages(exclude=("tests", "tests.*", "benchmarks")),
    python_requires=">=3.6",
    install_requires=get_install_requires(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache License 2.0",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    zip_safe=False,
    include_package_data=True,
)