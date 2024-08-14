# install the p2g module to the python path

from setuptools import find_packages, setup

setup(
    name="p2g",
    version="0.1",
    # pip requiresment in requirements.txt
    packages=find_packages(),
)
