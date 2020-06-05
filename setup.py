import os
from setuptools import Extension, find_packages, setup

cwd = os.path.dirname(os.path.abspath(__file__))
version = "0.0.2"

with open(os.path.join(cwd, "densetorch", "version.py"), "w") as f:
    f.write(f"__version__ = '{version}'\n")

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="densetorch",
    version=version,
    author="Vladimir Nekrasov",
    author_email="nekrasowladimir.at.gmail.com",
    description="Light-Weight PyTorch Wrapper for dense per-pixel tasks.",
    url="https://github.com/drsleep/densetorch",
    packages=find_packages(exclude=("examples/",)),
    setup_requires=["setuptools>=18.0", "cython"],
    install_requires=requirements,
    ext_modules=[
        Extension("densetorch.engine.miou", sources=["./densetorch/engine/miou.pyx"])
    ],
    classifiers=("Programming Language :: Python :: 3"),
    zip_safe=False,
)
