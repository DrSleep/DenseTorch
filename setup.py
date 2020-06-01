from setuptools import Extension, find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="densetorch",
    version="0.0.1",
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
