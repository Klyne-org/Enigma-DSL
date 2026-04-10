from setuptools import setup, find_packages

setup(
    name="enigma",
    version="0.1.0",
    description="A Python DSL for Apple Metal GPU kernels",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy"],
)
