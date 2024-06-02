from setuptools import setup, find_packages

setup(
    name="grafo",
    version="0.1.0",
    packages=find_packages(include=["grafo", "grafo.*"]),
    install_requires=["openai"],
    entry_points={},
)
