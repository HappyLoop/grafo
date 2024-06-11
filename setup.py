from setuptools import setup, find_packages

with open("readme.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="grafo",
    version="0.1.2",
    description="A library for building runnable asynchronous trees",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Paulo Mattos",
    author_email="paulomtts@outlook.com",
    url="https://github.com/paulomtts/grafo",
    packages=find_packages(include=["grafo", "grafo.*"]),
    install_requires=["openai", "instructor", "langsmith"],
    python_requires=">=3.6",
)
