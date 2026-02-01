from setuptools import setup, find_packages

setup(
    name="dendritic",
    version="0.1.0",
    description="Dendritic neural network layers for polynomial feature enhancement",
    author="Your Name",
    packages=find_packages(),
    install_requires=["torch", "transformers", "datasets", "tqdm"],
    python_requires=">=3.7",
)
