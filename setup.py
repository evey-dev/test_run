"""Package metadata and install configuration for the reproduction project.

This module defines the package name, dependencies, and the console entry point
for the baseline runner. It is typically used by pip or setuptools rather than
being executed directly.
"""

from setuptools import find_packages, setup

setup(
    name="mechanistic-interpretability-repro",
    version="0.1.0",
    description="Reproduction study for mechanistic interpretability circuits in Qwen3-4B-Instruct",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "pyyaml",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "mechanistic-baseline=src.baseline:main",
            "mechanistic-train=src.train:main",
        ]
    },
)
