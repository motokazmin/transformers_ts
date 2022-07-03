import os
from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="transformers_ts",
    version="1.0.0",
    author="Roman Kazmin",
    description="Huggingface transformers for time series task",
    packages=['transformers_ts'],
    python_requires=">=3.6.0"
)

print(find_packages("src"))