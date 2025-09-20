"""Setup configuration for quantum_conversations package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README for long description
readme_path = Path(__file__).parent.parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="quantum_conversations",
    version="0.2.0",
    description="Comprehensive toolkit for visualizing token generation paths in language models using particle filters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Contextual Dynamics Laboratory",
    author_email="",
    url="https://github.com/ContextLab/quantum-conversations",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",  # Updated for modern features
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords="language-models particle-filter visualization nlp transformers",
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ]
    },
    entry_points={
        "console_scripts": [
            "quantum-conversations=quantum_conversations.cli:main",
        ]
    },
)