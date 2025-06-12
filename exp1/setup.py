from setuptools import setup

setup(
    name="highway-env",
    version="1.0",
    description="A highway environment for reinforcement learning",
    author="Your Name",
    packages=["highway_env"],
    install_requires=[
        "gymnasium>=0.26.0",
        "numpy>=1.18.0",
        "pygame>=2.0.0",
        "matplotlib>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.8.0",
        ],
        "docs": [
            "sphinx>=3.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    python_requires=">=3.7",
) 