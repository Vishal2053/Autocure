from setuptools import setup, find_packages

setup(
    name="autocure",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "click"
    ],
    entry_points={
        "console_scripts": [
            "autocure=autocure.cli.main:cli"
        ]
    }
)
