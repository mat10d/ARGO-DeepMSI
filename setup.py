"""
Setup script for argo-deepmsi package.

This package provides utilities for multi-model MSI prediction from whole slide images.
"""

from setuptools import setup, find_packages

setup(
    packages=find_packages(exclude=["tests", "scripts", "STAMP", "HistoBistro"]),
)
