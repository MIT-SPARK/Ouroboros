from setuptools import setup, find_packages

setup(
    name="ouroboros",
    version="0.0.1",
    url="",
    author="Aaron Ray",
    author_email="aaronray@mit.edu",
    description="Storage backend for VLC",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*.yaml"]},
    install_requires=["numpy", "imageio", "pyyaml", "pytest", "importlib-resources"],
)
