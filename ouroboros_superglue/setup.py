from setuptools import setup, find_packages

setup(
    name="ouroboros_superglue",
    version="0.0.1",
    url="",
    author="Toya Takahashi",
    author_email="",
    description="Superglue integration for Ouroboros",
    package_dir={"": "src"},
    packages=find_packages("src"),
    #package_data={"": ["*.yaml"]},
    #install_requires=["numpy", "imageio", "pyyaml", "pytest"],
)
