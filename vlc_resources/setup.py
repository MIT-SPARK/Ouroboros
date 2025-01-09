from setuptools import setup, find_packages

setup(
    name="vlc_resources",
    version="0.0.1",
    url="",
    author="Aaron Ray",
    author_email="aaronray@mit.edu",
    description="Extra resource files for Ouroboros",
    # package_dir={"": "src"},
    packages=find_packages(),
    package_data={"": ["*.yaml", "*.png", "*.jpg"]},
    install_requires=["numpy"],
)
