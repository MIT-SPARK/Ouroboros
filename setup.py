import subprocess

from pybind11.setup_helpers import Pybind11Extension
from setuptools import find_packages, setup


def _get_flags(name):
    return (
        subprocess.run(["pkg-config", "--cflags", name], capture_output=True)
        .stdout.decode("utf-8")
        .strip("\n")
        .split(" ")
    )


ext_modules = [
    Pybind11Extension(
        "_ouroboros_opengv",
        [
            "src/ouroboros-opengv/_bindings.cpp",
            "third_party/opengv/src/relative_pose/CentralRelativeAdapter.cpp",
        ],
        extra_compile_args=["-Ithird_party/opengv/include"] + _get_flags("eigen3"),
    )
]

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
    install_requires=[
        "numpy",
        "imageio",
        "matplotlib",
        "opencv-python",
        "pyyaml",
        "pytest",
        "scipy",
    ],
    ext_modules=ext_modules,
)
