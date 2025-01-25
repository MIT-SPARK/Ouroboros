import glob
import pathlib
import subprocess

from pybind11.setup_helpers import ParallelCompile, Pybind11Extension, build_ext
from setuptools import find_packages, setup

ParallelCompile("NPY_NUM_BUILD_JOBS").install()


def _get_flags(name):
    return (
        subprocess.run(["pkg-config", "--cflags", name], capture_output=True)
        .stdout.decode("utf-8")
        .strip("\n")
        .split(" ")
    )


def _get_eigen_flags():
    flags = _get_flags("eigen3")
    assert len(flags) == 1, f"flags: {flags}"
    assert flags[0][:2] == "-I", f"flags: {flags}"
    return [flags[0], str(pathlib.Path(flags[0]) / "unsupported")]


ext_modules = [
    Pybind11Extension(
        "_ouroboros_opengv",
        ["src/ouroboros_opengv/_bindings.cpp"]
        + sorted(glob.glob("third_party/opengv/src/**/*.cpp", recursive=True)),
        extra_compile_args=["-Ithird_party/opengv/include"] + _get_eigen_flags(),
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
    cmdclass={"build_ext": build_ext},
)
