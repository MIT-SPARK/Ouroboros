import pathlib
import subprocess

from pybind11.setup_helpers import ParallelCompile, Pybind11Extension, build_ext
from setuptools import find_packages, setup

ParallelCompile("NPY_NUM_BUILD_JOBS").install()


def _get_flags(name):
    flags = (
        subprocess.run(["pkg-config", "--cflags", name], capture_output=True)
        .stdout.decode("utf-8")
        .strip("\n")
        .split(" ")
    )
    return [x for x in flags if len(x) > 0]


def _get_eigen_flags():
    flags = _get_flags("eigen3")
    assert len(flags) == 1, f"flags: {flags}"
    assert flags[0][:2] == "-I", f"flags: {flags}"
    return [flags[0], str(pathlib.Path(flags[0]) / "unsupported")]


ext_modules = []

opengv_exists = pathlib.Path("third_party/opengv/CMakeLists.txt").exists()
if not opengv_exists:
    ret = subprocess.run(["git", "submodule", "update", "--init"])
    opengv_exists = ret.returncode == 0

if opengv_exists:
    ext_modules.append(
        Pybind11Extension(
            "ouroboros_opengv._ouroboros_opengv",
            [
                "src/ouroboros_opengv/_bindings.cpp",
                "third_party/opengv/src/absolute_pose/methods.cpp",
                "third_party/opengv/src/absolute_pose/modules/Epnp.cpp",
                "third_party/opengv/src/absolute_pose/modules/gp3p/code.cpp",
                "third_party/opengv/src/absolute_pose/modules/gp3p/init.cpp",
                "third_party/opengv/src/absolute_pose/modules/gp3p/reductors.cpp",
                "third_party/opengv/src/absolute_pose/modules/gp3p/spolynomials.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp1/code.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp1/init.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp1/reductors.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp1/spolynomials.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp2/code.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp2/init.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp2/reductors.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp2/spolynomials.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp3/code.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp3/init.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp3/reductors.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp3/spolynomials.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp4/code.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp4/init.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp4/reductors.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp4/spolynomials.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp5/code.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp5/init.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp5/reductors.cpp",
                "third_party/opengv/src/absolute_pose/modules/gpnp5/spolynomials.cpp",
                "third_party/opengv/src/absolute_pose/modules/main.cpp",
                "third_party/opengv/src/absolute_pose/modules/upnp2.cpp",
                "third_party/opengv/src/absolute_pose/modules/upnp4.cpp",
                "third_party/opengv/src/math/Sturm.cpp",
                "third_party/opengv/src/math/arun.cpp",
                "third_party/opengv/src/math/cayley.cpp",
                "third_party/opengv/src/math/gauss_jordan.cpp",
                "third_party/opengv/src/math/quaternion.cpp",
                "third_party/opengv/src/math/roots.cpp",
                "third_party/opengv/src/point_cloud/methods.cpp",
                "third_party/opengv/src/relative_pose/methods.cpp",
                "third_party/opengv/src/relative_pose/modules/eigensolver/modules.cpp",
                "third_party/opengv/src/relative_pose/modules/fivept_kneip/code.cpp",
                "third_party/opengv/src/relative_pose/modules/fivept_kneip/init.cpp",
                "third_party/opengv/src/relative_pose/modules/fivept_kneip/reductors.cpp",
                "third_party/opengv/src/relative_pose/modules/fivept_kneip/spolynomials.cpp",
                "third_party/opengv/src/relative_pose/modules/fivept_nister/modules.cpp",
                "third_party/opengv/src/relative_pose/modules/fivept_stewenius/modules.cpp",
                "third_party/opengv/src/relative_pose/modules/ge/modules.cpp",
                "third_party/opengv/src/relative_pose/modules/main.cpp",
                "third_party/opengv/src/relative_pose/modules/sixpt/modules2.cpp",
                "third_party/opengv/src/sac_problems/absolute_pose/AbsolutePoseSacProblem.cpp",
                "third_party/opengv/src/sac_problems/point_cloud/PointCloudSacProblem.cpp",
                "third_party/opengv/src/sac_problems/relative_pose/CentralRelativePoseSacProblem.cpp",
                "third_party/opengv/src/triangulation/methods.cpp",
            ],
            extra_compile_args=["-Ithird_party/opengv/include"] + _get_eigen_flags(),
        )
    )

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
        "scipy>=1.4.0",
        "spark_config @ git+https://github.com/MIT-SPARK/Spark-Config@main",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
