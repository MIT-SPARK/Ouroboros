cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED CONFIG)
find_package(Eigen3 REQUIRED)

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/opengv/.git)
  find_package(Git REQUIRED)
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

get_target_property(EIGEN_INCLUDE_DIR Eigen3::Eigen
                    INTERFACE_INCLUDE_DIRECTORIES)

pybind11_add_module(
  _ouroboros_opengv
  src/ouroboros_opengv/_bindings.cpp
  third_party/opengv/src/absolute_pose/methods.cpp
  third_party/opengv/src/absolute_pose/modules/Epnp.cpp
  third_party/opengv/src/absolute_pose/modules/gp3p/code.cpp
  third_party/opengv/src/absolute_pose/modules/gp3p/init.cpp
  third_party/opengv/src/absolute_pose/modules/gp3p/reductors.cpp
  third_party/opengv/src/absolute_pose/modules/gp3p/spolynomials.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp1/code.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp1/init.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp1/reductors.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp1/spolynomials.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp2/code.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp2/init.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp2/reductors.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp2/spolynomials.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp3/code.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp3/init.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp3/reductors.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp3/spolynomials.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp4/code.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp4/init.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp4/reductors.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp4/spolynomials.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp5/code.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp5/init.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp5/reductors.cpp
  third_party/opengv/src/absolute_pose/modules/gpnp5/spolynomials.cpp
  third_party/opengv/src/absolute_pose/modules/main.cpp
  third_party/opengv/src/absolute_pose/modules/upnp2.cpp
  third_party/opengv/src/absolute_pose/modules/upnp4.cpp
  third_party/opengv/src/math/Sturm.cpp
  third_party/opengv/src/math/arun.cpp
  third_party/opengv/src/math/cayley.cpp
  third_party/opengv/src/math/gauss_jordan.cpp
  third_party/opengv/src/math/quaternion.cpp
  third_party/opengv/src/math/roots.cpp
  third_party/opengv/src/point_cloud/methods.cpp
  third_party/opengv/src/relative_pose/methods.cpp
  third_party/opengv/src/relative_pose/modules/eigensolver/modules.cpp
  third_party/opengv/src/relative_pose/modules/fivept_kneip/code.cpp
  third_party/opengv/src/relative_pose/modules/fivept_kneip/init.cpp
  third_party/opengv/src/relative_pose/modules/fivept_kneip/reductors.cpp
  third_party/opengv/src/relative_pose/modules/fivept_kneip/spolynomials.cpp
  third_party/opengv/src/relative_pose/modules/fivept_nister/modules.cpp
  third_party/opengv/src/relative_pose/modules/fivept_stewenius/modules.cpp
  third_party/opengv/src/relative_pose/modules/ge/modules.cpp
  third_party/opengv/src/relative_pose/modules/main.cpp
  third_party/opengv/src/relative_pose/modules/sixpt/modules2.cpp
  third_party/opengv/src/sac_problems/absolute_pose/AbsolutePoseSacProblem.cpp
  third_party/opengv/src/sac_problems/point_cloud/PointCloudSacProblem.cpp
  third_party/opengv/src/sac_problems/relative_pose/CentralRelativePoseSacProblem.cpp
  third_party/opengv/src/triangulation/methods.cpp)
target_link_libraries(_ouroboros_opengv PRIVATE Eigen3::Eigen)
target_include_directories(
  _ouroboros_opengv PRIVATE third_party/opengv/include
                            "${EIGEN_INCLUDE_DIR}/unsupported")

install(TARGETS _ouroboros_opengv DESTINATION ouroboros_opengv)
