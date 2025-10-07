cmake_minimum_required(VERSION 3.16)
project(DBoW3)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

option(BUILD_SHARED_LIBS "Set to OFF to build static libraries" ON)

find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} src/BowVector.cpp src/Vocabulary.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC src)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
