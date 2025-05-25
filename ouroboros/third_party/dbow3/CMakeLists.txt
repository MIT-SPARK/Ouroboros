cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(DBoW3)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

option(BUILD_SHARED_LIBS "Set to OFF to build static libraries" ON)

add_library(
  ${PROJECT_NAME}
  src/BowVector.cpp
  src/Database.cpp
  src/DescManip.cpp
  src/FeatureVector.cpp
  src/QueryResults.cpp
  src/ScoringObject.cpp
  src/Vocabulary.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC src)
