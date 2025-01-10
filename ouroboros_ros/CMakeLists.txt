cmake_minimum_required(VERSION 3.10)
project(ouroboros_ros)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  include
)

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES active_dsg_fsm
  CATKIN_DEPENDS
    roscpp
    std_msgs
    nav_msgs
    geometry_msgs
    message_runtime
)
