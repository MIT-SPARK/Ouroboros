cmake_minimum_required(VERSION 3.10)
project(ouroboros_ros)

find_package(catkin REQUIRED COMPONENTS
    rospy
    dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
  cfg/PlaceDescriptorDebugging.cfg
)

catkin_package(
  CATKIN_DEPENDS
    rospy
)
