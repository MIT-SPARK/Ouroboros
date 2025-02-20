cmake_minimum_required(VERSION 3.10)
project(ouroboros_ros)

find_package(catkin REQUIRED COMPONENTS
    rospy
    ouroboros_msgs
    dynamic_reconfigure
)

catkin_python_setup()

generate_dynamic_reconfigure_options(
  cfg/PlaceDescriptorDebugging.cfg
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    ouroboros_msgs
)
