cmake_minimum_required(VERSION 3.10)
project(ouroboros_ros)

find_package(catkin REQUIRED COMPONENTS
    rospy
)

catkin_package(
  CATKIN_DEPENDS
    rospy
)
