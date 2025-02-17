cmake_minimum_required(VERSION 3.10)
project(ouroboros_msgs)

find_package(catkin REQUIRED COMPONENTS
    message_generation
    std_msgs
    geometry_msgs
    sensor_msgs
)

add_message_files(
  DIRECTORY
    msg
  FILES
    SparkImageMsg.msg
    VlcImageMetadataMsg.msg
    VlcImageMsg.msg
)

add_service_files(
  DIRECTORY
    srv
  FILES
    VlcInfo.srv
    VlcKeypointQuery.srv
)

generate_messages(DEPENDENCIES std_msgs geometry_msgs sensor_msgs)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    std_msgs
    geometry_msgs
    sensor_msgs
    DEPENDS
    INCLUDE_DIRS
    LIBRARIES
)
