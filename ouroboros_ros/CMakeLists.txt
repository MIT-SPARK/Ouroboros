cmake_minimum_required(VERSION 3.22)
project(ouroboros_ros)

find_package(ament_cmake REQUIRED)
find_package(ouroboros_msgs REQUIRED)

install(
  PROGRAMS
    src/ouroboros_ros/vlc_server_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    src/ouroboros_ros/vlc_multirobot_server_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
