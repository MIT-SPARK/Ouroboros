cmake_minimum_required(VERSION 3.22)
project(ouroboros_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

set(msg_files
  "msg/SparkImageMsg.msg"
  "msg/VlcImageMetadataMsg.msg"
  "msg/VlcImageMsg.msg"
  "msg/VlcInfoMsg.msg"
)

set(srv_files
  "srv/VlcKeypointQuery.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

ament_package()
