#!/usr/bin/env python3
import rospy
from ouroboros_ros.utils import setup_ros_log_forwarding
from ouroboros_ros.vlc_server_ros import VlcServerRos

if __name__ == "__main__":
    rospy.init_node("vlc_server_node")
    setup_ros_log_forwarding()
    node = VlcServerRos()
    node.run()
