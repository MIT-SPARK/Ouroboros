#!/usr/bin/env python3
import rospy
from ouroboros_ros.vlc_server_ros import VlcServerRos

if __name__ == "__main__":
    rospy.init_node("vlc_server_node")
    node = VlcServerRos()
    node.run()
