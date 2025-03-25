#!/usr/bin/env python3
import rospy

from ouroboros_ros.vlc_multirobot_server_ros import VlcMultirobotServerRos

if __name__ == "__main__":
    rospy.init_node("vlc_server_node")
    node = VlcMultirobotServerRos()
    node.run()
