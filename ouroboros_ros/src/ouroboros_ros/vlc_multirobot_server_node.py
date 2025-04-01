#!/usr/bin/env python3
import rclpy
from server.vlc_multirobot_server_ros import VlcMultirobotServerRos


def main():
    rclpy.init()
    vlc_server = VlcMultirobotServerRos()
    rclpy.spin(vlc_server)

    vlc_server.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
