#!/usr/bin/env python3
import rclpy
from server.vlc_server_ros import VlcServerRos


def main():
    rclpy.init()
    vlc_server = VlcServerRos()
    rclpy.spin(vlc_server)

    vlc_server.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
