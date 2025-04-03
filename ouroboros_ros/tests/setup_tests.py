"""Test import setup"""

import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(curr_dir, "../src/ouroboros_ros")

sys.path.append(src_path)
# TODO(Yun) figure out cleaner way to get around ROS2 and importing
