from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import ouroboros as ob
from ouroboros.config import Config, register_config


class VlcServerRosDisplay:
    def __init__(self, config: VlcServerRosDisplayConfig):
        self.config = config

        self.image_pair_pub = rospy.Publisher("~image_pair_out", Image, queue_size=1)
        self.kp_match_pub = rospy.Publisher("~kp_match_out", Image, queue_size=1)

        self.bridge = CvBridge()

    def display_image_pair(self, left: ob.SparkImage, right: ob.SparkImage):
        if not self.config.display_place_matches:
            return

        image_pair = ob.utils.plotting_utils.create_image_pair(left, right)
        try:
            img_msg = self.bridge.cv2_to_imgmsg(image_pair.astype(np.uint8), "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        self.image_pair_pub.publish(img_msg)

    def display_kp_match_pair(
        self, left: ob.VlcImage, right: ob.VlcImage, left_kp, right_kp
    ):
        if not self.config.display_keypoint_matches:
            return

        image_pair = ob.utils.plotting_utils.create_kp_match_pair(
            left, right, left_kp, right_kp
        )
        try:
            img_msg = self.bridge.cv2_to_imgmsg(image_pair.astype(np.uint8), "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        self.kp_match_pub.publish(img_msg)


@register_config("vlc_server_display", name="ros", constructor=VlcServerRosDisplay)
@dataclass
class VlcServerRosDisplayConfig(Config):
    display_place_matches: bool = True
    display_keypoint_matches: bool = True
