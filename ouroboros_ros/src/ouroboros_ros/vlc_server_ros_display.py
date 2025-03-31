from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from spark_config import Config, register_config

import ouroboros as ob


class VlcServerRosDisplay:
    def __init__(self, config: VlcServerRosDisplayConfig):
        self.config = config

        self.image_pair_pub = rospy.Publisher("~image_pair_out", Image, queue_size=1)
        self.kp_match_pub = rospy.Publisher("~kp_match_out", Image, queue_size=1)
        self.inlier_kp_match_pub = rospy.Publisher(
            "~inlier_kp_match_out", Image, queue_size=1
        )

        self.bridge = CvBridge()

    def setup(self, log_path: str):
        pass

    def display_image_pair(self, left: ob.VlcImage, right: ob.VlcImage, time_ns: int):
        if not self.config.display_place_matches:
            return

        if right is None:
            image_pair = ob.utils.plotting_utils.create_image_pair(left.image.rgb, None)
        else:
            image_pair = ob.utils.plotting_utils.create_image_pair(
                left.image.rgb, right.image.rgb
            )
        try:
            img_msg = self.bridge.cv2_to_imgmsg(image_pair.astype(np.uint8), "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        self.image_pair_pub.publish(img_msg)

    def display_kp_match_pair(
        self, left: ob.VlcImage, right: ob.VlcImage, left_kp, right_kp, time_ns: int
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

    def display_inlier_kp_match_pair(
        self,
        left: ob.VlcImage,
        right: ob.VlcImage,
        query_to_match,
        inliers,
        time_ns: int,
    ):
        if not self.config.display_inlier_keypoint_matches:
            return
        inlier_mask = np.zeros(len(query_to_match), dtype=bool)
        inlier_mask[inliers] = True
        left_inliers = left.keypoints[query_to_match[inlier_mask, 0]]
        right_inliers = right.keypoints[query_to_match[inlier_mask, 1]]
        left_outliers = left.keypoints[query_to_match[np.logical_not(inlier_mask), 0]]
        right_outliers = right.keypoints[query_to_match[np.logical_not(inlier_mask), 1]]
        img = ob.utils.plotting_utils.create_inlier_kp_match_pair(
            left, right, left_inliers, right_inliers, left_outliers, right_outliers
        )
        try:
            img_msg = self.bridge.cv2_to_imgmsg(img.astype(np.uint8), "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        self.inlier_kp_match_pub.publish(img_msg)


@register_config("vlc_server_display", name="ros", constructor=VlcServerRosDisplay)
@dataclass
class VlcServerRosDisplayConfig(Config):
    display_place_matches: bool = True
    display_keypoint_matches: bool = True
    display_inlier_keypoint_matches: bool = True
