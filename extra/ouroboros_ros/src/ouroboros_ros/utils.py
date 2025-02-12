import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from ouroboros_msgs.msg import SparkImageMsg, VlcImageMetadataMsg, VlcImageMsg

from ouroboros import SparkImage, VlcImage, VlcImageMetadata, VlcPose


def vlc_image_metadata_from_msg(metadata_msg: VlcImageMetadataMsg):
    return VlcImageMetadata(
        image_uuid=metadata_msg.uuid,
        session_id=metadata_msg.session_id,
        epoch_ns=metadata_msg.epoch_ns,
    )


def spark_image_from_msg(image_msg: SparkImageMsg):
    bridge = CvBridge()
    return SparkImage(
        rgb=bridge.imgmsg_to_cv2(image_msg.rgb, desired_encoding="passthrough"),
        depth=bridge.imgmsg_to_cv2(image_msg.depth, desired_encoding="passthrough"),
    )


def vlc_image_from_msg(vlc_msg: VlcImageMsg):
    vlc_image = VlcImage(
        metadata=vlc_image_metadata_from_msg(vlc_msg.metadata),
        image=spark_image_from_msg(vlc_msg.metadata),
    )

    return vlc_image


def get_tf_as_pose(tf_buffer, fixed_frame, body_frame, time=None):
    if time is None:
        time = rospy.Time()
    try:
        trans = tf_buffer.lookup_transform(fixed_frame, body_frame, time)
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ):
        rospy.logwarn(" Could not transform %s from %s ", fixed_frame, body_frame)
        return

    current_pos = np.array(
        [
            trans.transform.translation.x,
            trans.transform.translation.y,
            trans.transform.translation.z,
        ]
    )

    current_rot = np.array(
        [
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w,
        ]
    )

    time_ns = trans.header.stamp.to_nsec()
    vlc_pose = VlcPose(time_ns=time_ns, position=current_pos, rotation=current_rot)
    return vlc_pose
