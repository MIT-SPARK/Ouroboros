import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from ouroboros_msgs.msg import SparkImageMsg, VlcImageMetadataMsg, VlcImageMsg

from ouroboros import SparkImage, VlcImage, VlcImageMetadata, VlcPose


def vlc_image_metadata_to_msg(metadata: VlcImageMetadata) -> VlcImageMetadataMsg:
    vlc_msg = VlcImageMetadataMsg()
    return vlc_msg


def spark_image_to_msg(image: SparkImage) -> SparkImageMsg:
    bridge = CvBridge()
    msg = SparkImageMsg()
    msg.rgb = bridge.cv2_to_imgmsg(image.rgb, encoding="passthrough")
    msg.depth = bridge.cv2_to_imgmsg(image.depth, encoding="passthrough")
    return msg


def vlc_pose_to_msg(pose: VlcPose) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = rospy.Time(nsecs=pose.time_ns)
    msg.pose.position.x = pose.position[0]
    msg.pose.position.y = pose.position[1]
    msg.pose.position.z = pose.position[2]
    msg.pose.orientation.x = pose.rotation[0]
    msg.pose.orientation.y = pose.rotation[1]
    msg.pose.orientation.z = pose.rotation[2]
    msg.pose.orientation.w = pose.rotation[3]
    return msg


def vlc_image_to_msg(vlc: VlcImage) -> VlcImageMsg:
    bridge = CvBridge()
    msg = VlcImageMsg()
    msg.embedding = bridge.cv2_to_imgmsg(vlc.embedding, encodin="passthrough")
    return msg


def vlc_image_metadata_from_msg(metadata_msg: VlcImageMetadataMsg) -> VlcImageMetadata:
    return VlcImageMetadata(
        image_uuid=metadata_msg.uuid,
        session_id=metadata_msg.session_id,
        epoch_ns=metadata_msg.epoch_ns,
    )


def spark_image_from_msg(image_msg: SparkImageMsg) -> SparkImage:
    bridge = CvBridge()
    return SparkImage(
        rgb=bridge.imgmsg_to_cv2(image_msg.rgb, desired_encoding="passthrough"),
        depth=bridge.imgmsg_to_cv2(image_msg.depth, desired_encoding="passthrough"),
    )


def vlc_pose_from_msg(pose_msg: PoseStamped) -> VlcPose:
    pos = (pose_msg.pose.position,)
    quat = (pose_msg.pose.orientation,)
    return VlcPose(
        time_ns=pose_msg.header.stamp.to_nsec(),
        position=np.array([pos.x, pos.y, pos.z]),
        rotation=np.array([quat.x, quat.y, quat.z, quat.w]),
    )


def vlc_image_from_msg(vlc_msg: VlcImageMsg) -> VlcImage:
    bridge = CvBridge()
    vlc_image = VlcImage(
        metadata=vlc_image_metadata_from_msg(vlc_msg.metadata),
        image=spark_image_from_msg(vlc_msg.metadata),
        embedding=bridge.imgmsg_to_cv2(
            vlc_msg.embedding, desired_encoding="passthrough"
        ),
        keypoints=bridge.imgmsg_to_cv2(
            vlc_msg.keypoints, desired_encoding="passthrough"
        ),
        descriptors=bridge.imgmsg_to_cv2(
            vlc_msg.descriptors, desired_encoding="passthrough"
        ),
        pose_hint=vlc_pose_from_msg(vlc_msg.pose_hint),
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
