import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from ouroboros_msgs.msg import SparkImageMsg, VlcImageMetadataMsg, VlcImageMsg

from ouroboros import SparkImage, VlcImage, VlcImageMetadata, VlcPose


def vlc_image_metadata_to_msg(metadata: VlcImageMetadata) -> VlcImageMetadataMsg:
    metadata_msg = VlcImageMetadataMsg()
    metadata_msg.image_uuid = metadata.image_uuid
    metadata_msg.session_id = metadata.session_id
    metadata_msg.epoch_ns = metadata.epoch_ns
    return metadata_msg


def spark_image_to_msg(image: SparkImage) -> SparkImageMsg:
    bridge = CvBridge()
    image_msg = SparkImageMsg()
    image_msg.rgb = bridge.cv2_to_imgmsg(image.rgb, encoding="passthrough")
    image_msg.depth = bridge.cv2_to_imgmsg(image.depth, encoding="passthrough")
    return image_msg


def vlc_pose_to_msg(pose: VlcPose) -> PoseStamped:
    pose_msg = PoseStamped()
    pose_msg.header.stamp = rospy.Time(nsecs=pose.time_ns)
    pose_msg.pose.position.x = pose.position[0]
    pose_msg.pose.position.y = pose.position[1]
    pose_msg.pose.position.z = pose.position[2]
    pose_msg.pose.orientation.x = pose.rotation[0]
    pose_msg.pose.orientation.y = pose.rotation[1]
    pose_msg.pose.orientation.z = pose.rotation[2]
    pose_msg.pose.orientation.w = pose.rotation[3]
    return pose_msg


def vlc_image_to_msg(vlc: VlcImage) -> VlcImageMsg:
    bridge = CvBridge()
    vlc_msg = VlcImageMsg()
    vlc_msg.image = spark_image_to_msg(vlc.image)
    vlc_msg.header = vlc_msg.image.header
    vlc_msg.metadata = vlc_image_metadata_to_msg(vlc.metadata)
    vlc_msg.embedding = vlc.embedding.tolist()
    vlc_msg.keypoints = bridge.cv2_to_imgmsg(vlc.keypoints, encoding="passthrough")
    vlc_msg.descriptors = bridge.cv2_to_imgmsg(vlc.descriptors, encoding="passthrough")
    if vlc.pose_hint is not None:
        vlc_msg.has_pose_hint = True
        vlc_msg.pose_hint = vlc_pose_to_msg(vlc.pose_hint)
    return vlc_msg


def vlc_image_metadata_from_msg(metadata_msg: VlcImageMetadataMsg) -> VlcImageMetadata:
    return VlcImageMetadata(
        image_uuid=metadata_msg.image_uuid,
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
    pos = pose_msg.pose.position
    quat = pose_msg.pose.orientation
    return VlcPose(
        time_ns=pose_msg.header.stamp.to_nsec(),
        position=np.array([pos.x, pos.y, pos.z]),
        rotation=np.array([quat.x, quat.y, quat.z, quat.w]),
    )


def vlc_image_from_msg(vlc_msg: VlcImageMsg) -> VlcImage:
    bridge = CvBridge()
    pose_hint = None
    if vlc_msg.has_pose_hint:
        pose_hint = vlc_pose_from_msg(vlc_msg.pose_hint)
    vlc_image = VlcImage(
        metadata=vlc_image_metadata_from_msg(vlc_msg.metadata),
        image=spark_image_from_msg(vlc_msg.image),
        embedding=np.array(vlc_msg.embedding),
        keypoints=bridge.imgmsg_to_cv2(
            vlc_msg.keypoints, desired_encoding="passthrough"
        ),
        descriptors=bridge.imgmsg_to_cv2(
            vlc_msg.descriptors, desired_encoding="passthrough"
        ),
        pose_hint=pose_hint,
    )

    return vlc_image
