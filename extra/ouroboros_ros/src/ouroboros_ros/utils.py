import rospy
import tf2_ros
import numpy as np

from ouroboros import VlcPose


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
