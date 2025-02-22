import logging

import numpy as np
import rospy
import tf2_ros
from pose_graph_tools_msgs.msg import PoseGraphEdge
from scipy.spatial.transform import Rotation as R

import ouroboros as ob


def build_robot_lc_message(
    key_from_ns,
    key_to_ns,
    robot_id,
    from_T_to,
    pose_cov,
    body_T_cam,
):
    return build_lc_message(
        key_from_ns,
        key_to_ns,
        robot_id,
        robot_id,
        from_T_to,
        pose_cov,
        body_T_cam,
        body_T_cam,
    )


def build_lc_message(
    key_from_ns,
    key_to_ns,
    robot_from,
    robot_to,
    from_T_to,
    pose_cov,
    robot_from_T_cam,
    robot_to_T_cam,
):
    bodyfrom_T_bodyto = robot_from_T_cam @ from_T_to @ ob.invert_pose(robot_to_T_cam)
    relative_position = bodyfrom_T_bodyto[:3, 3]
    relative_orientation = R.from_matrix(bodyfrom_T_bodyto[:3, :3])

    lc_edge = PoseGraphEdge()
    lc_edge.header.stamp = rospy.Time.now()
    lc_edge.key_from = key_from_ns
    lc_edge.key_to = key_to_ns
    lc_edge.robot_from = robot_from
    lc_edge.robot_to = robot_to
    lc_edge.type = PoseGraphEdge.LOOPCLOSE
    lc_edge.pose.position.x = relative_position[0]
    lc_edge.pose.position.y = relative_position[1]
    lc_edge.pose.position.z = relative_position[2]
    q = relative_orientation.as_quat()
    lc_edge.pose.orientation.x = q[0]
    lc_edge.pose.orientation.y = q[1]
    lc_edge.pose.orientation.z = q[2]
    lc_edge.pose.orientation.w = q[3]

    lc_edge.covariance = pose_cov.flatten()
    return lc_edge


def get_tf_as_pose(tf_buffer, fixed_frame, body_frame, time=None, timeout=1.0):
    if time is None:
        time = rospy.Time()
    try:
        trans = tf_buffer.lookup_transform(
            fixed_frame, body_frame, time, rospy.Duration(timeout)
        )
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ) as e:
        rospy.logwarn(
            " Could not transform %s from %s: %s ", fixed_frame, body_frame, str(e)
        )
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
    vlc_pose = ob.VlcPose(time_ns=time_ns, position=current_pos, rotation=current_rot)
    return vlc_pose


def parse_camera_info(info_msg):
    K = np.array(info_msg.K).reshape((3, 3))
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    return ob.PinholeCamera(fx=fx, fy=fy, cx=cx, cy=cy)


# adapted from https://gist.github.com/ablakey/4f57dca4ea75ed29c49ff00edf622b38
class RosForwarder(logging.Handler):
    """Class to forward logging to ros handler."""

    level_map = {
        logging.DEBUG: rospy.logdebug,
        logging.INFO: rospy.loginfo,
        logging.WARNING: rospy.logwarn,
        logging.ERROR: rospy.logerr,
        logging.CRITICAL: rospy.logfatal,
    }

    def emit(self, record):
        """Send message to ROS."""
        level = record.levelno if record.levelno in self.level_map else logging.CRITICAL
        self.level_map[level](f"{record.name}: {record.msg}")


def setup_ros_log_forwarding(level=logging.INFO):
    """Forward logging to ROS."""
    logger = logging.getLogger("ouroboros")
    logger.addHandler(RosForwarder())
    logger.setLevel(level)
