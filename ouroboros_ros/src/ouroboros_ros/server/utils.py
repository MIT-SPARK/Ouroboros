import logging

import numpy as np
import tf2_ros
from pose_graph_tools_msgs.msg import PoseGraphEdge
from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

import ouroboros as ob


def build_robot_lc_message(
    key_from_ns,
    key_to_ns,
    robot_id,
    from_T_to,
    pose_cov,
    body_T_cam,
    time,
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
        time,
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
    time,
):
    bodyfrom_T_bodyto = robot_from_T_cam @ from_T_to @ ob.invert_pose(robot_to_T_cam)
    relative_position = bodyfrom_T_bodyto[:3, 3]
    relative_orientation = R.from_matrix(bodyfrom_T_bodyto[:3, :3])

    lc_edge = PoseGraphEdge()
    lc_edge.header.stamp = time.to_msg()
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


def parse_tf_pose(tf):
    current_pos = np.array(
        [
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ]
    )

    current_rot = np.array(
        [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        ]
    )

    time_ns = Time().from_msg(tf.header.stamp).nanoseconds
    vlc_pose = ob.VlcPose(time_ns=time_ns, position=current_pos, rotation=current_rot)
    return vlc_pose


def get_tf_as_pose(tf_buffer, fixed_frame, body_frame, time, timeout=1.0):
    try:
        trans = tf_buffer.lookup_transform(
            fixed_frame, body_frame, time, Duration(seconds=timeout)
        )
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ) as e:
        logging.error(
            " Could not transform %s from %s: %s ", fixed_frame, body_frame, str(e)
        )
        return

    return parse_tf_pose(trans)


def parse_camera_info(info_msg):
    K = np.array(info_msg.k).reshape((3, 3))
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    return ob.PinholeCamera(fx=fx, fy=fy, cx=cx, cy=cy)
