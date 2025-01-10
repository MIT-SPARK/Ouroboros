#!/usr/bin/env python3
from dataclasses import dataclass
import functools
import rospy
import numpy as np
import matplotlib
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

from pose_graph_tools_msgs.msg import PoseGraph, PoseGraphEdge

import tf2_ros


from vlc_db.vlc_db import VlcDb
from vlc_db.spark_loop_closure import SparkLoopClosure


@dataclass
class VlcPose:
    time_ns: int
    position: np.ndarray  # x,y,z
    rotation: np.ndarray  # qx, qy, qz

    def to_descriptor(self):
        return np.hstack([[self.time_ns], self.position, self.rotation])

    @classmethod
    def from_descriptor(cls, d):
        return cls(time_ns=d[0], position=d[1:4], rotation=d[4:])


# Hydra takes too long to add agent poses to the backend, so if we send the LC
# immediately it will get rejected. To work around this, we can't send the loop
# closure until several seconds after it is detected
loop_closure_delayed_queue = []


def invert_pose(p):
    p_inv = np.zeros((4, 4))
    p_inv[:3, :3] = p[:3, :3].T
    p_inv[:3, 3] = -p[:3, :3].T @ p[:3, 3]
    p_inv[3, 3] = 1
    return p_inv


def pose_from_quat_trans(q, t):
    pose = np.zeros((4, 4))
    Rmat = R.from_quat(q).as_matrix()
    pose[:3, :3] = Rmat
    pose[:3, 3] = t
    pose[3, 3] = 1
    return pose


def mypause(interval):
    backend = plt.rcParams["backend"]
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


def get_tf_as_descriptor(tf_buffer, fixed_frame, body_frame):
    try:
        trans = tf_buffer.lookup_transform(fixed_frame, body_frame, rospy.Time())
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
    return time_ns, vlc_pose.to_descriptor()


def update_plot(line, pts, vlc_db):

    positions = []
    for image in vlc_db.iterate_images():
        positions.append(VlcPose.from_descriptor(image.embedding).position)

    positions = np.array(positions)

    line.set_data(positions[:, 0], positions[:, 1])
    pts.set_offsets(positions[:, :2])
    ax = plt.gca()
    ax.relim()
    ax.autoscale_view()
    plt.draw_all()
    mypause(0.05)


def compute_descriptor_distance(
    lc_recent_pose_lockout_ns, lc_distance_threshold, d_query, d_stored
):

    query_pose = VlcPose.from_descriptor(d_query)
    stored_pose = VlcPose.from_descriptor(d_stored)

    if stored_pose.time_ns > query_pose.time_ns - lc_recent_pose_lockout_ns:
        return np.inf

    d = np.linalg.norm(query_pose.position - stored_pose.position)

    if d > lc_distance_threshold:
        return np.inf
    else:
        return d


def plot_lc(qd, md):
    qp = VlcPose.from_descriptor(qd).position
    mp = VlcPose.from_descriptor(md).position
    plt.plot(
        [mp[0], qp[0]],
        [mp[1], qp[1]],
        color="r",
    )


def recover_pose(query_descriptors, match_descriptors):

    query_pose = VlcPose.from_descriptor(query_descriptors[0])
    match_pose = VlcPose.from_descriptor(match_descriptors[0])

    w_T_cur = pose_from_quat_trans(query_pose.rotation, query_pose.position)
    w_T_old = pose_from_quat_trans(match_pose.rotation, match_pose.position)

    old_T_new = invert_pose(invert_pose(w_T_old) @ w_T_cur)

    return old_T_new


def compute_lc(
    lc_recent_pose_lockout_ns,
    lc_distance_threshold,
    session_id,
    lc_send_delay,
    vlc_db,
    last_uid,
):

    if last_uid is None:
        return

    query_embedding = vlc_db.get_image(last_uid).embedding
    query_fn = functools.partial(
        compute_descriptor_distance, lc_recent_pose_lockout_ns, lc_distance_threshold
    )
    (nearest, distances) = vlc_db.query_embeddings(
        np.array([query_embedding]), 1, distance_metric=query_fn
    )

    if np.isinf(distances[0][0]):
        return

    match_uid = nearest[0].metadata.image_uuid
    match_time = nearest[0].metadata.epoch_ns

    query_kps, query_descriptors = vlc_db.get_keypoints(last_uid)
    match_kps, match_descriptors = vlc_db.get_keypoints(match_uid)

    from_T_to = recover_pose(query_descriptors, match_descriptors)
    lc = SparkLoopClosure(
        from_image_uuid=0, to_image_uuid=0, f_T_t=from_T_to, quality=1
    )
    vlc_db.add_lc(lc, session_id)

    plot_lc(query_descriptors[0], match_descriptors[0])

    relative_position = from_T_to[:3, 3]
    relative_orientation = R.from_matrix(from_T_to[:3, :3])

    lc_edge = PoseGraphEdge()
    lc_edge.header.stamp = rospy.Time.now()
    lc_edge.key_from = int(vlc_db.get_image(last_uid).metadata.epoch_ns)
    lc_edge.key_to = int(match_time)
    lc_edge.robot_from = 0
    lc_edge.robot_to = 0
    lc_edge.type = PoseGraphEdge.LOOPCLOSE
    lc_edge.pose.position.x = relative_position[0]
    lc_edge.pose.position.y = relative_position[1]
    lc_edge.pose.position.z = relative_position[2]
    q = relative_orientation.as_quat()
    lc_edge.pose.orientation.x = q[0]
    lc_edge.pose.orientation.y = q[1]
    lc_edge.pose.orientation.z = q[2]
    lc_edge.pose.orientation.w = q[3]

    cov = np.zeros((6, 6))
    pos_cov = 0.1
    cov[0, 0] = pos_cov
    cov[1, 1] = pos_cov
    cov[2, 2] = pos_cov
    rot_cov = 0.001
    cov[3, 3] = rot_cov
    cov[4, 4] = rot_cov
    cov[5, 5] = rot_cov
    lc_edge.covariance = cov.flatten()

    pg = PoseGraph()
    pg.header.stamp = rospy.Time.now()
    pg.edges.append(lc_edge)
    loop_closure_delayed_queue.append((rospy.Time.now().to_sec() + lc_send_delay, pg))


if __name__ == "__main__":
    rospy.init_node("lc_gt_node")

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    lc_pub = rospy.Publisher("~loop_closure_output", PoseGraph, queue_size=1)

    fixed_frame = rospy.get_param("~fixed_frame")
    body_frame = rospy.get_param("~body_frame")
    show_plots = rospy.get_param("~show_plots")
    loop_closure_period = rospy.get_param("~loop_closure_period_s")
    lc_recent_pose_lockout_s = rospy.get_param("~lc_recent_pose_lockout_s")
    lc_distance_threshold_m = rospy.get_param("~lc_distance_threshold_m")
    lc_send_delay_s = rospy.get_param("~lc_send_delay_s")

    if show_plots:
        plt.ion()
        plt.figure()
        plt.show()
        line = plt.plot([], [], color="g")[0]
        points = plt.scatter([], [], color="g")

    vlc_db = VlcDb(8)
    session_id = vlc_db.add_session(0)

    rate = rospy.Rate(10)
    t0 = rospy.Time.now().to_sec()
    last_lc = 0
    last_uid = None
    while not rospy.is_shutdown():
        rate.sleep()

        maybe_time_desc = get_tf_as_descriptor(tf_buffer, fixed_frame, body_frame)
        if not maybe_time_desc:
            continue
        image_time_ns, gt_lc_descriptor = maybe_time_desc
        last_uid = vlc_db.add_image(session_id, image_time_ns, None)
        vlc_db.update_embedding(last_uid, gt_lc_descriptor)
        vlc_db.update_keypoints(
            last_uid, np.zeros([1, 2]), descriptors=np.array([gt_lc_descriptor])
        )

        if show_plots:
            update_plot(line, points, vlc_db)

        if rospy.Time.now().to_sec() - last_lc >= loop_closure_period:
            compute_lc(
                lc_recent_pose_lockout_s * 1e9,
                lc_distance_threshold_m,
                session_id,
                lc_send_delay_s,
                vlc_db,
                last_uid,
            )
            last_lc = rospy.Time.now().to_sec()

        # This is a dumb hack because Hydra doesn't deal properly with
        # receiving loop closures where the agent nodes haven't been added to
        # the backend yet, which occurs a lot when the backend runs slightly
        # slowly. You need to wait up to several seconds to send a loop closure
        # before it will be accepted by Hydra.
        while len(loop_closure_delayed_queue) > 0:
            send_time, pg = loop_closure_delayed_queue[0]
            if rospy.Time.now().to_sec() >= send_time:
                lc_pub.publish(pg)
                loop_closure_delayed_queue = loop_closure_delayed_queue[1:]
            else:
                break
