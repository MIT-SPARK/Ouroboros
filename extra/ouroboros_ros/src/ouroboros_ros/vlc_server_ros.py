#!/usr/bin/env python3

import cv_bridge
import matplotlib.pyplot as plt
import numpy as np
import rospy
import tf2_ros
from ouroboros_ros.utils import get_tf_as_pose
from pose_graph_tools_msgs.msg import PoseGraph, PoseGraphEdge
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image

import ouroboros as ob
from ouroboros.utils.plotting_utils import plt_fast_pause


def update_plot(line, pts, images_to_pose):

    if len(images_to_pose) == 0:
        return

    positions = []
    for _, v in images_to_pose.items():
        positions.append(v.position)

    positions = np.array(positions)

    line.set_data(positions[:, 0], positions[:, 1])
    pts.set_offsets(positions[:, :2])
    ax = plt.gca()
    ax.relim()
    ax.autoscale_view()
    plt.draw_all()
    plt_fast_pause(0.05)


def plot_lc(lc, image_to_pose):

    query_pos = image_to_pose[lc.from_image_uuid].position
    match_pos = image_to_pose[lc.to_image_uuid].position
    plt.plot(
        [match_pos[0], query_pos[0]],
        [match_pos[1], query_pos[1]],
        color="r",
    )


def build_lc_message(
    key_from_ns,
    key_to_ns,
    robot_id,
    from_T_to,
    pose_cov,
):

    relative_position = from_T_to[:3, 3]
    relative_orientation = R.from_matrix(from_T_to[:3, :3])

    lc_edge = PoseGraphEdge()
    lc_edge.header.stamp = rospy.Time.now()
    lc_edge.key_from = key_from_ns
    lc_edge.key_to = key_to_ns
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

    lc_edge.covariance = pose_cov.flatten()
    return lc_edge


class VlcServerRos:

    def __init__(self):

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.lc_pub = rospy.Publisher("~loop_closure_output", PoseGraph, queue_size=1)

        self.fixed_frame = rospy.get_param("~fixed_frame")
        self.camera_frame = rospy.get_param("~camera_frame")
        self.show_plots = rospy.get_param("~show_plots")
        self.vlc_frame_period_s = rospy.get_param("~vlc_frame_period_s")
        lc_recent_pose_lockout_s = rospy.get_param("~lc_recent_pose_lockout_s")
        lc_similarity_threshold = rospy.get_param("~lc_similarity_threshold")
        self.lc_send_delay_s = rospy.get_param("~lc_send_delay_s")

        place_method = rospy.get_param("~place_method")
        keypoint_method = rospy.get_param("~keypoint_method")
        descriptor_method = rospy.get_param("~descriptor_method")
        pose_method = rospy.get_param("~pose_method")
        self.loop_rate = rospy.Rate(10)

        self.images_to_pose = {}
        self.last_vlc_frame_time = None

        self.robot_id = 0
        self.vlc_server = ob.VlcServer(
            place_method,
            keypoint_method,
            descriptor_method,
            pose_method,
            lc_recent_pose_lockout_s * 1e9,
            lc_similarity_threshold,
            robot_id=0,
            strict_keypoint_evaluation=True,
        )

        if self.show_plots:
            plt.ion()
            plt.figure()
            plt.show()
            self.position_line = plt.plot([], [], color="g")[0]
            self.position_points = plt.scatter([], [], color="g")

        # Hydra takes too long to add agent poses to the backend, so if we send the LC
        # immediately it will get rejected. To work around this, we can't send the loop
        # closure until several seconds after it is detected
        self.loop_closure_delayed_queue = []

        # TODO: Should support a mode where we synchronize rgb and d (with
        # topic synchronizer?) so that we can push combined rgbd frames to the
        # vlc_server

        self.image_sub = rospy.Subscriber("~image_in", Image, self.image_callback)

    def image_callback(self, msg):

        if not (
            self.last_vlc_frame_time is None
            or (rospy.Time.now() - self.last_vlc_frame_time).to_sec()
            > self.vlc_frame_period_s
        ):
            return
        self.last_vlc_frame_time = rospy.Time.now()

        bridge = cv_bridge.CvBridge()
        try:
            image = bridge.imgmsg_to_cv2(msg, "bgr8")
        except cv_bridge.CvBridgeError as e:
            print(e)
            raise e

        # An estimate of the current camera pose, which is optionally used to
        # inform the place recognition, keypoint detection, keypoint
        # descriptor, and pose recovery methods.
        camera_pose = get_tf_as_pose(
            self.tf_buffer, self.fixed_frame, self.camera_frame, msg.header.stamp
        )

        if camera_pose is None:
            print("Cannot get current pose, skipping frame!")
            return

        spark_image = ob.SparkImage(rgb=image)
        image_uuid, loop_closures = self.vlc_server.add_and_query_frame(
            spark_image, msg.header.stamp.to_nsec(), pose_hint=camera_pose
        )
        self.images_to_pose[image_uuid] = camera_pose

        if loop_closures is None:
            return

        pose_cov_mat = self.build_pose_cov_mat
        pg = PoseGraph()
        pg.header.stamp = rospy.Time.now()
        for lc in loop_closures:

            if self.show_plots:
                plot_lc(lc, self.images_to_pose)

            from_time_ns, to_time_ns = self.vlc_server.get_lc_times(lc.metadata.lc_uuid)

            lc_edge = build_lc_message(
                from_time_ns,
                to_time_ns,
                self.robot_id,
                lc.f_T_t,
                pose_cov_mat,
            )

            pg.edges.append(lc_edge)
        self.loop_closure_delayed_queue.append(
            (rospy.Time.now().to_sec() + self.lc_send_delay, pg)
        )

    def build_pose_cov_mat(self):
        pose_cov_mat = np.zeros((6, 6))
        pos_cov = 0.1
        pose_cov_mat[0, 0] = pos_cov
        pose_cov_mat[1, 1] = pos_cov
        pose_cov_mat[2, 2] = pos_cov
        rot_cov = 0.001
        pose_cov_mat[3, 3] = rot_cov
        pose_cov_mat[4, 4] = rot_cov
        pose_cov_mat[5, 5] = rot_cov
        return pose_cov_mat

    def run(self):
        while not rospy.is_shutdown():
            self.step()
            self.loop_rate.sleep()

    def step(self):

        if self.show_plots:
            update_plot(self.position_line, self.position_points, self.images_to_pose)

        # This is a dumb hack because Hydra doesn't deal properly with
        # receiving loop closures where the agent nodes haven't been added to
        # the backend yet, which occurs a lot when the backend runs slightly
        # slowly. You need to wait up to several seconds to send a loop closure
        # before it will be accepted by Hydra.
        while len(self.loop_closure_delayed_queue) > 0:
            send_time, pg = self.loop_closure_delayed_queue[0]
            if rospy.Time.now().to_sec() >= send_time:
                self.lc_pub.publish(pg)
                self.loop_closure_delayed_queue = self.loop_closure_delayed_queue[1:]
            else:
                break
