#!/usr/bin/env python3

import threading

import cv_bridge
import matplotlib.pyplot as plt
import message_filters
import numpy as np
import rospy
import spark_config as sc
import tf2_ros
from pose_graph_tools_msgs.msg import PoseGraph
from sensor_msgs.msg import CameraInfo, Image

import ouroboros as ob
from ouroboros.utils.plotting_utils import plt_fast_pause
from ouroboros_ros.utils import (
    build_robot_lc_message,
    get_tf_as_pose,
    parse_camera_info,
)


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


def plot_lc(query_position, match_position, color):
    plt.plot(
        [match_position[0], query_position[0]],
        [match_position[1], query_position[1]],
        color=color,
    )


def get_query_and_est_match_position(lc, image_to_pose, body_T_cam):
    query_pose = image_to_pose[lc.from_image_uuid]

    world_T_query = query_pose.as_matrix()
    query_T_match = body_T_cam @ ob.invert_pose(lc.f_T_t) @ ob.invert_pose(body_T_cam)
    world_T_match = world_T_query @ query_T_match

    query_position = query_pose.position
    est_match_position = world_T_match[:3, 3]
    return query_position, est_match_position


class VlcServerRos:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.lc_pub = rospy.Publisher("~loop_closure_output", PoseGraph, queue_size=1)

        self.fixed_frame = rospy.get_param("~fixed_frame")
        self.hint_body_frame = rospy.get_param("~hint_body_frame")
        self.body_frame = rospy.get_param("~body_frame", "")
        self.camera_frame = rospy.get_param("~camera_frame", "")
        if self.body_frame == "" and self.camera_frame == "":
            rospy.logwarn("Body and camera frames not specified: assuming identity!")
        elif self.body_frame == "" or self.camera_frame == "":
            frame_str = f"body='{self.body_frame}', camera='{self.camera_frame}'"
            raise ValueError(f"Body/camera frames must not be empty, got {frame_str}!")

        self.show_plots = rospy.get_param("~show_plots")
        self.vlc_frame_period_s = rospy.get_param("~vlc_frame_period_s")
        self.lc_send_delay_s = rospy.get_param("~lc_send_delay_s")
        self.robot_id = rospy.get_param("~robot_id")

        plugins = sc.discover_plugins("ouroboros_")
        print("Found plugins: ", plugins)
        config_path = rospy.get_param("~config_path")
        server_config = ob.VlcServerConfig.load(config_path)
        self.vlc_server = ob.VlcServer(
            server_config,
            robot_id=self.robot_id,
        )

        self.camera_config = self.get_camera_config_ros()
        print(f"camera config: {self.camera_config}")
        self.session_id = self.vlc_server.register_camera(
            self.robot_id, self.camera_config, rospy.Time.now().to_nsec()
        )

        self.loop_rate = rospy.Rate(10)
        self.images_to_pose = {}
        self.image_pose_lock = threading.Lock()
        self.last_vlc_frame_time = None

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

        self.image_sub = message_filters.Subscriber("~image_in", Image)
        self.depth_sub = message_filters.Subscriber("~depth_in", Image)

        self.image_depth_sub = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub], 10, 0.1
        )
        self.image_depth_sub.registerCallback(self.image_depth_callback)

        if self.body_frame != self.camera_frame:
            self.body_T_cam = self.get_body_T_cam_ros()
        else:
            self.body_T_cam = ob.VlcPose(
                time_ns=0, position=np.array([0, 0, 0]), rotation=np.array([0, 0, 0, 1])
            )

    def get_camera_config_ros(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                info_msg = rospy.wait_for_message("~camera_info", CameraInfo, timeout=5)
            except rospy.ROSException:
                rospy.logerr("Timed out waiting for camera info")
                rate.sleep()
                continue
            pinhole = parse_camera_info(info_msg)
            return pinhole
        return None

    def get_body_T_cam_ros(self, max_retries=10, retry_delay=0.5):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                return get_tf_as_pose(
                    self.tf_buffer, self.body_frame, self.camera_frame, rospy.Time.now()
                )
            except rospy.ROSException:
                rospy.logerr("Failed to get body_T_cam transform")
                rate.sleep()
        return None

    def process_new_frame(self, image, stamp_ns, hint_pose):
        return self.vlc_server.add_and_query_frame(
            self.session_id,
            image,
            stamp_ns,
            pose_hint=hint_pose,
        )

    def image_depth_callback(self, img_msg, depth_msg):
        if not (
            self.last_vlc_frame_time is None
            or (rospy.Time.now() - self.last_vlc_frame_time).to_sec()
            > self.vlc_frame_period_s
        ):
            return
        self.last_vlc_frame_time = rospy.Time.now()

        bridge = cv_bridge.CvBridge()
        try:
            color_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except cv_bridge.CvBridgeError as e:
            print(e)
            raise e

        try:
            depth_image = bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except cv_bridge.CvBridgeError as e:
            print(e)
            raise e

        if depth_msg.encoding == "16UC1":
            depth_image = depth_image / 1000

        # An estimate of the current camera pose, which is optionally used to
        # inform the place recognition, keypoint detection, keypoint
        # descriptor, and pose recovery methods.
        hint_pose = get_tf_as_pose(
            self.tf_buffer, self.fixed_frame, self.hint_body_frame, img_msg.header.stamp
        )

        if hint_pose is None:
            print("Cannot get current pose, skipping frame!")
            return

        spark_image = ob.SparkImage(rgb=color_image, depth=depth_image)
        image_uuid, loop_closures = self.process_new_frame(
            spark_image, img_msg.header.stamp.to_nsec(), hint_pose
        )

        with self.image_pose_lock:
            self.images_to_pose[image_uuid] = hint_pose

        if loop_closures is None:
            return

        pose_cov_mat = self.build_pose_cov_mat()
        pg = PoseGraph()
        pg.header.stamp = rospy.Time.now()
        for lc in loop_closures:
            if self.show_plots:
                with self.image_pose_lock:
                    query_pos, match_pos = get_query_and_est_match_position(
                        lc, self.images_to_pose, self.body_T_cam.as_matrix()
                    )
                    true_match_pos = self.images_to_pose[lc.to_image_uuid].position
                if not lc.is_metric:
                    plot_lc(query_pos, match_pos, "y")
                elif np.linalg.norm(true_match_pos - match_pos) < 1:
                    plot_lc(query_pos, match_pos, "b")
                else:
                    plot_lc(query_pos, match_pos, "r")

            if not lc.is_metric:
                rospy.logwarn("Skipping non-metric loop closure.")
                continue

            from_time_ns, to_time_ns = self.vlc_server.get_lc_times(lc.metadata.lc_uuid)

            lc_edge = build_robot_lc_message(
                from_time_ns,
                to_time_ns,
                self.robot_id,
                lc.f_T_t,
                pose_cov_mat,
                self.body_T_cam.as_matrix(),
            )

            pg.edges.append(lc_edge)
        self.loop_closure_delayed_queue.append(
            (rospy.Time.now().to_sec() + self.lc_send_delay_s, pg)
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
            with self.image_pose_lock:
                update_plot(
                    self.position_line, self.position_points, self.images_to_pose
                )

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
