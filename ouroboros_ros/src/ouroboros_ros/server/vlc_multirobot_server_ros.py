#!/usr/bin/env python3
from dataclasses import dataclass

import rclpy
import spark_config as sc
from pose_graph_tools_msgs.msg import PoseGraph
from sensor_msgs.msg import CameraInfo

from server.conversions import (
    vlc_image_from_msg,
    vlc_image_to_msg,
    vlc_pose_from_msg,
    vlc_pose_to_msg,
)
from server.utils import build_lc_message, parse_camera_info
from server.vlc_server_ros import VlcServerRos

import ouroboros as ob
from ouroboros_msgs.msg import VlcImageMsg, VlcInfoMsg
from ouroboros_msgs.srv import VlcKeypointQuery


@dataclass
class RobotInfo:
    session_id: str = None
    camera_config: sc.Config = None
    body_T_cam: ob.VlcPose = None

    @classmethod
    def from_info_msg(cls, resp: VlcInfoMsg):
        camera_config = parse_camera_info(resp.camera_info)
        body_T_cam = vlc_pose_from_msg(resp.body_tf_cam)
        return cls(
            session_id=None,
            camera_config=camera_config,
            body_T_cam=body_T_cam,
        )


class VlcMultirobotServerRos(VlcServerRos):
    def __init__(self, node_name="vlc_server_node"):
        super().__init__(node_name)

        self.track_new_uuids = []

        # Spin up robot info server
        self.vlc_clients = self.declare_parameter("clients", [0]).value

        # Publish embeddings
        self.embedding_timer = self.create_timer(5.0, self.embedding_timer_callback)
        self.embedding_publisher = self.create_publisher(
            VlcImageMsg, "vlc_embedding", 10
        )

        # Keypoint request server
        self.keypoint_server = self.create_service(
            VlcKeypointQuery,
            "vlc_keypoints_request",
            self.process_keypoints_request,
        )

        self.embedding_subscribers = []
        self.keypoint_clients = {}

        # Publish info as discovery
        self.info_timer = self.create_timer(5.0, self.info_timer_callback)
        self.info_publisher = self.create_publisher(VlcInfoMsg, "/vlc_info", 10)

        # Subscribe to other robots' infos as discovery
        self.info_subscriber = self.create_subscription(
            VlcInfoMsg, "/vlc_info", self.vlc_info_callback, 10
        )

        self.robot_infos = {}
        self.uuid_map = {}
        self.session_robot_map = {}

        self.get_logger().info(
            f"Initialized VLC server for Robot ID {self.robot_id} with clients {self.vlc_clients}"
        )

    def info_timer_callback(self):
        # NOTE(Yun) maybe should terminate this? But there's a case where a new server shows up
        info_msg = VlcInfoMsg()
        info_msg.robot_id = self.robot_id
        camera_info = CameraInfo()
        camera_info.k = self.camera_config.K.flatten()
        info_msg.camera_info = camera_info
        info_msg.body_tf_cam = vlc_pose_to_msg(self.body_T_cam)

        info_msg.embedding_topic = self.resolve_topic_name("vlc_embedding")
        info_msg.keypoints_service = self.resolve_topic_name("vlc_keypoints_request")
        self.info_publisher.publish(info_msg)

    def vlc_info_callback(self, info_msg):
        # Note(Yun) alternatively define server(s) in msg
        if self.vlc_clients is None or info_msg.robot_id not in self.vlc_clients:
            # Not handling this robot
            return

        if info_msg.robot_id in self.robot_infos:
            # Already initialized
            return

        self.get_logger().info(
            f"Established connection with client with robot id {info_msg.robot_id}"
        )
        self.robot_infos[info_msg.robot_id] = RobotInfo.from_info_msg(info_msg)
        # Assign session_id
        self.robot_infos[
            info_msg.robot_id
        ].session_id = self.vlc_server.register_camera(
            info_msg.robot_id,
            self.robot_infos[info_msg.robot_id].camera_config,
            self.get_clock().now().nanoseconds,
        )
        self.session_robot_map[self.robot_infos[info_msg.robot_id].session_id] = (
            info_msg.robot_id
        )
        # Subscribe to embeddings
        self.embedding_subscribers.append(
            self.create_subscription(
                VlcImageMsg,
                info_msg.embedding_topic,
                lambda embedding_msg: self.client_embedding_callback(
                    embedding_msg, info_msg.robot_id
                ),
                10,
            )
        )
        self.get_logger().info(
            f"Starting subscription to embedding topic: {info_msg.embedding_topic}"
        )

        # Keypoint request client
        self.keypoint_clients[info_msg.robot_id] = self.create_client(
            VlcKeypointQuery, info_msg.keypoints_service
        )
        while not self.keypoint_clients[info_msg.robot_id].wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().warning(
                f"Service {info_msg.keypoints_service} not available, trying again..."
            )
        self.get_logger().info(
            f"Starting service client to server: {info_msg.keypoints_service}"
        )

    def process_new_frame(self, image, stamp_ns, hint_pose):
        # Need a different one due to sometimes needing to request keypoints
        new_uuid = self.vlc_server.add_frame(
            self.session_id,
            image,
            stamp_ns,
            pose_hint=hint_pose,
        )
        self.track_new_uuids.append(new_uuid)

        # Find match using the embeddings.
        image_match = self.vlc_server.find_match(new_uuid, stamp_ns)

        if image_match is None:
            return new_uuid, None

        match_uuid = image_match.metadata.image_uuid

        interrobot = self.session_id != image_match.metadata.session_id
        if image_match.keypoints is None and interrobot:
            remapped_match_uuid = self.uuid_map[match_uuid]
            # Request keypoint and descriptors for match
            robot_id = self.session_robot_map[image_match.metadata.session_id]
            keypoints_request = VlcKeypointQuery.Request()
            keypoints_request.image_uuid = remapped_match_uuid
            future = self.keypoint_clients[robot_id].call_async(keypoints_request)
            rclpy.spin_until_future_complete(self, future)
            vlc_response = vlc_image_from_msg(future.result().vlc_image)
            self.vlc_server.update_keypoints_decriptors(
                match_uuid, vlc_response.keypoints, vlc_response.descriptors
            )
            self.vlc_server.update_keypoint_depths(
                match_uuid, vlc_response.keypoint_depths
            )

        elif not interrobot:
            self.vlc_server.compute_keypoints_descriptors(
                match_uuid, compute_depths=True
            )

        # Compute self keypoints and descriptors
        self.vlc_server.compute_keypoints_descriptors(new_uuid, compute_depths=True)

        lc_list = self.vlc_server.compute_loop_closure_pose(
            self.session_id, new_uuid, image_match.metadata.image_uuid, stamp_ns
        )

        return new_uuid, lc_list

    def embedding_timer_callback(self):
        while len(self.track_new_uuids) > 0:
            new_uuid = self.track_new_uuids.pop(0)
            vlc_img_msg = vlc_image_to_msg(
                self.vlc_server.get_image(new_uuid), convert_image=False
            )
            self.embedding_publisher.publish(vlc_img_msg)

    def client_embedding_callback(self, vlc_img_msg, robot_id):
        # Add image
        vlc_image = vlc_image_from_msg(vlc_img_msg)
        vlc_stamp = vlc_img_msg.metadata.epoch_ns
        new_uuid = self.vlc_server.add_embedding_no_image(
            self.robot_infos[robot_id].session_id,
            vlc_image.embedding,
            vlc_stamp,
            pose_hint=vlc_image.pose_hint,
        )
        self.uuid_map[new_uuid] = vlc_image.metadata.image_uuid

        # Find match
        matched_img = self.vlc_server.find_match(
            new_uuid, vlc_stamp, search_sessions=[self.session_id]
        )
        if matched_img is None:
            return

        self.get_logger().warning(
            f"Found match between robots {robot_id} and {self.robot_id}"
        )

        # Request keypoints / descriptors
        keypoints_request = VlcKeypointQuery.Request()
        keypoints_request.image_uuid = vlc_image.metadata.image_uuid
        future = self.keypoint_clients[robot_id].call_async(keypoints_request)
        rclpy.spin_until_future_complete(self, future)
        vlc_response = vlc_image_from_msg(future.result().vlc_image)
        self.vlc_server.update_keypoints_decriptors(
            new_uuid, vlc_response.keypoints, vlc_response.descriptors
        )
        self.vlc_server.update_keypoint_depths(new_uuid, vlc_response.keypoint_depths)

        # Detect loop closures
        self.vlc_server.compute_keypoints_descriptors(
            matched_img.metadata.image_uuid, compute_depths=True
        )
        lc_list = self.vlc_server.compute_loop_closure_pose(
            self.robot_infos[robot_id].session_id,
            new_uuid,
            matched_img.metadata.image_uuid,
            vlc_stamp,
        )

        if lc_list is None:
            return

        self.get_logger().warning(
            f"Found lc between robots {robot_id} and {self.robot_id}"
        )

        pose_cov_mat = self.build_pose_cov_mat()
        pg = PoseGraph()
        pg.header.stamp = self.get_clock().now().to_msg()
        for lc in lc_list:
            if not lc.is_metric:
                self.get_logger().warning("Skipping non-metric loop closure.")
                continue

            from_time_ns, to_time_ns = self.vlc_server.get_lc_times(lc.metadata.lc_uuid)

            lc_edge = build_lc_message(
                from_time_ns,
                to_time_ns,
                robot_id,
                self.robot_id,
                lc.f_T_t,
                pose_cov_mat,
                self.robot_infos[robot_id].body_T_cam.as_matrix(),
                self.body_T_cam.as_matrix(),
                self.get_clock().now(),
            )

            pg.edges.append(lc_edge)
        self.loop_closure_delayed_queue.append(
            (self.get_clock().now().nanoseconds * 1.0e-9 + self.lc_send_delay_s, pg)
        )

    def process_keypoints_request(self, request, response):
        if not self.vlc_server.has_image(request.image_uuid):
            self.get_logger().warning(f"Image ID {request.image_uuid} not found!")
            return response

        self.vlc_server.compute_keypoints_descriptors(
            request.image_uuid, compute_depths=True
        )
        vlc_img = self.vlc_server.get_image(request.image_uuid)
        response.vlc_image = vlc_image_to_msg(
            vlc_img, convert_image=False, convert_embedding=False
        )
        return response
