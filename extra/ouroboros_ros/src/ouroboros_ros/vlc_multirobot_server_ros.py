#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional

import rospy
from ouroboros_msgs.msg import VlcImageMsg
from ouroboros_msgs.srv import VlcKeypointQuery, VlcKeypointQueryResponse
from pose_graph_tools_msgs.msg import PoseGraph

import ouroboros as ob
from ouroboros_ros.conversions import (
    vlc_image_from_msg,
    vlc_image_to_msg,
    vlc_pose_from_msg,
)
from ouroboros_ros.utils import build_lc_message, get_tf_as_pose
from ouroboros_ros.vlc_server_ros import VlcServerRos


@dataclass
class RobotInfo:
    ns: str = ""
    session_id: Optional[str] = None
    camera_config: Optional[ob.config.Config]


class VlcMultirobotServerRos(VlcServerRos):
    def __init__(self):
        super().__init__()

        self.servers = rospy.get_param("~servers")
        self.clients = rospy.get_param("~clients")
        self.robot_infos = self.get_robot_infos(self.servers + self.clients)
        for robot_id, robot_info in self.robot_infos:
            robot_info.session_id = self.vlc_server.register_camera(
                robot_id, robot_info.camera_config, rospy.Time.now().to_nsec()
            )

        self.embedding_publishers = []
        self.keypoint_servers = []
        self.embedding_subscribers = []
        self.keypoint_clients = {}
        for server_id in self.servers:
            # Publish embeddings
            self.embedding_publishers.append(
                rospy.Publisher(
                    f"robot_{robot_id}/embedding", VlcImageMsg, queue_size=10
                )
            )
            # Keypoint request server
            self.keypoint_servers.append(
                rospy.Service(
                    f"robot_{robot_id}/keypoints_request",
                    VlcKeypointQuery,
                    self.process_keypoints_request,
                )
            )

        for client_id in self.clients:
            # Subscribe to embeddings
            self.embedding_subscribers.append(
                rospy.Subscriber(
                    f"robot_{robot_id}/embedding",
                    VlcImageMsg,
                    self.client_embedding_callback,
                    callback_args=client_id,
                )
            )
            # Keypoint request client
            self.keypoint_clients[client_id] = rospy.ServiceProxy(
                f"robot_{robot_id}/keypoints_request", VlcKeypointQuery
            )

    def client_embedding_callback(self, vlc_img_msg, robot_id):
        # Add image
        vlc_image = vlc_image_from_msg(vlc_img_msg)
        new_uuid = self.vlc_server.add_frame(
            self.robot_infos[robot_id].session_id,
            vlc_image.image,
            vlc_image.header.stamp.to_nsec(),
            pose_hint=vlc_image.pose_hint,
        )
        self.uuid_map[robot_id][new_uuid] = vlc_image.metadata.image_uuid

        # Find match
        matched_img = self.vlc_server.find_match(
            new_uuid, vlc_image.header.stamp.to_nsec()
        )
        if matched_img is None:
            return

        # Request
        response = self.keypoint_clients[robot_id](vlc_image.metadata.image_uuid)
        updated_vlc_img = vlc_image_from_msg(response.vlc_image)
        other_body_T_cam = vlc_pose_from_msg(response.body_T_cam)
        self.vlc_server.update_keypoints(
            updated_vlc_img.keypoints, updated_vlc_img.descriptors
        )

        # Detect loop closures
        self.compute_keypoints_descriptors(matched_img.metadata.uuid)
        lc_list = self.compute_loop_closure_pose(
            self.robot_infos[robot_id].session_id,
            new_uuid,
            matched_img.metadata.image_uuid,
            vlc_image.header.stamp.to_nsec(),
        )

        if lc_list is None:
            return

        body_T_cam = get_tf_as_pose(
            self.tf_buffer, self.body_frame, self.camera_frame, vlc_img_msg.header.stamp
        )

        pose_cov_mat = self.build_pose_cov_mat()
        pg = PoseGraph()
        pg.header.stamp = rospy.Time.now()
        for lc in lc_list:
            if not lc.is_metric:
                rospy.logwarn("Skipping non-metric loop closure.")
                continue

            from_time_ns, to_time_ns = self.vlc_server.get_lc_times(lc.metadata.lc_uuid)

            lc_edge = build_lc_message(
                from_time_ns,
                to_time_ns,
                robot_id,
                self.robot_id,
                lc.f_T_t,
                pose_cov_mat,
                other_body_T_cam,
                body_T_cam,
            )

            pg.edges.append(lc_edge)
        self.loop_closure_delayed_queue.append(
            (rospy.Time.now().to_sec() + self.lc_send_delay_s, pg)
        )

    def process_keypoints_request(self, request):
        vlc_img = self.vlc_server.get_image(request.image_uuid)
        if vlc_img is None:
            rospy.logwarn(f"Image ID {request.id} not found!")
            return VlcKeypointQueryResponse()
        return VlcKeypointQueryResponse(vlc_image=vlc_image_to_msg(vlc_img))

    def run(self):
        while not rospy.is_shutdown():
            self.step()
            self.loop_rate.sleep()

    def step(self):
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
