#!/usr/bin/env python3
from dataclasses import dataclass

import rospy
from ouroboros_msgs.msg import VlcImageMsg
from ouroboros_msgs.srv import (
    VlcInfo,
    VlcInfoResponse,
    VlcKeypointQuery,
    VlcKeypointQueryResponse,
)
from pose_graph_tools_msgs.msg import PoseGraph
from sensor_msgs.msg import CameraInfo

import ouroboros as ob
from ouroboros_ros.conversions import (
    vlc_image_from_msg,
    vlc_image_to_msg,
    vlc_pose_from_msg,
    vlc_pose_to_msg,
)
from ouroboros_ros.utils import build_lc_message, parse_camera_info
from ouroboros_ros.vlc_server_ros import VlcServerRos


@dataclass
class RobotInfo:
    session_id: str = ""
    camera_config: ob.config.Config = None
    body_T_cam: ob.VlcPose = None

    @classmethod
    def from_service_response(cls, resp: VlcInfoResponse):
        camera_config = parse_camera_info(resp.camera_info)
        body_T_cam = vlc_pose_from_msg(resp.body_T_cam)
        return cls(
            session_id=resp.session_id,
            camera_config=camera_config,
            body_T_cam=body_T_cam,
        )


class VlcMultirobotServerRos(VlcServerRos):
    def __init__(self):
        super().__init__()

        # Spin up robot info server
        self.servers = rospy.get_param("~servers")
        self.clients = rospy.get_param("~clients")

        self.embedding_publishers = []
        self.keypoint_servers = []
        self.embedding_subscribers = []
        self.keypoint_clients = {}
        for server_id in self.servers:
            # Publish embeddings
            self.embedding_publishers.append(
                rospy.Publisher(
                    f"robot_{server_id}/embedding", VlcImageMsg, queue_size=10
                )
            )
            # Keypoint request server
            self.keypoint_servers.append(
                rospy.Service(
                    f"robot_{server_id}/keypoints_request",
                    VlcKeypointQuery,
                    self.process_keypoints_request,
                )
            )

        for client_id in self.clients:
            # Subscribe to embeddings
            self.embedding_subscribers.append(
                rospy.Subscriber(
                    f"robot_{client_id}/embedding",
                    VlcImageMsg,
                    self.client_embedding_callback,
                    callback_args=client_id,
                )
            )
            # Keypoint request client
            self.keypoint_clients[client_id] = rospy.ServiceProxy(
                f"robot_{client_id}/keypoints_request", VlcKeypointQuery
            )
        self.info_server = rospy.Service(
            f"robot_{client_id}/vlc_info", VlcInfo, self.process_info_request
        )

        self.robot_infos.append(self.get_robot_infos(self.servers + self.clients))

    def get_robot_infos(self, robot_ids, timeout=5.0):
        for robot_id in robot_ids:
            service_name = f"robot_{robot_id}/vlc_info"
            try:
                rospy.wait_for_service(service_name, timeout)
            except rospy.ROSException as e:
                rospy.logerr(
                    f"Timeout: Service {service_name} not available within {timeout} seconds. Exception: {e}"
                )
            info_client = rospy.ServiceProxy(service_name, VlcInfo)
            response = info_client()
            self.robot_infos[robot_id].from_service_response(response)

    def process_info_request(self, request):
        response = VlcInfoResponse()
        response.session_id = self.session_id
        camera_info = CameraInfo()
        camera_info.K = self.camera_config.flatten()
        response.camera_info = camera_info
        response.body_T_cam = vlc_pose_to_msg(self.body_T_cam)
        return response

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
        vlc_image = self.vlc_server.get_image(new_uuid)
        vlc_image.keypoints = updated_vlc_img.keypoints
        vlc_image.descriptors = updated_vlc_img.descriptors

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
                self.robot_infos[robot_id].body_T_cam,
                self.body_T_cam,
            )

            pg.edges.append(lc_edge)
        self.loop_closure_delayed_queue.append(
            (rospy.Time.now().to_sec() + self.lc_send_delay_s, pg)
        )

    def process_keypoints_request(self, request):
        if not self.vlc_server.has_image(request.image_uuid):
            rospy.logwarn(f"Image ID {request.id} not found!")
            return VlcKeypointQueryResponse()

        self.vlc_server.compute_keypoints_descriptors(request.image_uuid)
        vlc_img = self.vlc_server.get_image(request.image_uuid)
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
