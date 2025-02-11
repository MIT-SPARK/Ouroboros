#!/usr/bin/env python3
import os
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from spark_dataset_interfaces.rosbag_dataloader import RosbagDataLoader

import ouroboros as ob
from ouroboros.config import Config, config_field
from ouroboros.utils.plotting_utils import plt_fast_pause

# Must be False in i3 if you want to save consistent-sized overhead plots for animation
LIVE_OVERHEAD_PLOT = True


@dataclass
class VlcDriverConfig(Config):
    camera_config: ob.PinholeCamera = config_field("camera")
    server_config: ob.VlcServerConfig = config_field("vlc_server", default="vlc_server")

    @classmethod
    def load(cls, path: str):
        return ob.config.Config.load(VlcDriverConfig, path)


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
    if LIVE_OVERHEAD_PLOT:
        plt.draw_all()
        plt_fast_pause(0.05)


def get_query_and_est_match_position(lc, image_to_pose):
    query_pose = image_to_pose[lc.from_image_uuid]
    world_T_query = query_pose.as_matrix()

    match_T_query = lc.f_T_t

    world_T_match = world_T_query @ ob.invert_pose(match_T_query)

    query_position = query_pose.position
    est_match_position = world_T_match[:3, 3]
    return query_position, est_match_position


def plot_lc(query_position, match_position, color):
    plt.plot(
        [query_position[0], match_position[0]],
        [query_position[1], match_position[1]],
        color=color,
    )


def plot_lc_est_pose(lc, image_to_pose):
    query_pose = image_to_pose[lc.from_image_uuid]
    world_T_query = query_pose.as_matrix()

    match_T_query = lc.f_T_t

    world_T_match = world_T_query @ ob.invert_pose(match_T_query)

    plt.plot(
        [world_T_match[0, 3], world_T_query[0, 3]],
        [world_T_match[1, 3], world_T_query[1, 3]],
        color="r",
    )


def plot_lc_gt_pose(lc, image_to_pose):
    query_pos = image_to_pose[lc.from_image_uuid].position
    match_pos = image_to_pose[lc.to_image_uuid].position
    plt.plot(
        [match_pos[0], query_pos[0]],
        [match_pos[1], query_pos[1]],
        color="r",
    )


plugins = ob.discover_plugins()
print("Discovered Plugins: ")
print(plugins)

data_path = "/home/yunchang/Downloads/uhumans2/office_multi/uHumans2_office_s1_00h.bag"
rgb_topic = "/tesse/left_cam/rgb/image_raw"
rgb_info_topic = "/tesse/left_cam/camera_info"
depth_topic = "/tesse/depth_cam/mono/image_raw"
body_frame = "base_link_gt"
body_frame = "base_link_gt"
map_frame = "world"

loader = RosbagDataLoader(
    data_path,
    rgb_topic,
    rgb_info_topic,
    body_frame=body_frame,
    map_frame=map_frame,
    depth_topic=depth_topic,
)

vlc_frame_period_s = 0.5

images_to_pose = {}
last_vlc_frame_time = None

driver_config = VlcDriverConfig.load("config/vlc_driver_config.yaml")

if not os.path.exists("output"):
    os.mkdir("output")
log_path = os.path.join("output", datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
os.mkdir(log_path)
overhead_plot_dir = os.path.join(log_path, "overhead_plots")
os.mkdir(overhead_plot_dir)

vlc_server = ob.VlcServer(
    driver_config.server_config,
    robot_id=0,
    log_path=log_path,
)


session_id = vlc_server.register_camera(0, driver_config.camera_config, datetime.now())

plt.ion()
plt.figure(figsize=(16, 9))
if LIVE_OVERHEAD_PLOT:
    plt.show()
position_line = plt.plot([], [], color="g")[0]
position_points = plt.scatter([], [], color="g")

uid_to_pose = {}
full_poses = []
with loader:
    for idx, data in enumerate(loader):
        image = data.color
        depth = data.depth
        pose = data.pose
        full_poses.append(pose.matrix())
        time = data.timestamp

        if not (
            last_vlc_frame_time is None
            or time - last_vlc_frame_time > vlc_frame_period_s * 1e9
        ):
            continue
        last_vlc_frame_time = time

        pose_flat = pose.flatten()
        camera_pose = ob.VlcPose(time, pose_flat[:3], pose_flat[3:])
        if camera_pose is None:
            print("Cannot get current pose, skipping frame!")
            continue

        spark_image = ob.SparkImage(rgb=image, depth=depth)
        image_uuid, loop_closures = vlc_server.add_and_query_frame(
            session_id, spark_image, time, pose_hint=camera_pose
        )
        images_to_pose[image_uuid] = camera_pose

        update_plot(position_line, position_points, images_to_pose)
        if loop_closures is not None:
            for lc in loop_closures:
                query_pos, match_pos = get_query_and_est_match_position(
                    lc, images_to_pose
                )
                true_match_pos = images_to_pose[lc.to_image_uuid].position
                if np.linalg.norm(true_match_pos - match_pos) < 1:
                    plot_lc(query_pos, match_pos, "b")
                else:
                    plot_lc(query_pos, match_pos, "r")
        plt.savefig(os.path.join(overhead_plot_dir, f"overhead_{time}.png"))
