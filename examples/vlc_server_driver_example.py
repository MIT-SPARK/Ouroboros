#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from spark_dataset_interfaces.rosbag_dataloader import RosbagDataLoader

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


plugins = ob.discover_plugins()
print(plugins)

data_path = "/home/aaron/lxc_datashare/uHumans2_apartment_s1_00h.bag"
rgb_topic = "/tesse/left_cam/rgb/image_raw"
rgb_info_topic = "/tesse/left_cam/camera_info"
body_frame = "base_link_gt"
body_frame = "base_link_gt"
map_frame = "world"

loader = RosbagDataLoader(
    data_path, rgb_topic, rgb_info_topic, body_frame=body_frame, map_frame=map_frame
)

vlc_frame_period_s = 0.5

images_to_pose = {}
last_vlc_frame_time = None

# server_config = ob.VlcServerConfig.load("config/gt_vlc_server_config.yaml")
server_config = ob.VlcServerConfig.load("config/vlc_server_config.yaml")
print("server config: ")
print(server_config)

vlc_server = ob.VlcServer(
    server_config,
    robot_id=0,
)

plt.ion()
plt.figure()
plt.show()
position_line = plt.plot([], [], color="g")[0]
position_points = plt.scatter([], [], color="g")

uid_to_pose = {}
full_poses = []
with loader:
    for idx, data in enumerate(loader):
        image = data.color
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

        spark_image = ob.SparkImage(rgb=image)
        image_uuid, loop_closures = vlc_server.add_and_query_frame(
            spark_image, time, pose_hint=camera_pose
        )
        images_to_pose[image_uuid] = camera_pose

        if loop_closures is None:
            continue
        for lc in loop_closures:
            plot_lc(lc, images_to_pose)

        update_plot(position_line, position_points, images_to_pose)
