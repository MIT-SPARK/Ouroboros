from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from salad_example import get_salad_model
from scipy.spatial.transform import Rotation as R
from spark_dataset_interfaces.rosbag_dataloader import RosbagDataLoader

import ouroboros as ob
from ouroboros.vlc_db.gt_lc_utils import (
    VlcPose,
    recover_pose,
)


def plot_heading(positions, rpy):
    for p, angles in zip(positions, rpy):
        y = angles[2] + np.pi / 2
        plt.plot([p[0], p[0] + np.cos(y)], [p[1], p[1] + np.sin(y)], color="k")


embedding_model = get_salad_model()


lc_lockout = 5  # minimal time between two frames in loop closure
gt_lc_max_dist = 5

# If s(a,b) is the similarity between image embeddings a and b, if s(a,b) < place_recognition_threshold are not considered putative loop closures
place_recognition_threshold = 0.55

# Load Data
data_path = "/home/aaron/lxc_datashare/uHumans2_apartment_s1_00h.bag"
rgb_topic = "/tesse/left_cam/rgb/image_raw"
rgb_info_topic = "/tesse/left_cam/camera_info"
body_frame = "base_link_gt"
body_frame = "base_link_gt"
map_frame = "world"

loader = RosbagDataLoader(
    data_path, rgb_topic, rgb_info_topic, body_frame=body_frame, map_frame=map_frame
)

images = None  # numpy array
poses = None  # 7d vector

vlc_db = ob.VlcDb(8448)
robot_id = 0
session_id = vlc_db.add_session(robot_id)

uid_to_pose = {}
full_poses = []

### Batch LCD

# Place embeddings
print("Loading...")
with loader:
    print("Loaded!")
    for idx, data in enumerate(loader):
        image = data.color
        pose = data.pose
        full_poses.append(pose.matrix())

        if not idx % 2 == 0:
            continue
        uid = vlc_db.add_image(session_id, datetime.now(), ob.SparkImage(rgb=image))
        embedding = embedding_model(image)
        # embedding = VlcPose(
        #    time_ns=data.timestamp,
        #    position=pose.translation,
        #    rotation=pose.rotation.as_quat(),
        # ).to_descriptor()  # Will go away when we use real model
        vlc_db.update_embedding(uid, embedding)

        # To check our estimate vs. GT later
        uid_to_pose[uid] = pose.matrix()


# Query for closest matches
query_embeddings = np.array([image.embedding for image in vlc_db.iterate_images()])

# query_fn = functools.partial(
#    compute_descriptor_similarity, lc_lockout * 1e9, gt_lc_max_dist
# )

# matches, similarities = vlc_db.query_embeddings(
#    np.array(query_embeddings), 1, similarity_metric=query_fn
# )


matches, similarities = vlc_db.query_embeddings(
    query_embeddings, -1, similarity_metric="ip"
)


# TODO: move this to db query or utility function
# Ignore matches that are too close temporally or too far in descriptor similarity
putative_loop_closures = []
for key, matches_for_query, similarities_for_query in zip(
    vlc_db.get_image_keys(), matches, similarities
):
    ts = vlc_db.get_image(key).metadata.epoch_ns
    match_uuid = None
    for match_image, similarity in zip(matches_for_query, similarities_for_query):
        match_ts = match_image.metadata.epoch_ns
        if abs(match_ts - ts) > lc_lockout * 1e9 and ts > match_ts:
            match_uuid = match_image.metadata.image_uuid
            break

        if similarity < place_recognition_threshold:
            break

    if match_uuid is None:
        continue

    putative_loop_closures.append((key, match_uuid))

# Recover poses from matching loop closures
for key_from, key_to in putative_loop_closures:
    img_from = vlc_db.get_image(key_from)
    if img_from.keypoints is None or img_from.descriptors is None:
        # TODO: replace with real keypoints and descriptors
        # keypoints = generate_keypoints(img_from.image)
        # descriptors = generate_descriptors(img_from.image)
        keypoints = np.zeros([1, 2])
        pose = uid_to_pose[key_from]
        desc = VlcPose(
            time_ns=img_from.metadata.epoch_ns,
            position=pose[:3, 3],
            rotation=R.from_matrix(pose[:3, :3]).as_quat(),
        ).to_descriptor()
        descriptors = [desc]
        # descriptors = [img_from.embedding]

        vlc_db.update_keypoints(key_from, keypoints, descriptors)
        img_from = vlc_db.get_image(key_from)

    img_to = vlc_db.get_image(key_to)
    if img_to.keypoints is None or img_to.descriptors is None:
        # TODO: replace with real keypoints and descriptors
        # keypoints = generate_keypoints(img_to.image)
        # descriptors = generate_descriptors(img_to.image)
        keypoints = np.zeros([1, 2])
        pose = uid_to_pose[key_to]
        desc = VlcPose(
            time_ns=img_to.metadata.epoch_ns,
            position=pose[:3, 3],
            rotation=R.from_matrix(pose[:3, :3]).as_quat(),
        ).to_descriptor()
        descriptors = [desc]
        # descriptors = [img_to.embedding]

        vlc_db.update_keypoints(key_to, keypoints, descriptors)
        img_to = vlc_db.get_image(key_to)

    # relative_pose = recover_pose(img_from, img_to)
    relative_pose = recover_pose(img_from.descriptors, img_to.descriptors)
    loop_closure = ob.SparkLoopClosure(
        from_image_uuid=key_from,
        to_image_uuid=key_to,
        f_T_t=relative_pose,
        quality=1,
    )
    vlc_db.add_lc(loop_closure, session_id, creation_time=datetime.now())


# Plot estimated loop closures and ground truth trajectory
positions = np.array([p[:3, 3] for p in full_poses])
rpy = np.array([R.from_matrix(p[:3, :3]).as_euler("xyz") for p in full_poses])

plt.ion()
plt.plot(positions[:, 0], positions[:, 1], color="k")
plt.scatter(positions[:, 0], positions[:, 1], color="k")
plot_heading(positions, rpy)


for lc in vlc_db.iterate_lcs():
    w_T_from = uid_to_pose[lc.from_image_uuid]
    w_T_to = uid_to_pose[lc.to_image_uuid]

    inferred_pose_to = w_T_from @ lc.f_T_t

    # Plot point for the ground truth poses involved in the loop closure
    plt.scatter(
        [w_T_from[0, 3], w_T_to[0, 3]],
        [w_T_from[1, 3], w_T_to[1, 3]],
        color="r",
    )

    plt.plot(
        [w_T_from[0, 3], inferred_pose_to[0, 3]],
        [w_T_from[1, 3], inferred_pose_to[1, 3]],
        color="r",
    )

plt.show()
