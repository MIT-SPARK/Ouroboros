from datetime import datetime

import imageio.v2 as imageio
import numpy as np

from vlc_db.vlc_db import VlcDb
from vlc_db.spark_loop_closure import SparkLoopClosure
from vlc_db.spark_image import SparkImage

lc_lockout = 30  # minimal time between two frames in loop closure

# If d(a,b) is the distance between image embeddings a and b, if d(a,b) > place_recognition_threshold are not considered putative loop closures
place_recognition_threshold = 1

# Load Data
image_dir = ""
pose_dir = ""

images = None  # numpy array
poses = None  # 7d vector

vlc_db = VlcDb(128)
robot_id = 0
session_id = vlc_db.add_session(robot_id)

uid_to_pose = {}

### Batch LCD

# Place embeddings
for image, pose in zip(images, poses):
    uid = vlc_db.add_image(session_id, datetime.now(), SparkImage(rgb=image))
    embedding = model(image)
    vlc_db.update_embedding(uid, embedding)

    # To check our estimate vs. GT later
    uid_to_pose[uid] = pose


# Query for closest matches
query_embeddings = np.array([image.embedding for image in vlc_db.iterate_images()])
matches, distances = vlc_db.query_embeddings(
    query_embeddings, -1
)  # TODO: specify which distance metric we want to use

# Ignore matches that are too close temporally or too far in descriptor distance
putative_loop_closures = []
for key, matches, distances in zip(vlc_db.get_image_keys(), matches, distances):
    ts = vlc_db.get_image(key).metadata.epoch_ns
    for match_image, distance in zip(matches, distances):
        match_ts = match_image.metadata.epoch_ns
        if abs(match_ts - ts) > lc_lockout and ts > match_ts:
            match_uuid = match_image.metadata.image_uuid
            break

        if distance > place_recognition_threshold:
            match_uuid = None
            break

    if match_uuid is None:
        continue

    putative_loop_closures.append((key, match_uuid))

# Recover poses from matching loop closures
for key_from, key_to in putative_loop_closures:
    img_from = vlc_db.get_image(key_from)
    if img_from.keypoints is None or img_from.descriptors is None:
        keypoints = generate_keypoints(img_from.image)
        descriptors = generate_descriptors(img_from.image)
        vlc_db.update_keypoints(key_from, keypoints, descriptors)
        img_from = vlc_db.get_image(key_from)

    img_to = vlc_db.get_image(key_to)
    if img_to.keypoints is None or img_to.descriptors is None:
        keypoints = generate_keypoints(img_to.image)
        descriptors = generate_descriptors(img_to.image)
        vlc_db.update_keypoints(key_to, keypoints, descriptors)
        img_to = vlc_db.get_image(key_to)

    relative_pose = recover_pose(img_from, img_to)

# uuids = vlc_db.get_image_keys()
# positions = np.array([pose.position for pose in poses])

# Plot estimated loop closures on ground truth trajectory
