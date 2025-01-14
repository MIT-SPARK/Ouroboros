from datetime import datetime

import cv2

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from spark_dataset_interfaces.rosbag_dataloader import RosbagDataLoader

from vlc_db.vlc_db import VlcDb
from vlc_db.spark_image import SparkImage

from salad_example import get_salad_model


embedding_model = get_salad_model()


def combine_images(left, right):
    r = left.shape[0]
    c = left.shape[1] * 2 + 10
    img_out = np.zeros((r, c, 3))
    img_out[: left.shape[0], : left.shape[1], :] = left
    img_out[:, left.shape[1] + 10 :, :] = right
    return img_out


lc_lockout = 10  # minimal time between two frames in loop closure

# If s(a,b) is the similarity between image embeddings a and b, if s(a,b) < place_recognition_threshold are not considered putative loop closures
place_recognition_threshold = 0.96

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

vlc_db = VlcDb(8448)
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

        uid = vlc_db.add_image(session_id, datetime.now(), SparkImage(rgb=image))
        embedding = embedding_model(image)
        vlc_db.update_embedding(uid, embedding)

        # To check our estimate vs. GT later
        uid_to_pose[uid] = pose.matrix()


# Query for closest matches
query_embeddings = np.array([image.embedding for image in vlc_db.iterate_images()])

matches, similarities = vlc_db.query_embeddings(
    query_embeddings, -1, similarity_metric="ip"
)


# TODO: move this to db query or utility function
# Ignore matches that are too close temporally or too far in descriptor similarity
putative_loop_closures = []
for key, matches_for_query, similarities_for_query in tqdm(
    zip(vlc_db.get_image_keys(), matches, similarities), total=len(matches)
):
    ts = vlc_db.get_image(key).metadata.epoch_ns
    match_uuid = None
    for match_image, similarity in zip(matches_for_query, similarities_for_query):
        match_ts = match_image.metadata.epoch_ns
        if abs(match_ts - ts) > lc_lockout * 1e9:
            match_uuid = match_image.metadata.image_uuid
            break

        if similarity < place_recognition_threshold:
            break

    if match_uuid is None:
        putative_loop_closures.append((key, None))
        continue

    putative_loop_closures.append((key, match_uuid))

for key, match_key in tqdm(putative_loop_closures):
    left = vlc_db.get_image(key).image.rgb
    if match_key is None:
        right = np.zeros(left.shape)
    else:
        right = vlc_db.get_image(match_key).image.rgb
    img = combine_images(left, right)
    cv2.imshow("matches", img.astype(np.uint8))
    cv2.waitKey(30)
