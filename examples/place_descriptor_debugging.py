from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from salad_example import get_salad_model
from spark_dataset_interfaces.rosbag_dataloader import RosbagDataLoader
from tqdm import tqdm

import ouroboros as ob

embedding_model = get_salad_model()


def combine_images(left, right):
    r = left.shape[0]
    c = left.shape[1] * 2 + 10
    img_out = np.zeros((r, c, 3))
    img_out[: left.shape[0], : left.shape[1], :] = left
    img_out[:, left.shape[1] + 10 :, :] = right
    return img_out


lc_lockout = 5  # minimal time between two frames in loop closure

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

        uid = vlc_db.add_image(session_id, datetime.now(), ob.SparkImage(rgb=image))
        embedding = embedding_model(image)
        vlc_db.update_embedding(uid, embedding)

        # To check our estimate vs. GT later
        uid_to_pose[uid] = pose.matrix()


times = []
closest = []
for img in vlc_db.iterate_images():
    ts = img.metadata.epoch_ns
    matches = vlc_db.query_embeddings_filter(
        img.embedding, 1, lambda m, s: abs(m.metadata.epoch_ns - ts) > lc_lockout * 1e9
    )
    times.append(ts / 1e9)
    closest.append(matches[0][0])
    if matches[0][0] > place_recognition_threshold:
        right = matches[0][1].image.rgb
    else:
        right = None
    img = combine_images(img.image.rgb, right)
    cv2.imshow("matches", img.astype(np.uint8))
    cv2.waitKey(30)

t = [ti - times[0] for ti in times]
plt.ion()
plt.plot(t, closest)
plt.xlabel("Time (s)")
plt.ylabel("Best Descriptor Similarity")
plt.title("Closest Descriptor s.t. Time Constraint")
