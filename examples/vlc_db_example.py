import pathlib
from datetime import datetime

import imageio.v3 as iio
import numpy as np

import ouroboros as ob

vlc_db = ob.VlcDb(3)

robot_id = 0
session_id = vlc_db.add_session(robot_id)


def load_resource_image(image_name):
    resource_path = pathlib.Path(__file__).absolute().parent.parent / "resources"
    return iio.imread(resource_path / image_name)


def insert_image(db, image):
    uid = db.add_image(session_id, datetime.now(), ob.SparkImage(rgb=image))

    # TODO: expand the example to generate these with a real VLC pipeline
    # db.update_embedding()
    db.update_keypoints(
        uid, np.random.random((30, 2)), descriptors=np.random.random((30, 512))
    )

    return uid


img_a = load_resource_image("left_img_0.png")
img_b = load_resource_image("left_img_1.png")
img_c = load_resource_image("right_img_1.png")
img_d = load_resource_image("arch.jpg")

a_id = insert_image(vlc_db, img_a)
vlc_db.update_embedding(a_id, np.array([1, 0, 0]))

b_id = insert_image(vlc_db, img_b)
vlc_db.update_embedding(b_id, np.array([0, 1, 1]))

c_id = insert_image(vlc_db, img_c)
vlc_db.update_embedding(c_id, np.array([0, 1.1, 1.1]))

d_id = insert_image(vlc_db, img_d)
vlc_db.update_embedding(d_id, np.array([10, 2, 2]))


print("Print: vlc_db.image_table.get_image[a_id]:")
print(vlc_db.get_image(a_id))

print("Querying 0,1,1")
imgs, dists = vlc_db.query_embeddings(np.array([0, 1, 1]), 2)

computed_ts = datetime.now()
loop_closure = ob.SparkLoopClosure(
    from_image_uuid=a_id,
    to_image_uuid=b_id,
    f_T_t=np.eye(4),
    quality=1,
)

# If you want to add a loop closure where the two poses do not correspond to
# keyframe images in the database (not totally clear why you would want to do
# this), you should just insert the two times into the database as images with
# image=None

lc_uuid = vlc_db.add_lc(loop_closure, session_id, creation_time=computed_ts)
