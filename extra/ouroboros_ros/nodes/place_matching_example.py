import rospy

from dynamic_reconfigure.server import Server
from sensor_msgs.msg import Image
import cv2
import cv_bridge
import numpy as np

from ouroboros import VlcDb, SparkImage
from ouroboros_salad import get_salad_model
from ouroboros_ros.cfg import PlaceDescriptorDebuggingConfig


def display_image_pair(left, right):
    r = left.shape[0]
    c = left.shape[1] * 2 + 10
    img_out = np.zeros((r, c, 3))
    img_out[: left.shape[0], : left.shape[1], :] = left
    if right is not None:
        img_out[:, left.shape[1] + 10 :, :] = right

    cv2.imshow("matches", img_out.astype(np.uint8))
    cv2.waitKey(1)


class PlaceDescriptorExample:
    def __init__(self):
        self.lc_lockout = 10
        self.place_match_threshold = 0.5
        self.reconfigure_server = Server(
            PlaceDescriptorDebuggingConfig, self.dynamic_reconfigure_cb
        )

        self.embedding_model = get_salad_model()

        # 1. initialize vlc db
        self.vlc_db = VlcDb(8448)
        robot_id = 0
        self.session_id = self.vlc_db.add_session(robot_id)

        # 2. subscribe to images
        rospy.Subscriber("~image_in", Image, self.image_cb)

    def dynamic_reconfigure_cb(self, config, level):
        self.lc_lockout = config["lc_lockout"]
        self.place_match_threshold = config["place_match_threshold"]
        return config

    def image_cb(self, msg):
        # call embedding function and insert to db, then display match if it's above threshold
        bridge = cv_bridge.CvBridge()
        try:
            image = bridge.imgmsg_to_cv2(msg, "bgr8")
        except cv_bridge.CvBridgeError as e:
            print(e)
            raise e

        uid = self.vlc_db.add_image(
            self.session_id, msg.header.stamp.to_nsec(), SparkImage(rgb=image)
        )
        current_image = self.vlc_db.get_image(uid)
        embedding = self.embedding_model(image)

        image_matches, similarities = self.vlc_db.query_embeddings_max_time(
            embedding, 1, msg.header.stamp.to_nsec() - self.lc_lockout
        )

        self.vlc_db.update_embedding(uid, embedding)

        if similarities[0] < self.place_match_threshold:
            image_match = None
        else:
            image_match = image_matches[0]

        display_image_pair(current_image, image_match)


if __name__ == "__main__":
    rospy.init_node("place_matching_example")
    node = PlaceDescriptorExample()
    rospy.spin()
