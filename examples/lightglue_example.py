import pathlib

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from lightglue.viz2d import plot_images, plot_matches

import ouroboros as ob
from ouroboros.utils.plotting_utils import display_kp_match_pair
from ouroboros_keypoints.lightglue_interface import LightglueModel
from ouroboros_keypoints.superpoint_interface import SuperPointModel


def resource_dir():
    return pathlib.Path(__file__).absolute().parent.parent / "resources"


if __name__ == "__main__":
    superpoint_model = SuperPointModel.load("config/superpoint_config.yaml")
    lightglue_model = LightglueModel.load("config/lightglue_config.yaml")

    img_l = iio.imread(resource_dir() / "left_img_1.png")
    img_l = np.expand_dims(img_l, axis=-1)
    img_r = iio.imread(resource_dir() / "right_img_1.png")
    img_r = np.expand_dims(img_r, axis=-1)

    keypoints_l, descriptors_l = superpoint_model.infer(ob.SparkImage(rgb=img_l))
    keypoints_r, descriptors_r = superpoint_model.infer(ob.SparkImage(rgb=img_r))

    vlc_image_l = ob.VlcImage(
        None, ob.SparkImage(rgb=img_l), keypoints=keypoints_l, descriptors=descriptors_l
    )
    vlc_image_r = ob.VlcImage(
        None, ob.SparkImage(rgb=img_r), keypoints=keypoints_r, descriptors=descriptors_r
    )

    m_keypoints_l, m_keypoints_r = lightglue_model.infer(vlc_image_l, vlc_image_r)

    plt.figure()
    plt.title("left")
    plt.imshow(img_l)
    x, y = m_keypoints_l.T
    plt.scatter(x, y)

    plt.figure()
    plt.title("right")
    plt.imshow(img_r)
    x, y = m_keypoints_r.T
    plt.scatter(x, y)

    plt.figure()
    plot_images([img_l, img_r], titles=["left", "right"])
    plot_matches(m_keypoints_l, m_keypoints_r)

    plt.ion()
    plt.show()

    display_kp_match_pair(vlc_image_l, vlc_image_r, m_keypoints_l, m_keypoints_r)
