import pathlib
import matplotlib.pyplot as plt
import numpy as np

import imageio.v3 as iio

import ouroboros as ob
from ouroboros_keypoints.superpoint_interface import get_superpoint_model


def resource_dir():
    return pathlib.Path(__file__).absolute().parent.parent / "resources"


if __name__ == "__main__":
    model = get_superpoint_model()

    # img_d = iio.imread(resource_dir() / "arch.jpg")
    img_d = iio.imread(resource_dir() / "right_img_1.png")
    img_d = np.expand_dims(img_d, axis=-1)

    simg = ob.SparkImage(rgb=img_d)

    keypoints, descriptors = model.infer(simg)

    kp = keypoints
    desc = descriptors
    plt.imshow(img_d)
    x, y = kp.T
    plt.scatter(x, y)
    plt.show()
