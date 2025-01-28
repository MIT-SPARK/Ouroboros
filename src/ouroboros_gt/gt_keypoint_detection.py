import numpy as np

import ouroboros as ob


def get_gt_keypoint_model():
    return GtKeypointModel()


class GtKeypointModel:
    def __init__(self):
        pass

    def infer(self, image: ob.SparkImage, pose_hint: ob.VlcPose):
        return np.array([[np.nan, np.nan]]), None
