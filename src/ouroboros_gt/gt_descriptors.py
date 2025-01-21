import numpy as np

import ouroboros as ob


def get_gt_descriptor_model():
    return GtDescriptorModel()


class GtDescriptorModel:
    def __init__(self):
        pass

    def infer(self, image: ob.SparkImage, keypoints: np.ndarray, pose_hint: ob.VlcPose):
        if pose_hint is None:
            raise Exception("GtDescriptorModel requires setting pose_hint")
        return np.array([pose_hint.to_descriptor()])
