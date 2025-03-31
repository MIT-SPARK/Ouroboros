from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from spark_config import Config, register_config

import ouroboros as ob


class GtDescriptorModel:
    def __init__(self, config: GtDescriptorModelConfig):
        pass

    def infer(self, image: ob.SparkImage, keypoints: np.ndarray, pose_hint: ob.VlcPose):
        if pose_hint is None:
            raise Exception("GtDescriptorModel requires setting pose_hint")
        return np.array([pose_hint.to_descriptor()])


@register_config("descriptor_model", name="ground_truth", constructor=GtDescriptorModel)
@dataclass
class GtDescriptorModelConfig(Config):
    pass
