from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from spark_config import Config, register_config

import ouroboros as ob


class GtKeypointModel:
    def __init__(self, config: GtKeypointModelConfig):
        pass

    def infer(self, image: ob.SparkImage, pose_hint: ob.VlcPose):
        return np.array([[np.nan, np.nan]]), None


@register_config("keypoint_model", name="ground_truth", constructor=GtKeypointModel)
@dataclass
class GtKeypointModelConfig(Config):
    pass
