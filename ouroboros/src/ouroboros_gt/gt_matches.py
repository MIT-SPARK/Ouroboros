from __future__ import annotations

from dataclasses import dataclass

from spark_config import Config, register_config

import ouroboros as ob


class GtMatchModel:
    def __init__(self, config: GtMatchModelConfig):
        self.returns_descriptors = True

    def infer(
        self, image0: ob.VlcImage, image1: ob.VlcImage, pose_hint: ob.VlcPose = None
    ):
        return None, None, None


@register_config("match_model", name="ground_truth", constructor=GtMatchModel)
@dataclass
class GtMatchModelConfig(Config):
    pass
