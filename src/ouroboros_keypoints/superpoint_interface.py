from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from lightglue import SuperPoint

import ouroboros as ob
from ouroboros.config import Config, register_config


class SuperPointModel:
    def __init__(self, config: SuperPointModelConfig):
        self.model = SuperPoint(max_num_keypoints=config.max_keypoints).eval().cuda()
        self.returns_descriptors = True

    def infer(self, image: ob.SparkImage, pose_hint: ob.VlcPose = None):
        img_float = torch.tensor(
            np.array([image.rgb.transpose() / 255.0]).astype(np.float32)
        ).cuda()
        with torch.no_grad():
            ret = self.model({"image": img_float})

        # Superpoint seems to return (y, x) keypoint coordinates, but our interface assumes keypoints in (x,y)
        return ret["keypoints"].cpu().numpy()[0, :, ::-1], ret[
            "descriptors"
        ].cpu().numpy()[0]

    @classmethod
    def load(cls, path: str):
        config = ob.config.Config.load(SuperPointModelConfig, path)
        return cls(config)


@register_config("keypoint_model", name="SuperPoint", constructor=SuperPointModel)
@dataclass
class SuperPointModelConfig(Config):
    max_keypoints: int = 1024
