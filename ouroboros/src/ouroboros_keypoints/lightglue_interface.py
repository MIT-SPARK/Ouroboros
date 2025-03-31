from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from lightglue import LightGlue
from spark_config import Config, register_config

import ouroboros as ob


class LightglueModel:
    def __init__(self, config: LightglueModelConfig):
        self.model = LightGlue(features=config.feature_type).eval().cuda()
        self.returns_descriptors = True

    def infer(
        self, image0: ob.VlcImage, image1: ob.VlcImage, pose_hint: ob.VlcPose = None
    ):
        # Lightglue / Superpoint seem to expect (y,x) keypoints, but ours are storted in (x,y)
        kp0 = torch.Tensor(
            np.expand_dims(image0.keypoints[:, ::-1].copy(), axis=0)
        ).cuda()
        desc0 = torch.Tensor(np.expand_dims(image0.descriptors, axis=0)).cuda()
        kp1 = torch.Tensor(
            np.expand_dims(image1.keypoints[:, ::-1].copy(), axis=0)
        ).cuda()
        desc1 = torch.Tensor(np.expand_dims(image1.descriptors, axis=0)).cuda()

        with torch.no_grad():
            matches01 = self.model(
                {
                    "image0": {"keypoints": kp0, "descriptors": desc0},
                    "image1": {"keypoints": kp1, "descriptors": desc1},
                }
            )

        matches = matches01["matches"][0].cpu().numpy()

        m_kpts0, m_kpts1 = (
            image0.keypoints[matches[:, 0]],
            image1.keypoints[matches[:, 1]],
        )
        return m_kpts0, m_kpts1, matches

    @classmethod
    def load(cls, path):
        config = ob.config.Config.load(LightglueModelConfig, path)
        return cls(config)


@register_config("match_model", name="Lightglue", constructor=LightglueModel)
@dataclass
class LightglueModelConfig(Config):
    feature_type: str = "superpoint"
