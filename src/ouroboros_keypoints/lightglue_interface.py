from __future__ import annotations
from dataclasses import dataclass
from ouroboros.config import register_config, Config
from lightglue import LightGlue
import ouroboros as ob
import torch
import numpy as np


class LightglueModel:
    def __init__(self, config: LightglueModelConfig):
        self.model = LightGlue(features=config.feature_type).eval().cuda()
        self.returns_descriptors = True

    def infer(
        self, image0: ob.VlcImage, image1: ob.VlcImage, pose_hint: ob.VlcPose = None
    ):
        # Lightglue / Superpoint seem to expect (y,x) keypoints, but ours are storted in (x,y)
        kp0 = torch.Tensor(np.expand_dims(image0.keypoints[:, ::-1], axis=0)).cuda()
        desc0 = torch.Tensor(np.expand_dims(image0.descriptors, axis=0)).cuda()
        kp1 = torch.Tensor(np.expand_dims(image1.keypoints[:, ::-1], axis=0)).cuda()
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
        return m_kpts0, m_kpts1

    @classmethod
    def load(cls, path):
        config = ob.config.Config.load(LightglueModelConfig, path)
        return cls(config)


@register_config("match_model", name="Lightglue", constructor=LightglueModel)
@dataclass
class LightglueModelConfig(Config):
    feature_type: str = "superpoint"
