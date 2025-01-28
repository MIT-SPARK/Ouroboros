from lightglue import LightGlue
import ouroboros as ob
import torch
import numpy as np


class OuroborosLightGlueWrapper:
    def __init__(self, model):
        self.model = model
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


def get_lightglue_model():
    # Magically loads some weights from torchub
    model = LightGlue(features="superpoint").eval().cuda()
    return OuroborosLightGlueWrapper(model)
